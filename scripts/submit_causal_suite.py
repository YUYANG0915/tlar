#!/usr/bin/env python3
"""Prepare/submit isolated full-data offline reruns. Never submits a GPU claim."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

MODELS = ("qwen3_8b_thinking", "llama31_8b_instruct", "mistral_small_3_2_24b")
DOMAINS = {"code_debug": 100, "math": 24, "open_ended": 24}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def select_file(root, name, expected=None):
    candidates = sorted(root.rglob(name))
    valid = []
    for path in candidates:
        if expected is not None:
            with path.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
            if len(rows) != expected:
                continue
        valid.append(path)
    hashes = {digest(p) for p in valid}
    if not valid or len(hashes) != 1:
        raise ValueError(f"Missing or ambiguous input {name}: {valid}. "
                         "Use --inputs with explicit paths; do not guess a dataset.")
    return valid[0].resolve()


def select_draft(root, name, trace):
    with Path(trace).open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    expected = {(r['trajectory_id'], i): tok for r in rows
                for i, tok in enumerate(r['generated_token_ids'])}
    valid = []
    for path in sorted(root.rglob(name)):
        with path.open() as f:
            hits = list(csv.DictReader(f))
        observed = {(r['trajectory_id'], int(r['token_index'])): int(r['target_token_id']) for r in hits}
        if len(hits) == len(observed) and observed == expected:
            valid.append(path)
    if not valid or len({digest(p) for p in valid}) != 1:
        raise ValueError(f'Missing or ambiguous token-matched draft hits for {trace}: {valid}; use --inputs')
    return valid[0].resolve()


def validate_inputs(entries, models=MODELS, domains=DOMAINS):
    required = set(itertools.product(models, domains))
    if {(e['model'], e['domain']) for e in entries} != required or len(entries) != len(required):
        raise ValueError("Inputs must exactly match the requested models and domains")
    for e in entries:
        trace, draft = Path(e['trace']).resolve(), Path(e['draft']).resolve()
        with trace.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        ids = {r['trajectory_id'] for r in rows}
        if len(rows) != DOMAINS[e['domain']] or len(ids) != len(rows):
            raise ValueError(f"Wrong count or duplicate trajectory IDs: {trace}")
        lengths = {r['trajectory_id']: len(r['generated_token_ids']) for r in rows}
        token_rows = {r['trajectory_id']: r['generated_token_ids'] for r in rows}
        seen = set()
        draft_ids = set()
        with draft.open() as f:
            for r in csv.DictReader(f):
                tid, t = r['trajectory_id'], int(r['token_index'])
                if tid not in lengths or not 0 <= t < lengths[tid] or (tid, t) in seen:
                    raise ValueError(f'Invalid/duplicate draft position: {draft}: {(tid, t)}')
                if int(r['target_token_id']) != token_rows[tid][t]:
                    raise ValueError(f'Draft hit tokens do not match trace: {draft}: {(tid, t)}')
                seen.add((tid, t))
                draft_ids.add(tid)
        if len(seen) != sum(lengths.values()):
            raise ValueError(f'Incomplete draft-hit coverage: {draft}')
        if draft_ids != ids or any(n == 0 for n in lengths.values()):
            raise ValueError(f"Draft/trace IDs differ or empty generation: {trace}")
        e.update(trace=str(trace), draft=str(draft), trace_sha256=digest(trace), draft_sha256=digest(draft))


def task_list(entries):
    tasks = []
    for i, e in enumerate(entries):
        for kind in ('baseline', 'plugin', 'adaptive', 'truncation'):
            tasks.append(dict(input=i, kind=kind))
        if e['domain'] == 'code_debug':
            for c, eps, k in itertools.product((2, 4, 8), (0, 1, 2), (2, 4, 8)):
                tasks.append(dict(input=i, kind='grid', context=c, edits=eps, topk=k))
    return tasks


def execute(manifest_path, stage, index):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest['snapshot'])
    for name, expected in manifest['source_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f"Snapshot changed: {name}")
    task = {'input': index, 'kind': 'audit'} if stage == 'audit' else manifest['tasks'][index]
    entry = manifest['inputs'][task['input']]
    trace, draft = entry['trace'], entry['draft']
    if digest(trace) != entry['trace_sha256'] or digest(draft) != entry['draft_sha256']:
        raise ValueError('Input changed after submission')
    out = Path(manifest['output']) / f"{entry['domain']}_{entry['model']}"
    out.mkdir(exist_ok=True)
    folder = out / (f"grid_c{task['context']}_e{task['edits']}_k{task['topk']}" if task['kind'] == 'grid' else task['kind'])
    folder.mkdir(exist_ok=False)
    def run(script, *args):
        command = [manifest['python'], str(root / 'scripts' / script), *map(str, args)]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True, cwd=root)
    common = ['--traces', trace, '--draft-hits', draft]
    kind = task['kind']
    if kind in ('audit', 'grid'):
        run('audit_section4_causality.py', '--traces', trace, '--output-dir', folder / 'provenance',
            '--limit', 0, '--max-positions', 0, '--context-n', task.get('context', 4),
            '--approx-max-edits', task.get('edits', 1), '--topk', task.get('topk', 4))
    if kind in ('baseline', 'grid'):
        run('compare_retrieval_baselines.py', *common, '--output', folder / 'summary.csv',
            '--context-n', task.get('context', 4), '--approx-max-edits', task.get('edits', 1),
            '--topk', task.get('topk', 4))
    elif kind == 'plugin':
        run('evaluate_tlar_plugin.py', *common, '--output', folder / 'summary.csv')
    elif kind == 'adaptive':
        run('evaluate_adaptive_controller.py', *common, '--output', folder / 'per_token_replay.csv',
            '--output-decay', folder / 'hit_correlation.csv')
    elif kind == 'truncation':
        filtered, filtered_draft = folder / 'uncapped.jsonl', folder / 'uncapped_draft.csv'
        run('filter_capped_trajectories.py', *common, '--output-traces', filtered,
            '--output-draft-hits', filtered_draft, '--cap', 4096)
        if filtered.stat().st_size == 0:
            raise ValueError('Uncapped trajectory set is empty; record NA')
        run('analyze_selfcopy.py', '--input', filtered, '--output', folder / 'selfcopy.csv',
            '--context-n', 4, '--max-accepted-len', 32, '--approx-max-edits', 1, '--seed', 0)
        run('compare_retrieval_baselines.py', '--traces', filtered, '--draft-hits', filtered_draft,
            '--output', folder / 'summary.csv')
    (folder / 'DONE.json').write_text(json.dumps({'task': task, 'status': 'completed',
        'scope': 'offline candidate and controller analysis'}) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--stage', choices=['audit', 'experiments'])
    p.add_argument('--index', type=int)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--inputs', type=Path, help='JSON list with model, domain, trace, draft')
    p.add_argument('--output', type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--partition', help='Scheduler partition; defaults to the cluster configuration')
    p.add_argument('--time', default='1-00:00:00')
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--domains', nargs='+', choices=DOMAINS, default=list(DOMAINS))
    a = p.parse_args()
    if a.worker:
        execute(a.worker, a.stage, a.index)
        return
    if not a.output or not (a.inputs or a.data_root) or a.concurrency < 1:
        p.error('Provide --output and either --data-root or --inputs; concurrency must be positive')
    if a.output.exists():
        p.error('Output directory already exists; choose a new run directory')
    if a.inputs:
        entries = json.loads(a.inputs.read_text())
    else:
        entries = []
        for m in a.models:
            for d in a.domains:
                trace = select_file(a.data_root, f'{d}_{m}.jsonl', DOMAINS[d])
                draft = select_draft(a.data_root, f'{d}_{m}_draft_hits.csv', trace)
                entries.append(dict(model=m, domain=d, trace=str(trace), draft=str(draft)))
    validate_inputs(entries, a.models, a.domains)
    py = str(Path(a.python).absolute())  # Do not resolve the virtualenv symlink.
    subprocess.run([py, '-c', 'import numpy,tqdm,transformers,datasets; print("Dependencies OK")'], check=True)
    root = Path(__file__).resolve().parents[1]
    subprocess.run([py, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_section4_causality.py'], cwd=root, check=True)
    subprocess.run([py, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_tlar_adaptive_tree.py'], cwd=root, check=True)
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'code'
    for name in ('scripts', 'configs'):
        shutil.copytree(root / name, snapshot / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root / 'tlar_adaptive_tree.py', snapshot)
    hashes = {str(f.relative_to(snapshot)): digest(f) for f in snapshot.rglob('*') if f.is_file()}
    manifest = dict(inputs=entries, tasks=task_list(entries), python=py, output=str(out),
                    snapshot=str(snapshot), source_sha256=hashes, execution_kind='offline_analysis')
    path = out / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Prepared {len(entries)} audits + {len(manifest['tasks'])} offline tasks: {path}")
    if not a.submit:
        print('Not submitted. Use a new output directory with --submit to submit.')
        return
    jobs = {}
    for stage, count in [('audit', len(entries)), ('experiments', len(manifest['tasks']))]:
        command = [py, str(snapshot / 'scripts/submit_causal_suite.py'), '--worker', str(path), '--stage', stage]
        wrap = shlex.join(command) + ' --index "$SLURM_ARRAY_TASK_ID"'
        wrap = 'source ' + shlex.quote(str(snapshot / 'scripts/configure_cache.sh')) + ' && ' + wrap
        wrap = 'bash -c ' + shlex.quote(wrap)
        cmd = ['sbatch', '--parsable', '--job-name=causal_' + stage,
               '--cpus-per-task=4', '--mem=32G', '--time=' + a.time, '--export=ALL',
               f'--array=0-{count - 1}%{a.concurrency}', '--chdir=' + str(snapshot),
               '--output=' + str(out / (stage + '_%A_%a.out')),
               '--error=' + str(out / (stage + '_%A_%a.err'))]
        if a.partition:
            cmd += ['--partition=' + a.partition]
        if stage == 'experiments':
            cmd += ['--dependency=afterok:' + jobs['audit']]
        result = subprocess.run(cmd + ['--wrap', wrap], text=True, capture_output=True, check=True)
        jid = result.stdout.strip().split(';')[0]
        if not jid.isdigit():
            raise ValueError(f'Unexpected sbatch response: {result.stdout}')
        jobs[stage] = jid
        (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
        print(f'{stage.upper()}={jid}', flush=True)
    print('sacct -X -j ' + ','.join(jobs.values()) + ' --format=JobID%24,JobName,State,ExitCode,Elapsed')


if __name__ == '__main__':
    main()
