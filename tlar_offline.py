"""Paper-aligned offline metrics, budget comparisons and causal controller replay."""
from collections import defaultdict
from dataclasses import dataclass, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from evaluate_adaptive_controller import build_position_cache, read_draft_hits
from compare_retrieval_baselines import (stand_tree_prefixes, round_robin_unique_prefixes,
    accepted_from_prefixes, stable_int_seed)
from tlar_adaptive_tree import Config, HistoryIndex


def load_inputs(traces_path, hits_path, count, model_key=None):
    traces = [json.loads(s) for s in Path(traces_path).read_text().splitlines() if s.strip()]
    ids = [r['trajectory_id'] for r in traces]
    if len(traces) != count or len(set(ids)) != count:
        raise ValueError('Expected exactly the configured number of unique trajectories')
    if len({r['task_id'] for r in traces}) != count:
        raise ValueError('Each problem must contribute one trajectory to this evaluation')
    if model_key and any(r['model_key'] != model_key for r in traces):
        raise ValueError('Input model differs from the selected experiment')
    expected = {}
    for tr in traces:
        tokens = tr['generated_token_ids']
        if len(tokens) < 6 or any(type(x) is not int or x < 0 for x in tokens):
            raise ValueError('Expected nonnegative integer tokens and at least six generated tokens')
        expected.update({(tr['trajectory_id'], i): x for i, x in enumerate(tokens)})
    seen = set()
    with Path(hits_path).open() as f:
        for row in csv.DictReader(f):
            key = (row['trajectory_id'], int(row['token_index']))
            if key in seen or key not in expected or int(row['target_token_id']) != expected[key]:
                raise ValueError('Duplicate, foreign or token-misaligned draft record')
            if int(row['draft_top1_hit']) not in (0, 1):
                raise ValueError('Draft hit must be binary')
            seen.add(key)
    if seen != set(expected):
        raise ValueError('Draft records must cover every generated token')
    return traces, read_draft_hits(Path(hits_path))


def add_rows(rows):
    out = defaultdict(float)
    for row in rows:
        for k, v in row.items(): out[k] += v
    return dict(out)


def metrics(s):
    def ratio(a, b): return s.get(a, 0) / s[b] if s.get(b, 0) else 0.
    return dict(recovery=ratio('recovered_misses','active_misses'),
                G_miss=ratio('miss_tokens','positions'),
                delta=ratio('gain','positions'),
                small_acceptance=ratio('small','positions'),
                nodes_per_position=ratio('nodes','positions'),
                active_fraction=ratio('active','positions'),
                gain_per_node=ratio('gain','nodes'),
                events=s.get('events',0), positions=s.get('positions',0))


def record(s, row, accepted, nodes, active):
    s['small'] += row.small_len
    s['nodes'] += nodes
    s['gain'] += max(row.small_len, accepted) - row.small_len
    s['active'] += int(active)
    s['events'] += 1
    if row.draft_hit == 0 and active:
        s['active_misses'] += 1
        s['recovered_misses'] += int(accepted > 0)
        s['miss_tokens'] += accepted


def bootstrap(by_trajectory, resamples=2000, seed=0):
    if resamples < 1 or not by_trajectory: raise ValueError('Positive resamples and trajectories required')
    ids = sorted(by_trajectory)
    point = metrics(add_rows(by_trajectory.values()))
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(resamples):
        m = metrics(add_rows(by_trajectory[rng.choice(ids)] for _ in ids))
        for key, value in m.items(): samples[key].append(value)
    return {key: {'point':value,'ci95_low':float(np.quantile(samples[key],.025)),
                 'ci95_high':float(np.quantile(samples[key],.975))} for key,value in point.items()}


def cache_inputs(traces, hits, cfg):
    return {tr['trajectory_id']:build_position_cache(tr,hits,cfg.context,cfg.edits,cfg.depth,cfg.k_max)
            for tr in traces}


def fixed(rows, cfg):
    s=defaultdict(float, positions=len(rows))
    for row in rows:
        active=row.token_index>cfg.history_gate
        k=cfg.k_max if active else 0
        # Every eligible historical continuation has exactly cfg.depth tokens.
        # Fixed-candidate table counts branch lengths, including shared prefixes.
        record(s,row,row.accepted[k],row.branches[k] if row.branches else row.nodes[k],active)
    return dict(s)


@dataclass(frozen=True)
class Policy:
    kind: str = 'hit_ema'
    threshold: float = .05
    half_life: float = 32
    probe_interval: int = 32
    probability: float = .1
    period: int = 32
    adaptive_width: bool = True
    probes: bool = True


def replay(rows, cfg, policy=Policy(), seed=0, event_driven=False):
    """Use outcomes after each decision; event jumps commit max accepted + correction."""
    s=defaultdict(float,positions=len(rows));rho=0.;last=False;next_position=-1
    rng=random.Random(seed);decay=2**(-1/policy.half_life)
    previous=None
    for row in rows:
        t=row.token_index
        if event_driven and t<next_position: continue
        if previous is not None: rho*=decay**max(0,t-previous-1)
        eligible=t>cfg.history_gate
        first=t==cfg.history_gate+1
        probe=eligible and policy.probes and (t-cfg.history_gate-1)%policy.probe_interval==0
        if policy.kind=='fixed': active=eligible
        elif policy.kind=='random': active=eligible and rng.random()<policy.probability
        elif policy.kind=='periodic': active=eligible and (t-cfg.history_gate-1)%policy.period==0
        elif policy.kind=='last_hit': active=eligible and (last or probe or first)
        else: active=eligible and (rho>=policy.threshold or last or probe or first)
        if active:
            k=cfg.k_min if probe else (min(cfg.k_max,max(cfg.k_min,math.ceil(cfg.k_max*rho)))
                if policy.adaptive_width else cfg.k_max)
        else:k=0
        accepted=row.accepted[k];nodes=row.nodes[k]
        record(s,row,accepted,nodes,active)
        reward=max(row.small_len,accepted)-row.small_len if policy.kind=='marginal_ema' else int(accepted>0)
        rho=decay*rho+(1-decay)*reward
        if active:last=accepted>0
        previous=t
        if event_driven:
            a=max(row.small_len,accepted)
            step=a+1 if a<cfg.depth else cfg.depth
            # Position-clock probes match the online backend; retain their boundaries.
            if policy.probes:
                first_probe=cfg.history_gate+1
                next_probe=first_probe if t<first_probe else first_probe+((t-first_probe)//policy.probe_interval+1)*policy.probe_interval
                step=min(step,next_probe-t)
            next_position=t+step
    return dict(s)


def cost_matched(caches,cfg,kind,target,seed=0):
    """Select a setting solely by aggregate node-cost distance from the reference."""
    if kind=='random': options=[Policy(kind=kind,probability=x/100,adaptive_width=False,probes=False) for x in range(101)]
    elif kind=='periodic': options=[Policy(kind=kind,period=x,adaptive_width=False,probes=False) for x in range(1,129)]
    elif kind=='last_hit': options=[Policy(kind=kind,probe_interval=x,adaptive_width=False) for x in range(1,129)]
    else: options=[Policy(kind=kind,threshold=float(x)) for x in np.linspace(0,cfg.depth,161)]
    best=None
    for index,p in enumerate(options):
        result={tid:replay(rows,cfg,p,stable_int_seed(seed,tid)) for tid,rows in caches.items()}
        cost=metrics(add_rows(result.values()))['nodes_per_position']
        key=(abs(cost-target),index)
        if best is None or key<best[0]:best=(key,p,result,cost)
    return best[1],best[2],best[3]


def matched_budgets(traces,hits,cfg,model_key,budgets=(4,8),seed=0):
    result=defaultdict(dict)
    strongest='suffix' if model_key=='llama31_8b_instruct' else 'stand'
    for tr in traces:
        tid=tr['trajectory_id'];tokens=tr['generated_token_ids']
        rows=build_position_cache(tr,hits,cfg.context,cfg.edits,cfg.depth,cfg.k_max)
        approx=HistoryIndex(cfg);exact=HistoryIndex(replace(cfg,edits=0))
        totals=defaultdict(lambda:defaultdict(float,positions=len(rows)))
        scores={b:{'tlar':0.,strongest:0.} for b in budgets}
        for row in rows:
            t=row.token_index;history=tokens[:t];eligible=t>cfg.history_gate
            candidates=approx.query(history,cfg.k_max if eligible else 0)
            paths=list(dict.fromkeys(c.tokens[:n] for c in candidates for n in range(1,cfg.depth+1)))
            if strongest=='suffix':
                cs=exact.query(history,cfg.k_max if eligible else 0)
                other=list(dict.fromkeys(c.tokens[:n] for c in cs for n in range(1,cfg.depth+1)))
            else:
                other=stand_tree_prefixes(history,t,cfg.k_max,cfg.depth,2,8,1.,stable_int_seed(seed,tid,t,'stand')) if eligible else []
            for b in budgets:
                first,second=(other,paths) if scores[b][strongest]>=scores[b]['tlar'] else (paths,other)
                groups={'tlar':paths[:b],strongest:other[:b],
                        'union':round_robin_unique_prefixes(first,second,b)}
                for name,ps in groups.items():
                    a=accepted_from_prefixes(set(ps),tokens,t,cfg.depth)
                    record(totals[f'{name}_B{b}'],row,a,len(ps),eligible)
                # Update each budget's independent history after its decision.
                selected=set(groups['union'])
                for name,ps in [('tlar',paths),(strongest,other)]:
                    a=accepted_from_prefixes(set(ps)&selected,tokens,t,cfg.depth)
                    decay=2**(-1/cfg.half_life)
                    scores[b][name]=decay*scores[b][name]+(1-decay)*int(a>0)
        for name,s in totals.items():result[name][tid]=dict(s)
    return dict(result)
