#!/usr/bin/env python3
"""Validate a local input manifest and build or execute the paper's offline matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlar_offline import load_inputs
MODELS=('qwen3_8b_thinking','mistral_small_3_2_24b','llama31_8b_instruct')


def plan(entries,output,python):
    jobs=[];seen=set()
    for e in entries:
        identity=(e['snapshot'],e['domain'],e['model_key'])
        if identity in seen:raise ValueError('Duplicate input group')
        seen.add(identity)
        if e['model_key'] not in MODELS or e['snapshot'] not in ('main','expanded') or e['domain'] not in ('code_debug','math','open_ended'):
            raise ValueError('Unknown paper group')
        if e['snapshot']=='expanded' and e['domain']=='code_debug':raise ValueError('Expanded snapshot covers math and writing')
        count=100 if e['domain']=='code_debug' or e['snapshot']=='expanded' else 24
        load_inputs(e['traces'],e['draft_hits'],count,e['model_key'])
        folder=Path(output)/'_'.join(identity)
        experiments=['fixed']
        if e['snapshot']=='main' and e['domain']=='code_debug':
            experiments+=['matched','grid','controller','signals','matching','events']
        for name in experiments:
            jobs.append([python,'scripts/run_offline.py','--traces',e['traces'],'--draft-hits',e['draft_hits'],
                         '--model-key',e['model_key'],'--domain',e['domain'],'--snapshot',e['snapshot'],
                         '--experiment',name,'--output',str(folder/(name+'.json'))])
        if e['snapshot']=='main':
            source=e['traces']
            if e.get('source_exclude_ids'):
                source=str(folder/'source_cleaned.jsonl')
                jobs.append([python,'scripts/audit_repetition.py','--traces',e['traces'],
                             '--exclude-ids',e['source_exclude_ids'],'--cleaned-traces',source,
                             '--output',str(folder/'repetition_audit.json')])
            jobs.append([python,'scripts/analyze_selfcopy.py','--input',source,'--output',str(folder/'source.csv')])
            jobs.append([python,'scripts/summarize_sources.py','--input',str(folder/'source.csv'),'--output',str(folder/'source_summary.json')])
        if e['snapshot']=='main' and e['domain']=='code_debug':
            for script in ['compare_retrieval_baselines','evaluate_tlar_plugin']:
                jobs.append([python,f'scripts/{script}.py','--traces',e['traces'],'--draft-hits',e['draft_hits'],
                             '--output',str(folder/(script+'.csv'))])
            jobs.append([python,'scripts/evaluate_adaptive_controller.py','--traces',e['traces'],
                         '--draft-hits',e['draft_hits'],'--output',str(folder/'controller_legacy.csv'),
                         '--output-decay',str(folder/'temporal_dependence.csv')])
            jobs.append([python,'scripts/audit_section4_causality.py','--traces',e['traces'],
                         '--output-dir',str(folder/'causality'),'--limit','0','--max-positions','0'])
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--require-complete',action='store_true')
    a=p.parse_args();entries=json.loads(a.inputs.read_text())
    if a.require_complete:
        expected={(s,d,m) for s,ds in [('main',['code_debug','math','open_ended']),('expanded',['math','open_ended'])] for d in ds for m in MODELS}
        if {(e['snapshot'],e['domain'],e['model_key']) for e in entries}!=expected:
            raise ValueError('Complete matrix requires 15 model/domain/snapshot groups')
    jobs=plan(entries,a.output,sys.executable)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'plan.json').write_text(json.dumps({'jobs':jobs,'input_manifest_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest()},indent=2)+'\n')
    if a.execute:
        root=Path(__file__).resolve().parents[1]
        for job in jobs:subprocess.run(job,cwd=root,check=True)
    print(f'{len(jobs)} commands; executed={a.execute}')

if __name__=='__main__':main()
