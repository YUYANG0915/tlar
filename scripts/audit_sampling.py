#!/usr/bin/env python3
"""Audit target-coupled sequences and tree logits on supplied held prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tlar_adaptive_tree import Config
from tlar_hf_tree import check_tree_logits
from tlar_execution import PAPER_MODES,decode_batch
from benchmark_hf_adaptive_tree import load_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefixes',type=Path,required=True,help='JSON list of held target-token prefixes')
    p.add_argument('--revisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--streams',type=int,default=100)
    p.add_argument('--prefix-count',type=int,default=100)
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--node-budget',type=int,required=True)
    p.add_argument('--temperature',type=float,default=.6)
    p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--atol',type=float,default=5e-2)
    p.add_argument('--rtol',type=float,default=1e-2)
    a=p.parse_args()
    if min(a.streams,a.prefix_count,a.tokens,a.batch_size,a.node_budget)<1:p.error('Positive audit settings required')
    prefixes=json.loads(a.prefixes.read_text())
    if len(prefixes)!=a.prefix_count or any(not x for x in prefixes):raise ValueError('Held prefix count differs from requested audit')
    pinned=json.loads(a.revisions.read_text());target,ti=load_model(pinned['target'],pinned['target_revision']);draft,di=load_model(pinned['draft'],pinned['draft_revision'])
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=Config(history_gate=0)
    max_error=0.;trials=0;mismatches=0
    with (a.output/'trials.jsonl').open('w') as f:
        for index,prefix in enumerate(prefixes):
            check=check_tree_logits(target,prefix,atol=a.atol,rtol=a.rtol)
            max_error=max(max_error,check['max_logit_error'])
            for start in range(0,a.streams,a.batch_size):
                ids=[f'{index}:{s}' for s in range(start,min(start+a.batch_size,a.streams))]
                group=[prefix]*len(ids)
                baseline,_,_=decode_batch(target,draft,group,a.tokens,'vanilla',cfg,request_ids=ids,
                                          temperature=a.temperature,top_p=a.top_p,node_budget=a.node_budget)
                for mode in PAPER_MODES:
                    actual,_,_=decode_batch(target,draft,group,a.tokens,mode,cfg,request_ids=ids,
                                            temperature=a.temperature,top_p=a.top_p,node_budget=a.node_budget)
                    for key,expected,observed in zip(ids,baseline,actual):
                        errors=sum(x!=y for x,y in zip(expected,observed))+abs(len(expected)-len(observed))
                        mismatches+=errors
                        f.write(json.dumps(dict(request_id=key,mode=mode,token_mismatches=errors,
                             expected=expected,observed=observed))+'\n')
                trials+=len(ids);f.flush()
    report=dict(trials_per_mode=trials,modes=PAPER_MODES,token_mismatches=mismatches,
                max_logit_error=max_error,passed=mismatches==0,
                input_sha256=hashlib.sha256(a.prefixes.read_bytes()).hexdigest(),
                checkpoints={'target':ti,'draft':di},
                settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if mismatches:raise RuntimeError('Shared-stream sequence agreement failed; inspect trials.jsonl')

if __name__=='__main__':main()
