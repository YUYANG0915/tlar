#!/usr/bin/env python3
"""Rebuild causal plans and controller state from recorded execution events."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
from tlar_adaptive_tree import AdaptiveTree,Config
from tlar_execution import compose_plan, observe_sources


def audit(manifest,rows):
    cfg=Config(**manifest['config']);seed=manifest['arguments']['seed']
    histories=defaultdict(list);controllers={};events=0
    for row in rows:
        key=(row['batch_size'],row['length'],row['repeat'],row['mode'],row['node_budget'],row['request_id'])
        history=histories[key];controller=controllers.setdefault(key,AdaptiveTree(cfg))
        if len(history)!=row['history_length']:raise ValueError('Event history boundary differs')
        plan=compose_plan(controller,row['request_id'],history,row['small'],row['mode'],row['node_budget'] or 1,seed)
        if len(plan.nodes)!=row['tree_nodes'] or [c.start for c in plan.candidates]!=row['candidate_starts']:
            raise ValueError('Rebuilt candidate tree differs from execution')
        emitted=row['emitted'];accepted=row['accepted_nodes']
        if len(accepted)>len(emitted):raise ValueError('Accepted path exceeds emitted tokens')
        parent=-1
        for offset,i in enumerate(accepted):
            node=plan.nodes[i]
            if node.parent!=parent or node.prefix[-1]!=emitted[offset]:raise ValueError('Accepted node path differs')
            parent=i
        if len(accepted)==cfg.depth and len(emitted)!=cfg.depth:raise ValueError('Full-depth event includes extra tokens')
        if row['mode'] in ('tlar','budgeted_union'):
            selected={n.prefix for n in plan.nodes if n.retrieval}
            observed=replace(plan,candidates=tuple(c for c in plan.candidates if (c.tokens[0],) in selected))
            controller.pending[row['request_id']]=observed
            feedback=controller.observe(observed,emitted)
            for name in ('rho_before','rho_after','width','active','probe'):
                if feedback[name]!=row[name]:raise ValueError('Replayed controller state differs: '+name)
        else:controller.finish(row['request_id'])
        observe_sources(controller,row['request_id'],emitted)
        history.extend(emitted);events+=1
    if not events:raise ValueError('Empty event log')
    return dict(passed=True,events=events,requests=len(histories),committed_tokens=sum(map(len,histories.values())))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text())
    with (a.run/'rounds.jsonl').open() as f:report=audit(manifest,(json.loads(s) for s in f if s.strip()))
    a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
