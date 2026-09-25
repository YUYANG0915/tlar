#!/usr/bin/env python3
"""Run the fixed, grid, matched-budget, controller and expanded-domain protocols."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tlar_adaptive_tree import Config
from tlar_offline import (load_inputs,cache_inputs,fixed,replay,Policy,bootstrap,
                         matched_budgets,cost_matched,metrics,add_rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--draft-hits',type=Path,required=True)
    p.add_argument('--model-key',choices=['qwen3_8b_thinking','mistral_small_3_2_24b','llama31_8b_instruct'],required=True)
    p.add_argument('--domain',choices=['code_debug','math','open_ended'],required=True)
    p.add_argument('--snapshot',choices=['main','expanded'],default='main')
    p.add_argument('--experiment',choices=['fixed','grid','matched','controller','signals','matching','events'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    count=100 if a.domain=='code_debug' or a.snapshot=='expanded' else 24
    traces,hits=load_inputs(a.traces,a.draft_hits,count,a.model_key)
    if any(tr.get('dataset') != a.domain for tr in traces):
        raise ValueError('Trace domain differs from selected experiment')
    cfg=Config();results={};settings={}
    def save(name,rows,setting):
        results[name]=bootstrap(rows,a.bootstrap,a.seed)
        settings[name]=setting
    if a.experiment=='matched':
        for name,rows in matched_budgets(traces,hits,cfg,a.model_key,seed=a.seed).items():
            save(name,rows,{'counter':'distinct prefixes','budgets':[4,8]})
    elif a.experiment=='grid':
        for c in [2,4,8]:
            for e in [0,1,2]:
                for k in [2,4,8]:
                    conf=replace(cfg,context=c,edits=e,k_max=k)
                    cache=cache_inputs(traces,hits,conf)
                    save(f'c{c}_e{e}_k{k}',{tid:fixed(rows,conf) for tid,rows in cache.items()},asdict(conf))
    else:
        cache=cache_inputs(traces,hits,cfg)
        if a.experiment=='fixed':save('fixed',{tid:fixed(rows,cfg) for tid,rows in cache.items()},asdict(cfg))
        elif a.experiment in ('signals','matching'):
            reference={tid:replay(rows,cfg) for tid,rows in cache.items()}
            target=metrics(add_rows(reference.values()))['nodes_per_position']
            save('default_hit_ema',reference,asdict(Policy()))
            kinds=['random','periodic','last_hit','marginal_ema'] if a.experiment=='signals' else ['marginal_ema']
            for kind in kinds:
                policy,rows,cost=cost_matched(cache,cfg,kind,target,a.seed)
                save(kind,rows,dict(asdict(policy),target_nodes=target,actual_nodes=cost,cost_gap=cost-target))
            if a.experiment=='matching':
                exact_cfg=replace(cfg,edits=0);exact_cache=cache_inputs(traces,hits,exact_cfg)
                # Same reward, width and half-life; activation threshold matches node cost.
                policy,rows,cost=cost_matched(exact_cache,exact_cfg,'marginal_ema',target,a.seed)
                save('exact_marginal_ema',rows,dict(asdict(policy),edits=0,target_nodes=target,actual_nodes=cost,cost_gap=cost-target))
        elif a.experiment=='events':
            for event in [False,True]:
                save('event_driven' if event else 'position_wise',
                     {tid:replay(rows,cfg,event_driven=event) for tid,rows in cache.items()},asdict(Policy()))
        else:
            policies={'full':Policy(),'fixed':Policy(kind='fixed',adaptive_width=False,probes=False),
                      'activation_only':Policy(adaptive_width=False),'no_probes':Policy(probes=False)}
            for h in [8,16,32,64,128]:policies[f'H{h}']=Policy(half_life=h)
            for interval in [8,16,32,64,128]:policies[f'P{interval}']=Policy(probe_interval=interval)
            for name,policy in policies.items():
                save(name,{tid:replay(rows,cfg,policy) for tid,rows in cache.items()},asdict(policy))
    artifact={'experiment':a.experiment,'model_key':a.model_key,'domain':a.domain,'snapshot':a.snapshot,
              'problems':count,'bootstrap':{'resamples':a.bootstrap,'seed':a.seed,'unit':'trajectory'},
              'input_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('traces',a.traces),('draft_hits',a.draft_hits)]},
              'settings':settings,'results':results}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(artifact,indent=2)+'\n')
    print(a.output)

if __name__=='__main__':main()
