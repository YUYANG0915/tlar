#!/usr/bin/env python3
"""Collect CPU/CUDA timelines and memory counters for the systems-analysis framework."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
import tlar_execution as execution
from tlar_adaptive_tree import Config
from benchmark_hf_adaptive_tree import load_model,prompt_ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts',type=Path,required=True);p.add_argument('--revisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,choices=[1,32],default=1)
    p.add_argument('--length',type=int,default=1024);p.add_argument('--node-budget',type=int,required=True)
    p.add_argument('--mode',choices=execution.PAPER_MODES,default='budgeted_union');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for CPU/GPU timeline collection')
    pinned=json.loads(a.revisions.read_text());rows=[json.loads(x) for x in a.prompts.read_text().splitlines() if x.strip()]
    if len(rows)<a.batch_size:raise ValueError('Insufficient prompts for requested batch')
    tok=AutoTokenizer.from_pretrained(pinned['target'],revision=pinned['target_revision'])
    draft_tok=AutoTokenizer.from_pretrained(pinned['draft'],revision=pinned['draft_revision'])
    if tok.get_vocab()!=draft_tok.get_vocab():raise ValueError('Target and draft token IDs differ')
    target,ti=load_model(pinned['target'],pinned['target_revision']);draft,di=load_model(pinned['draft'],pinned['draft_revision'])
    prompts=[prompt_ids(tok,r) for r in rows[:a.batch_size]]
    kw=dict(seed=a.seed,node_budget=a.node_budget)
    execution.decode_batch(target,draft,prompts,640,a.mode,Config(),**kw)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();execution.PROFILE_RANGES=True
    a.output.mkdir(parents=True,exist_ok=False)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                               profile_memory=True,record_shapes=True) as profile:
        generated,events,seconds=execution.decode_batch(target,draft,prompts,a.length,a.mode,Config(),**kw)
    execution.PROFILE_RANGES=False
    profile.export_chrome_trace(str(a.output/'timeline.json'))
    flat=[e for request in events for e in request]
    report=dict(gpu=torch.cuda.get_device_name(0),mode=a.mode,node_budget=a.node_budget,
        batch_size=a.batch_size,output_length=a.length,checkpoints={'target':ti,'draft':di},
        profiled_elapsed_seconds=seconds,committed_tokens=sum(map(len,generated)),request_events=len(flat),
        mean_committed_tokens_per_request_event=sum(map(len,generated))/len(flat),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        tree_nodes=sum(e['tree_nodes'] for e in flat),timing_scope='instrumented diagnostic run')
    (a.output/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'operators.txt').write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=100))

if __name__=='__main__':main()
