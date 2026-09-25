#!/usr/bin/env python3
"""Run the batched, target-coupled TLAR experiment matrix on supplied prompts."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoTokenizer
from tlar_adaptive_tree import Config
from tlar_execution import PAPER_MODES, decode_batch
from benchmark_hf_adaptive_tree import load_model, prompt_ids


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/phase1_config.json')
    p.add_argument('--model-key',choices=['qwen3_8b_thinking','mistral_small_3_2_24b'],required=True)
    p.add_argument('--revisions',type=Path,required=True)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,32])
    p.add_argument('--lengths',type=int,nargs='+',default=[512,1024,2048,4096])
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--node-budgets',type=int,nargs='+',required=True,
                   help='Retrieval-node caps, chosen explicitly from the experiment configuration')
    p.add_argument('--temperature',type=float,default=.6)
    p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--warmup-tokens',type=int,default=640)
    a=p.parse_args()
    if a.repeats<2 or any(x<1 for x in a.batch_sizes+a.lengths+a.node_budgets) or a.warmup_tokens<1:
        p.error('Positive matrix settings and at least two repeats required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the timing matrix')
    rows=[json.loads(s) for s in a.prompts.read_text().splitlines() if s.strip()]
    keys=[str(r.get('trajectory_id',r.get('task_id',i))) for i,r in enumerate(rows)]
    if len(rows)!=100 or len(set(keys))!=100:raise ValueError('Expected 100 unique prompts')
    models=json.loads(a.config.read_text())['models'][a.model_key]
    pinned=json.loads(a.revisions.read_text())
    if (pinned['target'],pinned['draft'])!=(models['model_id'],models['draft_model_id']):
        raise ValueError('Checkpoint manifest differs from selected pair')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(pinned['target'],revision=pinned['target_revision'])
    draft_tok=AutoTokenizer.from_pretrained(pinned['draft'],revision=pinned['draft_revision'])
    if tok.get_vocab()!=draft_tok.get_vocab():raise ValueError('Target and draft require identical token IDs')
    target,ti=load_model(pinned['target'],pinned['target_revision'])
    draft,di=load_model(pinned['draft'],pinned['draft_revision'])
    prompts=[prompt_ids(tok,r) for r in rows]
    cfg=Config()
    root=Path(__file__).resolve().parents[1]
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        config=asdict(cfg),input_sha256=digest(a.prompts),request_ids=keys,
        checkpoints={'target':ti,'draft':di},gpu=torch.cuda.get_device_name(0),
        versions={k:importlib.metadata.version(k) for k in ('torch','transformers','accelerate')},
        python=platform.python_version(),precision='bfloat16',attention='eager',
        sampling='target-coupled inverse CDF',bonus_tokens=0,ignore_eos=True,
        timing='synchronized wall time including prefill, drafting, retrieval, verification, control and cache updates',
        node_budget='distinct retrieval prefixes; SmallDraft prefixes reserved separately',
        source_sha256={str(f.relative_to(root)):digest(f) for f in root.rglob('*.py') if '.venv' not in f.parts})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    settings=[(m,b if m=='budgeted_union' else 0) for m in PAPER_MODES
              for b in (a.node_budgets if m=='budgeted_union' else [0])]
    output_hashes = {}
    with (a.output/'measurements.jsonl').open('w') as measurements, (a.output/'outputs.jsonl').open('w') as outputs, (a.output/'rounds.jsonl').open('w') as events:
        for batch_size in a.batch_sizes:
            for length in a.lengths:
                for mode,cap in settings:
                    decode_batch(target,draft,prompts[:batch_size],a.warmup_tokens,mode,cfg,
                                 seed=a.seed,request_ids=keys[:batch_size],temperature=a.temperature,
                                 top_p=a.top_p,node_budget=cap or 1)
                for repeat in range(a.repeats):
                    references={}
                    ordered=settings[repeat%len(settings):]+settings[:repeat%len(settings)]
                    for mode,cap in ordered:
                        elapsed=0.;tokens=0
                        for start in range(0,len(prompts),batch_size):
                            group=prompts[start:start+batch_size];ids=keys[start:start+batch_size]
                            generated,rounds,seconds=decode_batch(target,draft,group,length,mode,cfg,
                                seed=a.seed,request_ids=ids,temperature=a.temperature,top_p=a.top_p,node_budget=cap or 1)
                            elapsed+=seconds;tokens+=sum(map(len,generated))
                            for key,out,rr in zip(ids,generated,rounds):
                                if key in references and references[key]!=out:
                                    raise RuntimeError(f'Coupled output mismatch: {key}, {mode}')
                                references[key]=out
                                out_hash=hashlib.sha256(json.dumps(out).encode()).hexdigest()
                                identity=(length,key)
                                if identity in output_hashes and output_hashes[identity]!=out_hash:
                                    raise RuntimeError(f'Cross-batch or repeated-run sequence mismatch: {key}')
                                output_hashes[identity]=out_hash
                                meta=dict(batch_size=batch_size,length=length,repeat=repeat,
                                          mode=mode,node_budget=cap,request_id=key)
                                outputs.write(json.dumps(dict(meta,tokens=out))+'\n')
                                for event in rr:events.write(json.dumps(dict(meta,**event))+'\n')
                        result=dict(model_key=a.model_key,batch_size=batch_size,length=length,repeat=repeat,
                                    mode=mode,node_budget=cap,output_tokens=tokens,elapsed_sec=elapsed,tokens_per_sec=tokens/elapsed)
                        measurements.write(json.dumps(result)+'\n');measurements.flush();outputs.flush();events.flush()
                        print(json.dumps(result),flush=True)
    (a.output/'COMPLETE.json').write_text(json.dumps({'completed':True,'measurements_sha256':digest(a.output/'measurements.jsonl')})+'\n')

if __name__=='__main__':main()
