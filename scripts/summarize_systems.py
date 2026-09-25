#!/usr/bin/env python3
"""Derive event accounting and paired time savings from measured runtime records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(measurements,events):
    def key(r):return tuple(r[k] for k in ['model_key','batch_size','length','repeat','mode','node_budget'])
    # Event records inherit the model from their run's measurements.
    models={r['model_key'] for r in measurements}
    if len(models)!=1:raise ValueError('Use one run directory per target')
    model=next(iter(models));counts=defaultdict(lambda:[0,0]);seen=set()
    for e in events:
        identity=(e['batch_size'],e['length'],e['repeat'],e['mode'],e['node_budget'],e['request_id'],e['history_length'])
        if identity in seen:raise ValueError('Duplicate executed event')
        seen.add(identity)
        k=key(dict(e,model_key=model));counts[k][0]+=1;counts[k][1]+=len(e['emitted'])
    indexed={key(r):r for r in measurements}
    if len(indexed)!=len(measurements):raise ValueError('Duplicate runtime measurement')
    out=[]
    for k,row in sorted(indexed.items()):
        ref=indexed.get((*k[:4],'small_draft',0))
        if ref is None:raise ValueError('Missing paired SmallDraft duration')
        event_count,tokens=counts[k]
        if tokens!=row['output_tokens'] or not event_count:raise ValueError('Events and measured output count differ')
        baseline=float(ref['elapsed_sec']);elapsed=float(row['elapsed_sec'])
        if elapsed<=0 or baseline<=0:raise ValueError('Positive measured durations required')
        out.append(dict(row,request_events=event_count,committed_per_request_event=tokens/event_count,
                        seconds_per_request_event=elapsed/event_count,
                        saved_seconds=baseline-elapsed,time_saving_percent=100*(1-elapsed/baseline),
                        throughput_ratio=baseline/elapsed))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    read=lambda name:[json.loads(x) for x in (a.run/name).read_text().splitlines() if x.strip()]
    result=summarize(read('measurements.jsonl'),read('rounds.jsonl'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
