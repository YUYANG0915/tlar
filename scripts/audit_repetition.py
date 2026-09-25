#!/usr/bin/env python3
"""Report repeated token blocks and apply an explicit reviewed exclusion list."""
import argparse
import hashlib
import json
from pathlib import Path


def repetition(tokens,max_period=128):
    best={'period':0,'copies':0,'start':0,'span':0}
    for period in range(1,min(max_period,len(tokens)//2)+1):
        run=0
        for i in range(period,len(tokens)):
            run=run+1 if tokens[i]==tokens[i-period] else 0
            copies=1+run//period
            span=copies*period
            if copies>=2 and span>best['span']:
                best=dict(period=period,copies=copies,start=i-span+1,span=span)
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exclude-ids',type=Path,help='Reviewed JSON list of trajectory IDs')
    p.add_argument('--cleaned-traces',type=Path)
    a=p.parse_args();rows=[json.loads(x) for x in a.traces.read_text().splitlines() if x.strip()]
    excluded=set(json.loads(a.exclude_ids.read_text())) if a.exclude_ids else set()
    ids={r['trajectory_id'] for r in rows}
    if not excluded<=ids:raise ValueError('Exclusion list contains foreign IDs')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'input_sha256':hashlib.sha256(a.traces.read_bytes()).hexdigest(),
        'trajectories':[dict(trajectory_id=r['trajectory_id'],tokens=len(r['generated_token_ids']),
            reviewed_exclusion=r['trajectory_id'] in excluded,**repetition(r['generated_token_ids'])) for r in rows]},indent=2)+'\n')
    if a.cleaned_traces:
        if a.exclude_ids is None:raise ValueError('Cleaning requires an explicit reviewed exclusion list')
        a.cleaned_traces.parent.mkdir(parents=True,exist_ok=True)
        a.cleaned_traces.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['trajectory_id'] not in excluded))

if __name__=='__main__':main()
