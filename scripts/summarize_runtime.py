#!/usr/bin/env python3
"""Summarize measured runs with paired Student-t intervals."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean,median,stdev
from scipy.stats import t


def summarize(rows):
    groups=defaultdict(dict)
    for r in rows:
        key=(r['model_key'],r['batch_size'],r['length'],r['mode'],r['node_budget'])
        if r['repeat'] in groups[key]:raise ValueError('Duplicate measured run')
        seconds=float(r['elapsed_sec']);tokens=int(r['output_tokens'])
        if seconds<=0 or tokens<=0:raise ValueError('Invalid measured duration or output count')
        if tokens!=100*int(r['length']):raise ValueError('Expected 100 complete fixed-length outputs')
        groups[key][r['repeat']]=tokens/seconds
    result=[]
    for key,runs in sorted(groups.items()):
        reference=groups.get((*key[:3],'small_draft',0))
        if reference is None or set(reference)!=set(runs):raise ValueError('Complete paired SmallDraft runs required')
        if len(runs)<2:raise ValueError('At least two repeated measurements required')
        values=list(runs.values());gains=[100*(runs[i]/reference[i]-1) for i in sorted(runs)]
        half=float(t.ppf(.975,len(gains)-1))*stdev(gains)/len(gains)**.5
        result.append(dict(zip(('model_key','batch_size','length','mode','node_budget'),key),
            repeats=len(runs),median_tokens_per_sec=median(values),cv_percent=100*stdev(values)/mean(values),
            paired_mean_gain_percent=mean(gains),ci95_low_percent=mean(gains)-half,ci95_high_percent=mean(gains)+half))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measurements',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(s) for f in a.measurements for s in f.read_text().splitlines() if s.strip()]
    out=summarize(rows)
    if not out:raise ValueError('Empty measurements')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)

if __name__=='__main__':main()
