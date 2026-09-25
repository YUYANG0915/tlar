#!/usr/bin/env python3
"""Trajectory-cluster confidence intervals for the four retrieval sources."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();groups=defaultdict(lambda:defaultdict(list))
    with a.input.open() as f:
        for r in csv.DictReader(f):
            groups[(r['dataset'],r['model_key'],r['control_type'])][r['trajectory_id']].append(float(r['exact_accepted_length']))
    result=[]
    for (domain,model,source),by_id in sorted(groups.items()):
        ids=sorted(by_id);totals={k:(sum(v),len(v)) for k,v in by_id.items()};rng=random.Random(a.seed)
        def value(sample):return sum(totals[k][0] for k in sample)/sum(totals[k][1] for k in sample)
        draws=[value([rng.choice(ids) for _ in ids]) for _ in range(a.bootstrap)]
        result.append(dict(domain=domain,model=model,source=source,point=value(ids),
                           ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975))))
    if not result:raise ValueError('Empty source measurements')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
