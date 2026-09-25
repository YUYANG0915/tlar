#!/usr/bin/env python3
"""Prepare canonical, seeded prompt subsets from locally supplied dataset exports."""
import argparse
import hashlib
import json
from pathlib import Path
import random

MATH_TYPES={'Algebra','Counting & Probability','Geometry','Number Theory'}


def select(rows,domain,count,seed=20260217):
    normalized=[]
    for row in rows:
        if domain=='code_debug':
            key=str(row['instance_id'])
            prompt='Repository: '+row['repo']+'\nBase commit: '+row['base_commit']+'\n\nIssue:\n'+row['problem_statement']
        elif domain=='math':
            if row['type'] not in MATH_TYPES:continue
            key=str(row['canonical_id']);prompt=row['problem']
        else:
            key=str(row['canonical_id']);prompt=row['prompt']
        normalized.append({'task_id':key,'prompt':prompt,'dataset':domain})
    normalized.sort(key=lambda r:r['task_id'])
    if len({r['task_id'] for r in normalized})!=len(normalized):raise ValueError('Duplicate canonical identifiers')
    if count<1 or len(normalized)<count:raise ValueError('Insufficient eligible problems')
    random.Random(seed).shuffle(normalized)
    return normalized[:count]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--domain',choices=['code_debug','math','open_ended'],required=True)
    p.add_argument('--count',type=int,choices=[24,100],required=True)
    p.add_argument('--seed',type=int,default=20260217)
    p.add_argument('--dataset-revision',required=True,help='Dataset revision/hash recorded with the local export')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for s in a.input.read_text().splitlines() if s.strip()]
    chosen=select(rows,a.domain,a.count,a.seed)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in chosen))
    a.output.with_suffix('.manifest.json').write_text(json.dumps({
        'domain':a.domain,'problems':a.count,'selection_seed':a.seed,'dataset_revision':a.dataset_revision,
        'source_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
        'selected_ids':[r['task_id'] for r in chosen],
        'sampling':'canonical identifier sort, seeded shuffle, leading subset'},indent=2)+'\n')

if __name__=='__main__':main()
