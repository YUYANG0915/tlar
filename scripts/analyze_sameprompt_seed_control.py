#!/usr/bin/env python3
"""Compare trajectory-local retrieval with another seed for the same prompt."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from compare_retrieval_baselines import (
    accepted_from_prefixes,
    candidate_prefixes,
    iter_jsonl,
)

Context = Tuple[int, ...]


def masked_keys(context: Sequence[int], max_edits: int) -> List[Context]:
    keys = [tuple(context)]
    if max_edits >= 1:
        for index in range(len(context)):
            keys.append(tuple(context[:index]) + (-1,) + tuple(context[index + 1:]))
    return keys


def build_approx_index(tokens: Sequence[int], context_n: int, max_edits: int):
    index: Dict[Context, List[int]] = defaultdict(list)
    for position in range(context_n, len(tokens)):
        context = tokens[position - context_n:position]
        for key in masked_keys(context, max_edits):
            index[key].append(position)
    return index


def candidates(index, context, topk: int, before: int | None = None, depth: int = 1, max_edits: int = 1) -> List[int]:
    positions = set()
    for key in masked_keys(context, max_edits):
        positions.update(index.get(key, []))
    ordered = sorted(positions, reverse=True)
    if before is not None:
        ordered = [position for position in ordered if position + depth <= before]
    return ordered[:topk]


def aggregate(rows: List[dict], sample: Sequence[str]) -> Dict[str, float]:
    by_traj: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_traj[row["trajectory_id"]].append(row)
    subset = [row for tid in sample for row in by_traj[tid]]
    n = len(subset)
    own = sum(row["own_accept"] for row in subset)
    control = sum(row["control_accept"] for row in subset)
    return {
        "n_positions": float(n),
        "own_extra_per_position": own / n,
        "same_prompt_other_seed_extra_per_position": control / n,
        "paired_delta_extra_per_position": (own - control) / n,
        "own_hit_rate": sum(row["own_accept"] > 0 for row in subset) / n,
        "same_prompt_other_seed_hit_rate": sum(
            row["control_accept"] > 0 for row in subset
        ) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-problems", type=int, default=24)
    parser.add_argument("--traces-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-n", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-edits", type=int, default=0)
    parser.add_argument("--history-gate", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-history", choices=("full", "prefix"), default="prefix")
    args = parser.parse_args()
    if args.max_edits not in (0, 1):
        raise ValueError("Supported edit tolerances: 0 and 1")

    traces = []
    for path in sorted(Path(args.traces_dir).glob("*.jsonl")):
        traces.extend(
            trace for trace in iter_jsonl(path) if trace["model_key"] == args.model_key
        )
    by_task_seed = {(trace["task_id"], int(trace["seed"])): trace for trace in traces}
    if len(by_task_seed) != len(traces):
        raise ValueError("Duplicate task/seed records")
    if len({tr['task_id'] for tr in traces}) != args.expected_problems:
        raise ValueError("Unexpected same-prompt problem count")
    seeds = sorted({int(trace["seed"]) for trace in traces})
    if len(seeds) != 2:
        raise ValueError(f"Expected exactly two seeds, found {seeds}")

    indices = {
        trace["trajectory_id"]: build_approx_index(
            trace["generated_token_ids"], args.context_n, args.max_edits
        )
        for trace in traces
    }
    rows: List[dict] = []
    for trace in traces:
        other_seed = seeds[1] if int(trace["seed"]) == seeds[0] else seeds[0]
        control = by_task_seed.get((trace["task_id"], other_seed))
        if control is None:
            raise ValueError(f"Missing paired seed for task {trace['task_id']}")
        target = trace["generated_token_ids"]
        source = control["generated_token_ids"]
        own_index = indices[trace["trajectory_id"]]
        control_index = indices[control["trajectory_id"]]
        for t in range(max(args.context_n, args.history_gate + 1), len(target) - 1):
            context = target[t - args.context_n:t]
            own_candidates = candidates(own_index, context, args.topk, before=t, depth=args.depth, max_edits=args.max_edits)
            control_candidates = candidates(control_index, context, args.topk,
                before=min(t, len(source)) if args.control_history == "prefix" else len(source),
                depth=args.depth, max_edits=args.max_edits)
            own_prefixes = candidate_prefixes(target[:t], own_candidates, args.depth)
            control_prefixes = candidate_prefixes(source, control_candidates, args.depth)
            rows.append({
                "trajectory_id": trace["trajectory_id"],
                "own_accept": accepted_from_prefixes(
                    set(own_prefixes), target, t, args.depth
                ),
                "control_accept": accepted_from_prefixes(
                    set(control_prefixes), target, t, args.depth
                ),
            })

    trajectory_ids = sorted({row["trajectory_id"] for row in rows})
    point = aggregate(rows, trajectory_ids)
    rng = random.Random(args.seed)
    boot: Dict[str, List[float]] = defaultdict(list)
    for _ in range(args.bootstrap):
        sample = [rng.choice(trajectory_ids) for _ in trajectory_ids]
        for metric, value in aggregate(rows, sample).items():
            boot[metric].append(value)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "point", "ci95_low", "ci95_high"]
        )
        writer.writeheader()
        for metric, value in point.items():
            values = np.asarray(boot[metric], dtype=float)
            writer.writerow({
                "metric": metric,
                "point": f"{value:.8f}",
                "ci95_low": f"{np.quantile(values, 0.025):.8f}",
                "ci95_high": f"{np.quantile(values, 0.975):.8f}",
            })
    print("saved:", output)


if __name__ == "__main__":
    main()
