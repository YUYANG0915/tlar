#!/usr/bin/env python3
"""Trajectory-level bootstrap summaries for tree and hybrid proxy CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def percentile(xs: List[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return float(sum(vals) / len(vals)) if vals else 0.0


def summarize_tree(rows: List[dict], denom_positions: int) -> Dict[str, float]:
    if not rows:
        return {
            "active_rate": 0.0,
            "p_hit": 0.0,
            "e_len": 0.0,
            "p_len_ge_4": 0.0,
            "extra_token": 0.0,
            "nodes_token": 0.0,
            "accepted_per_tree_node": 0.0,
        }
    accepted = [float(r["accepted_len"]) for r in rows]
    nodes = [float(r["tree_size"]) for r in rows]
    return {
        "active_rate": len(rows) / max(1, denom_positions),
        "p_hit": mean(1.0 if x >= 1 else 0.0 for x in accepted),
        "e_len": mean(accepted),
        "p_len_ge_4": mean(1.0 if x >= 4 else 0.0 for x in accepted),
        "extra_token": sum(accepted) / max(1, denom_positions),
        "nodes_token": sum(nodes) / max(1, denom_positions),
        "accepted_per_tree_node": sum(accepted) / max(1.0, sum(nodes)),
    }


def summarize_hybrid(rows: List[dict], denom_positions: int) -> Dict[str, float]:
    if not rows:
        return {
            "active_rate": 0.0,
            "small_e_len": 0.0,
            "retrieval_e_len": 0.0,
            "hybrid_e_len": 0.0,
            "extra_over_small_token": 0.0,
            "nodes_token": 0.0,
        }
    small = [float(r["small_draft_len"]) for r in rows]
    retr = [float(r["retrieval_len"]) for r in rows]
    hybrid = [float(r["hybrid_len"]) for r in rows]
    extra = [float(r["extra_len_over_small"]) for r in rows]
    nodes = [float(r["tree_size"]) for r in rows]
    return {
        "active_rate": len(rows) / max(1, denom_positions),
        "small_e_len": mean(small),
        "retrieval_e_len": mean(retr),
        "hybrid_e_len": mean(hybrid),
        "extra_over_small_token": sum(extra) / max(1, denom_positions),
        "nodes_token": sum(nodes) / max(1, denom_positions),
    }


def gate_rows(rows: List[dict], gate: str) -> List[dict]:
    if gate == "always":
        return rows
    if gate == "history_gt_512":
        return [r for r in rows if int(float(r["history_length"])) > 512]
    if gate == "history_gt_1024":
        return [r for r in rows if int(float(r["history_length"])) > 1024]
    raise ValueError(f"Unknown gate: {gate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-hits", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    draft_rows = read_csv(Path(args.draft_hits))
    tree_rows = read_csv(Path(args.tree))
    hybrid_rows = read_csv(Path(args.hybrid))

    rng = np.random.default_rng(args.seed)
    traj_ids = sorted({r["trajectory_id"] for r in draft_rows})
    denom_by_traj: Dict[str, int] = {}
    for r in draft_rows:
        denom_by_traj[r["trajectory_id"]] = denom_by_traj.get(r["trajectory_id"], 0) + 1

    tree_by_key: Dict[tuple[str, str], List[dict]] = {}
    for r in tree_rows:
        tree_by_key.setdefault((r["config"], r["trajectory_id"]), []).append(r)
    hybrid_by_key: Dict[tuple[str, str], List[dict]] = {}
    for r in hybrid_rows:
        hybrid_by_key.setdefault((r["config"], r["trajectory_id"]), []).append(r)

    configs = sorted({r["config"] for r in tree_rows} | {r["config"] for r in hybrid_rows})
    out_rows: List[dict] = []
    for table_name in ["tree", "hybrid"]:
        for config in configs:
            for gate in ["always", "history_gt_512", "history_gt_1024"]:
                if table_name == "tree":
                    all_rows = gate_rows([r for r in tree_rows if r["config"] == config], gate)
                    point = summarize_tree(all_rows, len(draft_rows))
                else:
                    all_rows = gate_rows([r for r in hybrid_rows if r["config"] == config], gate)
                    point = summarize_hybrid(all_rows, len(draft_rows))

                boot: Dict[str, List[float]] = {k: [] for k in point}
                for _ in range(args.bootstrap):
                    sampled = rng.choice(traj_ids, size=len(traj_ids), replace=True)
                    denom = sum(denom_by_traj[t] for t in sampled)
                    sample_rows: List[dict] = []
                    for tid in sampled:
                        key = (config, tid)
                        if table_name == "tree":
                            sample_rows.extend(gate_rows(tree_by_key.get(key, []), gate))
                        else:
                            sample_rows.extend(gate_rows(hybrid_by_key.get(key, []), gate))
                    sample_summary = summarize_tree(sample_rows, denom) if table_name == "tree" else summarize_hybrid(sample_rows, denom)
                    for metric, value in sample_summary.items():
                        boot[metric].append(value)

                for metric, value in point.items():
                    lo, hi = percentile(boot[metric])
                    out_rows.append({
                        "table": table_name,
                        "config": config,
                        "gate": gate,
                        "metric": metric,
                        "point": f"{value:.8f}",
                        "ci95_low": f"{lo:.8f}",
                        "ci95_high": f"{hi:.8f}",
                    })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["table", "config", "gate", "metric", "point", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(out_rows)

    for row in out_rows:
        if row["config"] == "tree_k4_d4" and row["gate"] == "history_gt_512":
            print(",".join(row[k] for k in ["table", "metric", "point", "ci95_low", "ci95_high"]))


if __name__ == "__main__":
    main()
