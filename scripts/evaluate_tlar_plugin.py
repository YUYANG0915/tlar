#!/usr/bin/env python3
"""Paired evaluation of TLAR as a plug-in for retrieval baselines."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from compare_retrieval_baselines import (
    accepted_from_prefixes,
    build_exact_index,
    candidate_prefixes,
    consecutive_small_draft_hits,
    encode_prompt,
    generated_candidates,
    indexed_candidates,
    iter_jsonl,
    read_draft_hits,
    round_robin_unique_prefixes,
    stable_int_seed,
    stand_tree_prefixes,
)

Prefix = Tuple[int, ...]


def unique_union(*groups: Sequence[Prefix]) -> List[Prefix]:
    out: List[Prefix] = []
    seen: set[Prefix] = set()
    for group in groups:
        for prefix in group:
            if prefix not in seen:
                seen.add(prefix)
                out.append(prefix)
    return out


def summarize(rows: List[dict], bootstrap: int, seed: int) -> List[dict]:
    grouped: Dict[Tuple[str, str], Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[(row["baseline"], row["variant"])][row["trajectory_id"]].append(row)

    rng = random.Random(seed)
    output: List[dict] = []
    for (baseline, variant), by_traj in sorted(grouped.items()):
        trajectory_ids = sorted(by_traj)

        totals_by_traj: Dict[str, Dict[str, float]] = {}
        for tid, trajectory_rows in by_traj.items():
            totals_by_traj[tid] = {
                "n": float(len(trajectory_rows)),
                "baseline_accept": float(sum(
                    row["baseline_accept"] for row in trajectory_rows
                )),
                "plugin_accept": float(sum(
                    row["plugin_accept"] for row in trajectory_rows
                )),
                "baseline_hybrid": float(sum(
                    row["baseline_hybrid"] for row in trajectory_rows
                )),
                "plugin_hybrid": float(sum(
                    row["plugin_hybrid"] for row in trajectory_rows
                )),
                "baseline_nodes": float(sum(
                    row["baseline_nodes"] for row in trajectory_rows
                )),
                "plugin_nodes": float(sum(
                    row["plugin_nodes"] for row in trajectory_rows
                )),
                "improved": float(sum(
                    row["plugin_accept"] > row["baseline_accept"]
                    for row in trajectory_rows
                )),
                "degraded": float(sum(
                    row["plugin_accept"] < row["baseline_accept"]
                    for row in trajectory_rows
                )),
            }

        def aggregate(sample: Sequence[str]) -> Dict[str, float]:
            totals = {
                key: sum(totals_by_traj[tid][key] for tid in sample)
                for key in next(iter(totals_by_traj.values()))
            }
            n = totals["n"]
            baseline_accept = totals["baseline_accept"]
            plugin_accept = totals["plugin_accept"]
            baseline_hybrid = totals["baseline_hybrid"]
            plugin_hybrid = totals["plugin_hybrid"]
            baseline_nodes = totals["baseline_nodes"]
            plugin_nodes = totals["plugin_nodes"]
            return {
                "n_positions": float(n),
                "baseline_extra_per_position": baseline_accept / n,
                "plugin_extra_per_position": plugin_accept / n,
                "delta_extra_per_position": (plugin_accept - baseline_accept) / n,
                "baseline_hybrid_extra_per_position": baseline_hybrid / n,
                "plugin_hybrid_extra_per_position": plugin_hybrid / n,
                "delta_hybrid_extra_per_position": (plugin_hybrid - baseline_hybrid) / n,
                "baseline_nodes_per_position": baseline_nodes / n,
                "plugin_nodes_per_position": plugin_nodes / n,
                "delta_nodes_per_position": (plugin_nodes - baseline_nodes) / n,
                "improved_position_rate": totals["improved"] / n,
                "degraded_position_rate": totals["degraded"] / n,
            }

        point = aggregate(trajectory_ids)
        samples: Dict[str, List[float]] = defaultdict(list)
        for _ in range(bootstrap):
            sampled = [rng.choice(trajectory_ids) for _ in trajectory_ids]
            for metric, value in aggregate(sampled).items():
                samples[metric].append(value)
        for metric, value in point.items():
            values = np.asarray(samples[metric], dtype=float)
            output.append({
                "baseline": baseline,
                "variant": variant,
                "metric": metric,
                "point": f"{value:.8f}",
                "ci95_low": f"{np.quantile(values, 0.025):.8f}",
                "ci95_high": f"{np.quantile(values, 0.975):.8f}",
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--draft-hits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-n", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--approx-max-edits", type=int, default=1)
    parser.add_argument("--history-gate", type=int, default=512)
    parser.add_argument("--stand-min-n", type=int, default=2)
    parser.add_argument("--stand-max-n", type=int, default=8)
    parser.add_argument("--stand-temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    traces = list(iter_jsonl(Path(args.traces)))
    draft_hits = read_draft_hits(Path(args.draft_hits))
    tokenizers = {}
    prompt_by_traj = {}
    generated_by_traj = {}
    prompt_index_by_traj = {}
    generated_index_by_traj = {}
    for trace in traces:
        model_id = trace["model_id"]
        if model_id not in tokenizers:
            tokenizers[model_id] = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
        tid = trace["trajectory_id"]
        prompt = encode_prompt(tokenizers[model_id], trace)
        generated = trace["generated_token_ids"]
        prompt_by_traj[tid] = prompt
        generated_by_traj[tid] = generated
        prompt_index_by_traj[tid] = build_exact_index(prompt, args.context_n)
        generated_index_by_traj[tid] = build_exact_index(generated, args.context_n)

    separator = [-1, -2, -3, -4]
    prompt_corpus_by_traj = {}
    prompt_corpus_index_by_traj = {}
    for trace in traces:
        tid = trace["trajectory_id"]
        corpus: List[int] = []
        for other in traces:
            if other["trajectory_id"] != tid:
                corpus.extend(prompt_by_traj[other["trajectory_id"]])
                corpus.extend(separator)
        prompt_corpus_by_traj[tid] = corpus
        prompt_corpus_index_by_traj[tid] = build_exact_index(corpus, args.context_n)

    rows: List[dict] = []
    for trace in tqdm(traces, desc="TLAR plug-in"):
        tid = trace["trajectory_id"]
        generated = generated_by_traj[tid]
        prompt = prompt_by_traj[tid]
        prompt_corpus = prompt_corpus_by_traj[tid]
        for t in range(args.context_n, len(generated) - 1):
            gated = t > args.history_gate
            ours_candidates = generated_candidates(
                generated, t, args.context_n, args.topk, True,
                args.approx_max_edits, depth=args.depth,
            ) if gated else []
            ours_prefixes = candidate_prefixes(generated[:t], ours_candidates, args.depth)

            pld_candidates = indexed_candidates(
                generated, t, args.context_n, args.topk, prompt_index_by_traj[tid]
            )
            rest_candidates = indexed_candidates(
                generated, t, args.context_n, args.topk,
                prompt_corpus_index_by_traj[tid],
            )
            suffix_candidates = indexed_candidates(
                generated, t, args.context_n, args.topk,
                generated_index_by_traj[tid], before=t, depth=args.depth,
            )
            pld_prefixes = candidate_prefixes(prompt, pld_candidates, args.depth)
            rest_prefixes = candidate_prefixes(prompt_corpus, rest_candidates, args.depth)
            suffix_prefixes = candidate_prefixes(generated[:t], suffix_candidates, args.depth)
            stand_prefixes = stand_tree_prefixes(
                generated, t, args.topk, args.depth,
                args.stand_min_n, args.stand_max_n,
                args.stand_temperature,
                stable_int_seed(args.seed, tid, t, "stand"),
            ) if gated else []
            # Offline functional SAM proxy: exact dynamic suffix retrieval plus
            # the external static prompt datastore. Runtime is not claimed.
            sam_prefixes = unique_union(suffix_prefixes, rest_prefixes) if gated else []
            baselines = {
                "PLD": pld_prefixes,
                "REST": rest_prefixes,
                "SuffixDecoding": suffix_prefixes,
                "STAND": stand_prefixes,
                "SAM-proxy": sam_prefixes,
            }
            small_len = min(
                consecutive_small_draft_hits(draft_hits, tid, t, args.depth),
                args.depth,
            )
            ours_accept = accepted_from_prefixes(
                set(ours_prefixes), generated, t, args.depth
            )
            rows.append({
                "trajectory_id": tid,
                "baseline": "SmallDraft",
                "variant": "full_union",
                "baseline_accept": small_len,
                "plugin_accept": max(small_len, ours_accept),
                "baseline_hybrid": 0,
                "plugin_hybrid": max(small_len, ours_accept) - small_len,
                "baseline_nodes": 0,
                "plugin_nodes": len(ours_prefixes),
            })
            for baseline, baseline_prefixes in baselines.items():
                baseline_accept = accepted_from_prefixes(
                    set(baseline_prefixes), generated, t, args.depth
                )
                variants = (
                    ("full_union", unique_union(baseline_prefixes, ours_prefixes)),
                    (
                        "fixed_cap",
                        unique_union(baseline_prefixes, ours_prefixes)[
                            :max(args.topk, len(baseline_prefixes))
                        ],
                    ),
                    (
                        "matched_nodes",
                        round_robin_unique_prefixes(
                            baseline_prefixes, ours_prefixes, len(baseline_prefixes)
                        ),
                    ),
                )
                for variant, plugin_prefixes in variants:
                    plugin_accept = accepted_from_prefixes(
                        set(plugin_prefixes), generated, t, args.depth
                    )
                    rows.append({
                        "trajectory_id": tid,
                        "baseline": baseline,
                        "variant": variant,
                        "baseline_accept": baseline_accept,
                        "plugin_accept": plugin_accept,
                        "baseline_hybrid": max(small_len, baseline_accept) - small_len,
                        "plugin_hybrid": max(small_len, plugin_accept) - small_len,
                        "baseline_nodes": len(baseline_prefixes),
                        "plugin_nodes": len(plugin_prefixes),
                    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, args.bootstrap, args.seed)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline", "variant", "metric", "point", "ci95_low", "ci95_high"
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    print("saved:", output)


if __name__ == "__main__":
    main()
