#!/usr/bin/env python3
"""Evaluate TLAR's causal reuse meter and adaptive tree width.

The expensive retrieval search is performed once per trajectory.  The cached
per-position outcomes for k=1..k_max are then replayed under every controller
configuration, so half-life and probe sweeps do not repeat retrieval matching.

Only tokens strictly before position t may appear in a retrieved continuation.
This explicit causal boundary also guards against accidental teacher-forcing
leakage when a matched historical context lies close to t.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_draft_hits(path: Path) -> Dict[Tuple[str, int], int]:
    out: Dict[Tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["trajectory_id"], int(row["token_index"]))] = int(row["draft_top1_hit"])
    return out


def signatures(ctx: Sequence[int], max_edits: int) -> Iterable[Tuple[int | None, ...]]:
    if max_edits == 0:
        yield tuple(ctx)
        return
    for masked in itertools.combinations(range(len(ctx)), max_edits):
        sig: List[int | None] = list(ctx)
        for i in masked:
            sig[i] = None
        yield tuple(sig)


class ApproxHistoryIndex:
    def __init__(self, tokens: Sequence[int], context_n: int, max_edits: int) -> None:
        self.tokens = tokens
        self.context_n = context_n
        self.max_edits = max_edits
        self.index: Dict[Tuple[int | None, ...], List[int]] = defaultdict(list)

    def add(self, continuation_start: int) -> None:
        c = self.context_n
        if continuation_start < c:
            return
        ctx = self.tokens[continuation_start - c:continuation_start]
        for sig in signatures(ctx, self.max_edits):
            self.index[sig].append(continuation_start)

    def query(self, t: int, topk: int) -> List[int]:
        c = self.context_n
        if t < c:
            return []
        ctx = self.tokens[t - c:t]
        candidates = set()
        for sig in signatures(ctx, self.max_edits):
            candidates.update(self.index.get(sig, ()))
        ranked = sorted(candidates, reverse=True)
        out = []
        for j in ranked:
            hist = self.tokens[j - c:j]
            if sum(a != b for a, b in zip(ctx, hist)) <= self.max_edits:
                out.append(j)
                if len(out) == topk:
                    break
        return out


def causal_prefixes(tokens: Sequence[int], t: int, candidates: Sequence[int], depth: int) -> set[Tuple[int, ...]]:
    prefixes: set[Tuple[int, ...]] = set()
    for j in candidates:
        # The continuation must already exist in the generated prefix [0, t).
        available = min(depth, max(0, t - j))
        prefix: Tuple[int, ...] = ()
        for offset in range(available):
            prefix += (tokens[j + offset],)
            prefixes.add(prefix)
    return prefixes


def accepted_from_prefixes(prefixes: set[Tuple[int, ...]], tokens: Sequence[int], t: int, depth: int) -> int:
    accepted = 0
    prefix: Tuple[int, ...] = ()
    for offset in range(min(depth, len(tokens) - t)):
        prefix += (tokens[t + offset],)
        if prefix not in prefixes:
            break
        accepted += 1
    return accepted


def consecutive_hits(hits: Dict[Tuple[str, int], int], trajectory_id: str, t: int, depth: int) -> int:
    n = 0
    for offset in range(depth):
        if hits.get((trajectory_id, t + offset), 0) != 1:
            break
        n += 1
    return n


@dataclass
class PositionOutcome:
    token_index: int
    draft_hit: int
    small_len: int
    accepted: Tuple[int, ...]
    nodes: Tuple[int, ...]
    branches: Tuple[int, ...] = ()


@dataclass(frozen=True)
class Controller:
    name: str
    half_life: int
    probe_interval: int
    rho_min: float
    adaptive_activation: bool = True
    adaptive_width: bool = True
    probes: bool = True


def build_position_cache(
    trace: dict,
    draft_hits: Dict[Tuple[str, int], int],
    context_n: int,
    max_edits: int,
    depth: int,
    k_max: int,
) -> List[PositionOutcome]:
    tokens = trace["generated_token_ids"]
    trajectory_id = trace["trajectory_id"]
    index = ApproxHistoryIndex(tokens, context_n, max_edits)
    out: List[PositionOutcome] = []
    for t in range(context_n, len(tokens) - 1):
        # Section 4 requires full-depth continuations, not truncated paths.
        index.add(t - depth)
        key = (trajectory_id, t)
        if key not in draft_hits:
            continue
        candidates = index.query(t, k_max)
        acc = [0]
        nodes = [0]
        for k in range(1, k_max + 1):
            prefixes = causal_prefixes(tokens, t, candidates[:k], depth)
            acc.append(accepted_from_prefixes(prefixes, tokens, t, depth))
            nodes.append(len(prefixes))
        out.append(PositionOutcome(
            token_index=t,
            draft_hit=draft_hits[key],
            small_len=consecutive_hits(draft_hits, trajectory_id, t, depth),
            accepted=tuple(acc),
            nodes=tuple(nodes),
            branches=tuple(min(k, len(candidates)) * depth for k in range(k_max + 1)),
        ))
    return out


def replay(rows: Sequence[PositionOutcome], controller: Controller, tau: int, k_min: int, k_max: int) -> Dict[str, float]:
    alpha = 1.0 - 2.0 ** (-1.0 / controller.half_life)
    rho = 0.0
    last_attempt_success = False
    eligible_count = 0
    sums = defaultdict(float)
    sums["n_positions"] = float(len(rows))

    for row in rows:
        eligible = row.token_index > tau
        first_eligible = eligible and eligible_count == 0
        bootstrap_attempt = eligible and eligible_count == 0
        periodic = eligible and controller.probes and (
            first_eligible or eligible_count % controller.probe_interval == 0
        )
        if eligible:
            eligible_count += 1

        if not eligible:
            active = False
        elif controller.adaptive_activation:
            active = bootstrap_attempt or rho >= controller.rho_min or periodic or last_attempt_success
        else:
            active = True

        if active:
            if periodic and controller.probes:
                k = k_min
            elif controller.adaptive_width:
                k = min(k_max, max(k_min, int(math.ceil(k_max * rho))))
            else:
                k = k_max
            accepted = row.accepted[k]
            nodes = row.nodes[k]
            success = accepted >= 1
            rho = (1.0 - alpha) * rho + alpha * float(success)
            last_attempt_success = success
            sums["active_positions"] += 1
            sums["probe_positions"] += float(periodic)
            sums["sum_k"] += k
            sums["accepted_tokens"] += accepted
            sums["nodes"] += nodes
            sums["hybrid_gain"] += max(row.small_len, accepted) - row.small_len
            if row.draft_hit == 0:
                sums["active_misses"] += 1
                sums["recovered_misses"] += float(success)
        else:
            rho = (1.0 - alpha) * rho
        sums["draft_misses"] += float(row.draft_hit == 0)

    return dict(sums)


def metrics(s: Dict[str, float]) -> Dict[str, float]:
    n = max(1.0, s.get("n_positions", 0.0))
    active = s.get("active_positions", 0.0)
    misses = s.get("draft_misses", 0.0)
    nodes = s.get("nodes", 0.0)
    accepted = s.get("accepted_tokens", 0.0)
    return {
        "n_positions": s.get("n_positions", 0.0),
        "draft_misses": misses,
        "active_fraction": active / n,
        "probe_fraction": s.get("probe_positions", 0.0) / n,
        "mean_k_when_active": s.get("sum_k", 0.0) / active if active else 0.0,
        "recovery_over_all_misses": s.get("recovered_misses", 0.0) / misses if misses else 0.0,
        "recovery_given_active_miss": s.get("recovered_misses", 0.0) / s.get("active_misses", 1.0),
        "retrieval_accepted_per_position": accepted / n,
        "delta_over_small_draft": s.get("hybrid_gain", 0.0) / n,
        "nodes_per_position": nodes / n,
        "accepted_per_node": accepted / nodes if nodes else 0.0,
    }


def add_sums(parts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    total = defaultdict(float)
    for part in parts:
        for key, value in part.items():
            total[key] += value
    return dict(total)


def controllers(args: argparse.Namespace) -> List[Controller]:
    out: Dict[str, Controller] = {}

    def add(c: Controller) -> None:
        out[c.name] = c

    add(Controller("full_H32_P32", 32, 32, args.rho_min))
    for h in args.half_lives:
        add(Controller(f"half_life_H{h}", h, args.default_probe_interval, args.rho_min))
    for p in args.probe_intervals:
        add(Controller(f"probe_interval_P{p}", args.default_half_life, p, args.rho_min))
    for rho in args.rho_min_values:
        add(Controller(f"rho_min_{rho:g}", args.default_half_life, args.default_probe_interval, rho))
    add(Controller("fixed_gate_fixed_k4", 32, 32, args.rho_min, False, False, False))
    add(Controller("meter_activation_fixed_k4", 32, 32, args.rho_min, True, False, True))
    add(Controller("always_active_adaptive_k", 32, 32, args.rho_min, False, True, False))
    add(Controller("meter_adaptive_k_no_periodic_probes", 32, 32, args.rho_min, True, True, False))
    return list(out.values())


def write_summary(
    path: Path,
    per_config: Dict[str, Dict[str, Dict[str, float]]],
    n_bootstrap: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    output = []
    for config, by_traj in sorted(per_config.items()):
        trajectory_ids = sorted(by_traj)
        point = metrics(add_sums(by_traj.values()))
        boots = defaultdict(list)
        for _ in range(n_bootstrap):
            sample = [rng.choice(trajectory_ids) for _ in trajectory_ids]
            value = metrics(add_sums(by_traj[tid] for tid in sample))
            for metric, x in value.items():
                boots[metric].append(x)
        for metric, value in point.items():
            arr = np.asarray(boots[metric], dtype=float)
            output.append({
                "config": config,
                "metric": metric,
                "point": f"{value:.8f}",
                "ci95_low": f"{np.quantile(arr, 0.025):.8f}",
                "ci95_high": f"{np.quantile(arr, 0.975):.8f}",
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "metric", "point", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(output)


def write_decay(path: Path, caches: Dict[str, List[PositionOutcome]], tau: int, k: int, max_lag: int) -> None:
    rows = []
    for lag in range(1, max_lag + 1):
        paired = hit_then_hit = first_hit = second_hit = 0
        for positions in caches.values():
            seq = [int(r.accepted[k] >= 1) for r in positions if r.token_index > tau]
            for i in range(len(seq) - lag):
                a, b = seq[i], seq[i + lag]
                paired += 1
                first_hit += a
                second_hit += b
                hit_then_hit += a * b
        baseline = second_hit / paired if paired else 0.0
        conditional = hit_then_hit / first_hit if first_hit else 0.0
        rows.append({
            "lag": lag,
            "pairs": paired,
            "p_hit": f"{baseline:.8f}",
            "p_hit_given_hit_at_t": f"{conditional:.8f}",
            "excess_probability": f"{conditional - baseline:.8f}",
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--draft-hits", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-decay", required=True)
    ap.add_argument("--context-n", type=int, default=4)
    ap.add_argument("--approx-max-edits", type=int, default=1)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--history-threshold", type=int, default=512)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--rho-min", type=float, default=0.05)
    ap.add_argument("--default-half-life", type=int, default=32)
    ap.add_argument("--default-probe-interval", type=int, default=32)
    ap.add_argument("--half-lives", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--probe-intervals", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--rho-min-values", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.1, 0.2])
    ap.add_argument("--max-decay-lag", type=int, default=128)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (0.0 <= args.rho_min <= 1.0):
        ap.error("--rho-min must be in [0, 1]")
    if not (1 <= args.k_min <= args.k_max):
        ap.error("require 1 <= k-min <= k-max")

    traces = list(iter_jsonl(Path(args.traces)))
    draft_hits = read_draft_hits(Path(args.draft_hits))
    cache: Dict[str, List[PositionOutcome]] = {}
    for tr in tqdm(traces, desc="causal retrieval cache"):
        cache[tr["trajectory_id"]] = build_position_cache(
            tr, draft_hits, args.context_n, args.approx_max_edits,
            args.depth, args.k_max,
        )

    per_config: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for controller in controllers(args):
        for trajectory_id, positions in cache.items():
            per_config[controller.name][trajectory_id] = replay(
                positions, controller, args.history_threshold, args.k_min, args.k_max,
            )

    output = Path(args.output)
    write_summary(output, per_config, args.bootstrap, args.seed)
    decay = Path(args.output_decay)
    decay.parent.mkdir(parents=True, exist_ok=True)
    write_decay(decay, cache, args.history_threshold, args.k_max, args.max_decay_lag)
    print("saved:", output)
    print("saved:", decay)


if __name__ == "__main__":
    main()
