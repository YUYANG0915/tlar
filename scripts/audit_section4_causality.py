#!/usr/bin/env python3
"""Audit candidate boundaries and future invariance on saved traces (no GPU).

Output records the checked trajectories, positions and candidate boundaries.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tlar_adaptive_tree import AdaptiveTree, Config, retrieve
from compare_retrieval_baselines import (build_exact_index, candidate_prefixes,
                                        generated_candidates, indexed_candidates, iter_jsonl)
from evaluate_adaptive_controller import ApproxHistoryIndex


def audit_trace(trace, cfg, max_positions, stream):
    tokens = trace["generated_token_ids"]
    tid = trace["trajectory_id"]
    size = len(tokens)
    if max_positions:
        positions = {round(i * size / max_positions) for i in range(max_positions + 1)}
        positions.update(t for t in (cfg.context, cfg.history_gate, cfg.history_gate + 1) if t <= size)
    else:
        positions = set(range(size + 1))
    full_index = build_exact_index(tokens, cfg.context)
    incremental_exact = defaultdict(list)
    incremental_approx = ApproxHistoryIndex(tokens, cfg.context, cfg.edits)
    counts = {"positions": 0, "candidates": 0, "legacy_crossing_candidates": 0}
    for t in range(size + 1):
        j = t - cfg.depth
        if j >= cfg.context:
            incremental_exact[tuple(tokens[j - cfg.context:j])].append(j)
            incremental_approx.add(j)
        if t not in positions or t < cfg.context:
            continue
        counts["positions"] += 1
        selected = generated_candidates(tokens, t, cfg.context, cfg.k_max, True, cfg.edits, depth=cfg.depth)
        expected = retrieve(tokens[:t], cfg, cfg.k_max)
        assert selected == [c.start for c in expected], (tid, t, "offline/core mismatch")
        assert selected == incremental_approx.query(t, cfg.k_max), (tid, t, "incremental approximate mismatch")
        exact = indexed_candidates(tokens, t, cfg.context, cfg.k_max, full_index, before=t, depth=cfg.depth)
        assert exact == indexed_candidates(tokens, t, cfg.context, cfg.k_max, incremental_exact,
                                           before=t, depth=cfg.depth), (tid, t, "incremental exact mismatch")
        reference_plan = AdaptiveTree(cfg).propose(tid, tokens[:t], [17, 18])
        for changed in (tokens[:t], tokens[:t] + [999999] * (size - t)):
            result = generated_candidates(changed, t, cfg.context, cfg.k_max, True, cfg.edits, depth=cfg.depth)
            assert result == selected, (tid, t, "future changed ranking")
            assert candidate_prefixes(changed, result, cfg.depth) == candidate_prefixes(tokens, selected, cfg.depth)
            idx = build_exact_index(changed, cfg.context)
            assert indexed_candidates(changed, t, cfg.context, cfg.k_max, idx,
                                      before=t, depth=cfg.depth) == exact
            assert AdaptiveTree(cfg).propose(tid, changed[:t], [17, 18]) == reference_plan
        legacy = []
        for old_j in range(t - 1, cfg.context - 1, -1):
            if sum(a != b for a, b in zip(tokens[old_j - cfg.context:old_j], tokens[t - cfg.context:t])) <= cfg.edits:
                legacy.append(old_j)
                if len(legacy) == cfg.k_max:
                    break
        counts["legacy_crossing_candidates"] += sum(j + cfg.depth > t for j in legacy)
        for rank, c in enumerate(expected):
            assert c.start + cfg.depth <= t
            counts["candidates"] += 1
            stream.write(json.dumps({"trajectory_id": tid, "t": t, "rank": rank,
                                    "source_context_start": c.start - cfg.context,
                                    "source_start": c.start, "source_end_exclusive": c.start + cfg.depth,
                                    "candidate_tokens": c.tokens}) + "\n")
    return {"trajectory_id": tid, "trace_tokens": size, **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=3, help="0 checks every trajectory")
    parser.add_argument("--max-positions", type=int, default=128, help="0 checks every position")
    parser.add_argument("--context-n", type=int, default=4)
    parser.add_argument("--approx-max-edits", type=int, default=1)
    parser.add_argument("--topk", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0 or args.max_positions < 0:
        parser.error("Limits must be nonnegative")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    cfg = Config(context=args.context_n, edits=args.approx_max_edits, k_max=args.topk)
    records = []
    with (out / "candidate_provenance.jsonl").open("w") as stream:
        for i, trace in enumerate(iter_jsonl(Path(args.traces))):
            if args.limit and i >= args.limit:
                break
            records.append(audit_trace(trace, cfg, args.max_positions, stream))
    if not records:
        raise ValueError("No trajectories were audited")
    root = Path(__file__).resolve().parents[1]
    files = [Path(args.traces), root / "tlar_adaptive_tree.py", Path(__file__),
             root / "scripts/compare_retrieval_baselines.py", root / "scripts/evaluate_adaptive_controller.py"]
    report = {"status": "passed", "scope": "candidate construction and future-invariance audit",
              "config": asdict(cfg), "arguments": vars(args), "trajectories": records,
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "trajectories": len(records),
                      "positions": sum(r["positions"] for r in records),
                      "legacy_crossing_candidates": sum(r["legacy_crossing_candidates"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
