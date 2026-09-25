#!/usr/bin/env python3
"""Compare retrieval drafting baselines on existing traces.

The shared harness measures exact token matches on recorded trajectories.
It reports first-token-miss recovery and all-position hybrid acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm



def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def encode_prompt(tokenizer, trace: dict) -> List[int]:
    """Recover the target-side prompt for both HF and vLLM traces."""
    prompt_token_ids = trace.get("prompt_token_ids")
    if prompt_token_ids:
        return [int(token_id) for token_id in prompt_token_ids]
    if trace.get("prompt_token_ids"):
        return list(trace["prompt_token_ids"])
    prompt_text = trace.get("prompt_text")
    if prompt_text:
        return tokenizer.encode(prompt_text, add_special_tokens=False)
    raw_prompt = trace.get("prompt")
    if not raw_prompt:
        raise ValueError(f"Trace {trace.get('trajectory_id')} has no usable prompt")
    from generate_traces import build_messages
    encoded = tokenizer.apply_chat_template(
        build_messages(raw_prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return list(encoded)


def read_draft_misses(path: Path) -> set[Tuple[str, int]]:
    out: set[Tuple[str, int]] = set()
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["draft_top1_hit"]) == 0:
                out.add((row["trajectory_id"], int(row["token_index"])))
    return out


def read_draft_hits(path: Path) -> Dict[Tuple[str, int], int]:
    out: Dict[Tuple[str, int], int] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["trajectory_id"], int(row["token_index"]))] = int(row["draft_top1_hit"])
    return out


def consecutive_small_draft_hits(
    draft_hits: Dict[Tuple[str, int], int],
    trajectory_id: str,
    token_index: int,
    max_len: int,
) -> int:
    out = 0
    for offset in range(max_len):
        if draft_hits.get((trajectory_id, token_index + offset), 0) != 1:
            break
        out += 1
    return out


def hamming_leq(a: Sequence[int], b: Sequence[int], max_edits: int) -> bool:
    edits = 0
    for x, y in zip(a, b):
        if x != y:
            edits += 1
            if edits > max_edits:
                return False
    return True


def accepted_len(source: Sequence[int], source_start: int, target: Sequence[int], target_start: int, depth: int) -> int:
    out = 0
    while out < depth and source_start + out < len(source) and target_start + out < len(target):
        if source[source_start + out] != target[target_start + out]:
            break
        out += 1
    return out


def generated_candidates(
    tokens: Sequence[int],
    t: int,
    context_n: int,
    topk: int,
    approx: bool,
    max_edits: int,
    *,
    depth: int,
) -> List[int]:
    if depth < 1 or not 0 <= t <= len(tokens):
        raise ValueError("Invalid history boundary or continuation depth")
    if t < context_n or topk <= 0:
        return []
    ctx = tokens[t - context_n:t]
    out: List[int] = []
    # A complete continuation must exist before the current position.
    for j in range(t - depth, context_n - 1, -1):
        hist = tokens[j - context_n:j]
        ok = hamming_leq(ctx, hist, max_edits) if approx else tuple(ctx) == tuple(hist)
        if ok:
            out.append(j)
            if len(out) >= topk:
                break
    return out


def build_exact_index(tokens: Sequence[int], context_n: int) -> Dict[Tuple[int, ...], List[int]]:
    index: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for j in range(context_n, len(tokens)):
        ctx = tuple(tokens[j - context_n:j])
        # Ignore artificial separators inserted between documents.
        if any(tok < 0 for tok in ctx):
            continue
        index[ctx].append(j)
    return index


def indexed_candidates(
    target_tokens: Sequence[int],
    t: int,
    context_n: int,
    topk: int,
    index: Dict[Tuple[int, ...], List[int]],
    before: int | None = None,
    depth: int | None = None,
) -> List[int]:
    if before is not None and (depth is None or depth < 1):
        raise ValueError("History retrieval requires an explicit continuation depth")
    if t < context_n:
        return []
    ctx = tuple(target_tokens[t - context_n:t])
    positions = index.get(ctx, [])
    out: List[int] = []
    for j in reversed(positions):
        if before is not None and j + depth > before:
            continue
        out.append(j)
        if len(out) >= topk:
            break
    return out


def prompt_candidates(
    gen_tokens: Sequence[int],
    prompt_tokens: Sequence[int],
    t: int,
    context_n: int,
    topk: int,
) -> List[int]:
    if t < context_n or len(prompt_tokens) <= context_n:
        return []
    ctx = tuple(gen_tokens[t - context_n:t])
    out: List[int] = []
    for j in range(len(prompt_tokens) - 1, context_n - 1, -1):
        if tuple(prompt_tokens[j - context_n:j]) == ctx:
            out.append(j)
            if len(out) >= topk:
                break
    return out


def corpus_candidates(
    target_tokens: Sequence[int],
    corpus_tokens: Sequence[int],
    t: int,
    context_n: int,
    topk: int,
    approx: bool,
    max_edits: int,
) -> List[int]:
    if t < context_n or len(corpus_tokens) <= context_n:
        return []
    ctx = target_tokens[t - context_n:t]
    out: List[int] = []
    for j in range(len(corpus_tokens) - 1, context_n - 1, -1):
        hist = corpus_tokens[j - context_n:j]
        ok = hamming_leq(ctx, hist, max_edits) if approx else tuple(ctx) == tuple(hist)
        if ok:
            out.append(j)
            if len(out) >= topk:
                break
    return out


def best_from_candidates(
    source: Sequence[int],
    target: Sequence[int],
    target_t: int,
    cands: Sequence[int],
    depth: int,
) -> Tuple[int, int, int]:
    best = 0
    tree_size = 0
    for j in cands:
        if source is target and j + depth > target_t:
            raise ValueError("Candidate continuation crosses the generated-history boundary")
        tree_size += min(depth, max(0, len(source) - j))
        best = max(best, accepted_len(source, j, target, target_t, depth))
    return best, tree_size, len(cands)


def candidate_prefixes(
    source: Sequence[int],
    cands: Sequence[int],
    depth: int,
) -> List[Tuple[int, ...]]:
    """Expand retrieved continuation paths into token-tree prefixes."""
    out: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    for j in cands:
        prefix: Tuple[int, ...] = ()
        for offset in range(depth):
            if j + offset >= len(source):
                break
            prefix = prefix + (source[j + offset],)
            if prefix not in seen:
                seen.add(prefix)
                out.append(prefix)
    return out


def accepted_from_prefixes(
    prefixes: set[Tuple[int, ...]],
    target: Sequence[int],
    target_t: int,
    depth: int,
) -> int:
    accepted = 0
    true_prefix: Tuple[int, ...] = ()
    for offset in range(depth):
        if target_t + offset >= len(target):
            break
        true_prefix = true_prefix + (target[target_t + offset],)
        if true_prefix not in prefixes:
            break
        accepted = offset + 1
    return accepted


def round_robin_unique_prefixes(
    left: Sequence[Tuple[int, ...]],
    right: Sequence[Tuple[int, ...]],
    budget: int | None,
) -> List[Tuple[int, ...]]:
    """Merge two prefix lists without letting either source monopolize budget."""
    out: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    i = j = 0
    take_left = True
    while i < len(left) or j < len(right):
        if budget is not None and len(out) >= budget:
            break
        source = left if take_left else right
        idx = i if take_left else j
        if idx < len(source):
            item = source[idx]
            if take_left:
                i += 1
            else:
                j += 1
            if item not in seen:
                seen.add(item)
                out.append(item)
        else:
            if take_left:
                i = len(left)
            else:
                j = len(right)
        take_left = not take_left
    return out


def stable_int_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest()[:16], 16)


def stand_next_counts(
    history: Sequence[int],
    context_stream: Sequence[int],
    before: int,
    min_n: int,
    max_n: int,
) -> Tuple[Counter[int], int]:
    """Adaptive n-gram next-token distribution from generated history.

    This is a model-free offline proxy for STAND's adaptive/logit n-gram
    module: choose the longest available n-gram context, then use empirical
    next-token counts as log-count logits.
    """
    max_available = min(max_n, len(context_stream), before)
    for n in range(max_available, min_n - 1, -1):
        ctx = tuple(context_stream[-n:])
        counts: Counter[int] = Counter()
        for j in range(n, before):
            if tuple(history[j - n:j]) == ctx and j < len(history):
                counts[history[j]] += 1
        if counts:
            return counts, n
    return Counter(), 0


def gumbel_topk_from_counts(
    counts: Counter[int],
    k: int,
    rng: random.Random,
    temperature: float,
) -> List[Tuple[int, float]]:
    scored: List[Tuple[float, int, float]] = []
    temp = max(temperature, 1e-6)
    for tok, count in counts.items():
        # Empirical count distribution as a memory-efficient logit proxy.
        logit = math.log(float(count))
        u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
        gumbel = -math.log(-math.log(u))
        scored.append((logit + gumbel * temp, tok, logit))
    scored.sort(reverse=True)
    return [(tok, logit) for _, tok, logit in scored[:k]]


def stand_tree_acceptance(
    tokens: Sequence[int],
    t: int,
    topk: int,
    depth: int,
    min_n: int,
    max_n: int,
    temperature: float,
    seed: int,
) -> Tuple[int, int, int]:
    """STAND-style stochastic adaptive n-gram tree proxy.

    Implements the STAND ingredients available in an offline trace setting:
    stochastic drafting, adaptive n-gram backoff, log-count n-gram logits,
    Gumbel-Top-K sampling, and data-driven tree construction. Verification is
    still exact-token: accepted length is the deepest generated tree prefix
    matching the observed target continuation.
    """
    if t < min_n:
        return 0, 0, 0

    rng = random.Random(seed)
    tree_prefixes: set[Tuple[int, ...]] = set()
    frontier: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]

    for _depth in range(depth):
        proposals: List[Tuple[float, Tuple[int, ...]]] = []
        for prefix, prefix_score in frontier:
            context_stream = list(tokens[:t]) + list(prefix)
            counts, _used_n = stand_next_counts(tokens, context_stream, before=t, min_n=min_n, max_n=max_n)
            if not counts:
                continue
            for tok, logit in gumbel_topk_from_counts(counts, topk, rng, temperature):
                new_prefix = prefix + (tok,)
                tree_prefixes.add(new_prefix)
                # Data-driven global pruning: keep the strongest stochastic
                # partial continuations rather than expanding every branch.
                proposals.append((prefix_score + logit, new_prefix))
        if not proposals:
            break
        proposals.sort(reverse=True)
        frontier = [(prefix, score) for score, prefix in proposals[:topk]]

    accepted = 0
    true_prefix: Tuple[int, ...] = ()
    for offset in range(depth):
        if t + offset >= len(tokens):
            break
        true_prefix = true_prefix + (tokens[t + offset],)
        if true_prefix not in tree_prefixes:
            break
        accepted = offset + 1

    return accepted, len(tree_prefixes), len(frontier)


def stand_tree_prefixes(
    tokens: Sequence[int],
    t: int,
    topk: int,
    depth: int,
    min_n: int,
    max_n: int,
    temperature: float,
    seed: int,
) -> List[Tuple[int, ...]]:
    """Return STAND-style tree prefixes in deterministic priority order."""
    if t < min_n:
        return []

    rng = random.Random(seed)
    ordered: List[Tuple[int, ...]] = []
    seen: set[Tuple[int, ...]] = set()
    frontier: List[Tuple[Tuple[int, ...], float]] = [((), 0.0)]

    for _depth in range(depth):
        proposals: List[Tuple[float, Tuple[int, ...]]] = []
        for prefix, prefix_score in frontier:
            context_stream = list(tokens[:t]) + list(prefix)
            counts, _used_n = stand_next_counts(tokens, context_stream, before=t, min_n=min_n, max_n=max_n)
            if not counts:
                continue
            for tok, logit in gumbel_topk_from_counts(counts, topk, rng, temperature):
                new_prefix = prefix + (tok,)
                if new_prefix not in seen:
                    seen.add(new_prefix)
                    ordered.append(new_prefix)
                proposals.append((prefix_score + logit, new_prefix))
        if not proposals:
            break
        proposals.sort(reverse=True)
        frontier = [(prefix, score) for score, prefix in proposals[:topk]]

    return ordered


def summarize(rows: List[dict], bootstrap: int, seed: int) -> List[dict]:
    by_config: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_config[r["config"]][r["trajectory_id"]].append(r)

    rng = random.Random(seed)
    out = []
    for config, by_traj in sorted(by_config.items()):
        traj_ids = sorted(by_traj)

        def one(sample_ids: Sequence[str]) -> Dict[str, float]:
            subset = [r for tid in sample_ids for r in by_traj[tid]]
            n_miss = len(subset)
            if n_miss == 0:
                return {}
            n_all = sum(int(by_traj[tid][0]["total_positions"]) for tid in sample_ids)
            active = [r for r in subset if int(r["active"]) == 1]
            active_n = len(active)
            hit = sum(int(r["accepted_len"]) >= 1 for r in active)
            total_len = sum(int(r["accepted_len"]) for r in active)
            total_nodes = sum(int(r["tree_size"]) for r in active)
            return {
                "n_miss_positions": float(n_miss),
                "n_all_positions": float(n_all),
                "active_rate_over_miss": active_n / n_miss,
                "p_hit_given_active_miss": hit / active_n if active_n else 0.0,
                "e_len_given_active_miss": total_len / active_n if active_n else 0.0,
                "extra_per_draft_miss": total_len / n_miss,
                "extra_per_all_token": total_len / n_all if n_all else 0.0,
                "nodes_per_draft_miss": total_nodes / n_miss,
                "nodes_per_all_token": total_nodes / n_all if n_all else 0.0,
                "accepted_per_tree_node": total_len / total_nodes if total_nodes else 0.0,
            }

        point = one(traj_ids)
        boots: Dict[str, List[float]] = defaultdict(list)
        for _ in range(bootstrap):
            sample = [rng.choice(traj_ids) for _ in traj_ids]
            vals = one(sample)
            for k, v in vals.items():
                boots[k].append(v)

        for metric, val in point.items():
            arr = np.array(boots[metric], dtype=float)
            out.append({
                "config": config,
                "metric": metric,
                "point": f"{val:.8f}",
                "ci95_low": f"{np.quantile(arr, 0.025):.8f}",
                "ci95_high": f"{np.quantile(arr, 0.975):.8f}",
            })
    return out


def summarize_hybrid_rows(rows: List[dict], bootstrap: int, seed: int) -> List[dict]:
    by_config: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_config[r["config"]][r["trajectory_id"]].append(r)

    rng = random.Random(seed)
    out = []
    for config, by_traj in sorted(by_config.items()):
        traj_ids = sorted(by_traj)

        def one(sample_ids: Sequence[str]) -> Dict[str, float]:
            subset = [r for tid in sample_ids for r in by_traj[tid]]
            if not subset:
                return {}
            n_all = sum(int(by_traj[tid][0]["total_positions"]) for tid in sample_ids)
            small_hits = sum(int(r["small_draft_len"]) for r in subset)
            retrieval_len = sum(int(r["retrieval_len"]) for r in subset)
            hybrid_len = sum(int(r["hybrid_len"]) for r in subset)
            nodes = sum(int(r["tree_size"]) for r in subset)
            return {
                "n_all_positions": float(n_all),
                "small_e_len": small_hits / n_all if n_all else 0.0,
                "retrieval_extra_per_token": retrieval_len / n_all if n_all else 0.0,
                "hybrid_e_len": hybrid_len / n_all if n_all else 0.0,
                "extra_over_small_token": (hybrid_len - small_hits) / n_all if n_all else 0.0,
                "nodes_per_all_token": nodes / n_all if n_all else 0.0,
            }

        point = one(traj_ids)
        boots: Dict[str, List[float]] = defaultdict(list)
        for _ in range(bootstrap):
            sample = [rng.choice(traj_ids) for _ in traj_ids]
            vals = one(sample)
            for k, v in vals.items():
                boots[k].append(v)

        for metric, val in point.items():
            arr = np.array(boots[metric], dtype=float)
            out.append({
                "config": config,
                "metric": metric,
                "point": f"{val:.8f}",
                "ci95_low": f"{np.quantile(arr, 0.025):.8f}",
                "ci95_high": f"{np.quantile(arr, 0.975):.8f}",
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--draft-hits", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--context-n", type=int, default=4)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--approx-max-edits", type=int, default=1)
    ap.add_argument("--stand-min-n", type=int, default=2)
    ap.add_argument("--stand-max-n", type=int, default=8)
    ap.add_argument("--stand-temperature", type=float, default=1.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traces = list(iter_jsonl(Path(args.traces)))
    draft_miss = read_draft_misses(Path(args.draft_hits))
    draft_hits = read_draft_hits(Path(args.draft_hits))

    # Tokenizers are only needed for prompt lookup. Use each trace's own model id
    # because Qwen/DeepSeek/Llama tokenization differs.
    from transformers import AutoTokenizer

    tokenizers = {}
    prompt_by_traj: Dict[str, List[int]] = {}
    generated_by_traj: Dict[str, List[int]] = {}
    prompt_corpus_by_traj: Dict[str, List[int]] = {}
    generated_corpus_by_traj: Dict[str, List[int]] = {}
    prompt_index_by_traj: Dict[str, Dict[Tuple[int, ...], List[int]]] = {}
    generated_index_by_traj: Dict[str, Dict[Tuple[int, ...], List[int]]] = {}
    prompt_corpus_index_by_traj: Dict[str, Dict[Tuple[int, ...], List[int]]] = {}
    generated_corpus_index_by_traj: Dict[str, Dict[Tuple[int, ...], List[int]]] = {}

    sep = [-1, -2, -3, -4]
    for tr in traces:
        model_id = tr["model_id"]
        if model_id not in tokenizers:
            tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tid = tr["trajectory_id"]
        prompt_by_traj[tid] = encode_prompt(tokenizers[model_id], tr)
        generated_by_traj[tid] = tr["generated_token_ids"]
        prompt_index_by_traj[tid] = build_exact_index(prompt_by_traj[tid], args.context_n)
        generated_index_by_traj[tid] = build_exact_index(generated_by_traj[tid], args.context_n)

    for tr in traces:
        tid = tr["trajectory_id"]
        prompt_corpus: List[int] = []
        generated_corpus: List[int] = []
        for other in traces:
            if other["trajectory_id"] == tid:
                continue
            prompt_corpus.extend(prompt_by_traj[other["trajectory_id"]])
            prompt_corpus.extend(sep)
            generated_corpus.extend(generated_by_traj[other["trajectory_id"]])
            generated_corpus.extend(sep)
        prompt_corpus_by_traj[tid] = prompt_corpus
        generated_corpus_by_traj[tid] = generated_corpus
        prompt_corpus_index_by_traj[tid] = build_exact_index(prompt_corpus, args.context_n)
        generated_corpus_index_by_traj[tid] = build_exact_index(generated_corpus, args.context_n)

    rows: List[dict] = []
    hybrid_rows: List[dict] = []
    hybrid_configs = [
        "SuffixDecoding_generated_exact_k4_d4_plus_small_draft",
        "STAND_stochastic_adaptive_ngram_history_gt512_k4_d4_plus_small_draft",
        "Ours_approx_generated_history_gt512_k4_d4_plus_small_draft",
        "STAND_union_Ours_history_gt512_k4_d4_plus_small_draft",
    ]
    union_budgets = [4, 6, 8, 12]
    for tr in tqdm(traces, desc="baselines"):
        traj_id = tr["trajectory_id"]
        gen = tr["generated_token_ids"]
        prompt_tokens = prompt_by_traj[traj_id]
        prompt_index = prompt_index_by_traj[traj_id]
        generated_index = generated_index_by_traj[traj_id]
        prompt_corpus = prompt_corpus_by_traj[traj_id]
        generated_corpus = generated_corpus_by_traj[traj_id]
        prompt_corpus_index = prompt_corpus_index_by_traj[traj_id]
        generated_corpus_index = generated_corpus_index_by_traj[traj_id]

        for t in range(args.context_n, len(gen) - 1):
            small_hit = draft_hits.get((traj_id, t), 0)
            small_len = consecutive_small_draft_hits(draft_hits, traj_id, t, args.depth)
            total_positions = max(0, len(gen) - 1 - args.context_n)

            stand_acc, stand_size, stand_cands = stand_tree_acceptance(
                gen,
                t,
                topk=args.topk,
                depth=args.depth,
                min_n=args.stand_min_n,
                max_n=args.stand_max_n,
                temperature=args.stand_temperature,
                seed=stable_int_seed(args.seed, traj_id, t, "stand"),
            )
            stand_prefix_list = stand_tree_prefixes(
                gen,
                t,
                topk=args.topk,
                depth=args.depth,
                min_n=args.stand_min_n,
                max_n=args.stand_max_n,
                temperature=args.stand_temperature,
                seed=stable_int_seed(args.seed, traj_id, t, "stand"),
            )

            configs = [
                # Prompt Lookup Decoding / PLD: retrieve only from the current prompt.
                ("PLD_prompt_lookup_exact_k4_d4", True, prompt_tokens, indexed_candidates(gen, t, args.context_n, args.topk, prompt_index)),
                # REST-style retrieval datastore proxy: retrieve from external prompt corpus.
                ("REST_external_prompt_exact_k4_d4", True, prompt_corpus, indexed_candidates(gen, t, args.context_n, args.topk, prompt_corpus_index)),
                # SuffixDecoding-style current generated suffix tree, exact n-gram only.
                ("SuffixDecoding_generated_exact_k4_d4", True, gen, indexed_candidates(gen, t, args.context_n, args.topk, generated_index, before=t, depth=args.depth)),
                # Our retrieval-only component: generated-history approximate matching + history gate.
                ("Ours_approx_generated_history_gt512_k4_d4", t > 512, gen, generated_candidates(gen, t, args.context_n, args.topk, True, args.approx_max_edits, depth=args.depth)),
            ]
            suffix_acc, suffix_tree_size, _suffix_cand_count = best_from_candidates(
                gen,
                gen,
                t,
                indexed_candidates(gen, t, args.context_n, args.topk, generated_index, before=t, depth=args.depth),
                args.depth,
            )
            ours_acc, ours_tree_size, _ours_cand_count = (
                best_from_candidates(
                    gen,
                    gen,
                    t,
                    generated_candidates(gen, t, args.context_n, args.topk, True, args.approx_max_edits, depth=args.depth),
                    args.depth,
                )
                if t > 512 else (0, 0, 0)
            )
            ours_prefix_list = (
                candidate_prefixes(
                    gen,
                    generated_candidates(gen, t, args.context_n, args.topk, True, args.approx_max_edits, depth=args.depth),
                    args.depth,
                )
                if t > 512 else []
            )
            budgeted_union: Dict[int, Tuple[int, int]] = {}
            for budget in union_budgets:
                prefixes = round_robin_unique_prefixes(
                    stand_prefix_list if t > 512 else [],
                    ours_prefix_list,
                    budget,
                )
                budgeted_union[budget] = (
                    accepted_from_prefixes(set(prefixes), gen, t, args.depth),
                    len(prefixes),
                )

            hybrid_specs = [
                ("SuffixDecoding_generated_exact_k4_d4_plus_small_draft", suffix_acc, suffix_tree_size),
                (
                    "STAND_stochastic_adaptive_ngram_history_gt512_k4_d4_plus_small_draft",
                    stand_acc if t > 512 else 0,
                    stand_size if t > 512 else 0,
                ),
                ("Ours_approx_generated_history_gt512_k4_d4_plus_small_draft", ours_acc, ours_tree_size),
                (
                    "STAND_union_Ours_history_gt512_k4_d4_plus_small_draft",
                    max(stand_acc, ours_acc) if t > 512 else 0,
                    (stand_size + ours_tree_size) if t > 512 else 0,
                ),
            ]
            for config, retrieval_len, tree_size in hybrid_specs:
                hybrid_rows.append({
                    "trajectory_id": traj_id,
                    "task_id": tr["task_id"],
                    "model_key": tr["model_key"],
                    "token_index": t,
                    "total_positions": total_positions,
                    "config": config,
                    "small_draft_len": min(small_len, args.depth),
                    "retrieval_len": retrieval_len,
                    "hybrid_len": max(min(small_len, args.depth), retrieval_len),
                    "tree_size": tree_size,
                })
            for budget, (retrieval_len, tree_size) in budgeted_union.items():
                hybrid_rows.append({
                    "trajectory_id": traj_id,
                    "task_id": tr["task_id"],
                    "model_key": tr["model_key"],
                    "token_index": t,
                    "total_positions": total_positions,
                    "config": f"STAND_union_Ours_history_gt512_k4_d4_budget{budget}_plus_small_draft",
                    "small_draft_len": min(small_len, args.depth),
                    "retrieval_len": retrieval_len,
                    "hybrid_len": max(min(small_len, args.depth), retrieval_len),
                    "tree_size": tree_size,
                })

            if (traj_id, t) not in draft_miss:
                continue

            rows.append({
                "trajectory_id": traj_id,
                "task_id": tr["task_id"],
                "model_key": tr["model_key"],
                "token_index": t,
                "total_positions": total_positions,
                "config": "STAND_stochastic_adaptive_ngram_always_k4_d4",
                "active": 1,
                "accepted_len": stand_acc,
                "tree_size": stand_size,
                "candidate_count": stand_cands,
            })
            rows.append({
                "trajectory_id": traj_id,
                "task_id": tr["task_id"],
                "model_key": tr["model_key"],
                "token_index": t,
                "total_positions": total_positions,
                "config": "STAND_stochastic_adaptive_ngram_history_gt512_k4_d4",
                "active": int(t > 512),
                "accepted_len": stand_acc if t > 512 else 0,
                "tree_size": stand_size if t > 512 else 0,
                "candidate_count": stand_cands if t > 512 else 0,
            })
            rows.append({
                "trajectory_id": traj_id,
                "task_id": tr["task_id"],
                "model_key": tr["model_key"],
                "token_index": t,
                "total_positions": total_positions,
                "config": "STAND_union_Ours_history_gt512_k4_d4",
                "active": int(t > 512),
                "accepted_len": max(stand_acc, ours_acc) if t > 512 else 0,
                "tree_size": (stand_size + ours_tree_size) if t > 512 else 0,
                "candidate_count": (stand_cands + _ours_cand_count) if t > 512 else 0,
            })
            for budget, (accepted, tree_size) in budgeted_union.items():
                rows.append({
                    "trajectory_id": traj_id,
                    "task_id": tr["task_id"],
                    "model_key": tr["model_key"],
                    "token_index": t,
                    "total_positions": total_positions,
                    "config": f"STAND_union_Ours_history_gt512_k4_d4_budget{budget}",
                    "active": int(t > 512),
                    "accepted_len": accepted if t > 512 else 0,
                    "tree_size": tree_size if t > 512 else 0,
                    "candidate_count": tree_size if t > 512 else 0,
                })
            for name, active, source, cands in configs:
                acc, tree_size, cand_count = best_from_candidates(source, gen, t, cands, args.depth) if active else (0, 0, 0)
                rows.append({
                    "trajectory_id": traj_id,
                    "task_id": tr["task_id"],
                    "model_key": tr["model_key"],
                    "token_index": t,
                    "total_positions": total_positions,
                    "config": name,
                    "active": int(active),
                    "accepted_len": acc,
                    "tree_size": tree_size,
                    "candidate_count": cand_count,
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, args.bootstrap, args.seed)
    summary.extend(summarize_hybrid_rows(hybrid_rows, args.bootstrap, args.seed))
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "metric", "point", "ci95_low", "ci95_high"])
        writer.writeheader()
        writer.writerows(summary)
    print("saved:", out)


if __name__ == "__main__":
    main()
