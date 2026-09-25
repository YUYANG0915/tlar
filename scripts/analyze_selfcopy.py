#!/usr/bin/env python3
"""CPU exact/approx self-copy analysis with prefix-causal retrieval."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


TRIGGERS = {
    "weak_high_freq": ["not", "but", "so", "therefore"],
    "contrast": ["rather than", "instead of", "not but"],
    "backtracking": ["wait", "let me re-check", "let me redo", "alternatively"],
    "summary": ["the answer is", "we have shown", "we get"],
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def accepted_length(tokens: Sequence[int], draft_start: int, target_start: int, max_len: int) -> int:
    n = len(tokens)
    out = 0
    while out < max_len and draft_start + out < target_start and target_start + out < n:
        if tokens[draft_start + out] != tokens[target_start + out]:
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


def build_context_index(tokens: Sequence[int], context_n: int, upto: int) -> Dict[Tuple[int, ...], List[int]]:
    index: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for j in range(context_n, upto):
        index[tuple(tokens[j - context_n:j])].append(j)
    return index


def recent_exact_candidate(tokens: Sequence[int], t: int, context_n: int) -> Optional[int]:
    if t < context_n:
        return None
    ctx = tuple(tokens[t - context_n:t])
    for j in range(t - 1, context_n - 1, -1):
        if tuple(tokens[j - context_n:j]) == ctx:
            return j
    return None


def recent_approx_candidate(tokens: Sequence[int], t: int, context_n: int, max_edits: int) -> Optional[int]:
    if t < context_n:
        return None
    ctx = tokens[t - context_n:t]
    for j in range(t - 1, context_n - 1, -1):
        hist_ctx = tokens[j - context_n:j]
        if hamming_leq(ctx, hist_ctx, max_edits):
            return j
    return None


def cross_exact_candidate(
    target_tokens: Sequence[int],
    control_tokens: Sequence[int],
    t: int,
    context_n: int,
) -> Optional[int]:
    if t < context_n:
        return None
    ctx = tuple(target_tokens[t - context_n:t])
    hist_len = min(t, len(control_tokens))
    for j in range(hist_len - 1, context_n - 1, -1):
        if tuple(control_tokens[j - context_n:j]) == ctx:
            return j
    return None


def cross_approx_candidate(
    target_tokens: Sequence[int],
    control_tokens: Sequence[int],
    t: int,
    context_n: int,
    max_edits: int,
) -> Optional[int]:
    if t < context_n:
        return None
    ctx = target_tokens[t - context_n:t]
    hist_len = min(t, len(control_tokens))
    for j in range(hist_len - 1, context_n - 1, -1):
        if hamming_leq(ctx, control_tokens[j - context_n:j], max_edits):
            return j
    return None


def shuffled_tokens(tokens: Sequence[int], sentence_break_ids: set[int], seed_key: str) -> List[int]:
    """Cheap token-level sentence shuffle fallback.

    If sentence token ids are not known, this still gives a deterministic block shuffle
    and avoids using future target continuation for retrieval.
    """
    blocks: List[List[int]] = []
    cur: List[int] = []
    for tok in tokens:
        cur.append(tok)
        if tok in sentence_break_ids:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    if len(blocks) <= 1:
        block_size = 32
        blocks = [list(tokens[i:i + block_size]) for i in range(0, len(tokens), block_size)]
    rng = random.Random(seed_key)
    rng.shuffle(blocks)
    return [tok for block in blocks for tok in block]


def detect_trigger(text_lower: str) -> Tuple[str, str]:
    for trig_type, phrases in TRIGGERS.items():
        for phrase in phrases:
            if phrase in text_lower:
                return trig_type, phrase
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-n", type=int, default=4)
    parser.add_argument("--max-accepted-len", type=int, default=32)
    parser.add_argument("--approx-max-edits", type=int, default=1)
    parser.add_argument("--max-positions-per-trace", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    traces = list(iter_jsonl(Path(args.input)))
    if not traces:
        raise SystemExit("No traces found")

    by_group: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for tr in traces:
        by_group[(tr["dataset"], tr["model_key"])].append(tr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory_id", "dataset", "task_id", "model_key", "token_index",
        "relative_position", "history_length", "position_bucket",
        "trigger_type", "trigger_text", "control_type",
        "matched_control_trajectory_id", "exact_accepted_length",
        "approx_retrieval_score", "verifiable_accepted_length_after_approx_retrieval",
        "total_token_length", "thinking_flag",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tr in tqdm(traces, desc="analyzing"):
            tokens = tr["generated_token_ids"]
            n_tokens = len(tokens)
            if n_tokens <= args.context_n + 1:
                continue

            group = by_group[(tr["dataset"], tr["model_key"])]
            controls = [x for x in group if x["trajectory_id"] != tr["trajectory_id"]]
            control = controls[0] if controls else None
            control_tokens = control["generated_token_ids"] if control else []
            sentence_break_ids = set()
            max_t = n_tokens - 1
            positions = range(args.context_n, max_t, max(1, args.stride))
            if args.max_positions_per_trace:
                positions = list(positions)[: args.max_positions_per_trace]

            lower_text = tr.get("generated_text", "").lower()
            for t in positions:
                rel = t / max(1, n_tokens)
                if n_tokens < 2000:
                    bucket = "too_short_for_position_buckets"
                elif rel < 0.2:
                    bucket = "first_20"
                elif rel >= 0.8:
                    bucket = "last_20"
                else:
                    bucket = "middle_60"

                # Optional exact character offsets supplied by the trace producer.
                offsets = tr.get("token_char_offsets")
                trig_type, trig_text = "", ""
                if offsets is not None:
                    char_cut = int(offsets[t][0])
                    local_text = lower_text[max(0, char_cut - 80):char_cut + 80]
                    trig_type, trig_text = detect_trigger(local_text)

                exact_j = recent_exact_candidate(tokens, t, args.context_n)
                exact_len = accepted_length(tokens, exact_j, t, args.max_accepted_len) if exact_j is not None else 0

                approx_j = recent_approx_candidate(tokens, t, args.context_n, args.approx_max_edits)
                approx_score = 1 if approx_j is not None else 0
                approx_len = accepted_length(tokens, approx_j, t, args.max_accepted_len) if approx_j is not None else 0

                writer.writerow({
                    "trajectory_id": tr["trajectory_id"],
                    "dataset": tr["dataset"],
                    "task_id": tr["task_id"],
                    "model_key": tr["model_key"],
                    "token_index": t,
                    "relative_position": f"{rel:.6f}",
                    "history_length": t,
                    "position_bucket": bucket,
                    "trigger_type": trig_type,
                    "trigger_text": trig_text,
                    "control_type": "within",
                    "matched_control_trajectory_id": "",
                    "exact_accepted_length": exact_len,
                    "approx_retrieval_score": approx_score,
                    "verifiable_accepted_length_after_approx_retrieval": approx_len,
                    "total_token_length": n_tokens,
                    "thinking_flag": tr.get("thinking_flag"),
                })

                if control:
                    cross_j = cross_exact_candidate(tokens, control_tokens, t, args.context_n)
                    cross_exact_len = 0
                    if cross_j is not None:
                        cross_exact_len = 0
                        while cross_exact_len < args.max_accepted_len:
                            if cross_j + cross_exact_len >= min(t, len(control_tokens)) or t + cross_exact_len >= len(tokens):
                                break
                            if control_tokens[cross_j + cross_exact_len] != tokens[t + cross_exact_len]:
                                break
                            cross_exact_len += 1

                    cross_approx_j = cross_approx_candidate(
                        tokens, control_tokens, t, args.context_n, args.approx_max_edits
                    )
                    cross_approx_score = 1 if cross_approx_j is not None else 0
                    cross_approx_len = 0
                    if cross_approx_j is not None:
                        while cross_approx_len < args.max_accepted_len:
                            if cross_approx_j + cross_approx_len >= min(t, len(control_tokens)) or t + cross_approx_len >= len(tokens):
                                break
                            if control_tokens[cross_approx_j + cross_approx_len] != tokens[t + cross_approx_len]:
                                break
                            cross_approx_len += 1

                    writer.writerow({
                        "trajectory_id": tr["trajectory_id"],
                        "dataset": tr["dataset"],
                        "task_id": tr["task_id"],
                        "model_key": tr["model_key"],
                        "token_index": t,
                        "relative_position": f"{rel:.6f}",
                        "history_length": t,
                        "position_bucket": bucket,
                        "trigger_type": trig_type,
                        "trigger_text": trig_text,
                        "control_type": "cross",
                        "matched_control_trajectory_id": control["trajectory_id"],
                        "exact_accepted_length": cross_exact_len,
                        "approx_retrieval_score": cross_approx_score,
                        "verifiable_accepted_length_after_approx_retrieval": cross_approx_len,
                        "total_token_length": n_tokens,
                        "thinking_flag": tr.get("thinking_flag"),
                    })

                shuffled = shuffled_tokens(tokens[:t], sentence_break_ids, f"{args.seed}:{tr['trajectory_id']}:{t}")
                shuffled_j = cross_exact_candidate(tokens, shuffled, t, args.context_n)
                shuffled_exact_len = 0
                if shuffled_j is not None:
                    while shuffled_exact_len < args.max_accepted_len:
                        if shuffled_j + shuffled_exact_len >= len(shuffled) or t + shuffled_exact_len >= len(tokens):
                            break
                        if shuffled[shuffled_j + shuffled_exact_len] != tokens[t + shuffled_exact_len]:
                            break
                        shuffled_exact_len += 1

                writer.writerow({
                    "trajectory_id": tr["trajectory_id"],
                    "dataset": tr["dataset"],
                    "task_id": tr["task_id"],
                    "model_key": tr["model_key"],
                    "token_index": t,
                    "relative_position": f"{rel:.6f}",
                    "history_length": t,
                    "position_bucket": bucket,
                    "trigger_type": trig_type,
                    "trigger_text": trig_text,
                    "control_type": "shuffled",
                    "matched_control_trajectory_id": tr["trajectory_id"],
                    "exact_accepted_length": shuffled_exact_len,
                    "approx_retrieval_score": "",
                    "verifiable_accepted_length_after_approx_retrieval": "",
                    "total_token_length": n_tokens,
                    "thinking_flag": tr.get("thinking_flag"),
                })

                counts = {}
                for token in tokens[:t]:
                    counts[token] = counts.get(token, 0) + 1
                unigram_guess = max(counts, key=counts.get) if counts else None
                unigram_len = int(unigram_guess is not None and tokens[t] == unigram_guess)
                writer.writerow({
                    "trajectory_id": tr["trajectory_id"],
                    "dataset": tr["dataset"],
                    "task_id": tr["task_id"],
                    "model_key": tr["model_key"],
                    "token_index": t,
                    "relative_position": f"{rel:.6f}",
                    "history_length": t,
                    "position_bucket": bucket,
                    "trigger_type": trig_type,
                    "trigger_text": trig_text,
                    "control_type": "unigram",
                    "matched_control_trajectory_id": "",
                    "exact_accepted_length": unigram_len,
                    "approx_retrieval_score": "",
                    "verifiable_accepted_length_after_approx_retrieval": "",
                    "total_token_length": n_tokens,
                    "thinking_flag": tr.get("thinking_flag"),
                })


if __name__ == "__main__":
    main()
