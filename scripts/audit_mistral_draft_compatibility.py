#!/usr/bin/env python3
"""Verify that a Mistral draft shares the target's ordinary token IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = load_json(args.config)["models"][args.model_key]
    target_id = model["model_id"]
    draft_id = model["draft_model_id"]
    target = load_json(hf_hub_download(target_id, "tekken.json"))
    draft = load_json(hf_hub_download(draft_id, "tekken.json"))

    target_vocab = target["vocab"]
    draft_vocab = draft["vocab"]
    shared = min(len(target_vocab), len(draft_vocab))
    first_mismatch = next(
        (i for i, (left, right) in enumerate(zip(target_vocab, draft_vocab)) if left != right),
        None,
    )
    pattern_match = target["config"]["pattern"] == draft["config"]["pattern"]

    max_generated_id = -1
    special_id_count = 0
    trajectory_count = 0
    for trace_path in args.traces:
        with Path(trace_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                trajectory_count += 1
                ids = row["generated_token_ids"]
                if ids:
                    max_generated_id = max(max_generated_id, max(ids))
                    special_id_count += sum(token_id >= shared for token_id in ids)

    compatible = (
        first_mismatch is None
        and len(target_vocab) == len(draft_vocab)
        and pattern_match
        and special_id_count == 0
    )
    report = {
        "target_model_id": target_id,
        "draft_model_id": draft_id,
        "target_vocab_size": len(target_vocab),
        "draft_vocab_size": len(draft_vocab),
        "ordinary_vocab_exact_match": first_mismatch is None
        and len(target_vocab) == len(draft_vocab),
        "tokenization_pattern_match": pattern_match,
        "trajectory_count": trajectory_count,
        "max_generated_token_id": max_generated_id,
        "generated_ids_outside_shared_vocab": special_id_count,
        "compatible": compatible,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not compatible:
        raise SystemExit("Mistral target/draft tokenizer compatibility audit failed")


if __name__ == "__main__":
    main()
