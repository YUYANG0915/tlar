#!/usr/bin/env python3
"""Teacher-forcing draft top-1/top-5 hit analysis for complementarity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from tqdm import tqdm

from generate_traces import build_messages


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_prompt(tokenizer: Any, trace: dict) -> list[int]:
    """Recover the exact target-side prompt prefix used for generation."""
    if trace.get("prompt_token_ids"):
        return list(trace["prompt_token_ids"])
    prompt_text = trace.get("prompt_text")
    if prompt_text:
        return tokenizer.encode(prompt_text, add_special_tokens=False)

    raw_prompt = trace.get("prompt")
    if not raw_prompt:
        raise ValueError(f"Trace {trace.get('trajectory_id')} has no usable prompt")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-generated-tokens", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    model_config = config["models"][args.model_key]
    target_model_id = model_config["model_id"]
    draft_model_id = model_config["draft_model_id"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_tokenizer = AutoTokenizer.from_pretrained(
        target_model_id, revision=model_config.get("revision"), trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        revision=model_config.get("draft_revision"),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    rows = []
    for tr in tqdm(list(iter_jsonl(Path(args.input))), desc="draft hits"):
        if tr["model_key"] != args.model_key:
            continue
        prompt_ids = encode_prompt(target_tokenizer, tr)
        gen_ids = tr["generated_token_ids"]
        if args.max_generated_tokens:
            gen_ids = gen_ids[: args.max_generated_tokens]
        full_ids = prompt_ids + gen_ids
        if len(full_ids) < 2:
            continue
        if max(full_ids) >= model.config.vocab_size:
            raise ValueError(
                f"Token id exceeds draft vocabulary for {tr['trajectory_id']}: "
                f"max={max(full_ids)}, vocab={model.config.vocab_size}"
            )

        input_ids = torch.tensor([full_ids[:-1]], device=model.device)
        labels = full_ids[1:]
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0]

        prompt_offset = max(0, len(prompt_ids) - 1)
        for local_i, target_token in enumerate(gen_ids):
            logit_i = prompt_offset + local_i
            if logit_i >= logits.shape[0]:
                break
            top5 = torch.topk(logits[logit_i], k=5).indices.detach().cpu().tolist()
            top1 = top5[0]
            rows.append({
                "trajectory_id": tr["trajectory_id"],
                "dataset": tr["dataset"],
                "task_id": tr["task_id"],
                "model_key": tr["model_key"],
                "draft_model_id": draft_model_id,
                "token_index": local_i,
                "draft_top1_hit": int(top1 == target_token),
                "draft_top5_hit": int(target_token in top5),
                "draft_miss_flag": int(top1 != target_token),
                "target_token_id": target_token,
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory_id", "dataset", "task_id", "model_key", "draft_model_id",
        "token_index", "draft_top1_hit", "draft_top5_hit", "draft_miss_flag",
        "target_token_id",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
