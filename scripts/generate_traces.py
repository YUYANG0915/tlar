#!/usr/bin/env python3
"""Generate long-CoT traces with a stable JSONL schema and resume support."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset
from tqdm import tqdm


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def read_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["trajectory_id"])
            except Exception:
                continue
    return ids


def pick_field(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, str):
                return value
            return json.dumps(value, ensure_ascii=False)
    return None


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_tasks(config: Dict[str, Any], dataset_key: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    ds_cfg = config["datasets"][dataset_key]
    rows: Iterable[Dict[str, Any]]
    if ds_cfg["kind"] == "jsonl":
        path = Path(ds_cfg["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        rows = iter_jsonl(path)
    elif ds_cfg["kind"] == "hf":
        kwargs = {}
        if ds_cfg.get("name"):
            kwargs["name"] = ds_cfg["name"]
        split = ds_cfg.get("split", "test")
        rows = load_dataset(ds_cfg["path"], **kwargs, split=split)
    else:
        raise ValueError(f"Unknown dataset kind: {ds_cfg['kind']}")

    tasks: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        row = dict(row)
        prompt = pick_field(row, ds_cfg.get("prompt_fields", []))
        if not prompt:
            available = ", ".join(row.keys())
            raise ValueError(f"No prompt field found for {dataset_key}; available fields: {available}")
        answer = pick_field(row, ds_cfg.get("answer_fields", []))
        task_id = str(row.get("task_id") or row.get("id") or row.get("problem_id") or row.get("question_id") or i)
        tasks.append({
            "dataset": dataset_key,
            "task_id": task_id,
            "prompt": prompt,
            "reference_answer": answer,
            "raw": row,
        })
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are solving a reasoning task. Think carefully and show your reasoning. "
                "When useful, re-check assumptions and compare alternatives before giving the final answer."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def render_prompt(tokenizer: Any, prompt: str, thinking_flag: bool) -> str:
    messages = build_messages(prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking_flag,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {prompt}\nAssistant:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Load tasks and print examples without loading a model.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_json(config_path)
    model_cfg = config["models"][args.model_key]
    gen_cfg = dict(config["generation"])
    if args.max_new_tokens is not None:
        gen_cfg["max_new_tokens"] = args.max_new_tokens
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    random.seed(seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_ids(output)

    print(f"Loading tasks: dataset={args.dataset} limit={args.limit}")
    tasks = load_tasks(config, args.dataset, args.limit)
    print(f"Loaded {len(tasks)} tasks")
    if args.dry_run:
        for task in tasks[:5]:
            preview = task["prompt"].replace("\n", " ")[:240]
            print(json.dumps({
                "dataset": task["dataset"],
                "task_id": task["task_id"],
                "prompt_preview": preview,
                "has_reference_answer": task.get("reference_answer") is not None,
            }, ensure_ascii=False))
        return

    print(f"Loading model: {model_cfg['model_id']}")
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], revision=model_cfg.get("revision"), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        revision=model_cfg.get("revision"),
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
    )
    model.eval()

    decode_hash = stable_hash({"model_key": args.model_key, "generation": gen_cfg, "seed": seed})
    thinking_flag = bool(model_cfg.get("thinking_flag", False))

    with output.open("a", encoding="utf-8") as out:
        for task in tqdm(tasks, desc="generating"):
            trajectory_id = f"{task['dataset']}:{task['task_id']}:{args.model_key}:{seed}:{decode_hash}"
            if trajectory_id in existing:
                continue

            prompt_text = render_prompt(tokenizer, task["prompt"], thinking_flag)
            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            prompt_len = int(inputs["input_ids"].shape[-1])

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(gen_cfg["max_new_tokens"]),
                    temperature=float(gen_cfg.get("temperature", 1.0)),
                    top_p=float(gen_cfg.get("top_p", 1.0)),
                    do_sample=bool(gen_cfg.get("do_sample", True)),
                    pad_token_id=tokenizer.eos_token_id,
                )[0]

            generated_ids = output_ids[prompt_len:].detach().cpu().tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            record = {
                "trajectory_id": trajectory_id,
                "dataset": task["dataset"],
                "task_id": task["task_id"],
                "model_key": args.model_key,
                "model_id": model_cfg["model_id"],
                "model_revision": getattr(model.config, "_commit_hash", None),
                "draft_model_id": model_cfg.get("draft_model_id"),
                "seed": seed,
                "decode_config_hash": decode_hash,
                "decode_config": gen_cfg,
                "thinking_flag": thinking_flag,
                "prompt": task["prompt"],
                "prompt_text": prompt_text,
                "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
                "reference_answer": task.get("reference_answer"),
                "generated_text": generated_text,
                "generated_token_ids": generated_ids,
                "total_token_length": len(generated_ids),
                "answer_correctness": None,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
