#!/usr/bin/env python3
"""Generate trace JSONL with vLLM while preserving the Phase-1 schema."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from generate_traces import build_messages, load_json, load_tasks, read_existing_ids, stable_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--config-format", default="auto")
    parser.add_argument("--load-format", default="auto")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    config = load_json(Path(args.config))
    model_cfg = config["models"][args.model_key]
    gen_cfg = dict(config["generation"])
    if args.max_new_tokens is not None:
        gen_cfg["max_new_tokens"] = args.max_new_tokens
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    random.seed(seed)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_ids(output)
    decode_hash = stable_hash({"model_key": args.model_key, "generation": gen_cfg, "seed": seed})

    tasks = load_tasks(config, args.dataset, args.limit)
    pending = []
    for task in tasks:
        trajectory_id = f"{task['dataset']}:{task['task_id']}:{args.model_key}:{seed}:{decode_hash}"
        if trajectory_id not in existing:
            pending.append((task, trajectory_id))
    print(f"Loaded {len(tasks)} tasks; {len(pending)} remain")
    if not pending:
        return

    llm_kwargs = {
        "model": model_cfg["model_id"],
        "revision": model_cfg.get("revision"),
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.tokenizer_mode != "auto":
        llm_kwargs["tokenizer_mode"] = args.tokenizer_mode
    if args.config_format != "auto":
        llm_kwargs["config_format"] = args.config_format
    if args.load_format != "auto":
        llm_kwargs["load_format"] = args.load_format
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=float(gen_cfg.get("temperature", 1.0)),
        top_p=float(gen_cfg.get("top_p", 1.0)),
        max_tokens=int(gen_cfg["max_new_tokens"]),
        seed=seed,
    )
    conversations = [build_messages(task["prompt"]) for task, _ in pending]
    results = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)

    with output.open("a", encoding="utf-8") as out:
        for (task, trajectory_id), result in zip(pending, results):
            candidate = result.outputs[0]
            token_ids = list(candidate.token_ids)
            record = {
                "trajectory_id": trajectory_id,
                "dataset": task["dataset"],
                "task_id": task["task_id"],
                "model_key": args.model_key,
                "model_id": model_cfg["model_id"],
                "draft_model_id": model_cfg.get("draft_model_id"),
                "seed": seed,
                "decode_config_hash": decode_hash,
                "decode_config": gen_cfg,
                "thinking_flag": bool(model_cfg.get("thinking_flag", False)),
                "prompt": task["prompt"],
                "prompt_text": getattr(result, "prompt", ""),
                "reference_answer": task.get("reference_answer"),
                "generated_text": candidate.text,
                "generated_token_ids": token_ids,
                "total_token_length": len(token_ids),
                "answer_correctness": None,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
