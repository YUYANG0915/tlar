#!/usr/bin/env python3
"""Single-request greedy diagnostic benchmark."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from tlar_adaptive_tree import Config
from tlar_hf_tree import Backend, MODES, check_tree_logits, decode


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def prompt_ids(tokenizer, row):
    if row.get("prompt_token_ids"):
        return list(row["prompt_token_ids"])
    if row.get("prompt_text"):
        return tokenizer.encode(row["prompt_text"], add_special_tokens=False)
    if row.get("prompt"):
        from generate_traces import build_messages
        encoded = tokenizer.apply_chat_template(build_messages(row["prompt"]),
                    tokenize=True, add_generation_prompt=True, return_dict=False)
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        return list(encoded)
    # Never read the recorded answer or generated tokens for online generation.
    raise ValueError("No usable prompt")


def load_model(model_id, revision):
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    cls = AutoModelForImageTextToText if cfg.model_type == "mistral3" else AutoModelForCausalLM
    model, info = cls.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="eager",
        output_loading_info=True,
    )
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Incomplete checkpoint: {info}")
    return Backend(model), dict(revision=cfg._commit_hash, config=cfg.to_dict(), loading=info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--revisions", type=Path, required=True)
    parser.add_argument("--target", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument("--draft", default="mistralai/Ministral-8B-Instruct-2410")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = Config()
    code = Path(__file__).resolve().parents[1]
    identity = dict(
        target=args.target, draft=args.draft, input_sha256=sha(args.traces),
        source_sha256={name: sha(code / name) for name in (
            "tlar_adaptive_tree.py", "tlar_hf_tree.py", "scripts/benchmark_hf_adaptive_tree.py",
            "scripts/generate_traces.py")},
        config=asdict(cfg), modes=MODES, batch_size=1, greedy=True,
        ignore_eos=True, attention="eager", dtype="bfloat16",
        versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
    )
    # JSON normalization makes tuple/list comparison independent of stage.
    identity = json.loads(json.dumps(identity))
    gate = None
    if args.stage == "full":
        if not args.gate:
            raise ValueError("Full run requires a successful smoke gate")
        gate = json.loads(args.gate.read_text())
        if gate.get("passed") is not True or gate["identity"] != identity:
            raise ValueError("Smoke failed or source/data/config/environment changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model experiment requires CUDA")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write(args.output / "manifest.json", dict(identity=identity, stage=args.stage,
          python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
          scope="HF eager single-request greedy diagnostic"))
    pinned = json.loads(args.revisions.read_text())
    if pinned["target"] != args.target or pinned["draft"] != args.draft:
        raise ValueError("Prepared checkpoints differ from requested models")
    target_revision = pinned["target_revision"]
    draft_revision = pinned["draft_revision"]
    if gate and (gate["target_revision"] != target_revision or gate["draft_revision"] != draft_revision):
        raise ValueError("Prepared revisions differ from the smoke revisions")
    tokenizer = AutoTokenizer.from_pretrained(args.target, revision=target_revision)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft, revision=draft_revision)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Target/draft token IDs differ: vocabulary bridge NOT implemented")
    target, target_info = load_model(args.target, target_revision)
    draft, draft_info = load_model(args.draft, draft_revision)
    if (getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") not in (None, target_info["revision"]) or
        getattr(draft_tokenizer, "init_kwargs", {}).get("_commit_hash") not in (None, draft_info["revision"])):
        raise ValueError("Tokenizer/model revisions changed during startup")
    write(args.output / "checkpoints.json", dict(target=target_info, draft=draft_info))
    with args.traces.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 100 or len({r["trajectory_id"] for r in rows}) != 100:
        raise ValueError("Expected 100 unique code-debug prompts")
    count, budget, repeats = (2, 640, 1) if args.stage == "smoke" else (100, 1024, 3)
    prompts = [prompt_ids(tokenizer, row) for row in rows[:count]]
    if any(not p or min(p) < 0 for p in prompts):
        raise ValueError("Invalid prompt IDs")
    write(args.output / "prompts.json", prompts)
    checks = check_tree_logits(target, prompts[0][-16:])
    write(args.output / "tree_checks.json", checks)
    # Identical short warmups; all timings below include both models' prefill.
    for mode in MODES:
        decode(target, draft, prompts[0], 16, mode, cfg)
    summaries, adaptive_attempts, adaptive_candidates = [], 0, 0
    with (args.output / "rounds.jsonl").open("w") as events, (args.output / "outputs.jsonl").open("w") as outputs:
        for repeat in range(repeats):
            for i, prompt in enumerate(prompts):
                reference = None
                # Rotate order to avoid always measuring the hybrid last.
                shift = (repeat + i) % len(MODES)
                modes = MODES[shift:] + MODES[:shift]
                for mode in modes:
                    tokens, rounds, elapsed = decode(target, draft, prompt, budget, mode, cfg)
                    record = dict(repeat=repeat, trajectory_id=rows[i]["trajectory_id"],
                                  mode=mode, tokens=tokens, elapsed_sec=elapsed)
                    outputs.write(json.dumps(record) + "\n")
                    outputs.flush()
                    if reference is None:
                        reference = tokens
                    elif tokens != reference:
                        first = next((j for j, (a, b) in enumerate(zip(tokens, reference)) if a != b), min(len(tokens), len(reference)))
                        write(args.output / "mismatch.json", dict(record=record, reference=reference, first_difference=first))
                        raise AssertionError("Paired greedy outputs differ; full run blocked")
                    for event in rounds:
                        event.update(repeat=repeat, trajectory_id=rows[i]["trajectory_id"], mode=mode)
                        events.write(json.dumps(event) + "\n")
                    events.flush()
                    if mode == "adaptive_union":
                        adaptive_attempts += sum(r["active"] for r in rounds)
                        adaptive_candidates += sum(bool(r["candidate_starts"]) for r in rounds)
                    summaries.append(dict(repeat=repeat, prompt=i, mode=mode,
                                          generated_tokens=len(tokens), elapsed_sec=elapsed,
                                          tokens_per_sec=len(tokens) / elapsed))
                    print(f"{args.stage}: repeat={repeat} prompt={i + 1}/{count} mode={mode} sec={elapsed:.2f}", flush=True)
    write(args.output / "measurements.json", summaries)
    if not adaptive_attempts or not adaptive_candidates:
        raise AssertionError("No actual adaptive retrieval candidates; smoke insufficient")
    write(args.output / "PASS.json", dict(passed=True, identity=identity,
          target_revision=target_info["revision"], draft_revision=draft_info["revision"],
          adaptive_attempts=adaptive_attempts, adaptive_candidate_rounds=adaptive_candidates,
          note="Greedy HF single-request diagnostic with exact-history ablation."))


if __name__ == "__main__":
    main()
