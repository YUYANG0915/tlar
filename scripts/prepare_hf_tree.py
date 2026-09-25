#!/usr/bin/env python3
"""Download/check checkpoints on CPU before reserving a GPU."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    parser.add_argument("--draft", default="mistralai/Ministral-8B-Instruct-2410")
    args = parser.parse_args()
    ids = dict(target=args.target, draft=args.draft)
    api = HfApi()
    tokenizers = []
    for role in ("target", "draft"):
        model_id = ids[role]
        revision = api.model_info(model_id).sha
        ids[role + "_revision"] = revision
        # Load/check token IDs before downloading tens of GB of weights.
        tokenizers.append(AutoTokenizer.from_pretrained(model_id, revision=revision))
    if tokenizers[0].get_vocab() != tokenizers[1].get_vocab():
        raise RuntimeError("Token-ID mismatch; no unvalidated vocabulary bridge is allowed")
    for role in ("target", "draft"):
        snapshot_download(ids[role], revision=ids[role + "_revision"],
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.tiktoken"],
                          max_workers=2)
    with args.output.open("x") as stream:
        json.dump(ids, stream, indent=2)
    print("CHECKPOINTS READY", flush=True)


if __name__ == "__main__":
    main()
