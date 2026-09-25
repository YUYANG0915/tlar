#!/usr/bin/env python3
"""Filter length-capped trajectories and their position-level draft records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--draft-hits")
    parser.add_argument("--output-traces", required=True)
    parser.add_argument("--output-draft-hits")
    parser.add_argument("--cap", type=int, default=4096)
    args = parser.parse_args()

    output_traces = Path(args.output_traces)
    output_traces.parent.mkdir(parents=True, exist_ok=True)
    retained_ids: set[str] = set()
    kept = dropped = 0
    with Path(args.traces).open(encoding="utf-8") as source, output_traces.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["total_token_length"]) >= args.cap:
                dropped += 1
                continue
            retained_ids.add(row["trajectory_id"])
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    if bool(args.draft_hits) != bool(args.output_draft_hits):
        raise SystemExit("--draft-hits and --output-draft-hits must be provided together")
    kept_draft_rows = 0
    if args.draft_hits:
        output_draft = Path(args.output_draft_hits)
        output_draft.parent.mkdir(parents=True, exist_ok=True)
        with Path(args.draft_hits).open(newline="", encoding="utf-8") as source, output_draft.open(
            "w", newline="", encoding="utf-8"
        ) as destination:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["trajectory_id"] in retained_ids:
                    writer.writerow(row)
                    kept_draft_rows += 1

    print(
        json.dumps(
            {
                "cap": args.cap,
                "kept_trajectories": kept,
                "dropped_trajectories": dropped,
                "kept_draft_rows": kept_draft_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
