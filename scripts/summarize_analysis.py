#!/usr/bin/env python3
"""Small summaries for smoke-test self-copy CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcopy", required=True)
    args = parser.parse_args()

    stats = defaultdict(lambda: {"n": 0, "exact_sum": 0.0, "approx_sum": 0.0})
    with Path(args.selfcopy).open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["model_key"], row["control_type"])
            stats[key]["n"] += 1
            stats[key]["exact_sum"] += float(row["exact_accepted_length"])
            approx = row.get("verifiable_accepted_length_after_approx_retrieval") or 0
            stats[key]["approx_sum"] += float(approx)

    print("dataset,model_key,control_type,n,E_exact,E_verifiable_after_approx")
    for key, val in sorted(stats.items()):
        n = max(1, val["n"])
        print(",".join([
            key[0],
            key[1],
            key[2],
            str(val["n"]),
            f"{val['exact_sum'] / n:.4f}",
            f"{val['approx_sum'] / n:.4f}",
        ]))


if __name__ == "__main__":
    main()
