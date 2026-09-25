#!/usr/bin/env python3
"""Submit reference CPU preparation -> GPU smoke -> paired batch-one experiment."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cpu-partition")
    parser.add_argument("--gpu-partition")
    args = parser.parse_args()
    if not args.traces.is_file():
        raise ValueError("Missing trace input")
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "code"
    files = ["tlar_adaptive_tree.py", "tlar_hf_tree.py",
             "scripts/benchmark_hf_adaptive_tree.py", "scripts/prepare_hf_tree.py",
             "scripts/configure_cache.sh", "scripts/generate_traces.py",
             "tests/test_tlar_hf_tree.py", "tests/test_tlar_adaptive_tree.py"]
    for name in files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    # Freeze the prompt source; old generated answers are never replayed online.
    shutil.copy2(args.traces, out / "input.jsonl")
    py = args.python
    jobs = {}
    for stage in ("prepare", "smoke", "full"):
        commands = ["set -euo pipefail",
                    "source scripts/configure_cache.sh", "export PYTHONUNBUFFERED=1"]
        if stage == "prepare":
            commands += [shlex.join([py, "-m", "unittest", "discover", "-s", "tests", "-p", pattern])
                         for pattern in ("test_tlar_adaptive_tree.py", "test_tlar_hf_tree.py")]
            commands += [shlex.join([py, "scripts/prepare_hf_tree.py", "--output", str(out / "revisions.json")])]
        else:
            commands += ["export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"]
            command = [py, "scripts/benchmark_hf_adaptive_tree.py", "--stage", stage,
                       "--traces", str(out / "input.jsonl"), "--output", str(out / stage),
                       "--revisions", str(out / "revisions.json")]
            if stage == "full":
                command += ["--gate", str(out / "smoke/PASS.json")]
            commands += [shlex.join(command)]
        wrap = "bash -c " + shlex.quote("\n".join(commands))
        cmd = ["sbatch", "--parsable", "--export=ALL", "--cpus-per-task=4",
               "--job-name=hf_tree_" + stage, "--chdir=" + str(snapshot),
               "--output=" + str(out / (stage + "_%j.out")),
               "--error=" + str(out / (stage + "_%j.err")),
               "--mem=" + ("8G" if stage == "prepare" else "32G"),
               "--time=" + {"prepare": "04:00:00", "smoke": "02:00:00", "full": "1-00:00:00"}[stage]]
        partition = args.cpu_partition if stage == "prepare" else args.gpu_partition
        if partition:
            cmd += ["--partition=" + partition]
        if stage != "prepare":
            cmd += ["--gres=gpu:1", "--dependency=afterok:" + jobs["prepare" if stage == "smoke" else "smoke"]]
        result = subprocess.run(cmd + ["--wrap", wrap], check=True, capture_output=True, text=True)
        jid = result.stdout.strip().split(";")[0]
        if not jid.isdigit():
            raise ValueError(result.stdout)
        jobs[stage] = jid
        (out / "jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
        print(stage.upper() + "=" + jid, flush=True)
    print("RUN=" + str(out))


if __name__ == "__main__":
    main()
