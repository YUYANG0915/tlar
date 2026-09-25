# Trajectory-Local Adaptive Retrieval (TLAR)

Anonymous research implementation of trajectory-local retrieval, adaptive
activation and width, prefix-sharing verification trees, and target-coupled
speculative execution.

## Install and verify

Run from the repository root with Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python scripts/check_release.py
```

For model generation, tiny-model tests and GPU execution:

```bash
python -m pip install -r requirements-models.txt
python scripts/check_release.py --models
```

## Experiments

The release covers source controls, fixed-draft complementarity, matched
verification budgets, adaptive control and reward ablations, exact/approximate
matching, event replay, expanded 100-problem mathematics/writing evaluation,
and batched end-to-end timing. `configs/paper_protocol.json` records the paper
settings; `configs/phase1_config.json` selects checkpoints and private input paths.

- [Experiment commands](docs/Experiments.md)
- [Input schemas and dataset preparation](docs/Data.md)

Original inputs, generated trajectories and raw measured logs are external to
this package. Evaluation scripts consume supplied data and write fresh results.
B200 runtime replication additionally uses the original checkpoint revisions
and serving configuration. The paper-to-code map identifies settings specified
by the manuscript and implementation choices exposed by this release.

## Complete offline matrix

Create a private JSON input manifest as described in `docs/DATA.md`, then:

```bash
python scripts/plan_experiments.py \
  --inputs data/private/inputs.json --output outputs/offline --require-complete
```

This validates 15 model/domain/snapshot groups and writes a command plan. Add
`--execute` on a new output directory to execute the matrix. Same-prompt paired
rollouts and GPU audits have separate entry points in the experiment guide.