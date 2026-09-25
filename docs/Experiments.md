# Experiment entry points

Run commands from the repository root. Model execution uses eager attention,
bfloat16 and full dynamic KV caches on CUDA. Model loading and checkpoint
preparation precede timing. Timed execution includes target/draft prefill,
drafting, retrieval and index updates, tree construction, target verification,
sampling, controller feedback and KV compaction. JSON serialization follows timing.

## Prepare checkpoint revisions

```bash
python scripts/prepare_hf_tree.py \
  --target Qwen/Qwen3-8B --draft Qwen/Qwen3-0.6B \
  --output /path/to/qwen_revisions.json

python scripts/prepare_hf_tree.py \
  --target mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --draft mistralai/Ministral-8B-Instruct-2410 \
  --output /path/to/mistral_revisions.json
```

The tokenizer check verifies identical token-ID vocabularies. Preparation pins
both models and tokenizers to checkpoint commit hashes.

## Throughput and length matrix

The input is JSONL with 100 unique `trajectory_id` or `task_id` values and one of
`prompt_token_ids`, `prompt_text`, or `prompt`. `prompt_token_ids` are exact input
IDs. `prompt_text` is an already-rendered prompt. `prompt` is rendered through the
model chat template. Recorded answers are outside the online generation path.

```bash
python scripts/benchmark_paper.py \
  --model-key qwen3_8b_thinking \
  --prompts /path/to/qwen_prompts.jsonl \
  --revisions /path/to/qwen_revisions.json \
  --output /path/to/qwen_run \
  --batch-sizes 1 32 --lengths 512 1024 2048 4096 \
  --repeats 5 --node-budgets 4 6 8 12
```

Use `mistral_small_3_2_24b` and the corresponding input/revision paths for Mistral.
The retrieval-node cap is explicit: use the cap from the experiment being
compared, or pass all caps for a sweep. SmallDraft prefixes are reserved; STAND
and TLAR retrieval prefixes share the specified cap through feedback-ordered
round-robin merging.
Manifest files record the chosen cap, sampling parameters, seed and backend.

Each configuration warms its execution path. Method order rotates across five
runs. Shared per-request uniform streams couple methods and batch sizes. Sampling
uses target probabilities (temperature 0.6, top-p 0.95 by default). Full-depth
acceptance ends the event; a target-sampled token outside the tree is committed
as a correction. Each prompt generates the requested fixed token count.

```bash
python scripts/summarize_runtime.py \
  --measurements /path/to/qwen_run/measurements.jsonl \
  --output /path/to/qwen_summary.csv
python scripts/audit_execution_records.py \
  --run /path/to/qwen_run --output /path/to/qwen_event_audit.json
```

Summaries use total emitted tokens divided by synchronized elapsed seconds,
medians, sample CV and paired percentage gains with Student-t 95% intervals.

## Distribution audit

Provide a JSON list of 100 held token-ID prefixes. Each is evaluated against
100 shared-uniform streams. Model revisions come from checkpoint preparation.

```bash
python scripts/audit_sampling.py \
  --prefixes /path/to/held_prefixes.json \
  --revisions /path/to/qwen_revisions.json \
  --output /path/to/qwen_sampling_audit \
  --prefix-count 100 --streams 100 --batch-size 32 --node-budget 8
```

The audit records observed logit errors and token mismatches for each method.
The prefix, stream and budget settings are saved with the report. Configure the
node cap and numerical tolerances to match the experiment under inspection.

## Source and seed controls

`analyze_selfcopy.py` restricts Within/Cross continuations to available history.
Shuffled blocks are drawn from the available prefix and queried with the target
context. Unigram counts use the available prefix. Optional trigger metadata uses
supplied `token_char_offsets`; absent offsets produce empty trigger fields.

`analyze_sameprompt_seed_control.py` exposes `--max-edits 0` for exact matching
and `--control-history prefix` for equal-length alternative history. All three targets use exact matching and equal-length alternative prefixes.
Each candidate requires its complete depth
before the source-history boundary.

For offline inputs, configure `configs/phase1_config.json` with local prompt
paths. `generate_traces.py` seeds Python and PyTorch; the vLLM generator passes
its seed to SamplingParams. Install vLLM in the CUDA environment used for that
optional generation backend. `analyze_draft_hits.py` computes draft-model hits;
`submit_causal_suite.py` prepares the offline parameter matrix and audits.

## Cluster configuration

`submit_hf_tree.py` accepts `--python`, `--cpu-partition` and `--gpu-partition`.
`submit_causal_suite.py` accepts `--partition`. Omitted partitions use scheduler
defaults. Activate the desired software environment before submission.

`configure_cache.sh` uses `TLAR_CACHE_ROOT` when supplied, then the user's
XDG cache directory. Account names and installation paths are supplied at runtime.

## Prepare and generate the selected data

Prepare local exports with `prepare_inputs.py` following `docs/DATA.md`. Update
the three relative paths in `configs/phase1_config.json`, then:

```bash
python scripts/generate_traces.py --config configs/phase1_config.json \
  --dataset math --model-key qwen3_8b_thinking --limit 100 --seed 0 \
  --output data/private/math100_qwen.jsonl
python scripts/analyze_draft_hits.py --config configs/phase1_config.json \
  --model-key qwen3_8b_thinking --input data/private/math100_qwen.jsonl \
  --output data/private/math100_qwen_hits.csv
```

Checkpoint revisions for generation should match the archived experiment. Model
revisions can be pinned through `revision` and `draft_revision` in each model's
configuration. The optional vLLM generator follows the same configuration.

## Fixed and expanded-domain evaluation

```bash
python scripts/run_offline.py --traces data/private/math100_qwen.jsonl \
  --draft-hits data/private/math100_qwen_hits.csv \
  --model-key qwen3_8b_thinking --domain math --snapshot expanded \
  --experiment fixed --output outputs/math100_qwen.json
```

Use `--snapshot main` for 24-problem math/writing inputs. Code uses 100 problems.
Recovery, G_miss, Delta and cost are calculated directly and bootstrapped over
trajectory clusters (2,000 resamples). Summaries include input digests.

For code-debugging inputs, `--experiment` also accepts:

- `grid`: all 27 context/tolerance/width settings.
- `matched`: B=4 and B=8 applied separately to each source and union.
- `controller`: full/fixed/activation-only/removed-probe and H/P sweeps.
- `signals`: equal-cost random/periodic/last-hit/hit-EMA/marginal-EMA comparisons.
- `matching`: exact versus approximate under marginal-gain feedback.
- `events`: position-wise and event-driven evaluation.

Cost matching selects by node cost and records the residual difference. Numerical
threshold search is specified in `tlar_offline.py`. The main matrix planner covers
these experiments and the legacy baseline/proxy analysis entry points.

## Same-prompt source control

Place the two 24-problem rollout files in a private directory. Each task must have
both seeds and the model's original tokenization.

```bash
python scripts/analyze_sameprompt_seed_control.py \
  --traces-dir data/private/paired_qwen --model-key qwen3_8b_thinking \
  --max-edits 0 --control-history prefix --context-n 4 --topk 4 --depth 4 \
  --history-gate 512 --output outputs/paired_qwen.csv
```

These dimensions are explicit release defaults; the original source-run manifest
determines their values for archival replication. The same exact/prefix protocol
applies to Mistral and Llama in the current manuscript.

## Systems diagnostics

```bash
python scripts/profile_execution.py --prompts data/private/qwen_prompts.jsonl \
  --revisions data/private/qwen_revisions.json --batch-size 32 \
  --length 1024 --node-budget 8 --output outputs/qwen_profile
python scripts/summarize_systems.py --run outputs/qwen_run \
  --output outputs/qwen_systems.json
```

The profiling command's example budget 8 is a chosen configuration; supply the
cap associated with the run being studied. CPU/CUDA timelines include ranges for
drafting, retrieval/tree construction, metadata transfer, model calls and KV
packing/extraction. `resources.json` records peak allocated/reserved CUDA memory.
Instrumented durations belong to the diagnostic run; the throughput matrix uses
uninstrumented execution. Request-event counts are explicitly distinguished from
batched forward calls. Profiling changes runtime overhead and should be run as a
separate diagnostic experiment.

## Historical snapshots

`analyze_selfcopy.py` accepts traces from the archived DeepSeek source snapshot.
The archived EAGLE-3 run used separate checkpoints and serving settings. Its
original execution archive is required to replicate that table. It serves as a
historical reference in the paper; the matched online TLAR comparison uses the
SmallDraft checkpoint pairs listed in the current configuration.
