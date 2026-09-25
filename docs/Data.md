# Input contracts

Keep experiment inputs in `data/private/`, excluded by `.gitignore`. Checkpoint
credentials come from the environment/local Hugging Face login. Configuration
files contain experiment parameters and paths.

## Dataset exports and selection

Supply local JSONL exports for:

- SWE-bench Verified: `instance_id`, `repo`, `base_commit`, `problem_statement`.
- MATH test: `canonical_id`, `type`, `problem`. Preserve dataset-relative canonical
  IDs and the original subject labels. Accepted subjects: Algebra, Counting &
  Probability, Geometry, Number Theory.
- WritingPrompts: `canonical_id`, `prompt`.

Gold patches, tests, solutions and reference stories stay in the original export.
The prompt builder copies the documented input fields. Record the dataset revision
or export hash in the selection manifest.

```bash
python scripts/prepare_inputs.py --input data/private/math_export.jsonl \
  --domain math --count 100 --dataset-revision EXPORT_REVISION \
  --output data/private/math.jsonl
```

Use `--count 24` for the original cross-domain set. Canonical sorting followed by
a seeded permutation makes the first 24 a subset of the 100 for the same export.
The seed is 20260217. For exact archival replication, use the original selected IDs
and rendered prompts; prompt formatting and permutation implementation are explicit
release conventions. New paper runs should retain the selection manifest.

## Generated trace JSONL

Each row contains:

- `trajectory_id`: unique string; `task_id`: stable problem ID.
- `dataset`: `code_debug`, `math` or `open_ended`.
- `model_key`, `model_id`, `seed`.
- `generated_token_ids`: target token IDs, excluding the prompt.
- `prompt_token_ids`: exact rendered target prompt IDs (preferred); alternatively
  `prompt_text`, or the raw `prompt` with recorded template settings.
- Optional `generated_text`, `thinking_flag`, `decode_config`.

The trace generator records prompt IDs and sampling settings. Offline comparisons
share the same target traces and draft predictions. Two same-prompt seeds live in
a separate directory and share `task_id` values.

## Draft-hit CSV

Required columns: `trajectory_id,token_index,target_token_id,draft_top1_hit`.
Indices are zero-based in generated tokens. Supply every token exactly once.
`draft_top1_hit` is 0 or 1. The validator checks the target ID against the trace.

## Matrix input manifest

A JSON list, one object per model/domain/snapshot:

```json
[
  {
    "snapshot": "expanded",
    "domain": "math",
    "model_key": "qwen3_8b_thinking",
    "traces": "data/private/math100_qwen.jsonl",
    "draft_hits": "data/private/math100_qwen_hits.csv"
  }
]
```

A complete matrix has nine main groups (3 models x 3 domains) and six expanded
groups (3 models x math/writing). Main counts are 100/24/24; expanded counts are
100 per domain and target. The repository ships the schema, leaving private
records to the experiment owner.

## Online prompts and audits

Online prompts contain a unique `task_id` or `trajectory_id`, plus exact
`prompt_token_ids` or rendered `prompt_text`. The timing matrix requires 100
unique prompts. Recorded generated answers stay outside its generation path.
Sampling audits take a JSON array of 100 nonempty lists of held-prefix token IDs.
Repetition filtering takes an explicit reviewed JSON list of excluded trajectory
IDs; the seven historical exclusions require the original review record.

The matrix manifest accepts an optional `source_exclude_ids` path for a reviewed
source-only repetition exclusion list. This filters the source comparison while
retaining the full task set for the separate drafter-complementarity evaluation.
