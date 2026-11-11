# DAPO-Math boxed training dataset preparation

Last updated: 11/11/2026.

`examples/data_preprocess/dapo_math_boxed.py` prepares the deduplicated 12,691-prompt boxed math training set from [`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter) (`train/dapo_filter.parquet`).

```bash
python examples/data_preprocess/dapo_math_boxed.py --output-dir ~/data/dapo-math-boxed
```

## Why a preparation step is required

In the pinned upstream file (`train/dapo_filter.parquet`), every row already carries a single user message ending with `\n\nLet's think step by step and output the final answer within \boxed{}.` and a rule-based `reward_model.ground_truth`, but `data_source` is set to `"dapo_filter"`, which raises `NotImplementedError` in `verl.utils.reward_score.default_compute_score`.

`dapo_math_boxed.py` performs an auditable, single-field normalization:

- Pins the exact Hugging Face revision (`d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd`), file size (`2,428,814` bytes), and SHA-256 (`47a9806fdb10be56336651384f0c1a7d2c7de6be2883b1b64a7d559bf6f9e39c`).
- Verifies all 12,691 rows have unique `extra_info.index` UUIDs, unique prompts, and non-empty boxed problem statements.
- Rewrites only `data_source` from `"dapo_filter"` to `"math_dapo"` while preserving `prompt`, `ability`, `reward_model`, `extra_info`, and row order verbatim.
- Verifies `default_compute_score("math_dapo", ...)` on every row (`\boxed{ground_truth}` -> `+1.0`, incorrect sentinel -> `-1.0`), checks Parquet round-trip equality, and writes `train.parquet` alongside a provenance `manifest.json`.
- Supports `--source-dir` for offline environments where `Jiawei415--DPAO_filter/train/dapo_filter.parquet` is pre-staged; byte size and SHA-256 verification still apply.

## Keeping training and validation separate

This script intentionally prepares **training data only** and never downloads or consumes `test/aime24.parquet` or `test/aime25.parquet` from `Jiawei415/DPAO_filter` (those two files are byte-for-byte identical in the upstream repository). Prepare AIME or other benchmark validation splits with dedicated validation preprocessing scripts so training and evaluation artifacts remain cleanly decoupled.

## Provenance and licensing note

The pinned `Jiawei415/DPAO_filter` dataset metadata does not declare a license; `manifest.json` records `license: null` explicitly rather than inferring usage rights. Do not assume that generated Parquet files inherit `verl`'s Apache-2.0 code license—verify dataset usage terms independently before redistribution.
