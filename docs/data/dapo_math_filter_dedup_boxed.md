# DAPO-Math filtered & deduplicated boxed training data (`dapo_math_filter_dedup_boxed`)

Last updated: 11/12/2026.

`examples/data_preprocess/dapo_math_filter_dedup_boxed.py` prepares the 12,691-prompt difficulty-filtered and prompt-deduplicated boxed math training dataset (`dapo_math_filter_dedup_boxed`) from [`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter) (`train/dapo_filter.parquet`).

```bash
python examples/data_preprocess/dapo_math_filter_dedup_boxed.py \
    --output-dir ~/data/dapo_math_filter_dedup_boxed
```

## Dataset lineage

`Jiawei415/DPAO_filter` reflects two upstream curation stages on top of [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k):

1. **Initial-model accuracy filtering ([`aaabiao/dapo_filter`](https://huggingface.co/datasets/aaabiao/dapo_filter), 13,044 rows)**: Used in the [SimpleTIR](https://arxiv.org/abs/2509.02479) setup, tasks from `DAPO-Math-17k` were filtered according to an initial model's solve rate to discard overly easy or overly hard problems, and formatted with the `\n\nLet's think step by step and output the final answer within \boxed{}.` instruction (`data_source="dapo_filter"`). That split contains 13,044 rows (`12,691` unique prompts plus `353` duplicate-prompt entries inherited from repeated problem texts across distinct UUIDs).
2. **First-occurrence prompt deduplication ([`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter), 12,691 rows)**: Linked in `docs/algo/otb.md` ([OTB](https://arxiv.org/abs/2602.07078)), `train/dapo_filter.parquet` deduplicates `aaabiao/dapo_filter` by first-occurrence `prompt` (`downsample/dedupe_downsample.py --dedupe_keys prompt`), removing the 353 duplicate-prompt rows while preserving all 12,691 unique rows, `extra_info.index` UUIDs, ground-truth answers, and row order verbatim.

## Why a preparation step is required

In the pinned upstream file (`train/dapo_filter.parquet`), `data_source` is set to `"dapo_filter"`, which raises `NotImplementedError` in `verl.utils.reward_score.default_compute_score`.

`dapo_math_filter_dedup_boxed.py` performs an auditable, single-field normalization:

- Pins the exact Hugging Face revision (`d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd`), file size (`2,428,814` bytes), and SHA-256 (`47a9806fdb10be56336651384f0c1a7d2c7de6be2883b1b64a7d559bf6f9e39c`).
- Verifies all 12,691 rows have unique `extra_info.index` UUIDs, unique prompts, and non-empty boxed problem statements.
- Rewrites only `data_source` from `"dapo_filter"` to `"math_dapo"` while preserving `prompt`, `ability`, `reward_model`, `extra_info`, and row order verbatim.
- Verifies `default_compute_score("math_dapo", ...)` on every row (`\boxed{ground_truth}` -> `+1.0`, incorrect sentinel -> `-1.0`), checks Parquet round-trip equality, and writes `train.parquet` alongside a provenance `manifest.json`.
- Supports `--source-dir` for offline environments where `Jiawei415--DPAO_filter/train/dapo_filter.parquet` is pre-staged; byte size and SHA-256 verification still apply.

## Keeping training and validation separate

This script intentionally prepares **training data only** and never downloads or consumes `test/aime24.parquet` or `test/aime25.parquet` from `Jiawei415/DPAO_filter` (those two files are byte-for-byte identical in the upstream repository). Prepare AIME or other benchmark validation splits with dedicated validation preprocessing scripts so training and evaluation artifacts remain cleanly decoupled.

## Provenance and licensing note

The pinned `Jiawei415/DPAO_filter` dataset metadata does not declare a license; `manifest.json` records `license: null` explicitly rather than inferring usage rights. Do not assume that generated Parquet files inherit `verl`'s Apache-2.0 code license—verify dataset usage terms independently before redistribution.
