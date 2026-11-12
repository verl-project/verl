# DAPO-Math filtered & deduplicated boxed training dataset preparation

Last updated: 11/12/2026.

`examples/data_preprocess/dapo_math_boxed.py` prepares the 12,691-prompt difficulty-filtered and prompt-deduplicated boxed math training dataset from [`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter) (`train/dapo_filter.parquet`, default) or [`aaabiao/dapo_filter`](https://huggingface.co/datasets/aaabiao/dapo_filter) (`train.parquet`).

```bash
python examples/data_preprocess/dapo_math_boxed.py --output-dir ~/data/dapo-math-boxed
```

## Dataset lineage: initial-model accuracy filtering and prompt deduplication

This dataset is **not** simply a prompt-deduplicated copy of [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k); it reflects two upstream curation stages:

1. **Initial-model accuracy filtering ([`aaabiao/dapo_filter`](https://huggingface.co/datasets/aaabiao/dapo_filter), 13,044 rows)**: Curated in the [SimpleTIR](https://arxiv.org/abs/2509.02479) setup by filtering `BytedTsinghua-SIA/DAPO-Math-17k` according to an initial model's empirical solve rate—discarding tasks whose accuracy is overly high or overly low—and formatting prompts with `\n\nLet's think step by step and output the final answer within \boxed{}.` (`data_source="dapo_filter"`). This split retains 13,044 rows (`12,691` unique prompts plus `353` duplicate-prompt entries inherited from repeated problem texts across distinct UUIDs in `DAPO-Math-17k`).
2. **First-occurrence prompt deduplication ([`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter), 12,691 rows)**: Linked in `docs/algo/otb.md` ([OTB](https://arxiv.org/abs/2602.07078)), `train/dapo_filter.parquet` applies deterministic first-occurrence deduplication on `prompt` (`downsample/dedupe_downsample.py --dedupe_keys prompt`) to `aaabiao/dapo_filter`, removing the 353 duplicate-prompt rows and preserving all 12,691 unique rows, `extra_info.index` UUIDs, ground-truth answers, and row order verbatim.

When `--source aaabiao/dapo_filter` is passed, `dapo_math_boxed.py` applies the exact same first-occurrence `prompt` deduplication (`13,044 -> 12,691` rows), producing **row-for-row identical** prepared rows (`canonical_rows_sha256`) to `--source Jiawei415/DPAO_filter`.

## Why a preparation step is required

In both pinned upstream files, every row stores `data_source="dapo_filter"`, which raises `NotImplementedError` in `verl.utils.reward_score.default_compute_score`.

`dapo_math_boxed.py` performs an auditable normalization:

- Pins exact Hugging Face revisions, byte sizes, and SHA-256 digests for both `Jiawei415/DPAO_filter` (`d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd`, `2,428,814` bytes, SHA-256 `47a9806fdb10be56336651384f0c1a7d2c7de6be2883b1b64a7d559bf6f9e39c`) and `aaabiao/dapo_filter` (`0b10fea35812902fe4f6985f2064424ec460f9c0`, `2,395,267` bytes, SHA-256 `78b9c05dfa36982ac4b44c76c9a05335979c1d37371425271fa7f29fd2e7e436`).
- Verifies all 12,691 unique rows have distinct `extra_info.index` UUIDs, distinct prompts, and non-empty boxed problem statements.
- Rewrites only `data_source` from `"dapo_filter"` to `"math_dapo"` while preserving `prompt`, `ability`, `reward_model`, `extra_info`, and row order verbatim.
- Verifies `default_compute_score("math_dapo", ...)` on every row (`\boxed{ground_truth}` -> `+1.0`, incorrect sentinel -> `-1.0`), checks Parquet round-trip equality, and writes `train.parquet` alongside a provenance `manifest.json`.
- Supports `--source-dir` for offline environments where `<publisher>--<dataset>/<filename>` is pre-staged; byte size and SHA-256 verification still apply.

## Keeping training and validation separate

This script intentionally prepares **training data only** and never downloads or consumes `test/aime24.parquet` or `test/aime25.parquet` from `Jiawei415/DPAO_filter` (those two files are byte-for-byte identical in the upstream repository). Prepare AIME or other benchmark validation splits with dedicated validation preprocessing scripts so training and evaluation artifacts remain cleanly decoupled.

## Provenance and licensing note

Neither `Jiawei415/DPAO_filter` nor `aaabiao/dapo_filter` declares a license in its pinned Hugging Face dataset metadata; `manifest.json` records `license: null` explicitly rather than inferring usage rights. Do not assume that generated Parquet files inherit `verl`'s Apache-2.0 code license—verify dataset usage terms independently before redistribution.
