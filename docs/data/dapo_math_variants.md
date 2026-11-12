# DAPO-Math Variants

Last updated: 11/12/2026.

Several math RL recipes in `verl` (DAPO, SimpleTIR, OTB, DPPO, Score Centering, ReTool) train on variants derived from [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) that differ in **row deduplication**, **initial-model solve-rate filtering**, **answer formatting (`Answer: $Answer` vs. `\boxed{}`)**, and **`data_source` reward-dispatch labels**.

This page catalogs the public Hugging Face datasets in this family and provides `examples/data_preprocess/dapo_math_variants.py` for both reproducible pinned variants (such as `dapo_math_filter_dedup_boxed`) and ad-hoc format conversions (such as converting `Answer:` prompts to `\boxed{}`).

Validation benchmarks (such as AIME 2024–2026) are intentionally kept **separate** from DAPO-Math training preparation.

---

## Summary of Public Hugging Face Datasets

| Hugging Face Dataset | Raw Rows | Unique Prompts | Difficulty Filtering | Prompt Format | Raw `data_source` | Direct `verl` Compatibility & Notes |
| --- | ---: | ---: | --- | --- | --- | --- |
| [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) (`data/dapo-math-17k.parquet`) | 1,791,700 | 17,398 (17,917 UUIDs) | Unfiltered pool | `Answer: $Answer` prefix + suffix | `"math_dapo"` | Works directly with `math_dapo.compute_score`, but each of the 17,917 UUIDs is repeated 100 times (`1,791,700` rows) for multi-epoch streaming. |
| [`aaabiao/dapo_filter`](https://huggingface.co/datasets/aaabiao/dapo_filter) (`train.parquet`) | 13,044 | 12,691 | Filtered by initial-model solve rate ([SimpleTIR](https://arxiv.org/abs/2509.02479)) | `\boxed{}` suffix | `"dapo_filter"` | Needs `data_source -> "math_dapo"` rewrite (`"dapo_filter"` raises `NotImplementedError` in `default_compute_score`); still contains 353 duplicate-prompt rows across distinct UUIDs. |
| [`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter) (`train/dapo_filter.parquet`) | 12,691 | 12,691 | Same SimpleTIR solve-rate filter + first-occurrence `prompt` dedup ([OTB](https://arxiv.org/abs/2602.07078)) | `\boxed{}` suffix | `"dapo_filter"` | Exact prompt-deduplicated `aaabiao/dapo_filter` (`dapo_math_filter_dedup_boxed`). Needs `data_source -> "math_dapo"` rewrite. **Do not** use its `test/aime24.parquet` or `test/aime25.parquet` (byte-identical). |
| [`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed) (`all` / `en` / `cn`) | 17,398 (`en`: 14,116, `cn`: 3,282) | 17,398 | Unfiltered, prompt-deduplicated & language-split | Raw problem string (`prompt: str`, no instruction wrapper) | `"math_dapo"` | Stores `prompt` as a plain string rather than a chat-message list (`source_prompt` holds the original chat list), so it requires wrapping into `[{"role": "user", "content": ...}]` before `RLHFDataset` loading. |
| [`pe-nlp/DAPO-Math-17k-filtered`](https://huggingface.co/datasets/pe-nlp/DAPO-Math-17k-filtered) | 1,790,700 | ~17,388 | Removes 10 problematic UUIDs (1,000 repeated rows) | `Answer: $Answer` prefix + suffix | `"math_dapo"` | Same 100x-repeated layout as `DAPO-Math-17k` minus 10 UUIDs. |

### Lineage of `dapo_math_filter_dedup_boxed`

`dapo_math_filter_dedup_boxed` (prepared from `Jiawei415/DPAO_filter`) reflects two upstream curation stages on top of `BytedTsinghua-SIA/DAPO-Math-17k`:

1. **Initial-model accuracy filtering (`aaabiao/dapo_filter`, 13,044 rows)**: Curated in the [SimpleTIR](https://arxiv.org/abs/2509.02479) setup by evaluating an initial model on `DAPO-Math-17k` and discarding tasks whose solve rate was overly high or overly low, while replacing the `Answer:` wrapper with `\n\nLet's think step by step and output the final answer within \boxed{}.`.
2. **First-occurrence prompt deduplication (`Jiawei415/DPAO_filter`, 12,691 rows)**: Linked in `docs/algo/otb.md` ([OTB](https://arxiv.org/abs/2602.07078)), `train/dapo_filter.parquet` deduplicates `aaabiao/dapo_filter` by first-occurrence `prompt` (`downsample/dedupe_downsample.py --dedupe_keys prompt`), removing 353 duplicate-prompt rows while preserving all 12,691 unique rows, `extra_info.index` UUIDs, ground-truth answers, and row order verbatim.

---

## Preparing Variants with `dapo_math_variants.py`

`examples/data_preprocess/dapo_math_variants.py` verifies pinned Hugging Face revisions/SHA-256 digests, applies exact `data_source` / deduplication / prompt-format transforms, checks `default_compute_score` reward dispatch on every row, and writes `train.parquet` plus an auditable `manifest.json`.

### 1. `dapo_math_filter_dedup_boxed` (Recommended for `\boxed{}` single-turn math RL)

Prepares the pinned 12,691-row `Jiawei415/DPAO_filter` training split by rewriting only `data_source: "dapo_filter" -> "math_dapo"` (preserving all prompts, answers, UUIDs, and row order verbatim):

```bash
python examples/data_preprocess/dapo_math_variants.py \
    --variant dapo_math_filter_dedup_boxed \
    --output-dir ~/data/dapo_math_filter_dedup_boxed
```

### 2. Deduplicating `BytedTsinghua-SIA/DAPO-Math-17k` (`Answer:` or `\boxed{}`)

To collapse the 100x exact repeats in `BytedTsinghua-SIA/DAPO-Math-17k` (`1,791,700 -> 17,917` unique UUIDs, or `17,398` unique problem texts with `--dedupe-prompts`):

```bash
# Keep native "Answer: $Answer" prompt format (17,917 unique UUID rows)
python examples/data_preprocess/dapo_math_variants.py \
    --variant dapo_math_17k_dedup \
    --output-dir ~/data/dapo_math_17k_dedup

# Convert prompts from "Answer: $Answer" wrapper to "\boxed{}" suffix (17,398 unique prompts)
python examples/data_preprocess/dapo_math_variants.py \
    --variant dapo_math_17k_boxed \
    --dedupe-prompts \
    --output-dir ~/data/dapo_math_17k_boxed
```

### 3. Ad-hoc transforms on a local Parquet file (`--variant custom`)

If you already have a local DAPO-format Parquet file and want to convert its prompts to `\boxed{}`, deduplicate by `index` or `prompt`, or normalize `data_source` to `"math_dapo"`:

```bash
python examples/data_preprocess/dapo_math_variants.py \
    --variant custom \
    --input-parquet /path/to/raw_dapo.parquet \
    --dedupe prompt \
    --prompt-format boxed \
    --target-data-source math_dapo \
    --output-dir ~/data/dapo_custom_boxed
```

For multi-turn tool-integrated reasoning (`code_interpreter` `tools_kwargs`), see `examples/data_preprocess/dapo_multiturn_w_tool.py`.

### Offline cache & provenance notes

- **Offline mode**: Pass `--source-dir /path/to/cache` where `<publisher>--<dataset>/<filename>` is pre-staged; byte size and SHA-256 checks still run before processing.
- **Validation separation**: Never use `test/aime24.parquet` or `test/aime25.parquet` from `Jiawei415/DPAO_filter` (their bytes and SHA-256 digests are identical). Prepare AIME or other evaluation benchmarks using dedicated validation scripts.
- **Licensing**: `BytedTsinghua-SIA/DAPO-Math-17k` declares `apache-2.0`; `Jiawei415/DPAO_filter` and `aaabiao/dapo_filter` do not declare a license in their pinned dataset cards (`manifest.json` records `license: null` without inferring usage rights). Verify dataset terms independently before redistribution.
