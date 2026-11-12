# DAPO-Math Variants

Last updated: 11/12/2026.

Several math RL recipes in `verl` (DAPO, SimpleTIR, OTB, DPPO, Score Centering, ReTool) train on variants derived from [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k). Public Hugging Face releases already cover **UUID deduplication**, **prompt deduplication**, **`\boxed{}` vs. `Answer:` formatting**, and **initial-model solve-rate filtering**, so single-turn training does **not** require a custom preprocessing script.

Validation benchmarks (such as AIME 2024–2026) are intentionally kept **separate** from DAPO-Math training datasets.

---

## Summary of Public Hugging Face Datasets

| Hugging Face Dataset | Parquet Path | Rows | Curation / Filtering | Prompt Format | Raw `data_source` | Direct `verl` CLI Usage |
| --- | --- | ---: | --- | --- | --- | --- |
| [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | `data/dapo-math-17k.parquet` | 1,791,700 | Raw release: 100× exact repeats of 17,917 UUIDs (17,398 unique prompts) | `Answer: $Answer` prefix + suffix | `"math_dapo"` | Ready out-of-the-box (`default_compute_score` routes `"math_dapo"` to `math_dapo.compute_score`). |
| [`tongyx361/DAPO-Math-Raw-17k`](https://huggingface.co/datasets/tongyx361/DAPO-Math-Raw-17k) | `dapo-math-raw-17k.parquet` | 17,917 | Deduplicated by `extra_info.index` UUID | `Answer: $Answer` prefix + suffix | `"MATH##*"` (10 sub-sources) | Zero-edit with `reward.preset=math_dapo` or `'reward.data_source_presets=["MATH##*=math_dapo"]'`. |
| [`tongyx361/DAPO-Math-Unique-17k`](https://huggingface.co/datasets/tongyx361/DAPO-Math-Unique-17k) | `data/dapo-math-unique-17k.parquet` | 17,398 | Deduplicated by `raw_problem_id` (unique problem text) | `Answer: $Answer` prefix + suffix | `"MATH##*"` | Zero-edit with `reward.preset=math_dapo` or `'reward.data_source_presets=["MATH##*=math_dapo"]'`. |
| [`YouJiacheng/DAPO-Math-17k-dedup`](https://huggingface.co/datasets/YouJiacheng/DAPO-Math-17k-dedup) | `distinct-prompts-with-rewards.parquet` | 17,398 | Deduplicated by distinct `prompt` | `Answer: $Answer` prefix + suffix | `"math_dapo"` | Ready out-of-the-box. |
| [`tongyx361/DAPO-Math-Unique-Boxed-17k`](https://huggingface.co/datasets/tongyx361/DAPO-Math-Unique-Boxed-17k) | `data/dapo-math-unique-boxed-17k.parquet` | 17,398 | Deduplicated by `raw_problem_id` | `\boxed{...}` suffix | `"MATH##*"` | Zero-edit with `reward.preset=math_dapo` or `'reward.data_source_presets=["MATH##*=math_dapo"]'`. |
| [`tongyx361/DAPO-Math-Unique-Clean-17k`](https://huggingface.co/datasets/tongyx361/DAPO-Math-Unique-Clean-17k) | `data/dapo-math-unique-clean-17k.parquet` | 17,398 | Deduplicated by `raw_problem_id` | Clean chat prompt (no wrapper) | `"MATH##*"` | Useful when your model chat template or system prompt supplies the answer instruction; pair with `reward.preset=math_dapo` or `math_verify`. |
| [`aaabiao/dapo_filter`](https://huggingface.co/datasets/aaabiao/dapo_filter) | `train.parquet` | 13,044 | Filtered by initial-model solve rate ([SimpleTIR](https://arxiv.org/abs/2509.02479)); 12,691 unique prompts + 353 duplicate-prompt rows | `\boxed{}` suffix | `"dapo_filter"` | Zero-edit with `reward.preset=math_dapo` or `+reward.data_source_presets.dapo_filter=math_dapo`. |
| [`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter) | `train/dapo_filter.parquet` | 12,691 | `aaabiao/dapo_filter` + first-occurrence `prompt` dedup ([OTB](https://arxiv.org/abs/2602.07078)) | `\boxed{}` suffix | `"dapo_filter"` | Zero-edit with `reward.preset=math_dapo` or `+reward.data_source_presets.dapo_filter=math_dapo`. **Ignore** `test/aime24.parquet` & `test/aime25.parquet` (byte-identical). |
| [`pe-nlp/DAPO-Math-17k-filtered`](https://huggingface.co/datasets/pe-nlp/DAPO-Math-17k-filtered) | `data/train-0000*-of-00003.parquet` | 1,790,700 | Removes 10 UUIDs (1,000 repeated rows) from the 100× pool | `Answer: $Answer` prefix + suffix | `"math_dapo"` | Ready out-of-the-box. |
| [`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed) | `all/` / `en/` / `cn/` | 17,398 (`en`: 14,116, `cn`: 3,282) | Prompt-deduplicated & split by language | Plain `str` column (`prompt: str`) | `"math_dapo"` | Not directly loadable by `RLHFDataset` without wrapping `prompt` into `[{"role": "user", "content": ...}]` (prefer `DAPO-Math-Unique-Clean-17k` if you want ready-made chat messages). |

---

## Lineage of Filtered & Deduplicated Variants

1. **Base pool (`BytedTsinghua-SIA/DAPO-Math-17k`)**: Contains `17,917` distinct `extra_info.index` UUIDs (`17,398` unique problem texts) repeated `100` times (`1,791,700` rows) with the Minerva-style `Answer: $Answer` wrapper.
2. **Initial-model accuracy filtering (`aaabiao/dapo_filter`, `13,044` rows)**: Curated in the [SimpleTIR](https://arxiv.org/abs/2509.02479) setup by evaluating an initial model on `DAPO-Math-17k` to discard tasks with overly high or overly low solve rates, and formatting prompts with `\n\nLet's think step by step and output the final answer within \boxed{}.` (`data_source="dapo_filter"`). It retains `12,691` unique prompts plus `353` duplicate-prompt rows across distinct UUIDs.
3. **First-occurrence prompt deduplication (`Jiawei415/DPAO_filter`, `12,691` rows)**: Linked in `docs/algo/otb.md` ([OTB](https://arxiv.org/abs/2602.07078)), `train/dapo_filter.parquet` applies first-occurrence `prompt` deduplication (`downsample/dedupe_downsample.py --dedupe_keys prompt`) to `aaabiao/dapo_filter`, removing the 353 duplicate-prompt entries while preserving all `12,691` unique rows, `extra_info.index` UUIDs, ground-truth answers, and row order verbatim.

---

## Zero-Edit Usage Examples

Download the desired Parquet artifact directly from Hugging Face and select the built-in scorer on the command line (no Parquet rewrite needed):

```bash
# Example 1: Filtered + deduplicated boxed 12,691-row split (Jiawei415/DPAO_filter)
hf download Jiawei415/DPAO_filter train/dapo_filter.parquet \
    --repo-type dataset \
    --revision d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd \
    --local-dir ~/data/dapo_filter

python3 -m verl.trainer.main_ppo \
    data.train_files=~/data/dapo_filter/train/dapo_filter.parquet \
    +reward.data_source_presets.dapo_filter=math_dapo \
    ...

# Example 2: Full 17,398-row unique boxed split (tongyx361/DAPO-Math-Unique-Boxed-17k)
hf download tongyx361/DAPO-Math-Unique-Boxed-17k data/dapo-math-unique-boxed-17k.parquet \
    --repo-type dataset \
    --revision b873644c6a2ee5f8226effebd6e0dfa6327ac84e \
    --local-dir ~/data/dapo_unique_boxed

python3 -m verl.trainer.main_ppo \
    data.train_files=~/data/dapo_unique_boxed/data/dapo-math-unique-boxed-17k.parquet \
    'reward.data_source_presets=["MATH##*=math_dapo"]' \
    ...
```

For multi-turn tool-integrated reasoning (`code_interpreter` `tools_kwargs`), see `examples/data_preprocess/dapo_multiturn_w_tool.py`.

## Validation Separation & Licensing Notes

- **Keep training and validation separate**: Do **not** use `test/aime24.parquet` or `test/aime25.parquet` inside `Jiawei415/DPAO_filter`—both files have identical bytes and SHA-256 (`3fc9f2a8c3c27f8e86a99dc4b6d13629a4851f81ce7dffc5dfad59feeb6f5804`) and contain AIME 2024. Prepare or download AIME validation benchmarks separately.
- **Licensing**: `BytedTsinghua-SIA/DAPO-Math-17k`, `tongyx361/DAPO-Math-Raw-17k`, and `tongyx361/DAPO-Math-Unique-17k` declare `apache-2.0`; several community derivatives (`aaabiao/dapo_filter`, `Jiawei415/DPAO_filter`) do not declare a license in their dataset cards. Verify each dataset's usage rights independently before redistribution.
