# DAPO-Math Variants

Last updated: 11/12/2026.

Math RL recipes in `verl` (DAPO, SimpleTIR, OTB, DPPO, Score Centering, ReTool) commonly train on [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) or its difficulty-filtered derivatives. Below are the primary, most widely used public Hugging Face releases—especially the initial-model solve-rate filtered splits that cannot be constructed on-the-fly without running rollout filtering.

Validation benchmarks (such as AIME 2024–2026) are kept **separate** from DAPO-Math training datasets.

---

## Primary Public Hugging Face Datasets

| Hugging Face Dataset | Parquet Path | Rows | Curation / Why Use It | Prompt Format | Raw `data_source` | `verl` Usage Notes |
| --- | --- | ---: | --- | --- | --- | --- |
| [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | `data/dapo-math-17k.parquet` | 1,791,700 | Official release: 100× exact repeats of 17,917 UUIDs (17,398 unique prompts) | `Answer: $Answer` prefix + suffix | `"math_dapo"` | Ready out-of-the-box (`default_compute_score` routes `"math_dapo"` to `math_dapo.compute_score`). |
| [`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed) | `all/` / `en/` / `cn/` | 17,398 (`en`: 14,116, `cn`: 3,282) | Widely used prompt-deduplicated & language-split (`en` / `cn`) release | Plain `str` (`prompt: str`, unwrapped problem text) | `"math_dapo"` | `prompt` is a plain string (`source_prompt` stores the original chat list); wrap `prompt` into `[{"role": "user", "content": ...}]` before loading with `RLHFDataset`. |
| [`aaabiao/dapo_filter`](https://huggingface.co/datasets/aaabiao/dapo_filter) | `train.parquet` | 13,044 | **Non-trivial offline curation**: filtered by initial-model solve rate ([SimpleTIR](https://arxiv.org/abs/2509.02479)) to drop overly easy/hard tasks; 12,691 unique prompts + 353 duplicate-prompt rows | `\boxed{}` suffix | `"dapo_filter"` | Route `"dapo_filter"` to `math_dapo` (see below), or use `Jiawei415/DPAO_filter` for the prompt-deduplicated split. |
| [`Jiawei415/DPAO_filter`](https://huggingface.co/datasets/Jiawei415/DPAO_filter) | `train/dapo_filter.parquet` | 12,691 | `aaabiao/dapo_filter` + first-occurrence `prompt` dedup ([OTB](https://arxiv.org/abs/2602.07078), linked in `docs/algo/otb.md`) | `\boxed{}` suffix | `"dapo_filter"` | Route `"dapo_filter"` to `math_dapo` (see below). **Ignore** `test/aime24.parquet` & `test/aime25.parquet` (byte-identical). |

---

## Lineage of `aaabiao/dapo_filter` and `Jiawei415/DPAO_filter`

Unlike simple UUID/prompt deduplication or prompt-template swaps, the `dapo_filter` splits depend on offline rollout evaluation and cannot be reconstructed purely by text preprocessing:

1. **Base pool (`BytedTsinghua-SIA/DAPO-Math-17k`)**: `17,917` distinct `extra_info.index` UUIDs (`17,398` unique problem texts) repeated `100` times (`1,791,700` rows) with the Minerva-style `Answer: $Answer` wrapper.
2. **Initial-model accuracy filtering (`aaabiao/dapo_filter`, `13,044` rows)**: Curated in the [SimpleTIR](https://arxiv.org/abs/2509.02479) setup by evaluating an initial model on `DAPO-Math-17k` to discard tasks with overly high or overly low solve rates, and formatting prompts with `\n\nLet's think step by step and output the final answer within \boxed{}.` (`data_source="dapo_filter"`). It retains `12,691` unique prompts plus `353` duplicate-prompt rows across distinct UUIDs.
3. **First-occurrence prompt deduplication (`Jiawei415/DPAO_filter`, `12,691` rows)**: Linked in `docs/algo/otb.md` ([OTB](https://arxiv.org/abs/2602.07078)), `train/dapo_filter.parquet` applies first-occurrence `prompt` deduplication (`downsample/dedupe_downsample.py --dedupe_keys prompt`) to `aaabiao/dapo_filter`, removing the 353 duplicate-prompt entries while preserving all `12,691` unique rows, `extra_info.index` UUIDs, ground-truth answers, and row order verbatim.

---

## Using `Jiawei415/DPAO_filter` in `verl`

Download the pinned training Parquet file:

```bash
hf download Jiawei415/DPAO_filter train/dapo_filter.parquet \
    --repo-type dataset \
    --revision d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd \
    --local-dir ~/data/dapo_filter
```

Because `train/dapo_filter.parquet` sets `data_source="dapo_filter"` (which is not in `default_compute_score`'s hardcoded table), route it to `math_dapo` in either of two ways:

- **Option A — One-line Parquet `data_source` normalization** (works on any `verl` version):

  ```bash
  python3 -c '
  import pyarrow as pa, pyarrow.parquet as pq
  p = "train/dapo_filter.parquet"
  t = pq.read_table(p)
  pq.write_table(t.set_column(t.schema.get_field_index("data_source"), "data_source", pa.array(["math_dapo"] * len(t))), "train_math_dapo.parquet")
  '
  ```

- **Option B — Custom reward wrapper** via `reward.custom_reward_function.path` delegating `(data_source, solution_str, ground_truth, extra_info=None)` to `verl.utils.reward_score.math_dapo.compute_score(solution_str, ground_truth)` (or CLI preset routing `+reward.data_source_presets.dapo_filter=math_dapo` when available).

For multi-turn tool-integrated reasoning (`code_interpreter` `tools_kwargs`), see `examples/data_preprocess/dapo_multiturn_w_tool.py`.

## Validation Separation & Licensing Notes

- **Keep training and validation separate**: Do **not** use `test/aime24.parquet` or `test/aime25.parquet` inside `Jiawei415/DPAO_filter`—both files have identical bytes and SHA-256 (`3fc9f2a8c3c27f8e86a99dc4b6d13629a4851f81ce7dffc5dfad59feeb6f5804`) and contain AIME 2024. Prepare AIME validation benchmarks separately.
- **Licensing**: `BytedTsinghua-SIA/DAPO-Math-17k` declares `apache-2.0`; `aaabiao/dapo_filter` and `Jiawei415/DPAO_filter` do not declare a license in their dataset cards. Verify each dataset's usage rights independently before redistribution.
