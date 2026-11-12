# AIME Variants and Exam-Session Splits

Last updated: 11/12/2026.

AIME (American Invitational Mathematics Examination) is the standard validation benchmark across math RL recipes in `verl`. Public Hugging Face repositories fall into two categories:

1. **Ready-to-use `verl` `RLHFDataset` Parquet datasets** (currently available for **AIME 2024** combined 30 problems).
2. **Raw benchmark problem/answer repositories** for **AIME 2024, 2025, and 2026** (including separate I/II contest sessions), which require formatting into `verl`'s `prompt` + `reward_model` Parquet schema via `examples/data_preprocess/aime_exam_splits.py`.

AIME validation datasets are kept strictly separate from training dataset variants.

---

## Summary of Public Hugging Face AIME Datasets

### 1. Ready-to-Use `verl` Parquet Datasets (AIME 2024)

All `data_source` values starting with `"aime"` automatically route to `verl.utils.reward_score.math_dapo.compute_score` in `default_compute_score`, so these files work out-of-the-box without editing the Parquet file:

| Hugging Face Dataset | Parquet Path | Rows | Sampling Note | Prompt Format | `data_source` |
| --- | --- | ---: | --- | --- | --- |
| [`tongyx361/AIME-2024-Boxed`](https://huggingface.co/datasets/tongyx361/AIME-2024-Boxed) | `data/aime-2024-boxed.parquet` | 30 | Unique 30 problems; set `actor_rollout_ref.rollout.val_kwargs.n=32` for `mean@32` | `\boxed{...}` suffix | `"aime-2024"` |
| [`tongyx361/AIME-2024`](https://huggingface.co/datasets/tongyx361/AIME-2024) | `data/aime-2024.parquet` | 30 | Unique 30 problems; set `actor_rollout_ref.rollout.val_kwargs.n=32` for `mean@32` | `Answer: $Answer` prefix + suffix | `"aime-2024"` |
| [`BytedTsinghua-SIA/AIME-2024`](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024) | `data/aime-2024.parquet` | 960 | Repeats the 30 problems 32× (`960` rows); keep `val_kwargs.n=1` to avoid 32×32 oversampling | `Answer: $Answer` prefix + suffix | `"aime_2024"` |

### 2. Raw Upstream Benchmark Sources (2024 / 2025 / 2026 & I/II Sessions)

For **AIME 2025**, **AIME 2026**, and **per-session I/II splits (`aime2024-I`..`aime2026-II`)**, public Hugging Face repositories store raw `(problem, answer)` tables rather than `RLHFDataset` chat messages:

| Hugging Face Dataset | Year / Session | Rows | Raw Schema | Declared License | Notes |
| --- | --- | ---: | --- | --- | --- |
| [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | 2024 (I + II) | 30 | `id, problem, solution, answer, url, year` | Not declared in pinned card | `url` identifies exact I/II session & problem number (`id` is lexicographical, not exam order). |
| [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) | 2024 (I + II) | 30 | `ID, Problem, Solution, Answer` | MIT | Alternative 2024 raw transcription (`ID` e.g. `2024-I-1`). |
| [`MathArena/aime_2024_I`](https://huggingface.co/datasets/MathArena/aime_2024_I), [`MathArena/aime_2024_II`](https://huggingface.co/datasets/MathArena/aime_2024_II) | 2024 I / II | 15 + 15 | `problem_idx, problem, answer` | `cc-by-nc-sa-4.0` | Separate per-session reference tables. |
| [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025) (`_I` / `_II`) | 2025 (I + II) | 30 (15 + 15) | `problem_idx, problem, answer` | `cc-by-nc-sa-4.0` | Combined `1..15` = I, `16..30` = II; matches `_I` / `_II` verbatim. |
| [`math-ai/aime25`](https://huggingface.co/datasets/math-ai/aime25), [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025) | 2025 (I + II) | 30 | Raw JSONL / Parquet (`problem, answer`) | Apache-2.0 / MIT | Raw evaluation tables, not `RLHFDataset` chat format. |
| [`MathArena/aime_2026`](https://huggingface.co/datasets/MathArena/aime_2026) (`_I`) | 2026 (I + II) | 30 (`_I`: 15) | `problem_idx, problem, answer` | `cc-by-nc-sa-4.0` | Retains publisher's regional II #10 variant (sum = `850`); see [discussion #2](https://huggingface.co/datasets/MathArena/aime_2026/discussions/2). |

> **Warning**: Do **not** use `test/aime24.parquet` or `test/aime25.parquet` inside `Jiawei415/DPAO_filter`—those two files are byte-for-byte identical (`sha256: 3fc9f2a8c3c27f8e86a99dc4b6d13629a4851f81ce7dffc5dfad59feeb6f5804`) and both contain AIME 2024.

---

## Preparing 2024–2026 & I/II Splits with `aime_exam_splits.py`

Because ready-made `verl` Parquet releases do not cover AIME 2025, AIME 2026, or separate I/II session metric identities, `examples/data_preprocess/aime_exam_splits.py` builds them reproducibly from pinned, SHA-256-verified upstream sources:

```bash
# Default: boxed suffix ("\n\nLet's think step by step and output the final answer within \boxed{}.")
python examples/data_preprocess/aime_exam_splits.py --output-dir ~/data/aime-exams

# Minerva / DAPO "Answer: $Answer" wrapper
python examples/data_preprocess/aime_exam_splits.py \
    --prompt-format dapo_answer \
    --output-dir ~/data/aime-exams-dapo-answer

# Clean problem text without instruction wrapper (when using system prompt / chat template)
python examples/data_preprocess/aime_exam_splits.py \
    --prompt-format clean \
    --output-dir ~/data/aime-exams-clean
```

Generated files under `--output-dir`:

- **Per-session splits (15 problems each)**: `aime2024-I.parquet`, `aime2024-II.parquet`, `aime2025-I.parquet`, `aime2025-II.parquet`, `aime2026-I.parquet`, `aime2026-II.parquet` (distinct `data_source` values such as `aime2026-I` report per-exam validation metrics).
- **Per-year combined files (30 problems each)**: `aime2024.parquet`, `aime2025.parquet`, `aime2026.parquet`.
- **Full 3-year suite (90 problems)**: `aime2024-2026.parquet`, plus `manifest.json`.

For multi-turn tool-integrated AIME 2024 evaluation (`code_interpreter`), see `examples/data_preprocess/aime2024_multiturn_w_tool.py`.

## Verification, Offline Cache & Provenance

- **Pinned revisions & cross-verification**: Every upstream file's Git commit SHA, byte length, and SHA-256 are verified before loading; question membership and answers are cross-checked against separate session sources, Parquet round-trips are verified, and `default_compute_score` is checked on every row.
- **Preserving legacy files**: Optional `--existing-aime2024` and `--existing-aime2025` verify and preserve existing prompts/metadata verbatim while adding `year`, `session`, and `problem_number` identity. `--source-dir` supports offline pre-staged caches with full byte/hash verification.
- **Licensing**: MathArena sources declare `CC BY-NC-SA 4.0`. The pinned `HuggingFaceH4/aime_2024` card does not declare a license (`manifest.json` records `license: null` and links the pinned card). Do not infer that MathArena's license covers H4 text, or that generated Parquet files inherit `verl`'s Apache-2.0 license—check each source's usage rights independently.
