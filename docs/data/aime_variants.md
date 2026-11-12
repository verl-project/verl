# AIME Variants

Last updated: 11/12/2026.

AIME (American Invitational Mathematics Examination) is the standard validation benchmark across math RL recipes in `verl`. On Hugging Face, only the 32×-repeated **AIME 2024** DAPO release ships directly in `verl`'s `RLHFDataset` Parquet format, whereas **AIME 2025**, **AIME 2026**, and **per-session I/II splits (`aime2024-I`..`aime2026-II`)** are published as raw `(problem, answer)` tables and prepared via `examples/data_preprocess/aime.py`.

AIME validation datasets are kept strictly separate from training dataset variants.

---

## Primary Public Hugging Face AIME Datasets

| Hugging Face Dataset | Scope | Rows | Format & Schema | `data_source` / License | Community Harness Alignment & `verl` Notes |
| --- | --- | ---: | --- | --- | --- |
| [`BytedTsinghua-SIA/AIME-2024`](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024) (`data/aime-2024.parquet`) | 2024 (I + II, 32× repeated) | 960 | **Ready-to-use `RLHFDataset` Parquet** (`Answer: $Answer` wrapper) | `data_source="math_dapo"` | Ready out-of-the-box; keep `actor_rollout_ref.rollout.val_kwargs.n=1` because the 30 problems are already repeated 32 times (`960` rows). |
| [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | 2024 (I + II) | 30 | Raw benchmark table (`id, problem, solution, answer, url, year`) | License not declared in pinned card | Pinned by `aime.py`; 30/30 problem texts, IDs (`60..89`), and unpadded integer answers match [Harbor](https://github.com/laude-institute/harbor/tree/main/adapters/aime)'s `test2024.jsonl` verbatim, and answers corroborate [`MathArena/aime_2024_I`](https://huggingface.co/datasets/MathArena/aime_2024_I) & [`_II`](https://huggingface.co/datasets/MathArena/aime_2024_II). |
| [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025) (`_I` / `_II`) | 2025 (I + II) | 30 (15 + 15) | Raw benchmark table (`problem_idx, problem, answer`) | `cc-by-nc-sa-4.0` | Pinned by `aime.py`; 30/30 answers and `2025-II` problem texts match Harbor's `test2025-{I,II}.jsonl` verbatim, while `2025-I` additionally retains full LaTeX/TikZ/Asymptote diagrams on Q2/Q10/Q11. |
| [`MathArena/aime_2026`](https://huggingface.co/datasets/MathArena/aime_2026) (`_I`) | 2026 (I + II) | 30 (`_I`: 15) | Raw benchmark table (`problem_idx, problem, answer`) | `cc-by-nc-sa-4.0` | Pinned by `aime.py` at the same commit (`d2de22f3...`) used by MathArena and DeepMind `gbench`; retains publisher's regional II #10 variant (sum = `850`, see [discussion #2](https://huggingface.co/datasets/MathArena/aime_2026/discussions/2)). |

---

## Preparing 2024–2026 Datasets with `examples/data_preprocess/aime.py`

`examples/data_preprocess/aime.py` builds deduplicated 30-problem yearly files (`val_kwargs.n=32` friendly), 15-problem I/II session files, and the 90-problem 2024–2026 suite from pinned, SHA-256-verified upstream sources. All generated `data_source` values start with `"aime"` (`aime2024-I`..`aime2026-II`), which route automatically to `verl.utils.reward_score.math_dapo.compute_score` in `default_compute_score`:

```bash
# Default: boxed suffix ("\n\nLet's think step by step and output the final answer within \boxed{}.")
python examples/data_preprocess/aime.py --output-dir ~/data/aime

# Minerva / DAPO "Answer: $Answer" wrapper
python examples/data_preprocess/aime.py \
    --prompt-format dapo_answer \
    --output-dir ~/data/aime-dapo-answer

# Clean problem text without instruction wrapper (when using system prompt / chat template)
python examples/data_preprocess/aime.py \
    --prompt-format clean \
    --output-dir ~/data/aime-clean
```

Generated files under `--output-dir`:

- **Per-session splits (15 problems each)**: `aime2024-I.parquet`, `aime2024-II.parquet`, `aime2025-I.parquet`, `aime2025-II.parquet`, `aime2026-I.parquet`, `aime2026-II.parquet`.
- **Per-year combined files (30 problems each)**: `aime2024.parquet`, `aime2025.parquet`, `aime2026.parquet`.
- **Full 3-year suite (90 problems)**: `aime2024-2026.parquet`, plus `manifest.json`.

For multi-turn tool-integrated AIME 2024 evaluation (`code_interpreter`), see `examples/data_preprocess/aime2024_multiturn_w_tool.py`.

## Verification, Offline Cache & Provenance

- **Pinned revisions & cross-verification**: Every upstream file's Git commit SHA, byte length, and SHA-256 are verified before loading; question membership and answers are cross-checked against separate session sources, Parquet round-trips are verified, and `default_compute_score` is checked on every row.
- **Preserving legacy files & offline cache**: Optional `--existing-aime2024` and `--existing-aime2025` accept previously prepared 30-row Parquet files whose `data_source` is `aime{year}` (or matching `aime{year}-{session}`), string `extra_info.index` matches the pinned source IDs (`"60"`..`"89"` for H4 2024, `"1"`..`"30"` for MathArena 2025), `prompt` matches `--prompt-format`, and `reward_model` equals `{"style": "rule", "ground_truth": "<answer>"}`, preserving any extra metadata while adding `year`, `session`, and `problem_number`. `--source-dir` accepts an offline source cache (`<publisher>--<dataset>/data/...`) with full byte/hash verification.
- **Licensing**: MathArena sources declare `CC BY-NC-SA 4.0`. The pinned `HuggingFaceH4/aime_2024` card does not declare a license (`manifest.json` records `license: null` and links the pinned card). Do not infer that MathArena's license covers H4 text, or that generated Parquet files inherit `verl`'s Apache-2.0 license—check each source's usage rights independently.
