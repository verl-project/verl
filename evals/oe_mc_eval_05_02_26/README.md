# oe_mc_eval_05_02_26

This folder contains evaluation scripts for the OE (open-ended) and MC (multiple-choice) Qwen models trained in this repo.

Goals:
- Generate model outputs **once** per example.
- Score the same outputs with **two different families of parsers**:
  - **VERL methods** (from `verl/utils/reward_score/*`)
  - **Qwen-original style** parsing (replicates `evals/qwen_original/evaluate_chat_gsm8k.py` last-number extraction)
- Save **all per-example details** into a single JSON file and print a terminal summary.

## Scripts

### 0) One-command runner (venv + deps + W&B + artifacts)

`run_evals.sh` creates a small eval venv, installs deps, optionally logs into W&B, runs the evals, and saves outputs under `evals/oe_mc_eval_05_02_26/outputs/<timestamp>/`.

```bash
export OE_MODEL="/path/to/oe_model_dir_or_hf_id"
export MC_MODEL="/path/to/mc_model_dir_or_hf_id"
export WANDB_PROJECT="gsm8k-evaluation"
export WANDB_ENTITY="your_entity"   # optional
export WANDB_API_KEY="..."          # recommended
./evals/oe_mc_eval_05_02_26/run_evals.sh
```

### 1) Cross-eval your trained OE + MC models (2×2)

Runs:
- OE model → GSM8K
- OE model → GSM8K-MC
- MC model → GSM8K
- MC model → GSM8K-MC

```bash
python evals/oe_mc_eval_05_02_26/evaluate_oe_mc_models_dual_parse.py \
  --oe_model /path/to/oe_model_dir_or_hf_id \
  --mc_model /path/to/mc_model_dir_or_hf_id \
  --out_json evals/oe_mc_eval_05_02_26/results_oe_mc.json
```

You can also pass W&B model artifacts directly (entity/project/artifact:alias), e.g.:

```bash
python evals/oe_mc_eval_05_02_26/evaluate_oe_mc_models_dual_parse.py \
  --oe_model tommaso-bendinelli-eth-zurich/multiple_choice_question_study/qwen25_3B_gsm8k:v0 \
  --mc_model tommaso-bendinelli-eth-zurich/multiple_choice_question_study/qwen25_3B_mc_gsm8k:v0
```

In that case, models are downloaded into `${VERL_RUN_DIR:-~/.cache/verl}/models/`.

### 2) Eval the base/original Qwen model on GSM8K + GSM8K-MC

```bash
python evals/oe_mc_eval_05_02_26/evaluate_base_qwen_dual_parse.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --out_json evals/oe_mc_eval_05_02_26/results_base_qwen.json
```

## Important flags

- `--num_samples X` (or env `NUM_SAMPLES=X`)
  - If `X > 0`: evaluate only the first `X` instances from each split.
  - If omitted: evaluate on the full split (default).
  - `X` must be positive when provided/set.
- `--inference_id <id>`
  - Use a named decoding/profile config instead of manual decoding flags.
  - Available: `greedy`, `sample_balanced`, `sample_creative`, `greedy_different_system_prompt_test`, `greedy_bad_math_system_prompt_test`.
  - Cannot be combined with manual decoding flags:
    `--max_new_tokens`, `--max_length`, `--do_sample`, `--temperature`, `--top_p`,
    `--repetition_penalty`, `--no_chat_template`, `--system_prompt`.
- Multi-GPU behavior
  - The eval scripts support `torchrun` data-parallel evaluation (`WORLD_SIZE > 1`).
  - Each rank processes a shard of prompts, and rank 0 gathers/merges outputs before scoring + writing JSON.
  - In `run_evals.sh`, `BATCH_SIZE` is per-GPU process (default `32`), and `MULTI_GPU=1` auto-launches `torchrun` when multiple GPUs are available.
- W&B (artifacts)
  - `--wandb_project <name>` enables W&B logging and uploads the output JSON as an artifact.
  - `--wandb_entity <entity>` optional.
  - `--wandb_mode online|offline|disabled`
- `--parse_methods strict flexible`
  - **VERL strict** expects `#### <answer>` and is *very* sensitive to formatting.
  - **VERL flexible** extracts the last number / last A-D letter without requiring `####`.
- `--prompt_style train|raw`
  - `train`: matches your training-style prompts (includes `####` instruction).
  - `raw`: passes the dataset question/options without extra instructions.
- Chat formatting
  - Default uses `tokenizer.apply_chat_template(...)` (recommended for Qwen-*Instruct* checkpoints).
  - `--no_chat_template` disables chat templating and treats prompts as raw text completion.
- FlashAttention (optional)
  - If available, you can set `--attn_implementation flash_attention_2` for speed.
  - You still need a compatible CUDA + GPU + `flash-attn` install; otherwise use `sdpa` (default in many envs) or `eager`.
  - If `flash_attention_2` is requested but `flash-attn` is missing, the scripts fall back to `sdpa` and print a warning.

## Common issue: passing W&B artifacts to Transformers

If you pass `--oe_model entity/project/artifact:alias`, Transformers will error unless the script downloads the artifact first.
These scripts handle that automatically, but you must have:
- `wandb` installed
- a valid W&B login (`wandb login` or `WANDB_API_KEY=...`)
- `WANDB_MODE` not set to `disabled`
