# reliable-gsm8k-builder

Small standalone GSM8K dataset builder.

It does exactly this:

1. Load GSM8K questions.
2. Use one generator model (`qwen` or `gpt`) to produce:
   - `4` incorrect numeric answers
   - `1` correct numeric answer
3. Check generated MC answers against the GSM8K gold numeric answer.
4. Export three datasets:
   - `gsm8k_mc_allwrong`: numeric options `A-D` are all incorrect, correct target is `NONE`
   - `gsm8k_mc_onecorrect`: three incorrect numeric options and one correct option
   - `gsm8k_oe`: source GSM8K chain-of-thought solution

This project is intentionally not connected to the old stage graph in `research-mc-reasoning`.

## Layout

- `run_build.py`: main CLI entry point
- `src/reliable_gsm8k/`: minimal package
- `runs/<run_id>/`: generated artifacts

## Built-in profiles

Generator profiles:

- `qwen25_3b`
- `gpt4o_mini`

Inference profiles:

- `greedy` (default)
- `sample_balanced`
- `sample_creative`
- `greedy_different_system_prompt_test`
- `greedy_bad_math_system_prompt_test`

Inference profiles define decoding and prompt formatting behavior:

- `max_new_tokens`, `max_length`
- `do_sample`, `temperature`, `top_p`, `repetition_penalty`
- `use_chat_template`
- `system_prompt`

By default, built-in profiles use `use_chat_template=true` and
`system_prompt="You are a helpful assistant."`.

Judge profiles:

- `judgelm_7b`
- `gpt41_mini`
- `gpt4o_mini`

## Example

From this folder:

```bash
export OPENAI_API_KEY="..."

python run_build.py \
  --run-id smoke-qwen \
  --split train \
  --num-samples 20 \
  --generator-profile qwen25_3b \
  --inference-profile greedy
```

Or with OpenAI for generation:

```bash
export OPENAI_API_KEY="..."

python run_build.py \
  --run-id smoke-gpt \
  --split train \
  --num-samples 20 \
  --generator-profile gpt4o_mini \
  --inference-profile sample_balanced
```

Decoding is profile-only (`--inference-profile`); legacy per-run temperature/max-token flags are disabled.

Just 10 samples:

```bash
python run_build.py \
  --run-id smoke-10 \
  --split train \
  --num-samples 10 \
  --generator-profile qwen25_3b
```

Use a local checkpoint/model path as the generator:

```bash
python run_build.py \
  --run-id from-checkpoint \
  --split train \
  --num-samples 100 \
  --generator-profile qwen25_3b \
  --generator-model-path /path/to/actor/huggingface
```

## Multi-GPU

If multiple GPUs are visible and either the generator or the optional judge uses a local Transformers model,
`run_build.py` now shards the GSM8K items automatically:

- one worker process per visible GPU
- each worker gets `CUDA_VISIBLE_DEVICES=<single_gpu>`
- shard outputs are merged back into the normal `runs/<run_id>/` layout

You do not need a separate launcher.

To restrict which GPUs are used:

```bash
python run_build.py \
  --run-id full-qwen \
  --split train \
  --generator-profile qwen25_3b \
  --gpu-ids 0,1,2
```

To force single-process execution:

```bash
python run_build.py \
  --run-id debug-single \
  --split train \
  --generator-profile qwen25_3b \
  --disable-multi-gpu
```

If you want to re-enable judge verification:

```bash
python run_build.py \
  --run-id judged-qwen \
  --split train \
  --num-samples 20 \
  --generator-profile qwen25_3b \
  --use-judge \
  --judge-profile judgelm_7b
```

## FlashAttention

For local Qwen or JudgeLM inference on Linux:

```bash
./run/install_flash_attn.sh
```

If no prebuilt wheel is available and you want to allow a source build:

```bash
ALLOW_FLASH_ATTN_SOURCE_BUILD=1 MAX_JOBS=4 ./run/install_flash_attn.sh
```

## Outputs

Each run writes to `runs/<run_id>/`:

- `items/<split>.jsonl`: per-question accepted solutions and stats
- `artifacts/answers_<timestamp>.json`: one record per question with every generated response attempt
- `artifacts/evaluation_<timestamp>.json`: rich parser/judge labels for each generated response attempt
- `datasets/gsm8k_oe_<generator>/<split>.jsonl`
- `datasets/gsm8k_mc_onecorrect_<generator>/<split>.jsonl`
- `datasets/gsm8k_mc_allwrong_<generator>/<split>.jsonl`
- matching `.parquet` files for `items` and every generated dataset
- `manifest.json`

Every item and dataset row includes both:

- `item_id`: positional id such as `gsm8k:train:42`
- `question_id`: stable `md5(question)` hash for joining generated answers, evaluations, and final datasets

Candidate-source metadata is explicit:

- OE targets use `original`
- MC options currently use live `sampled_from_model` generations
- the manifest reserves the source names `original`, `sampled_from_model`, `programmatic`, and `path_to_answers` for future source-mode extensions

When multi-GPU mode is used, worker artifacts are also kept under:

- `runs/<run_id>/shards/shard_00/`
- `runs/<run_id>/shards/shard_01/`
- ...

## Notes

- The generator prompt requires this format:

```text
FINAL_ANSWER: ...
```

- MC generation is numeric-only.
  Qwen is asked to emit just `FINAL_ANSWER: <number>`.

- Acceptance is strict:
  - correct MC answers must parse to the GSM8K gold answer
  - incorrect MC answers must parse to a different numeric answer
  - duplicate incorrect numeric answers are rejected
- OE targets come directly from the original GSM8K chain-of-thought answer and end with `FINAL_ANSWER: <gold answer>`
- `gsm8k_mc_allwrong` is a real all-wrong dataset:
  - the prompt tells the model to output `NONE` if no option is correct
  - the ground truth is always `NONE`

## On-Policy Loop

`run_on_policy_loop.py` automates:

1. generate `N` fresh data samples from the current model
2. train with VERL/GRPO on the generated parquet
3. find the saved HF actor checkpoint
4. use that checkpoint as the generator for the next iteration

Example dry run:

```bash
python run_on_policy_loop.py \
  --run-prefix onpol-smoke \
  --iterations 2 \
  --num-samples 100 \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --generator-profile qwen25_3b \
  --inference-profile sample_balanced \
  --train-dataset mc_onecorrect \
  --gpu-ids 0,1,2 \
  --dry-run
```

Real run:

```bash
python run_on_policy_loop.py \
  --run-prefix onpol-qwen \
  --iterations 3 \
  --num-samples 1000 \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --generator-profile qwen25_3b \
  --inference-profile sample_balanced \
  --train-dataset mc_onecorrect \
  --val-file ~/data/gsm8k/test.parquet \
  --gpu-ids 0,1,2
```

The loop stores generated data under `runs_on_policy/<run>-iterXX-data/` and checkpoints under
`../checkpoints/on_policy/<run>-iterXX-train/` by default.

## VERL-Native Dynamic MC Loop

`bash/20260604_gsm8k_dynamic_mc_verl_native.sh` is the faster on-policy variant. It starts VERL once and keeps the actor/rollout model resident on GPU. The seed parquet only contains Stage 1 GSM8K prompts:

- each seed row asks the current actor to solve the GSM8K question with chain-of-thought
- `STAGE1_PROMPT_COUNT` controls how many neutral candidate-generation prompts are seeded per question
- each rollout is parsed with `FINAL_ANSWER: <number>` and verified against the GSM8K gold answer
- correct Stage 1 completions get reward `1`; incorrect neutral completions get reward `0` but are still buffered as Stage 2 distractors
- rewards are computed by `reliable_gsm8k.verl_dynamic_mc.compute_score`

After each VERL batch, `GSM8KDynamicMCDataset.on_batch_end(...)` reads the generated rollouts from the same `DataProto`, verifies final numeric answers against GSM8K gold, and buffers candidates by `question_id = md5(question)`. Once a question has one verified correct candidate and three distinct verified incorrect candidates, the dataset queues a Stage 2 MC prompt with four CoT options and exactly one correct letter. Later candidate sets consume newer verified wrong answers in groups of three and newer verified correct CoTs when available, so Stage 2 can keep refreshing instead of being frozen to the first successful candidate set. At epoch end, VERL calls `on_epoch_end(...)`, and the queued Stage 2 rows are prepended for the next dataloader pass. This avoids mutating numeric sampler indices in the middle of an active epoch.

Run a dry command preview:

```bash
DRY_RUN=1 NUM_SAMPLES=10 GPU_IDS=0 GPUS=1 \
TRAIN_BATCH_SIZE=4 GEN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 \
  ../bash/20260604_gsm8k_dynamic_mc_verl_native.sh
```

Run a small real smoke:

```bash
NUM_SAMPLES=10 GPU_IDS=0 GPUS=1 \
TRAIN_BATCH_SIZE=4 GEN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4 \
TOTAL_EPOCHS=3 TEST_FREQ=999999 SAVE_FREQ=999999 \
  ../bash/20260604_gsm8k_dynamic_mc_verl_native.sh
```

Then validate that the run actually reached the dynamic Stage 2 path:

```bash
python check_dynamic_mc_smoke_log.py ../logs/20260604_gsm8k_dynamic_mc_verl_native_grpo.log
```

The checker requires evidence that Stage 1 rollouts were parsed, correct and incorrect candidates were accepted, Stage 2 rows were queued and inserted, and the train dataloader was rebuilt without restarting VERL.

Useful parameters:

- `NUM_SAMPLES`: number of GSM8K questions used to create the initial Stage 1 seed parquet.
- `ROLLOUT_N`: number of VERL rollouts per prompt. Default `4`, which is useful for GRPO grouping.
- `STAGE1_PROMPT_COUNT`: number of neutral Stage 1 candidate-generation prompts seeded per GSM8K question. Default `4`. Actual Stage 1 sampled completions per question are `STAGE1_PROMPT_COUNT * ROLLOUT_N`, so the default is `16` attempts per question.
- `STAGE1_PROMPT_MODE`: prompt style for Stage 1 seed rows. Default `neutral`. `role` keeps the older candidate-mining behavior with one correct-solution prompt and wrong-solution prompts, but the cleaner on-policy path is `neutral`.
- `INCORRECT_PROMPT_COUNT` / `INCORRECT_TARGET_COUNT`: deprecated aliases from the role-prompt version. If one is set, the launcher maps it to total neutral seed rows as `old_value + 1`.
- `STAGE2_INCORRECT_COUNT`: number of verified wrong candidates used in each Stage 2 MC prompt. This must be `3`, because Stage 2 is always one correct option plus three wrong options.
- `MAX_STAGE2_PER_QUESTION`: how many MC prompts can be created from one GSM8K question. Default `4`.
- `MAX_NEW_STAGE2_PER_BATCH`: cap on new Stage 2 rows queued after one VERL batch. Default `256`.
- `STAGE2_CANDIDATE_MAX_CHARS`: maximum characters kept from each candidate CoT when building Stage 2 options. Default `2000`; the final `FINAL_ANSWER` tail is preserved when clipping.
- `TOTAL_EPOCHS`: number of dataloader passes. Default `3`. Use at least `2`, because Stage 2 rows are produced after Stage 1 rollouts and consumed on a later pass. Use `3+` when you want to see the updated actor generate a second wave of Stage 1 candidates and train on the promoted Stage 2 rows from that wave.
- `TOTAL_TRAINING_STEPS`: optional explicit VERL step budget. If unset, VERL starts from the seed dataloader length and expands the automatic step budget after epoch-end Stage 2 insertion. If set, it is treated as a hard cap.
- `VAL_FILE`: optional validation parquet. If unset, the launcher creates a small Stage 1 GSM8K test parquet under `runs_verl_native_dynamic_mc/seed/`.
- `VAL_NUM_SAMPLES`: number of GSM8K test questions used for that auto-created validation parquet. Default `128`.
- `TRAIN_BATCH_SIZE`, `GEN_BATCH_SIZE`, `PPO_MINI_BATCH_SIZE`, `ACTOR_MICRO_BATCH_SIZE`: VERL batch controls.
- `MODEL_PATH`: base model or local actor checkpoint path.
- `GPU_IDS`, `GPUS`: visible CUDA devices and VERL GPU count.
- `TRAINER_LOGGER`: VERL logger list. Default `["console"]`; use `TRAINER_LOGGER='["console","wandb"]'` when W&B is configured.

This path is different from `run_on_policy_loop.py`: it does not regenerate a static dataset, stop VERL, save an HF checkpoint, reload it, and start again. It keeps the update loop inside one VERL run. Checkpoints are still saved according to `SAVE_FREQ` and at the final step, but they are not used as the mechanism for the next policy update.

Trainer resume is intentionally disabled for this dynamic dataset. The candidate buffer is in memory and is not part of VERL's dataloader checkpoint, so `trainer.resume_mode=disable` is required for this path.
