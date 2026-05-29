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
