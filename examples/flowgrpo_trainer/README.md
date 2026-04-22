# FlowGRPO Trainer

This example shows how to post-train `Qwen-Image` with FlowGRPO on an OCR-style image generation task using `vllm-omni` rollout and a visual generative reward model (`Qwen3-VL-8B-Instruct` in this example).

For the full installation and quickstart guide, see `docs/start/flowgrpo_quickstart.rst`. For algorithm details and rule-based reward training (e.g. JPEG incompressibility), see `docs/algo/flowgrpo.md`.

## Installation

Install `verl` following the installation guide in `docs/start/install.rst`.

Then install the FlowGRPO example-specific dependencies in the same environment:

```bash
pip install "vllm==0.18" "vllm-omni==0.18" Levenshtein
```

The provided script is configured for a single node with `4` GPUs.

## Prepare the dataset

Obtain the raw OCR dataset from the original Flow-GRPO repository:

- https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

Place the raw dataset under `$HOME/data/ocr`, then preprocess it into parquet files:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --local_dataset_path $HOME/data/ocr \
  --local_save_dir $HOME/data/ocr
```

This produces:

- `$HOME/data/ocr/train.parquet`
- `$HOME/data/ocr/test.parquet`

## Prepare the models

The example script expects the following local model paths:

```bash
$HOME/models/Qwen/Qwen-Image
$HOME/models/Qwen/Qwen-Image/tokenizer
$HOME/models/Qwen/Qwen3-VL-8B-Instruct
```

If your models live elsewhere, update the path overrides in `examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh`.

## Run training

Launch the example from the repository root:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh
```

Optional KL loss tuning:

- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`

The script runs `python3 -m verl.trainer.main_flowgrpo` with:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `reward.reward_manager.name=visual`
- `reward.custom_reward_function.name=compute_score_ocr`
- LoRA fine-tuning for `Qwen-Image`
- `trainer.n_gpus_per_node=4`

## Logging

W&B logging is enabled by default in the example script:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

The script sets:

```bash
trainer.logger='["console", "wandb"]'
trainer.project_name=flow_grpo
trainer.experiment_name=qwen_image_ocr_lora
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

The following metrics are specific to diffusion FlowGRPO training.

**`critic/rewards/zero_std_ratio`** — the fraction of prompt groups (out of
`train_batch_size` prompts) where every one of the `n` generated images
received the same reward, giving a within-group standard deviation of zero.
GRPO derives its learning signal from *relative* rewards within a group, so a
group with zero std contributes no gradient regardless of the absolute reward
value. A persistently high ratio (e.g. above 0.5) means the reward model is
saturated or the task difficulty is poorly calibrated — either all images are
rewarded or none are — and the policy is not receiving useful training signal.

**`critic/rewards/std_mean`** — the mean of the per-prompt reward standard
deviations across all prompt groups in the batch. Complements
`zero_std_ratio`: while `zero_std_ratio` flags completely collapsed groups,
`std_mean` tracks the average reward spread within a group across the whole
batch. A healthy, rising `std_mean` indicates the reward model is producing
diverse signal; a declining `std_mean` is an early warning of reward
saturation before `zero_std_ratio` spikes.

**`actor/pg_clipfrac_higher`** and **`actor/pg_clipfrac_lower`** — these
break down PPO clipping by direction. `pg_clipfrac_higher` is the fraction of
`(image, denoising-timestep)` pairs where the probability ratio
`π_new / π_old` exceeded `1 + clip_ratio`, meaning the policy is trying to
increase the probability of high-advantage images more than the clip allows.
`pg_clipfrac_lower` is the fraction where the ratio fell below
`1 - clip_ratio`, meaning the policy is trying to suppress low-advantage
images more aggressively than allowed. A large asymmetry between the two
(e.g. `higher` >> `lower`) indicates the dominant learning direction and can
guide tuning of `clip_ratio` or the learning rate.

**`timing_per_image_ms/{stage}`** — per-image latency in milliseconds for
each core compute stage: `gen` (rollout), `ref` (reference log-prob),
`old_log_prob`, `adv` (advantage computation), and `update_actor`. Use
these to pinpoint which stage dominates step time.

**`perf/throughput`** — images processed per second per GPU, computed as
`(train_batch_size × rollout.n) / (time_per_step × n_gpus)`.

## Variants

For reward models that are expensive to evaluate (e.g., a VLM judge), the reward model can be allocated its own dedicated GPU resource pool and run asynchronously alongside the policy. This avoids blocking policy training on reward computation.

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_async_reward.sh
```


## Performance

> All experiments were conducted on *NVIDIA H800* GPUs using the OCR reward.

The experiment settings and throughputs are shown in the table below.

| Script | Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | `rollout.n` | lr   | # Val Samples | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / Seconds) | Time per Step (Seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- |-------------------------| --- | --- |------| --- | --- | --- |--------------------------------| --------------------------------|
| `run_qwen_image_ocr_lora.sh` | Qwen-Image | Flow-GRPO | True | 4 | qwenvl-ocr-vllm | 4 | 4 | 0 (sync)                | 32 | 16 | 3e-4 | 1k (full set) | 32×16=512 | 16 | 0.0305                         | 420 |
| `run_qwen_image_ocr_lora_async_reward.sh` | Qwen-Image | Flow-GRPO | True | 5 | qwenvl-ocr-vllm | 4 | 4 | 1                       | 32 | 16 | 3e-4 | 1k (full set) | 32×16=512 | 16 | 0.0280                         | 360 |

- Validation reward curve (evaluated with `trainer.val_before_train=True`):

<div align="center">
<img width="600" alt="2p_comparison" src="https://github.com/user-attachments/assets/1094beaf-fed9-4661-8a6a-1c3983150648" />
<br>
qwen_image_ocr_lora: corresponding with the script `run_qwen_image_ocr_lora.sh`; 
<br>
qwen_image_ocr_lora_async_reward: corresponding with the script `run_qwen_image_ocr_lora_async_reward.sh`.
</div>
