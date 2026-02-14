# DeepSpeed Worker Integration (Current PR Scope)

## Goal

Make DeepSpeed a first-class training backend in verl for RL training, with reproducible PPO/GRPO runs and clear feature boundaries.

Rollout backend changes:

- `vLLM` is migrated to native server mode (HTTP-based) to align with upstream architecture direction.
- Inference backend in this path is `vLLM` async rollout (`actor_rollout_ref.rollout.name=vllm`, `mode=async`).
- Training algorithms covered in this scope: `PPO` (actor+critic), `GRPO` (actor-only, `critic.enable=False`).

## Feature Matrix

| Feature | Status | Notes |
| --- | --- | --- |
| Data Parallel (DP) | Supported | Actor/ref/critic/reward workers run in DP layout. |
| ZeRO-1 | Supported | Training path available. |
| ZeRO-2 | Supported | Training path available. |
| ZeRO-3 | Supported | Training path available. |
| Offload | Stage-aware | ZeRO-2: optimizer offload; ZeRO-3: optimizer + param offload (param offload needs env gate). |
| Checkpoint save/load | Supported | Native DS checkpoint manager (`DeepSpeedCheckpointManager`). |
| Activation checkpointing | Supported | Controlled by model/deepspeed flags. |
| Ulysses / SP | Not supported | Forced to `ulysses_sequence_parallel_size=1` in DS workers. |

Important offload behavior:

- `zero1_cpu_offload` in current DS worker path is normalized to no effective offload.
- Real CPU offload should use ZeRO-2/ZeRO-3 commands below.
- ZeRO-3 param offload requires `VERL_ENABLE_PARAM_OFFLOAD=1`.

## Environment Knobs

- `VERL_ENABLE_PARAM_OFFLOAD=1`: enable ZeRO-3 parameter offload.
- `VERL_DS_ZERO2_FP32_ACCUM_PATCH=1`: enable ZeRO-2 accumulation patch.
- `VERL_DS_ZERO2_STEP_EACH_MICRO=0|1`: switch ZeRO-2 micro-step behavior.
- `VERL_DS_ROLLOUT_SYNC_PROFILE=1`: log rollout weight-sync phase timings (helps diagnose ZeRO-3 slowness).

## Command Setup (Relative Paths)

Run from repo root (`verl/`):

```bash
export DATA_TRAIN=../data/gsm8k_ppo/train.parquet
export DATA_VAL=../data/gsm8k_ppo/test.parquet

export BASE_ARGS="\
actor@actor_rollout_ref.actor=ds_actor \
critic=ds_critic \
actor_rollout_ref.actor.strategy=deepspeed \
trainer.use_legacy_worker_impl=enable \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.load_format=auto \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.max_model_len=512 \
actor_rollout_ref.rollout.max_num_seqs=8 \
actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
critic.model.path=Qwen/Qwen3-0.6B \
actor_rollout_ref.model.trust_remote_code=True \
critic.model.trust_remote_code=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
critic.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
critic.ppo_micro_batch_size_per_gpu=2 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
critic.ppo_max_token_len_per_gpu=2048 \
actor_rollout_ref.actor.use_kl_loss=False \
algorithm.use_kl_in_reward=False \
data.train_files=${DATA_TRAIN} \
data.val_files=${DATA_VAL} \
data.train_batch_size=64 \
data.max_prompt_length=512 \
data.max_response_length=512 \
data.trust_remote_code=True \
trainer.n_gpus_per_node=2 \
trainer.nnodes=1 \
trainer.save_freq=0 \
trainer.test_freq=-1 \
trainer.val_before_train=False \
trainer.total_training_steps=60 \
trainer.logger=[console]"
```

## Six Training Cases (Single Commands)

### 1) `zero1_no_offload`

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 \
VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  ${BASE_ARGS} \
  trainer.experiment_name=zero1_no_offload \
  actor_rollout_ref.actor.deepspeed.zero_stage=1 \
  critic.deepspeed_config.zero_stage=1 \
  actor_rollout_ref.actor.deepspeed.offload=none \
  critic.deepspeed_config.offload=none
```

### 2) `zero2_no_offload`

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 \
VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  ${BASE_ARGS} \
  trainer.experiment_name=zero2_no_offload \
  actor_rollout_ref.actor.deepspeed.zero_stage=2 \
  critic.deepspeed_config.zero_stage=2 \
  actor_rollout_ref.actor.deepspeed.offload=none \
  critic.deepspeed_config.offload=none
```

### 3) `zero3_no_offload`

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 \
VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  ${BASE_ARGS} \
  trainer.experiment_name=zero3_no_offload \
  actor_rollout_ref.actor.deepspeed.zero_stage=3 \
  critic.deepspeed_config.zero_stage=3 \
  actor_rollout_ref.actor.deepspeed.offload=none \
  critic.deepspeed_config.offload=none
```

### 4) `zero1_cpu_offload` (current behavior note)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 \
VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  ${BASE_ARGS} \
  trainer.experiment_name=zero1_cpu_offload \
  actor_rollout_ref.actor.deepspeed.zero_stage=1 \
  critic.deepspeed_config.zero_stage=1 \
  actor_rollout_ref.actor.deepspeed.offload=cpu \
  critic.deepspeed_config.offload=cpu
```

This case is currently normalized to no effective offload in DS worker logic.

### 5) `zero2_cpu_offload` (real offload)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 \
VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  ${BASE_ARGS} \
  trainer.experiment_name=zero2_cpu_offload \
  actor_rollout_ref.actor.deepspeed.zero_stage=2 \
  critic.deepspeed_config.zero_stage=2 \
  actor_rollout_ref.actor.deepspeed.offload=cpu \
  critic.deepspeed_config.offload=cpu
```

### 6) `zero3_cpu_offload` (real offload, includes param offload)

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VERL_ENABLE_PARAM_OFFLOAD=1 \
VERL_DS_ZERO2_FP32_ACCUM_PATCH=1 \
VERL_DS_ZERO2_STEP_EACH_MICRO=0 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  ${BASE_ARGS} \
  trainer.experiment_name=zero3_cpu_offload \
  actor_rollout_ref.actor.deepspeed.zero_stage=3 \
  critic.deepspeed_config.zero_stage=3 \
  actor_rollout_ref.actor.deepspeed.offload=cpu \
  critic.deepspeed_config.offload=cpu
```

## GRPO 60-step Reference Command (DeepSpeed Actor Path)

This configuration avoids the `128-token` reward sparsity issue and enables configurable GSM8K reward kwargs.

```bash
CUDA_VISIBLE_DEVICES=4,5 \
python3 -m verl.trainer.main_ppo -cn ppo_trainer \
  actor@actor_rollout_ref.actor=ds_actor \
  critic=ds_critic \
  actor_rollout_ref.actor.strategy=deepspeed \
  actor_rollout_ref.actor.deepspeed.zero_stage=2 \
  actor_rollout_ref.actor.deepspeed.offload=none \
  critic.enable=False \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  +reward.reward_kwargs.method=flexible \
  +reward.reward_kwargs.format_score=0.1 \
  +reward.reward_kwargs.score=1.0 \
  reward.num_workers=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.max_model_len=1024 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  data.train_files=${DATA_TRAIN} \
  data.val_files=${DATA_VAL} \
  data.train_batch_size=64 \
  data.max_prompt_length=256 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.trust_remote_code=True \
  trainer.experiment_name=grpo_fix_60 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_training_steps=60 \
  trainer.save_freq=0 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.logger=[console]
```

## GRPO Six-Case Benchmark + Curves

Run full six-case GRPO benchmark (default behavior):

```bash
STEPS=60 \
RUN_TAG=grpo_zero_six_60 \
./scripts/bench/run_ds_zero_six_case_grpo_bench.sh
```

Optional fast mode: reuse a completed `zero2_no_offload` baseline and run the other five cases:

```bash
STEPS=60 \
REUSE_BASELINE=1 \
BASELINE_CASE=zero2_no_offload \
BASELINE_LOG=outputs/<baseline_run>/zero2_no_offload/train.log \
RUN_TAG=grpo_zero_six_60_reuse \
./scripts/bench/run_ds_zero_six_case_grpo_bench.sh
```

Outputs:

- `outputs/<run_tag>_<timestamp>/summary.tsv`
- `outputs/<run_tag>_<timestamp>/grpo_zero_six_curves.png`
- `outputs/<run_tag>_<timestamp>/grpo_zero_six_curves_metrics.tsv`

Plot-only command:

```bash
python3 ./scripts/bench/plot_ds_zero_six_grpo_curves.py \
  --run-root outputs/<run_tag>_<timestamp>
```

## PPO + GRPO Twelve-Case Benchmark (60-step)

Run PPO six-case + GRPO six-case end-to-end:

```bash
SEED=7777 \
STEPS=60 \
DS_ROLLOUT_SYNC_PROFILE=0 \
PPO_TAG=ppo_zero_six_60_seed7777 \
GRPO_TAG=grpo_zero_six_60_seed7777 \
./scripts/bench/run_ds_ppo_grpo_twelve_case_bench.sh
```

Main outputs:

- `outputs/<ppo_tag>_<timestamp>/summary.tsv`
- `outputs/<ppo_tag>_<timestamp>/zero_six_curves.png`
- `outputs/<grpo_tag>_<timestamp>/summary.tsv`
- `outputs/<grpo_tag>_<timestamp>/grpo_zero_six_curves.png`
- `outputs/twelve_case_manifest_<timestamp>.txt`

## PR Curve Export (GRPO-30 + PPO-60)

Export a PR-friendly curve bundle from existing run logs:

```bash
./scripts/bench/export_pr_curves.sh
```

Custom run roots/output directory:

```bash
PPO_RUN_ROOT=outputs/<ppo_run_dir> \
GRPO_RUN_ROOT=outputs/<grpo_run_dir> \
OUT_DIR=outputs/pr_curves_manual \
./scripts/bench/export_pr_curves.sh
```

Generated artifacts:

- `outputs/pr_curves_<timestamp>/grpo_zero_six_30step_curves.png`
- `outputs/pr_curves_<timestamp>/grpo_zero_six_30step_metrics.tsv`
- `outputs/pr_curves_<timestamp>/ppo_zero_six_60step_curves.png`
- `outputs/pr_curves_<timestamp>/ppo_zero_six_60step_metrics.tsv`

## ZeRO-3 Slowdown Profiling Workflow

1) Enable runtime sync-phase logs for actor->rollout weight updates:

```bash
VERL_DS_ROLLOUT_SYNC_PROFILE=1 python3 -m verl.trainer.main_ppo ...
```

Or enable it for six-case bench runs:

```bash
DS_ROLLOUT_SYNC_PROFILE=1 \
STEPS=60 \
RUN_TAG=zero_six_60_sync_profile \
./scripts/bench/run_ds_zero_six_case_bench.sh
```

2) Build timing breakdown from benchmark logs:

```bash
python3 ./scripts/bench/profile_zero_case_timing_breakdown.py \
  --run-root outputs/zero_six_60_seed7777_fix_20260209_120716 \
  --tail-steps 5 \
  --baseline-case zero2_no_offload \
  --output-tsv outputs/zero_six_60_seed7777_fix_20260209_120716/timing_breakdown_tail5.tsv
```

Typical observation in this PR branch: `zero3_no_offload` is mainly slowed by
`timing_s/update_weights` (rollout sync) and by `old_log_prob/values` recomputation overhead.
