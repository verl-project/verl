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
