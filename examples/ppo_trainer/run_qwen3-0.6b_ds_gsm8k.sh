#!/usr/bin/env bash
# DeepSpeed PPO example (ZeRO-1/2/3 + optional offload) on GSM8K.
set -euo pipefail
set -x

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k_ppo/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k_ppo/test.parquet}

STEPS=${STEPS:-60}
ZERO_STAGE=${ZERO_STAGE:-2}          # 1 | 2 | 3
OFFLOAD=${OFFLOAD:-none}             # none | cpu
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export CUDA_VISIBLE_DEVICES

if [[ "$ZERO_STAGE" -ge 3 && "$OFFLOAD" == "cpu" ]]; then
  export VERL_ENABLE_PARAM_OFFLOAD=${VERL_ENABLE_PARAM_OFFLOAD:-1}
fi

python3 -m verl.trainer.main_ppo -cn ppo_trainer \
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
  actor_rollout_ref.model.path="$MODEL_PATH" \
  critic.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.trust_remote_code=True \
  critic.model.trust_remote_code=True \
  data.trust_remote_code=True \
  +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
  +critic.model.override_config.attn_implementation=flash_attention_2 \
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
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  trainer.project_name=verl_ds_ppo_example \
  trainer.experiment_name=qwen3_0p6b_gsm8k_ds_ppo \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.resume_mode=disable \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq=0 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.logger='[console]' \
  actor_rollout_ref.actor.deepspeed.zero_stage="$ZERO_STAGE" \
  critic.deepspeed_config.zero_stage="$ZERO_STAGE" \
  actor_rollout_ref.actor.deepspeed.offload="$OFFLOAD" \
  critic.deepspeed_config.offload="$OFFLOAD" \
  "${@}"
