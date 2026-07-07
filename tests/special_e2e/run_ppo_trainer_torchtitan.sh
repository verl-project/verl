#!/usr/bin/env bash
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-0.6B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
#huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}

# torchtitan parallelism
FSDP_SIZE=${FSDP_SIZE:-1}
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

# torchtitan attention backend (sdpa is not a valid language-model backend):
#   "flex"       - FlexAttention; needs torch.compile to be fast (eager is slow)
#   "flex_flash" - FlexAttention FLASH kernel; needs flash-attn-4/CUTE, Hopper/Blackwell only
#   "varlen"     - fast eager, flash-style; needs FA3 (flash_attn_interface)
ATTN_TYPE=${ATTN_TYPE:-flex}
# activation checkpointing: "selective" | "full" | "none".
# Use "none" for spmd_backend=spmd_types with use_torch_compile=False (eager AC
# recompute runs off the SPMD mesh context and crashes in spmd.assert_type).
AC_MODE=${AC_MODE:-none}
# torchtitan SPMD backend:
#   "default"      - legacy per-parallelism sharding (no full-DTensor mesh)
#   "full_dtensor" - all params/buffers/inputs are DTensors on a dense multi-axis mesh
#   "spmd_types"   - spmd_types typed collectives on a dense mesh
SPMD_BACKEND=${SPMD_BACKEND:-spmd_types}

VERL_EXP_NAME=${VERL_EXP_NAME:-qwen3-0.6b-torchtitan}

common_params=(
    model_engine=torchtitan
    algorithm.adv_estimator=grpo
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=256
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.seed=42
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.min_lr_factor=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${FSDP_SIZE}"
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TP_SIZE}"
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${EP_SIZE}"
    actor_rollout_ref.actor.torchtitan.attn_type="${ATTN_TYPE}"
    actor_rollout_ref.actor.torchtitan.activation_checkpoint="${AC_MODE}"
    actor_rollout_ref.actor.torchtitan.spmd_backend="${SPMD_BACKEND}"
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False
    actor_rollout_ref.actor.torchtitan.param_offload=False
    actor_rollout_ref.actor.torchtitan.optimizer_offload=False
    actor_rollout_ref.ref.torchtitan.use_torch_compile=False
    actor_rollout_ref.ref.torchtitan.attn_type="${ATTN_TYPE}"
    actor_rollout_ref.ref.torchtitan.activation_checkpoint="${AC_MODE}"
    actor_rollout_ref.ref.torchtitan.spmd_backend="${SPMD_BACKEND}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.n=5
    critic.optim.lr=1e-5
    critic.model.path="${MODEL_PATH}"
    critic.ppo_micro_batch_size_per_gpu=4
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.logger=['console','file','wandb']
    trainer.project_name='verl_grpo_example_gsm8k_0217'
    trainer.experiment_name="${VERL_EXP_NAME}"
    trainer.val_before_train="${VAL_BEFORE_TRAIN}"
    trainer.n_gpus_per_node="${NUM_GPUS}"
    trainer.nnodes=1
    trainer.total_training_steps=100
)

python3 -m verl.trainer.main_ppo "${common_params[@]}" $@
