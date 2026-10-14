#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Fast 5-step smoke test: GRPO | Qwen3-0.6B | GSM8K | TorchTitan Training & vLLM Rollout | TPU v6e-4 x2 Slices
# V1 PPOTrainer (Separate Async Overlap) — requires verl-hardware-plugin.
# See examples/tpu/grpo/README.md for full instructions.

set -xeuo pipefail

export VERL_PLATFORM=tpu
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"

project_name="${PROJECT_NAME:-verl_tpu_grpo}"
exp_name="${EXPERIMENT_NAME:-qwen3_0.6b_fast_smoke_test}"

# Paths
RAY_DATA_HOME="${RAY_DATA_HOME:-/data}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-0.6B}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/gsm8k/train.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/gsm8k/test.parquet}"

# TPU topology: 2 single-host v6e-4 slices (4 trainer chips + 4 rollout chips).
# For 2 v6e-8 slices (8 + 8 chips), set NNODES_TRAINER=2 NNODES_ROLLOUT=2.
export NNODES_TRAINER="${NNODES_TRAINER:-1}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${TOTAL_TRAINER_CHIPS}}"
# The tpu-sync checkpoint engine supports a single rollout replica, so the rollout TP spans all rollout chips.
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-${TOTAL_ROLLOUT_CHIPS}}"

# Smoke-test training & validation parameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
TEST_FREQ="${TEST_FREQ:-2}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    transfer_queue.backend.SimpleStorage.total_storage_size=100000 \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    +data.max_length=4096 \
    +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.pad_mode=no_padding \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=tpu-sync \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir=/tmp/verl_dump/rollout \
    trainer.validation_data_dir=/tmp/verl_dump/validation \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" "$@"
