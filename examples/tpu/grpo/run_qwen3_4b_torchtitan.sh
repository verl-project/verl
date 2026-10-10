#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# GRPO | Qwen3-4B-Base | GSM8K | TorchTitan Training & vLLM Rollout | 32 trainer + 32 rollout TPUs
# V1 PPOTrainer (Separate Async Overlap) — requires verl-hardware-plugin:
#   ray job submit --working-dir . --runtime-env-json '{
#     "py_modules": ["/path/to/verl-hardware-plugin/verl_hardware_plugin"],
#     "env_vars": {"VERL_PLATFORM": "tpu", "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin", ...}
#   }' -- bash examples/tpu/grpo/run_qwen3_4b_torchtitan.sh
# See examples/tpu/grpo/README.md for submission instructions.
#
# By default this runs a 250-step GRPO job using the training settings of the
# submitted TPU run 4ed4c5, with GSM8K-only validation. Append --print-command
# to inspect the command locally.
# Model and parquet paths below must already be accessible on every worker.
#
# The main training settings, and what they control:
#
#   train_batch_size      128   Prompts per training step.
#   max_response_length  2048   Token budget for each sampled response.
#   total_training_steps  250   Optimizer updates in the default run.
#
# Each prompt samples rollout.n=16 responses, giving 2048 responses per optimizer
# update. GRPO estimates the advantage from the spread of rewards within each
# group of responses to the same prompt.
#
# Parallelism: the actor runs pure FSDP (tensor_parallel_size=1,
# data_parallel_shard_size=32) across eight four-chip hosts. Rollout uses another
# eight four-chip hosts with tensor_model_parallel_size=1.
#
# The original run also used TorchTitan steps/seed fixes, TPU FP32 log-probs,
# and TP1 physical-chip binding fixes. Clean PR50 does not contain those fixes;
# matching these settings alone does not provide those runtime changes.

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS=1
export VERL_PLATFORM=tpu
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_TPU_VLLM_ATTN_FP32_STATS=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd -- "${script_dir}/../../.." && pwd -P)"
cd "${repo_dir}"
PYTHON="${PYTHON:-python3}"

print_command=0
if [[ "${1:-}" == "--print-command" ]]; then
    print_command=1
    shift
fi

# Training: 128 prompts x 16 responses = 2048 responses per optimizer update.
project_name="${PROJECT_NAME:-verl_tpu_grpo}"
exp_name="${EXPERIMENT_NAME:-qwen3_4b_base_grpo_250steps_seed1}"
SEED="${SEED:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-16}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-250}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"

# Validate the complete GSM8K test split (1319 questions).
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1319}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_VALIDATION_PROMPT_LEN="${MAX_VALIDATION_PROMPT_LEN:-1024}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# Paths: the Base model and the exact prepared parquet inputs from that run.
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-4B-Base}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/gsm8k/train.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/gsm8k/test.parquet}"
LOG_ROOT="${LOG_ROOT:-/tmp/verl_dump/${exp_name}}"

# Eight four-chip hosts per role: pure FSDP32 training and 32 independent TP1 generators.
export NNODES_TRAINER="${NNODES_TRAINER:-8}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-8}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-32}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"

export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

command=("${PYTHON}" -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.loss_type=ppo_clip \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=3.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.rollout_rs=null \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.val_max_samples=-1 \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    +data.max_length=4096 \
    +data.max_token_len_per_gpu=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.pad_mode=no_padding \
    data.dataloader_num_workers=0 \
    data.seed="${SEED}" \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_splash_attention=True \
    actor_rollout_ref.actor.torchtitan.tpu_eager_mode=DEFER_AND_FUSE \
    actor_rollout_ref.actor.torchtitan.use_simple_fsdp=True \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.decay_type=cosine \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.data_loader_seed="${SEED}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    '+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}' \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.seed="${SEED}" \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.spmd_backend=default \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.reshard_after_forward=always \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.torchtitan.seed="${SEED}" \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=True \
    actor_rollout_ref.ref.torchtitan.spmd_backend=default \
    actor_rollout_ref.ref.torchtitan.max_seq_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.seed="${SEED}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.seed="${SEED}" \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=raiden \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.prompt_length="${MAX_VALIDATION_PROMPT_LEN}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.logger="['console','tensorboard','file']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.log_val_generations=4 \
    trainer.rollout_data_dir="${LOG_ROOT}/rollout" \
    trainer.validation_data_dir="${LOG_ROOT}/validation" \
    trainer.default_local_dir="/tmp/verl_checkpoints/${exp_name}" \
    hydra.run.dir="/tmp/verl_hydra/${exp_name}" \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" "$@")

if [[ "${print_command}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
