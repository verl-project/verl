#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# GRPO | Qwen3-4B | GSM8K | TorchTitan Training & vLLM Rollout | TPU v6e-4 x2 Slices
# V1 PPOTrainer (Separate Async Overlap) — requires verl-hardware-plugin.
# See examples/tpu/grpo/README.md for full instructions.

set -xeuo pipefail

export VERL_PLATFORM=tpu
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_TPU_VLLM_ATTN_FP32_STATS=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
# RL timeline: per-step trainer / weight-sync / rollout spans and TPU HBM usage in the job log; verl-hardware-plugin's
# profilers.make_report turns the log into an HTML report. Set VERL_TPU_TIMELINE=0 to turn it off.
export VERL_TPU_TIMELINE="${VERL_TPU_TIMELINE:-1}"
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
# PROMOTE_F32 promotes sigmoid/SiLU/GELU to f32 (instead of bf16) on TPU 7x, for RL stability; a no-op on v6e.
export LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32"

project_name="${PROJECT_NAME:-verl_tpu_grpo}"
exp_name="${EXPERIMENT_NAME:-qwen3_4b_gsm8k_grpo}"
SEED="${SEED:-1}"

# Paths
RAY_DATA_HOME="${RAY_DATA_HOME:-/data}"
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d "${RAY_DATA_HOME}/assets/hf/Qwen3-4B-Base" ]]; then
        MODEL_PATH="${RAY_DATA_HOME}/assets/hf/Qwen3-4B-Base"
    else
        MODEL_PATH="${RAY_DATA_HOME}/assets/hf/Qwen3-4B"
    fi
fi
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/gsm8k/train.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/gsm8k/test.parquet}"
LOG_ROOT="${LOG_ROOT:-/tmp/verl_dump/${exp_name}}"

# TPU topology: 2 single-host v6e-4 slices (4 trainer chips + 4 rollout chips).
# For 2 v6e-8 slices (8 + 8 chips), set NNODES_TRAINER=2 NNODES_ROLLOUT=2.
export NNODES_TRAINER="${NNODES_TRAINER:-1}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
# all_gather: load-balanced KV all-gather CP with Splash Attention; ulysses: DeepSpeed-Ulysses all-to-all CP.
CONTEXT_PARALLEL_STRATEGY="${CONTEXT_PARALLEL_STRATEGY:-all_gather}"
if (( TENSOR_PARALLEL_SIZE < 1 || CONTEXT_PARALLEL_SIZE < 1 || TOTAL_TRAINER_CHIPS % (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE) != 0 )); then
    echo "ERROR: TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} x CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE} must be a positive divisor of the ${TOTAL_TRAINER_CHIPS} trainer chips." >&2
    exit 1
fi
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-$((TOTAL_TRAINER_CHIPS / TENSOR_PARALLEL_SIZE / CONTEXT_PARALLEL_SIZE))}"
# The tpu-sync checkpoint engine supports a single rollout replica, so the rollout TP spans all rollout chips.
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-${TOTAL_ROLLOUT_CHIPS}}"
FUSE_QKV="${FUSE_QKV:-True}"

# Training & validation settings
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

# Rollout & training overlap settings
# Batches queued ahead of the trainer. With 2, the rollout always has requests to refill its
# MAX_NUM_SEQS decode slots, also while the trainer still waits for the long responses of a batch.
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
# vLLM token budget per engine step. Every weight sync aborts the in-flight rollouts, which then
# re-prefill their prompt and partial response; a larger budget gets them decoding again sooner.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
# The actor update pads each packed micro-batch to a multiple of this many tokens. Smaller buckets
# waste less compute on padding as responses get shorter, at the cost of a few more compiled shapes.
PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-512}"

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    trainer.v1.separate_async.parameter_sync_step=1 \
    transfer_queue.enable=True \
    transfer_queue.backend.SimpleStorage.total_storage_size=100000 \
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
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
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
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_splash_attention=True \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tpu_eager_mode=DEFER_AND_FUSE \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=True \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.fuse_qkv="${FUSE_QKV}" \
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
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.context_parallel_strategy="${CONTEXT_PARALLEL_STRATEGY}" \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.context_parallel_strategy="${CONTEXT_PARALLEL_STRATEGY}" \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.fuse_qkv="${FUSE_QKV}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.reshard_after_forward=always \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.actor.torchtitan.pad_to_length=True \
    actor_rollout_ref.actor.torchtitan.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.torchtitan.seed="${SEED}" \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=True \
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
    actor_rollout_ref.rollout.checkpoint_engine.backend=tpu-sync \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LEN}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LEN}" \
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
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" "$@"
