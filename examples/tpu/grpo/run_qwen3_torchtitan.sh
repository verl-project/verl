#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# GRPO | Dense Qwen3 (0.6B / 1.7B / 4B / 8B / 14B / 32B) | GSM8K | TorchTitan Training & vLLM Rollout | Cloud TPU
# V1 PPOTrainer (Separate Async Overlap) — requires verl-hardware-plugin.
# See examples/tpu/grpo/README.md for full instructions.
#
# Select the model size with MODEL_SIZE=0.6B|1.7B|4B|8B|14B|32B (defaults to 0.6B, a 5-step smoke test on 2 x v6e-1):
#   - 0.6B : 2 x v6e-1  (1x1,  1 +  1 chips), FSDP1  trainer, 1 x TP1  rollout, 5-step smoke test
#   - 1.7B : 2 x v6e-4  (2x2,  4 +  4 chips), FSDP4  trainer, 1 x TP4  rollout, 100 steps
#   - 4B   : 2 x v6e-4  (2x2,  4 +  4 chips), FSDP4  trainer, 1 x TP4  rollout, 100 steps
#   - 8B   : 2 x v6e-8  (2x4,  8 +  8 chips), FSDP8  trainer, 1 x TP8  rollout, 100 steps, USE_KL_LOSS=False
#   - 14B  : 2 x v6e-16 (4x4, 16 + 16 chips), FSDP16 trainer, 4 x TP4  rollout, 100 steps, USE_KL_LOSS=False
#   - 32B  : 2 x v6e-32 (4x8, 32 + 32 chips), FSDP32 trainer, 1 x TP32 rollout,  20 steps, USE_KL_LOSS=False
#
# HBM per trainer chip on v6e (32 GiB): fp32 params + grads + Adam m/v take ~16.4 GB at 8B/8 chips, ~14.8 GB at
# 14B/16 chips, and ~16.4 GB at 32B/32 chips. To fit within 32 GiB HBM, the 8B / 14B / 32B presets train on 1 sequence
# per micro-batch and turn the KL-to-reference loss off by default. tpu-sync sends from a bf16 copy of each rank's
# local shards plus the full q/k/v that torchtitan's fused-wqkv state_dict hook all-gathers, and unbinds them after
# every sync (engine_kwargs.tpu-sync.release_buffers_after_sync, on by default).

set -xeuo pipefail

RAW_MODEL_SIZE="${MODEL_SIZE:-0.6B}"
MODEL_SIZE="${RAW_MODEL_SIZE^^}"
MODEL_SIZE="${MODEL_SIZE#QWEN3[-_]}"
MODEL_SIZE="${MODEL_SIZE/_/.}"

case "${MODEL_SIZE}" in
    0.6B)
        DEFAULT_MODEL_NAME="Qwen3-0.6B"
        DEFAULT_NUM_HEADS=16
        DEFAULT_EXP_NAME="qwen3_0.6b_fast_smoke_test"
        DEFAULT_NNODES=1
        DEFAULT_N_CHIPS=1
        DEFAULT_ROLLOUT_TP_PER_HOST=0
        DEFAULT_TRAIN_BSZ=4
        DEFAULT_MINI_BSZ=4
        DEFAULT_MICRO_BSZ=8
        DEFAULT_ROLLOUT_LOGPROB_MICRO_BSZ=""
        DEFAULT_LR="1e-6"
        DEFAULT_LR_WARMUP_STEPS=0
        DEFAULT_STEPS=5
        DEFAULT_VAL_BSZ=4
        DEFAULT_VAL_SAMPLES=8
        DEFAULT_TEST_FREQ=2
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_MAX_RESPONSE_LEN=512
        DEFAULT_MAX_NUM_SEQS=64
        DEFAULT_NUM_WARMUP_BATCHES=1
        DEFAULT_PAD_BUCKET=1024
        DEFAULT_USE_KL_LOSS=True
        DEFAULT_ENABLE_THINKING=True
        ;;
    1.7B)
        DEFAULT_MODEL_NAME="Qwen3-1.7B"
        DEFAULT_NUM_HEADS=16
        DEFAULT_EXP_NAME="qwen3_1.7b_gsm8k_grpo"
        DEFAULT_NNODES=1
        DEFAULT_N_CHIPS=4
        DEFAULT_ROLLOUT_TP_PER_HOST=0
        DEFAULT_TRAIN_BSZ=32
        DEFAULT_MINI_BSZ=32
        DEFAULT_MICRO_BSZ=2
        DEFAULT_ROLLOUT_LOGPROB_MICRO_BSZ=4
        DEFAULT_LR="2e-6"
        DEFAULT_LR_WARMUP_STEPS=5
        DEFAULT_STEPS=100
        DEFAULT_VAL_BSZ=64
        DEFAULT_VAL_SAMPLES=128
        DEFAULT_TEST_FREQ=10
        DEFAULT_VAL_BEFORE_TRAIN=True
        DEFAULT_MAX_RESPONSE_LEN=1024
        DEFAULT_MAX_NUM_SEQS=256
        DEFAULT_NUM_WARMUP_BATCHES=2
        DEFAULT_PAD_BUCKET=512
        DEFAULT_USE_KL_LOSS=True
        DEFAULT_ENABLE_THINKING=True
        ;;
    4B)
        DEFAULT_MODEL_NAME="Qwen3-4B"
        DEFAULT_NUM_HEADS=32
        DEFAULT_EXP_NAME="qwen3_4b_gsm8k_grpo"
        DEFAULT_NNODES=1
        DEFAULT_N_CHIPS=4
        DEFAULT_ROLLOUT_TP_PER_HOST=0
        DEFAULT_TRAIN_BSZ=32
        DEFAULT_MINI_BSZ=32
        DEFAULT_MICRO_BSZ=2
        DEFAULT_ROLLOUT_LOGPROB_MICRO_BSZ=4
        DEFAULT_LR="2e-6"
        DEFAULT_LR_WARMUP_STEPS=5
        DEFAULT_STEPS=100
        DEFAULT_VAL_BSZ=64
        DEFAULT_VAL_SAMPLES=128
        DEFAULT_TEST_FREQ=10
        DEFAULT_VAL_BEFORE_TRAIN=True
        DEFAULT_MAX_RESPONSE_LEN=1024
        DEFAULT_MAX_NUM_SEQS=256
        DEFAULT_NUM_WARMUP_BATCHES=2
        DEFAULT_PAD_BUCKET=512
        DEFAULT_USE_KL_LOSS=True
        DEFAULT_ENABLE_THINKING=True
        ;;
    8B)
        DEFAULT_MODEL_NAME="Qwen3-8B"
        DEFAULT_NUM_HEADS=32
        DEFAULT_EXP_NAME="qwen3_8b_gsm8k_grpo"
        DEFAULT_NNODES=2
        DEFAULT_N_CHIPS=4
        DEFAULT_ROLLOUT_TP_PER_HOST=0
        DEFAULT_TRAIN_BSZ=32
        DEFAULT_MINI_BSZ=32
        DEFAULT_MICRO_BSZ=1
        DEFAULT_ROLLOUT_LOGPROB_MICRO_BSZ=""
        DEFAULT_LR="1e-6"
        DEFAULT_LR_WARMUP_STEPS=5
        DEFAULT_STEPS=100
        DEFAULT_VAL_BSZ=64
        DEFAULT_VAL_SAMPLES=128
        DEFAULT_TEST_FREQ=10
        DEFAULT_VAL_BEFORE_TRAIN=True
        DEFAULT_MAX_RESPONSE_LEN=1024
        DEFAULT_MAX_NUM_SEQS=256
        DEFAULT_NUM_WARMUP_BATCHES=2
        DEFAULT_PAD_BUCKET=512
        DEFAULT_USE_KL_LOSS=False
        DEFAULT_ENABLE_THINKING=False
        ;;
    14B)
        DEFAULT_MODEL_NAME="Qwen3-14B"
        DEFAULT_NUM_HEADS=40
        DEFAULT_EXP_NAME="qwen3_14b_gsm8k_grpo"
        DEFAULT_NNODES=4
        DEFAULT_N_CHIPS=4
        # Qwen3-14B has 40 attention heads (not divisible by 16), so default to one single-host TP=4 vLLM replica per
        # rollout host (4 x TP4 on v6e-16).
        DEFAULT_ROLLOUT_TP_PER_HOST=1
        DEFAULT_TRAIN_BSZ=32
        DEFAULT_MINI_BSZ=32
        DEFAULT_MICRO_BSZ=1
        DEFAULT_ROLLOUT_LOGPROB_MICRO_BSZ=""
        DEFAULT_LR="1e-6"
        DEFAULT_LR_WARMUP_STEPS=5
        DEFAULT_STEPS=100
        DEFAULT_VAL_BSZ=64
        DEFAULT_VAL_SAMPLES=128
        DEFAULT_TEST_FREQ=10
        DEFAULT_VAL_BEFORE_TRAIN=True
        DEFAULT_MAX_RESPONSE_LEN=1024
        DEFAULT_MAX_NUM_SEQS=256
        DEFAULT_NUM_WARMUP_BATCHES=2
        DEFAULT_PAD_BUCKET=512
        DEFAULT_USE_KL_LOSS=False
        DEFAULT_ENABLE_THINKING=False
        ;;
    32B)
        DEFAULT_MODEL_NAME="Qwen3-32B"
        DEFAULT_NUM_HEADS=64
        DEFAULT_EXP_NAME="qwen3_32b_gsm8k_grpo"
        DEFAULT_NNODES=8
        DEFAULT_N_CHIPS=4
        DEFAULT_ROLLOUT_TP_PER_HOST=0
        DEFAULT_TRAIN_BSZ=32
        DEFAULT_MINI_BSZ=32
        DEFAULT_MICRO_BSZ=1
        DEFAULT_ROLLOUT_LOGPROB_MICRO_BSZ=""
        DEFAULT_LR="1e-6"
        DEFAULT_LR_WARMUP_STEPS=5
        DEFAULT_STEPS=20
        DEFAULT_VAL_BSZ=64
        DEFAULT_VAL_SAMPLES=128
        DEFAULT_TEST_FREQ=10
        DEFAULT_VAL_BEFORE_TRAIN=True
        DEFAULT_MAX_RESPONSE_LEN=1024
        DEFAULT_MAX_NUM_SEQS=256
        DEFAULT_NUM_WARMUP_BATCHES=2
        DEFAULT_PAD_BUCKET=512
        DEFAULT_USE_KL_LOSS=False
        DEFAULT_ENABLE_THINKING=False
        ;;
    *)
        echo "ERROR: MODEL_SIZE=${RAW_MODEL_SIZE} must be one of 0.6B, 1.7B, 4B, 8B, 14B, 32B." >&2
        exit 1
        ;;
esac

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
# Added to any caller-supplied LIBTPU_INIT_ARGS unless the flag is already present.
DEFAULT_LIBTPU_INIT_ARGS="--xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32"
for flag in ${DEFAULT_LIBTPU_INIT_ARGS}; do
    if [[ " ${LIBTPU_INIT_ARGS:-} " != *" ${flag%%=*}="* ]]; then
        LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }${flag}"
    fi
done
if [[ -n "${EXTRA_LIBTPU_INIT_ARGS:-}" ]]; then
    LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }${EXTRA_LIBTPU_INIT_ARGS}"
fi
export LIBTPU_INIT_ARGS

project_name="${PROJECT_NAME:-verl_tpu_grpo}"
exp_name="${EXPERIMENT_NAME:-${DEFAULT_EXP_NAME}}"
SEED="${SEED:-1}"

# Paths
# Note: loading the 62 GB Qwen3-32B checkpoint straight from GCS FUSE is slow; staging it to a pod-visible local path on
# every host (e.g. /tmp/Qwen3-32B, mounted from host /tmp in ray-tpu-v6e-2slice.yaml) and setting MODEL_PATH speeds up load.
RAY_DATA_HOME="${RAY_DATA_HOME:-/data}"
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -n "${MODEL_NAME:-}" ]]; then
        MODEL_PATH="${RAY_DATA_HOME}/assets/hf/${MODEL_NAME}"
    elif [[ "${MODEL_SIZE}" == "4B" && -d "${RAY_DATA_HOME}/assets/hf/Qwen3-4B-Base" ]]; then
        MODEL_PATH="${RAY_DATA_HOME}/assets/hf/Qwen3-4B-Base"
    else
        MODEL_PATH="${RAY_DATA_HOME}/assets/hf/${DEFAULT_MODEL_NAME}"
    fi
fi
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/gsm8k/train.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/gsm8k/test.parquet}"
LOG_ROOT="${LOG_ROOT:-/tmp/verl_dump/${exp_name}}"

# TPU topology (defaults depend on MODEL_SIZE; override NNODES_* / N_CHIPS_* to run any size on a different slice).
export NNODES_TRAINER="${NNODES_TRAINER:-${DEFAULT_NNODES}}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-${DEFAULT_N_CHIPS}}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-${DEFAULT_NNODES}}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-${DEFAULT_N_CHIPS}}"

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

# Rollout TP: defaults to one vLLM replica over all rollout chips (at TP=32 for 32B, vLLM replicates each of the 8 KV
# heads on 4 ranks and tpu-sync sends each rank its KV head), except 14B (40 Q heads, not divisible by 16), which
# defaults to one single-host replica per rollout host. Set ROLLOUT_TENSOR_PARALLEL_SIZE < TOTAL_ROLLOUT_CHIPS to run
# multiple rollout replicas in parallel.
if (( DEFAULT_ROLLOUT_TP_PER_HOST )); then
    DEFAULT_ROLLOUT_TP="${N_CHIPS_ROLLOUT}"
else
    DEFAULT_ROLLOUT_TP="${TOTAL_ROLLOUT_CHIPS}"
fi
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-${DEFAULT_ROLLOUT_TP}}"
if (( ROLLOUT_TENSOR_PARALLEL_SIZE < 1 || TOTAL_ROLLOUT_CHIPS % ROLLOUT_TENSOR_PARALLEL_SIZE != 0 )); then
    echo "ERROR: ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_TENSOR_PARALLEL_SIZE} must be a positive divisor of the ${TOTAL_ROLLOUT_CHIPS} rollout chips." >&2
    exit 1
fi
if (( DEFAULT_NUM_HEADS % ROLLOUT_TENSOR_PARALLEL_SIZE != 0 )); then
    echo "ERROR: Qwen3-${MODEL_SIZE} has ${DEFAULT_NUM_HEADS} attention heads, which is not divisible by ROLLOUT_TENSOR_PARALLEL_SIZE=${ROLLOUT_TENSOR_PARALLEL_SIZE}." >&2
    exit 1
fi
FUSE_QKV="${FUSE_QKV:-True}"

# Training & validation settings
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BSZ}}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${DEFAULT_MINI_BSZ}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${DEFAULT_MICRO_BSZ}}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-${DEFAULT_ROLLOUT_LOGPROB_MICRO_BSZ:-${MICRO_BATCH_SIZE}}}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LR}}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-${DEFAULT_LR_WARMUP_STEPS}}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${DEFAULT_STEPS}}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${DEFAULT_VAL_BSZ}}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-${DEFAULT_VAL_SAMPLES}}"
TEST_FREQ="${TEST_FREQ:-${DEFAULT_TEST_FREQ}}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${DEFAULT_VAL_BEFORE_TRAIN}}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-${DEFAULT_MAX_RESPONSE_LEN}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))}"
MIN_TOKEN_LEN=$((MICRO_BATCH_SIZE * MAX_MODEL_LEN))
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$(( MIN_TOKEN_LEN > 4096 ? MIN_TOKEN_LEN : 4096 ))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
# Hybrid thinking mode in Qwen3's chat template: off by default at 8B / 14B / 32B, whose thinking traces often run
# past the 1024-token response budget before emitting the final answer.
ENABLE_THINKING="${ENABLE_THINKING:-${DEFAULT_ENABLE_THINKING}}"

# Rollout & training overlap settings
# Batches queued ahead of the trainer. With 2, the rollout always has requests to refill its MAX_NUM_SEQS decode slots.
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-${DEFAULT_NUM_WARMUP_BATCHES}}"
# vLLM token budget per engine step. Every weight sync aborts in-flight rollouts, which then re-prefill their prompt
# and partial response; a larger budget gets them decoding again sooner.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
# The actor update pads each packed micro-batch to a multiple of this many tokens.
PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-${DEFAULT_PAD_BUCKET}}"

# TorchTitan trainer: SimpleFSDP (FSDP all-gather / reduce-scatter traced into each compiled TransformerBlock) with
# per-block torch.compile and Splash Attention. DEFER_AND_FUSE fuses neighbouring eager ops outside the compiled blocks
# (LM head, log-prob / loss, grad clipping, optimizer) into larger XLA programs so HBM stays within 32 GiB at 8B+.
# Note: on an older fork image (with TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=false), update_actor could hang
# at 32 ranks under DEFER_AND_FUSE; if a 32-rank run ever hangs in update_actor, set TPU_EAGER_MODE=DEFER_NEVER.
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_AND_FUSE}"
# KL-to-reference loss: on by default for 0.6B / 1.7B / 4B; off by default for 8B / 14B / 32B because the fp32
# reference weights add ~3.7-4.1 GB per trainer chip and leave too little HBM at the update_actor peak on v6e.
USE_KL_LOSS="${USE_KL_LOSS:-${DEFAULT_USE_KL_LOSS}}"

# tpu-sync weight-sync check: off (plugin default) | norm (compare trainer / rollout weight norms after every sync) |
# exact (on the step-0 sync, compare every received tensor bit for bit with the checkpoint vLLM loaded) | all.
EXTRA_OVERRIDES=()
if [[ -n "${VERIFY_PARITY:-}" ]]; then
    case "${VERIFY_PARITY}" in
        off | norm | exact | all) ;;
        *)
            echo "ERROR: VERIFY_PARITY=${VERIFY_PARITY} must be one of off, norm, exact, all." >&2
            exit 1
            ;;
    esac
    EXTRA_OVERRIDES+=("++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.verify_parity=${VERIFY_PARITY}")
fi

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
    +data.max_length="${MAX_TOKEN_LEN_PER_GPU}" \
    +data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    +data.pad_mode=no_padding \
    data.dataloader_num_workers=0 \
    data.seed="${SEED}" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    actor_rollout_ref.actor.strategy=torchtitan \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_splash_attention=True \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tpu_eager_mode="${TPU_EAGER_MODE}" \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp=True \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.fuse_qkv="${FUSE_QKV}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.decay_type=cosine \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.data_loader_seed="${SEED}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode \
    '+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}' \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.clip_ratio_c=3.0 \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.torchtitan.seed="${SEED}" \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.context_parallel_strategy="${CONTEXT_PARALLEL_STRATEGY}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.reshard_after_forward=always \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.actor.torchtitan.pad_to_length=True \
    actor_rollout_ref.actor.torchtitan.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.torchtitan.seed="${SEED}" \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=True \
    actor_rollout_ref.ref.torchtitan.max_seq_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.ref.torchtitan.pad_to_length=True \
    actor_rollout_ref.ref.torchtitan.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.use_splash_attention=True \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.tpu_eager_mode="${TPU_EAGER_MODE}" \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.use_simple_fsdp=True \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.fuse_qkv="${FUSE_QKV}" \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.context_parallel_strategy="${CONTEXT_PARALLEL_STRATEGY}" \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
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
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} "$@"
