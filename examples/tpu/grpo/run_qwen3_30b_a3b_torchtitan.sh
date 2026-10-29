#!/usr/bin/env bash
# Copyright (c) 2026 Google LLC. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# GRPO | Qwen3-30B-A3B (MoE) | GSM8K | TorchTitan Training & vLLM Rollout | TPU: a trainer slice and a rollout slice
# (default TPU v7x 2x2x1 + 2x2x1).
# V1 PPOTrainer (Separate Async Overlap) - requires verl-hardware-plugin with Qwen3 MoE support.
# See examples/tpu/grpo/README.md (Qwen3-30B-A3B on TPU) for full instructions.
#
# Qwen3-30B-A3B: 48 layers, hidden 2048, 128 experts (8 active per token, moe_intermediate_size 768);
# 30.5B parameters, 3.3B active per token.
#
# Trainer: FSDP only (tensor parallel 1, expert parallel 1) over all trainer chips, as in torchtitan's FSDP-only
# Qwen3-30B-A3B TPU recipe. fp32 params + grads + Adam m/v take 30.5B * 16 B / <trainer chips> per chip
# (~61 GB on the 8 TensorCores of a v7x 2x2x1 slice, 96 GiB HBM each).
# Rollout: one vLLM replica with TP over all rollout chips and expert parallelism over the same group
# (ROLLOUT_EP = ROLLOUT_TP, whole experts per rank). The tpu-sync weight sync then installs every rank's experts
# directly into vLLM's fused expert weights instead of going through vLLM's loaders.

set -xeuo pipefail

export VERL_PLATFORM=tpu
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x}"
export VERL_TPU_VLLM_ATTN_FP32_STATS=1
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
# RL timeline: per-step trainer / weight-sync / rollout spans and TPU HBM usage in the job log; verl-hardware-plugin's
# profilers.make_report turns the log into an HTML report. Set VERL_TPU_TIMELINE=0 to turn it off.
export VERL_TPU_TIMELINE="${VERL_TPU_TIMELINE:-1}"
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99
# PROMOTE_F32 promotes sigmoid/SiLU/GELU to f32 (instead of bf16) on TPU 7x, for RL stability; a no-op on v6e.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_use_enhanced_launch_barrier=false --xla_tpu_bf16_emission_mode=PROMOTE_F32}"

project_name="${PROJECT_NAME:-verl_tpu_grpo}"
exp_name="${EXPERIMENT_NAME:-qwen3_30b_a3b_gsm8k_grpo}"
SEED="${SEED:-1}"

# Paths
RAY_DATA_HOME="${RAY_DATA_HOME:-/data}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-30B-A3B}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/gsm8k/train.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/gsm8k/test.parquet}"
LOG_ROOT="${LOG_ROOT:-/tmp/verl_dump/${exp_name}}"

# Optional: copy the checkpoint (~61 GB) to a local path on every TPU host before training. TorchTitan's HF reader and
# vLLM mmap the safetensors shards, which is slow over GCS Fuse; a parallel copy is much faster. Hosts that already
# hold a complete copy are skipped, and the CPU head only gets the config and tokenizer files.
STAGE_MODEL="${STAGE_MODEL:-0}"
if [[ "${STAGE_MODEL}" == "1" ]]; then
    LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-/tmp/models/$(basename "${MODEL_PATH}")}"
    python3 "$(dirname "${BASH_SOURCE[0]}")/stage_model_local.py" "${MODEL_PATH}" "${LOCAL_MODEL_PATH}"
    MODEL_PATH="${LOCAL_MODEL_PATH}"
fi

# TPU topology. N_CHIPS_* counts torch_tpu processes per host: 8 on v7x (4 chips x 2 TensorCores), 4 on v5p / v6e.
# Default: a v7x 2x2x1 trainer slice (1 host, 8 TensorCores) and a v7x 2x2x1 rollout slice.
export NNODES_TRAINER="${NNODES_TRAINER:-1}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-8}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))
TOTAL_ROLLOUT_CHIPS=$((NNODES_ROLLOUT * N_CHIPS_ROLLOUT))

# The tpu-sync checkpoint engine supports a single rollout replica, so the rollout TP spans all rollout chips.
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-${TOTAL_ROLLOUT_CHIPS}}"
# vLLM expert parallelism over the whole TP group: 1 (off) or ROLLOUT_TENSOR_PARALLEL_SIZE. With EP every rank holds
# whole experts (no 768-column split of each expert) and tpu-sync installs them directly; with 1, the weight sync
# falls back to vLLM's loaders, which is much slower.
ROLLOUT_EXPERT_PARALLEL_SIZE="${ROLLOUT_EXPERT_PARALLEL_SIZE:-${ROLLOUT_TENSOR_PARALLEL_SIZE}}"

# Training & validation settings
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
# Constant learning rate (no warmup), as in the Qwen3-30B-A3B runs this recipe was tuned with.
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"

VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-128}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
# Qwen3 chat-template thinking mode, off: with thinking on, many responses hit the 1024-token budget before the
# "#### <answer>" line that the GSM8K reward needs, and score 0.
ENABLE_THINKING="${ENABLE_THINKING:-False}"

# Rollout & training overlap settings (see run_qwen3_4b_torchtitan.sh)
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-512}"

# Packed micro-batches: up to MAX_TOKEN_LEN_PER_GPU tokens of whole sequences per micro-batch (use_dynamic_bsz) for
# the actor update and the reference log-probs. Every micro-batch pays the FSDP collectives of all 48 layers, so a few
# packed micro-batches are much faster than one sequence each. Must stay >= MAX_MODEL_LEN.
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-4096}"
# Packed micro-batch lengths are padded to a multiple of this, which keeps the set of compiled shapes small. The
# plugin requires SEQ_BUCKET_SIZE * 8 (top-k) to be a multiple of its 256-row grouped-matmul tile.
SEQ_BUCKET_SIZE="${SEQ_BUCKET_SIZE:-1024}"

# TorchTitan trainer: SimpleFSDP with per-block torch.compile and the splash attention kernel; DEFER_AND_FUSE fuses
# the ops outside the compiled blocks into larger XLA programs (see run_qwen3_4b_torchtitan.sh).
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_AND_FUSE}"
# KL-to-reference loss, off by default: the reference model keeps a second fp32 copy of the weights on every
# trainer chip.
USE_KL_LOSS="${USE_KL_LOSS:-False}"

# tpu-sync weight-sync check: off | norm (compare trainer / rollout weight norms after every sync) | exact (on the
# step-0 sync, compare every received tensor bit for bit with the checkpoint vLLM loaded) | all (norm + exact).
VERIFY_PARITY="${VERIFY_PARITY:-off}"
case "${VERIFY_PARITY}" in
    off | norm | exact | all) ;;
    *)
        echo "ERROR: VERIFY_PARITY=${VERIFY_PARITY} must be one of off, norm, exact, all." >&2
        exit 1
        ;;
esac

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
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.seq_bucket_size="${SEQ_BUCKET_SIZE}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.decay_type=cosine \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
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
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${TOTAL_TRAINER_CHIPS}" \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.reshard_after_forward=always \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.actor.torchtitan.pad_to_length=True \
    actor_rollout_ref.actor.torchtitan.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.torchtitan.seed="${SEED}" \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=True \
    actor_rollout_ref.ref.torchtitan.max_seq_len="${MAX_MODEL_LEN}" \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.use_splash_attention=True \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.tpu_eager_mode="${TPU_EAGER_MODE}" \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.use_simple_fsdp=True \
    +actor_rollout_ref.ref.torchtitan.engine_kwargs.tpu.seq_bucket_size="${SEQ_BUCKET_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EXPERT_PARALLEL_SIZE}" \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=tpu-sync \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.verify_parity="${VERIFY_PARITY}" \
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
