#!/usr/bin/env bash
# GSPO | Qwen3-8B | GSM8K | vLLM rollout | FSDP2 training | Ascend NPU

set -xeuo pipefail

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=0
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/${MODEL_ID}}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}

SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}" .sh)"
LOG_DIR=/root/.cache/nightly_log/$SCRIPT_NAME
mkdir -p "$LOG_DIR"
rm -rf "$LOG_DIR/$SCRIPT_NAME.log"

# ---- user-adjustable (CI defaults keep runtime short) ----
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ROLLOUT_N=${ROLLOUT_N:-4}
SP_SIZE=${SP_SIZE:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
FSDP_SIZE=${FSDP_SIZE:--1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-15}

CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}
ACTOR_LR=${ACTOR_LR:-1e-6}
SEED=${SEED:-42}

PROJECT_NAME=${PROJECT_NAME:-GSPO-Qwen3-8B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GSPO-Qwen3-8B-FSDP-vLLM-nightly}
# ---- end user-adjustable ----

seq_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ppo_max_token_len_per_gpu=$((seq_len / SP_SIZE))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation='left'
    data.shuffle=False
    data.validation_shuffle=False
    data.seed=${SEED}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.data_loader_seed=${SEED}
    actor_rollout_ref.actor.fsdp_config.seed=${SEED}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    actor_rollout_ref.actor.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.seed=${SEED}
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_batched_tokens=${seq_len}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4, 8, 12, 16, 20, 24, 28, 32]"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.nnodes=1
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.device=npu
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.balance_batch=False
    trainer.critic_warmup=0
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

########################### launch ###########################
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "$LOG_DIR/$SCRIPT_NAME.log"