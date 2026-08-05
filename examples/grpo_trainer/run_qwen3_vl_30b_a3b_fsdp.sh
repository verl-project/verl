#!/usr/bin/env bash
# GRPO | Qwen3-VL-30B-A3B (MoE) | FSDP training | GPU/NPU
set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_geo3k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_30b_a3b_grpo_${INFER_BACKEND}_fsdp_$(date +%Y%m%d_%H%M)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-VL-30B-A3B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-10}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
FSDP_SIZE=${FSDP_SIZE:-}
SP_SIZE=${SP_SIZE:-2}

ROLLOUT_TP=${ROLLOUT_TP:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-}
ROLLOUT_N=${ROLLOUT_N:-5}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-20000}

ROLLOUT_IS=${ROLLOUT_IS:-sequence}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-true}
ROLLOUT_RS=${ROLLOUT_RS:-token_k1}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-0.6_1.6}

SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}

case "${DEVICE}" in
    gpu)
        nnodes=${NNODES:-1}
        fsdp_size=${FSDP_SIZE:-8}
        rollout_tp=${ROLLOUT_TP:-4}
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

        nnodes=${NNODES:-2}
        fsdp_size=${FSDP_SIZE:-16}
        rollout_tp=${ROLLOUT_TP:-8}
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

ckpts_dir=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}
    algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE}
    algorithm.rollout_correction.rollout_rs=${ROLLOUT_RS}
    algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD}
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.calculate_log_probs=True
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console", "wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${nnodes}
    trainer.default_local_dir=${ckpts_dir}
    trainer.resume_mode=auto
    trainer.val_before_train=True
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=()

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
