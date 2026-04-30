#!/usr/bin/env bash
# GSPO | text | vLLM rollout | FSDP training | GPU/NPU
# GSPO is a sequence-mean policy-loss variant on top of GRPO (paper: arXiv:2507.18071).

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}

SP_SIZE=${SP_SIZE:-}
ROLLOUT_TP=${ROLLOUT_TP:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-}
ROLLOUT_N=${ROLLOUT_N:-16}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}

PROJECT_NAME=${PROJECT_NAME:-verl_gspo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_vllm_fsdp}

GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
MATH_TEST_FILE=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}
save_freq=${SAVE_FREQ}
test_freq=${TEST_FREQ}

case "${DEVICE}" in
    gpu)
        train_batch_size=${TRAIN_BATCH_SIZE:-512}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
        sp_size=${SP_SIZE:-1}
        rollout_tp=${ROLLOUT_TP:-2}
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
        ;;
    npu)
        ulimit -n 32768
        export RAY_DEDUP_LOGS=0
        export HYDRA_FULL_ERROR=1
        export TASK_QUEUE_ENABLE=1
        export HCCL_EXEC_TIMEOUT=3600
        export HCCL_CONNECT_TIMEOUT=3600
        export HCCL_ASYNC_ERROR_HANDLING=0
        export CPU_AFFINITY_CONF=1
        export VLLM_USE_V1=1

        train_batch_size=${TRAIN_BATCH_SIZE:-256}
        ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
        sp_size=${SP_SIZE:-4}
        rollout_tp=${ROLLOUT_TP:-4}
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.7}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$GSM8K_TRAIN_FILE', '$MATH_TRAIN_FILE']"
    data.val_files="['$GSM8K_TEST_FILE', '$MATH_TEST_FILE']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# Per-device extras (single trailing array, never empty under set -u).
EXTRA=()
if [ "${DEVICE}" = npu ]; then
    EXTRA+=(
        actor_rollout_ref.actor.strategy=fsdp2
        actor_rollout_ref.actor.use_torch_compile=False
        actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
        actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
        actor_rollout_ref.actor.entropy_checkpointing=True
        actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
        actor_rollout_ref.rollout.val_kwargs.n=1
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0
        actor_rollout_ref.rollout.val_kwargs.top_p=0.7
        actor_rollout_ref.ref.strategy=fsdp2
        actor_rollout_ref.ref.use_torch_compile=False
        actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
        trainer.val_before_train=False
    )
fi

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
