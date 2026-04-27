#!/usr/bin/env bash
# PPO | text | vLLM rollout | FSDP training | GPU/NPU
# Canonical PPO (actor + critic) baseline on GSM8K + MATH.

set -xeuo pipefail

# ---- user-adjustable ----
DEVICE=${DEVICE:-gpu}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-$MODEL_PATH}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-1e-5}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}

GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
MATH_TEST_FILE=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
device_trainer_args=()

case "${DEVICE}" in
    gpu)
        n_devices_per_node=${NDEVICES_PER_NODE:-${NGPUS_PER_NODE:-8}}
        project_name=${PROJECT_NAME:-verl_ppo_gsm8k_math}
        experiment_name=${EXPERIMENT_NAME:-qwen3_8b_vllm_fsdp}
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=2400
        export HCCL_EXEC_TIMEOUT=2400
        export HCCL_OP_EXPANSION_MODE=AIV
        export CLOSE_MATMUL_K_SHIFT=1

        n_devices_per_node=${NDEVICES_PER_NODE:-${NPUS_PER_NODE:-${NGPUS_PER_NODE:-8}}}
        project_name=${PROJECT_NAME:-verl_ppo_gsm8k_math_npu}
        experiment_name=${EXPERIMENT_NAME:-qwen3_8b_vllm_fsdp_npu}
        device_trainer_args+=("trainer.device=npu")
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

train_files="['$GSM8K_TRAIN_FILE', '$MATH_TRAIN_FILE']"
val_files="['$GSM8K_TEST_FILE', '$MATH_TEST_FILE']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="$CRITIC_MODEL_PATH" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=${CRITIC_LR} \
    critic.use_dynamic_bsz=True \
    critic.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    critic.fsdp.param_offload=False \
    critic.fsdp.optimizer_offload=False \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_devices_per_node} \
    trainer.nnodes=${NNODES} \
    "${device_trainer_args[@]}" \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"
