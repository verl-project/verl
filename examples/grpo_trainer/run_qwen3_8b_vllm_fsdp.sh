#!/usr/bin/env bash
# GRPO | text | vLLM rollout | FSDP training | GPU/NPU
#
# Canonical GRPO baseline: Qwen3-8B on GSM8K + MATH.
# Use DEVICE=gpu or DEVICE=npu to select hardware-specific defaults.

set -xeuo pipefail

# ---- user-adjustable ----
DEVICE=${DEVICE:-gpu}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-}
ROLLOUT_N=${ROLLOUT_N:-5}
SP_SIZE=${SP_SIZE:-1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}

GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-$HOME/data/gsm8k/test.parquet}
MATH_TRAIN_FILE=${MATH_TRAIN_FILE:-$HOME/data/math/train.parquet}
MATH_TEST_FILE=${MATH_TEST_FILE:-$HOME/data/math/test.parquet}
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
device_actor_args=()
device_rollout_args=()
device_ref_args=()
device_trainer_args=()

case "${DEVICE}" in
    gpu)
        n_devices_per_node=${NDEVICES_PER_NODE:-${NGPUS_PER_NODE:-8}}
        rollout_tp=${ROLLOUT_TP:-2}
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
        experiment_name=${EXPERIMENT_NAME:-qwen3_8b_vllm_fsdp}
        device_actor_args+=(
            "actor_rollout_ref.actor.fsdp_config.param_offload=False"
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
        )
        device_rollout_args+=("actor_rollout_ref.rollout.mode=async")
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

        n_devices_per_node=${NDEVICES_PER_NODE:-${NPUS_PER_NODE:-8}}
        rollout_tp=${ROLLOUT_TP:-4}
        rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
        experiment_name=${EXPERIMENT_NAME:-qwen3_8b_vllm_fsdp_npu}
        device_actor_args+=(
            "actor_rollout_ref.actor.use_torch_compile=False"
            "actor_rollout_ref.actor.fsdp_config.param_offload=True"
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
            "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}"
        )
        device_rollout_args+=(
            "actor_rollout_ref.rollout.enable_chunked_prefill=False"
            "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096"
        )
        device_ref_args+=("actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}")
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
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    "${device_actor_args[@]}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    "${device_rollout_args[@]}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    "${device_ref_args[@]}" \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_devices_per_node} \
    trainer.nnodes=${NNODES} \
    "${device_trainer_args[@]}" \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"
