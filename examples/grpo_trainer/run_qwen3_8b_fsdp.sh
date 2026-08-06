#!/usr/bin/env bash
# GRPO | Qwen3-8B | FSDP training | NVIDIA GPUs or Ascend NPUs
#
# Knobs:
#   INFER_BACKEND   rollout backend: vllm | sglang | trtllm        (default: vllm)
#   MACHINE         free-form tag for hardware tweaks (e.g. gb200)  (default: unset)
# (DEVICE is auto-detected from torch_npu; export DEVICE=gpu|npu only to override.)
#
# TensorRT-LLM is GPU-only.
# `MACHINE=gb200` (Blackwell SM100) bundles: enforce_eager=True, FSDP
# model_dtype=bfloat16, SGLang attention_backend=flashinfer (FA3 unsupported on
# SM>90), and ray_init.num_gpus pinned (Docker --privileged bypasses GPU
# autodetect). Unknown MACHINE values are accepted and only affect experiment_name.

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MACHINE=${MACHINE:-}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-}
NPUS_PER_NODE=${NPUS_PER_NODE:-}

train_batch_size=${TRAIN_BATCH_SIZE:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-}
rollout_n=${ROLLOUT_N:-5}
sp_size=${SP_SIZE:-1}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_grpo_${INFER_BACKEND}_fsdp_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
case "${DEVICE}" in
    gpu | npu) ;;
    *)
        echo "DEVICE must be gpu or npu, got: ${DEVICE}" >&2
        exit 1
        ;;
esac

if [ "${DEVICE}" = npu ] && [ "${INFER_BACKEND}" = trtllm ]; then
    echo "INFER_BACKEND=trtllm is only supported with DEVICE=gpu" >&2
    exit 1
fi

# Defaults and extras grouped by device. Backend / machine refinements stay
# nested in the device branch they apply to.
EXTRA=()
case "${DEVICE}" in
    gpu)
        actor_param_offload=False
        actor_optimizer_offload=False
        rollout_tp=${rollout_tp:-2}
        rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.6}

        case "${MACHINE}" in
            gb200)
                NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
                # Blackwell SM100: see header comment for rationale of each override.
                EXTRA+=(
                    actor_rollout_ref.rollout.enforce_eager=True
                    actor_rollout_ref.rollout.free_cache_engine=True
                    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
                    "+ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}"
                )
                if [ "${INFER_BACKEND}" = sglang ]; then
                    EXTRA+=(+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer)
                fi
                ;;
            *)
                NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
                ;;
        esac
        n_trainer_devices=${NGPUS_PER_NODE}
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

        NPUS_PER_NODE=8
        n_trainer_devices=${NPUS_PER_NODE}
        actor_param_offload=True
        actor_optimizer_offload=True
        rollout_tp=${rollout_tp:-4}
        sp_size=4
        train_batch_size=16
        max_prompt_length=$((1024 * 2))
        max_response_length=$((1024 * 32))
        ppo_mini_batch_size=16
        rollout_gpu_mem_util=0.3
        EXTRA+=(
            actor_rollout_ref.actor.use_torch_compile=False
            actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
            actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
        )
        if [ "${INFER_BACKEND}" = sglang ]; then
            EXTRA+=(
                +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=ascend
            )
        fi
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$HOME/data/gsm8k/train.parquet', '$HOME/data/math/train.parquet']"
    data.val_files="['$HOME/data/gsm8k/test.parquet', '$HOME/data/math/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

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
