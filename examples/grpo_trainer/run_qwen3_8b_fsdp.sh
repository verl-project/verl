#!/usr/bin/env bash
# GRPO | Qwen3-8B | FSDP training | NVIDIA GPUs or Ascend NPUs
#
# INFER_BACKEND controls rollout backend: vllm, sglang, or trtllm.
# DEVICE controls hardware path: gpu or npu. TensorRT-LLM is GPU-only.

set -xeuo pipefail

# ---- user-adjustable ----
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}

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

project_name=${PROJECT_NAME:-verl_grpo_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-}
# ---- end user-adjustable ----

# ---- device / backend defaults (normally leave as-is) ----
case "${DEVICE}" in
    gpu | npu) ;;
    *)
        echo "DEVICE must be gpu or npu, got: ${DEVICE}" >&2
        exit 1
        ;;
esac

case "${INFER_BACKEND}" in
    vllm | sglang | trtllm) ;;
    *)
        echo "INFER_BACKEND must be vllm, sglang, or trtllm, got: ${INFER_BACKEND}" >&2
        exit 1
        ;;
esac

optional_ppo_args=()
actor_param_offload=False
actor_optimizer_offload=False

if [ "${DEVICE}" = npu ]; then
    if [ "${INFER_BACKEND}" = trtllm ]; then
        echo "INFER_BACKEND=trtllm is only supported with DEVICE=gpu" >&2
        exit 1
    fi

    export HCCL_CONNECT_TIMEOUT=1500
    export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
    export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

    NPUS_PER_NODE=${NPUS_PER_NODE:-8}
    n_trainer_devices=${NPUS_PER_NODE}
    actor_param_offload=True
    actor_optimizer_offload=True
    experiment_name=${experiment_name:-qwen3_8b_${INFER_BACKEND}_fsdp_npu}

    optional_ppo_args+=(
        actor_rollout_ref.actor.use_torch_compile=False
        actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size="${sp_size}"
        actor_rollout_ref.rollout.enable_chunked_prefill=False
        actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size="${sp_size}"
        trainer.device=npu
    )

    if [ "${INFER_BACKEND}" = vllm ]; then
        rollout_tp=${rollout_tp:-4}
        rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.5}
        optional_ppo_args+=(actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096)
    else
        rollout_tp=${rollout_tp:-4}
        rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.3}
        optional_ppo_args+=(+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=ascend)
    fi
else
    NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
    n_trainer_devices=${NGPUS_PER_NODE}
    rollout_tp=${rollout_tp:-2}
    rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.6}
    experiment_name=${experiment_name:-qwen3_8b_${INFER_BACKEND}_fsdp}

    if [ "${INFER_BACKEND}" = trtllm ]; then
        optional_ppo_args+=(actor_rollout_ref.hybrid_engine=True)
    else
        optional_ppo_args+=(actor_rollout_ref.rollout.mode=async)
    fi
fi
# ---- end device / backend defaults ----

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet
math_train=$HOME/data/math/train.parquet
math_test=$HOME/data/math/test.parquet

train_files="['$gsm8k_train', '$math_train']"
val_files="['$gsm8k_test', '$math_test']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload} \
    actor_rollout_ref.rollout.name=${INFER_BACKEND} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_trainer_devices} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    "${optional_ppo_args[@]}" \
    "$@"
