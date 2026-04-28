#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B | Megatron training | NVIDIA GPUs
# DAPO-style recipe on DAPO-Math-17k / AIME-2024.
#
# INFER_BACKEND controls rollout backend: vllm or sglang.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- user-adjustable ----
INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}   # optional dist-checkpoint path
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_tp=${ACTOR_TP:-4}
actor_pp=${ACTOR_PP:-2}
actor_ep=${ACTOR_EP:-4}
all_offload=${ALL_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-8}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_grpo_dapo_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_${INFER_BACKEND}_megatron}
# ---- end user-adjustable ----

# ---- backend defaults (normally leave as-is) ----
case "${INFER_BACKEND}" in
    vllm | sglang) ;;
    *)
        echo "INFER_BACKEND must be vllm or sglang, got: ${INFER_BACKEND}" >&2
        exit 1
        ;;
esac

optional_ppo_args=(actor_rollout_ref.rollout.mode=async)
if [ "${INFER_BACKEND}" = vllm ]; then
    optional_ppo_args+=(actor_rollout_ref.rollout.enable_chunked_prefill=True)
fi
# ---- end backend defaults ----

train_files=$HOME/data/dapo-math-17k.parquet
val_files=$HOME/data/aime-2024.parquet

dist_ckpt_args=""
if [ -n "$MCORE_MODEL_PATH" ]; then
    dist_ckpt_args="actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH} \
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True"
fi

python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep} \
    actor_rollout_ref.actor.megatron.param_offload=${all_offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${all_offload} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.name=${INFER_BACKEND} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep} \
    actor_rollout_ref.ref.megatron.param_offload=${all_offload} \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    ${dist_ckpt_args} \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    "${optional_ppo_args[@]}" \
    "$@"
