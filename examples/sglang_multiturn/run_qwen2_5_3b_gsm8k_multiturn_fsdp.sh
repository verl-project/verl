#!/usr/bin/env bash
# GRPO multiturn | GSM8K + tool-call | FSDP training
#
# Set INFER_BACKEND=sglang (default) or vllm.

set -xeuo pipefail

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# ---- user-adjustable ----
INFER_BACKEND=${INFER_BACKEND:-sglang}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-}

ROLLOUT_MODE=${ROLLOUT_MODE:-async}     # async | server (SGLang only)
ROLLOUT_TP=${ROLLOUT_TP:-}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-}
ROLLOUT_N=${ROLLOUT_N:-}
OVER_SAMPLE_RATE=${OVER_SAMPLE_RATE:-0.1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
ACTOR_LR=${ACTOR_LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TEST_FREQ=${TEST_FREQ:-20}

PROJECT_NAME=${PROJECT_NAME:-}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
# ---- end user-adjustable ----

# ---- backend defaults (normally leave as-is) ----
case "${INFER_BACKEND}" in
    sglang)
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
        ROLLOUT_TP=${ROLLOUT_TP:-1}
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
        ROLLOUT_N=${ROLLOUT_N:-16}
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
        PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
        PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}
        PROJECT_NAME=${PROJECT_NAME:-multi-turn-grpo-qwen2_5_3b-sglang}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_3b_gsm8k_multiturn_$(date '+%d-%H-%M')}
        if [ "${ROLLOUT_MODE}" = "server" ]; then
            CONFIG_NAME=gsm8k_multiturn_grpo_server
        else
            CONFIG_NAME=gsm8k_multiturn_grpo
        fi
        rollout_extra_args=(
            actor_rollout_ref.rollout.multi_stage_wake_up=True
            actor_rollout_ref.rollout.over_sample_rate=${OVER_SAMPLE_RATE}
        )
        actor_extra_args=()
        ;;
    vllm)
        export VLLM_USE_V1=1
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
        ROLLOUT_TP=${ROLLOUT_TP:-2}
        ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
        ROLLOUT_N=${ROLLOUT_N:-8}
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
        PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
        PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
        PROJECT_NAME=${PROJECT_NAME:-gsm8k_async_rl}
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5-3b_function_rm-gsm8k-sgl-multi-w-tool-verify-n16}
        CONFIG_NAME=gsm8k_multiturn_grpo
        rollout_extra_args=(
            actor_rollout_ref.rollout.trace.token2text=False
            actor_rollout_ref.rollout.enforce_eager=True
            actor_rollout_ref.rollout.free_cache_engine=True
        )
        actor_extra_args=(
            actor_rollout_ref.actor.use_torch_compile=False
        )
        ;;
    *)
        echo "INFER_BACKEND must be sglang or vllm, got: ${INFER_BACKEND}" >&2
        exit 1
        ;;
esac
# ---- end backend defaults ----

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    "${actor_extra_args[@]}" \
    actor_rollout_ref.rollout.name=${INFER_BACKEND} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    "${rollout_extra_args[@]}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=${TEST_FREQ} \
    trainer.val_before_train=True \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"
