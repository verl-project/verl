#!/usr/bin/env bash
# GRPO scale demo | Qwen3-235B-A22B | vLLM rollout | Megatron training | GPU/NPU
# Requires multi-node clusters. DEVICE is auto-detected from torch_npu; override
# with DEVICE=gpu|npu only when the auto-detection is wrong.

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-235B-A22B}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}
NNODES=${NNODES:-8}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}

ACTOR_TP=${ACTOR_TP:-4}
ACTOR_PP=${ACTOR_PP:-8}
ACTOR_EP=${ACTOR_EP:-4}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_DP=${ROLLOUT_DP:-}
ROLLOUT_EP=${ROLLOUT_EP:-64}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
ROLLOUT_N=${ROLLOUT_N:-16}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:--1}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_scale_demo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_235b_a22b_grpo_vllm_megatron_$(date +%Y%m%d_%H%M)}
CKPTS_DIR=${CKPTS_DIR:-.ckpt}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
n_devices_per_node=${NDEVICES_PER_NODE:-8}

case "${DEVICE}" in
    gpu)
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.free_cache_engine=True
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

TRAINER=(
    actor_rollout_ref.nccl_timeout=7200
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${CKPTS_DIR}"
)

# ---- conditional / per-device extras (rolled into a single trailing array) ----
# Seed with the always-present rollout mode so the array is never empty (Bash 3.x + set -u safe).
EXTRA=(
    model_engine=megatron
)

if [ "${DEVICE}" = npu ]; then
    ACTOR+=(
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    )
    TRAINER+=(
        trainer.n_gpus_per_node=16
    )

    EXTRA+=(
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
        actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP:-8}
        actor_rollout_ref.rollout.enforce_eager=False
        +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[8,16,32,64,128]
        +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
    )
elif [ -n "${ROLLOUT_DP}" ]; then
    EXTRA+=(actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP})
fi

if [ -n "$MCORE_MODEL_PATH" ]; then
    EXTRA+=(
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
    )
fi

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
