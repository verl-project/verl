#!/usr/bin/env bash
# GRPO scale demo | Qwen3-235B-A22B | vLLM rollout | Megatron training | GPU/NPU
# Requires multi-node clusters. Use DEVICE=gpu or DEVICE=npu to select hardware-specific defaults.

set -xeuo pipefail

# ---- user-adjustable ----
DEVICE=${DEVICE:-gpu}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-235B-A22B}
MCORE_MODEL_PATH=${MCORE_MODEL_PATH:-}   # path to Megatron dist checkpoint
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
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
CKPTS_DIR=${CKPTS_DIR:-.ckpt}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
device_rollout_args=()
device_trainer_args=()
device_actor_megatron_args=()
device_ref_megatron_args=()

case "${DEVICE}" in
    gpu)
        export CUDA_DEVICE_MAX_CONNECTIONS=1

        n_devices_per_node=${NDEVICES_PER_NODE:-${NGPUS_PER_NODE:-8}}
        experiment_name=${EXPERIMENT_NAME:-qwen3_235b_a22b_vllm_megatron}
        device_rollout_args+=("actor_rollout_ref.rollout.mode=async")
        if [ -n "${ROLLOUT_DP}" ]; then
            device_rollout_args+=("actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}")
        fi
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

        n_devices_per_node=${NDEVICES_PER_NODE:-${NPUS_PER_NODE:-16}}
        rollout_dp=${ROLLOUT_DP:-8}
        experiment_name=${EXPERIMENT_NAME:-qwen3_235b_a22b_vllm_megatron_npu}
        device_rollout_args+=(
            "actor_rollout_ref.rollout.data_parallel_size=${rollout_dp}"
            "actor_rollout_ref.rollout.enforce_eager=False"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[8,16,32,64,128]"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY"
        )
        device_trainer_args+=("trainer.device=npu")
        # MindSpeed's TransformerConfig still accepts `use_flash_attn`; upstream
        # Megatron-Core (used on GPU) removed it in favor of `attention_backend`,
        # so only inject this override on NPU runs.
        device_actor_megatron_args+=(
            "+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True"
        )
        device_ref_megatron_args+=(
            "+actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True"
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

dist_ckpt_args=()
if [ -n "$MCORE_MODEL_PATH" ]; then
    dist_ckpt_args+=(
        "actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}"
        "actor_rollout_ref.actor.megatron.use_dist_checkpointing=True"
        "actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}"
        "actor_rollout_ref.ref.megatron.use_dist_checkpointing=True"
    )
fi

python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    "${device_actor_megatron_args[@]}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    "${device_rollout_args[@]}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD} \
    "${device_ref_megatron_args[@]}" \
    "${dist_ckpt_args[@]}" \
    actor_rollout_ref.nccl_timeout=7200 \
    trainer.balance_batch=True \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_devices_per_node} \
    trainer.nnodes=${NNODES} \
    "${device_trainer_args[@]}" \
    trainer.val_before_train=False \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CKPTS_DIR}" "$@"
