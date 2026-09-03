#!/usr/bin/env bash
# GRPO | NVIDIA Nemotron 3.5 Lightning 30B-A3B | Megatron actor | vLLM rollout
# DAPO-style recipe on DAPO-Math-17k / AIME-2024.
#
# These settings completed a four-hour, 261-step BF16 soak on the v0.7 backport
# with R3 router replay and one-token MTP speculative rollout enabled. This
# rebased-main port still needs a GPU smoke test. Convergence, checkpoint restore,
# quantized synchronization, and alternate topologies remain unvalidated.
#
# This is a verl adaptation; NVIDIA has not published a Lightning-specific
# GRPO recipe. The 2x8 H100 actor topology starts from NVIDIA's verified SFT
# topology. Run this script once on the head of an already-started Ray cluster.
#
# Pinned pre-release dependency snapshot:
#   Megatron-Bridge r0.6.0: c93251151adeeadbae3ff2a2bf5ee7a1c34cff01
#   Megatron-Core 0.19.0 (Bridge submodule): cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54
#   Transformers: >=5.8.1,<5.11 (5.10.4 is compatible with verl and Bridge)
#   vLLM: 6e448d0ea9bf3d88d898b65449ca6dc2aec170ac (hardware-tested)
#
# Use the BF16 customization checkpoint. Quantized Lightning checkpoints and
# quantized actor-to-rollout weight updates are intentionally not enabled here.
# One-token MTP rollout speculation can be combined with R3 when vLLM >= 0.26;
# older versions may capture draft-model router decisions as target routes.
# Checkpoint revision used for hardware validation:
#   d468880b6ad3c6e0d21377ce7242adaea4cc884d
# Dataset revisions used for hardware validation:
#   DAPO-Math-17k: 65877096c24ffa7abc4e4fa5edb95cf3413a5674
#   AIME-2024:     aa49075e24ad594b79fdf0bdcefa735c2181be67
# Download the exact model and dataset snapshots before launch:
#   hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
#     --revision d468880b6ad3c6e0d21377ce7242adaea4cc884d \
#     --local-dir "$HOME/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
#   hf download BytedTsinghua-SIA/DAPO-Math-17k data/dapo-math-17k.parquet \
#     --repo-type dataset --revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674 \
#     --local-dir "$HOME/verl/data/DAPO-Math-17k"
#   hf download BytedTsinghua-SIA/AIME-2024 data/aime-2024.parquet \
#     --repo-type dataset --revision aa49075e24ad594b79fdf0bdcefa735c2181be67 \
#     --local-dir "$HOME/verl/data/AIME-2024"
# Then set MODEL_PATH to the downloaded model directory. Override DATA_DIR if
# the datasets were downloaded under a root other than "$HOME/verl".

set -xeuo pipefail

export VLLM_USE_V1=1
# Nemotron's hybrid Mamba2 attention is not supported by vLLM's
# batch-invariant mode. Keep the setting explicit so inherited environments
# cannot turn it on accidentally.
export VLLM_BATCH_INVARIANT=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### user-adjustable ###########################
PYTHON_BIN=${PYTHON_BIN:-python3}
INFER_BACKEND=${INFER_BACKEND:-vllm}
ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-R3}
MTP_ROLLOUT_SPEC=${MTP_ROLLOUT_SPEC:-1}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-1}
SEED=${SEED:-42}

DATA_DIR=${DATA_DIR:-${HOME}/verl}
# For exact reproduction, point MODEL_PATH at a local snapshot downloaded at
# d468880b6ad3c6e0d21377ce7242adaea4cc884d. Current main does not yet expose a
# model revision override through HFModelConfig.
MODEL_PATH=${MODEL_PATH:-nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/data/DAPO-Math-17k/data/dapo-math-17k.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/data/AIME-2024/data/aime-2024.parquet}

NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${GPUS_PER_NODE:-8}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}

ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}

ACTOR_TP=${ACTOR_TP:-2}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_CP=${ACTOR_CP:-1}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_ETP=${ACTOR_ETP:-1}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.70}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-4096}
ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING:-False}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}

LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-4096}

MTP_NUM_LAYERS=${MTP_NUM_LAYERS:-2}
MTP_LOSS_SCALING_FACTOR=${MTP_LOSS_SCALING_FACTOR:-0.3}
MTP_DETACH_ENCODER=${MTP_DETACH_ENCODER:-True}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-10}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
LOGGER=${LOGGER:-'["console","wandb"]'}
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_dapo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-nemotron-3-5-lightning-30b-a3b-megatron}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${HOME}/verl/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
########################### end user-adjustable ###########################

########################### validation and derived defaults ###########################
if [ "${INFER_BACKEND}" != vllm ]; then
    echo "Nemotron 3.5 Lightning GRPO currently supports only INFER_BACKEND=vllm." >&2
    exit 1
fi

case "${ROUTER_REPLAY_MODE}" in
    disabled)
        ROLLOUT_ROUTING_REPLAY_ENABLED=False
        ;;
    R3)
        ROLLOUT_ROUTING_REPLAY_ENABLED=True
        ;;
    *)
        echo "ROUTER_REPLAY_MODE must be disabled or R3, got: ${ROUTER_REPLAY_MODE}" >&2
        exit 1
        ;;
esac

case "${MTP_ROLLOUT_SPEC}" in
    0)
        MTP_ROLLOUT_ENABLED=False
        ;;
    1)
        MTP_ROLLOUT_ENABLED=True
        ;;
    *)
        echo "MTP_ROLLOUT_SPEC must be 0 or 1, got: ${MTP_ROLLOUT_SPEC}" >&2
        exit 1
        ;;
esac

if [ "${ROUTER_REPLAY_MODE}" = R3 ] && [ "${ACTOR_CP}" -ne 1 ]; then
    echo "R3 router replay for Lightning currently requires ACTOR_CP=1, got: ${ACTOR_CP}" >&2
    exit 1
fi

if [ "${ROUTER_REPLAY_MODE}" = R3 ] && [[ ! "${ROLLOUT_TEMPERATURE}" =~ ^1([.]0+)?$ ]]; then
    echo "R3 raw-logprob parity requires ROLLOUT_TEMPERATURE=1.0, got: ${ROLLOUT_TEMPERATURE}" >&2
    exit 1
fi

if [ "${MTP_ROLLOUT_SPEC}" = 1 ] && [ "${NUM_SPECULATIVE_TOKENS}" -ne 1 ]; then
    echo "Lightning's Nemotron-H MTP rollout currently supports NUM_SPECULATIVE_TOKENS=1 only." >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_DIR}"

########################### parameter arrays ###########################
ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=1024
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.seed=${SEED}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation=left
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.mtp.enable=True
    actor_rollout_ref.model.mtp.enable_train=True
    actor_rollout_ref.model.mtp.enable_rollout=${MTP_ROLLOUT_ENABLED}
    actor_rollout_ref.model.mtp.detach_encoder=${MTP_DETACH_ENCODER}
    actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR}
    actor_rollout_ref.model.mtp.method=mtp
    actor_rollout_ref.model.mtp.num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}
)

ACTOR=(
    actor_rollout_ref.actor.data_loader_seed=${SEED}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.betas=[0.9,0.95]
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=null
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.seed=${SEED}
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=fused
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.init_method_std=0.0173
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_weighted_squared_relu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.calculate_per_token_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=none
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.cuda_graph_impl=none
    +actor_rollout_ref.actor.megatron.override_transformer_config.cuda_graph_scope=[]
    +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=${MTP_NUM_LAYERS}
    '+actor_rollout_ref.actor.megatron.override_transformer_config.mtp_hybrid_override_pattern="*E"'
    +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_use_repeated_layer=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.keep_mtp_spec_in_bf16=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.seed=${SEED}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${ROLLOUT_ROUTING_REPLAY_ENABLED}
    actor_rollout_ref.rollout.enable_prefix_caching=${ROLLOUT_ENABLE_PREFIX_CACHING}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
)

TRAINER=(
    trainer.balance_batch=False
    trainer.critic_warmup=0
    "trainer.logger=${LOGGER}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.val_before_train=False
    trainer.log_val_generations=10
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
