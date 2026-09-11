#!/usr/bin/env bash
# GRPO + LoRA | DeepSeek-V4-Flash | vLLM rollout | Megatron training | NVIDIA GPUs
#
# LoRA (rank-16, merge=False) on DeepSeek-V4-Flash with DAPO-Math-17k.
# FP8 training (hybrid blockwise) is opt-in via the USE_FP8 toggle below.
#
# Megatron-Bridge must be installed and importable on every node.
# With:
# - Megatron-Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
# - Megatron-LM:     https://github.com/NVIDIA/Megatron-LM/commit/fd1121b8ff7e3a4f83a28d35aed172d7bc0260e1
#
# Knobs:
#   USE_FP8        1 to enable FP8 hybrid blockwise training+rollout   (default: unset=BF16)
#   LORA_MERGE     True to merge LoRA into base weights (faster rollout,
#                  no online refit); False for online-refit merge=False  (default: False)

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

############################### configs ################################

MODEL_PATH=${MODEL_PATH:-$HDFS_ROOT/model/DeepSeek-V4-Flash}
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

USE_FP8=${USE_FP8:-0}
LORA_MERGE=${LORA_MERGE:-False}
LORA_DTYPE=${LORA_DTYPE:-float32}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

ACTOR_LR=${ACTOR_LR:-5e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-3}
CLIP_GRAD=${CLIP_GRAD:-1.0}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1}

# DeepSeek-V4-Flash has o_groups=8, so rollout TP must not exceed 8
# (n_local_groups = o_groups // tp_size would be 0 and crash fp8 BMM post-process).
ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-2}
ACTOR_VPP=${ACTOR_VPP:-null}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_ETP=${ACTOR_ETP:-1}
ACTOR_CP=${ACTOR_CP:-4}

REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_ETP=${REF_ETP:-${ACTOR_ETP}}
REF_CP=${REF_CP:-${ACTOR_CP}}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_EP=${ROLLOUT_EP:-8}
ROLLOUT_PP=${ROLLOUT_PP:-1}
ROLLOUT_DCP=${ROLLOUT_DCP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-4096}
ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-R3}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

PROJECT_NAME=${PROJECT_NAME:-DAPO}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-dsv4_flash_lora_megatron}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/verl/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-$DATA_ROOT/dataset/aime25_test.parquet}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-$((1024 * 2))}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-10}

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.gamma=1.0
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.rollout_is=token
    algorithm.rollout_correction.rollout_is_threshold="0.5_5.0"
    algorithm.rollout_correction.rollout_is_batch_normalize=true
    algorithm.rollout_correction.bypass_mode=false
    algorithm.rollout_correction.loss_type=ppo_clip
)

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.prompt_key=prompt
    data.trust_remote_code=True
    data.filter_overlong_prompts=False
    data.truncation=left
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    +data.apply_chat_template_kwargs.enable_thinking=False
    +data.continuous_token.enable=True
    +data.continuous_token.model_family=deepseekv4
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.lora.rank=${LORA_RANK}
    actor_rollout_ref.model.lora.alpha=${LORA_ALPHA}
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
    actor_rollout_ref.model.lora.merge=${LORA_MERGE}
    actor_rollout_ref.model.lora.dtype=${LORA_DTYPE}
    actor_rollout_ref.model.lora.experts_shared_outer_loras=True
    actor_rollout_ref.model.lora.lora_plus_ratio=16.0
    "actor_rollout_ref.model.lora.target_modules=['linear_wkv','linear_wgate','linear_wq_b','linear_weights_proj','linear_kv_proj','linear_q_down_proj','linear_q_up_proj','linear_proj','linear_fc1','linear_fc2','router']"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}
    actor_rollout_ref.actor.optim.weight_decay=0
    actor_rollout_ref.actor.optim.clip_grad=${CLIP_GRAD}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.mtp_num_layers=0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

# Context parallelism needs three extra transformer-config settings for DeepSeek-V4 on top of
# `context_parallel_size`. Each of them is enforced by Megatron-Core, so without them the run
# aborts at model build or in the first attention forward. They are appended only when CP > 1.
CP_ARGS=()
if [ "${ACTOR_CP}" -gt 1 ]; then
    CP_ARGS=(
        ++actor_rollout_ref.actor.megatron.override_transformer_config.cp_partition_mode=contiguous
        ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_packing_scheduler=dp_balanced
        ++actor_rollout_ref.actor.megatron.override_transformer_config.max_seqlen_per_dp_cp_rank=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / ACTOR_CP))
    )
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${ROLLOUT_PP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.top_p=1
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size=${ROLLOUT_DCP}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${ROLLOUT_KV_CACHE_DTYPE}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP}
    actor_rollout_ref.ref.megatron.use_remove_padding=True
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=True
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.test_freq=${TEST_FREQ}
    trainer.save_freq=${SAVE_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.resume_mode=disable
    trainer.log_val_generations=8
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    actor_rollout_ref.nccl_timeout=3600
    model_engine=megatron
)

# FP8 hybrid blockwise training + FP8 rollout. When USE_FP8=1 the actor forward
# uses E4M3 / backward E5M2 blockwise quantization, and the vLLM rollout runs in
# FP8 with FP8 KV cache. The TE backward amax_epsilon=1e-12 fix (in Megatron-LM
# fp8_utils.py) is required to avoid NaN grad_norm from all-zero gradient blocks.
if [ "${USE_FP8}" = 1 ]; then
    ACTOR+=(
        +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid
        +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=blockwise
        +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
        +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
        +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe=blockwise
    )
    ROLLOUT+=(+actor_rollout_ref.rollout.quantization=fp8)
    ROLLOUT+=(actor_rollout_ref.rollout.enforce_eager=True)
else
    ROLLOUT+=(actor_rollout_ref.rollout.enforce_eager=False)
fi

########################### launch ###########################

# uv (set VERL_USE_UV=0 for system python): the GPU vllm × megatron driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the matching extras of the committed uv.lock.
# Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra megatron")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${CP_ARGS[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
