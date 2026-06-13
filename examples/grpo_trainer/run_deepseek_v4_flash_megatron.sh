#!/usr/bin/env bash
# GRPO | DeepSeek-V4-Flash | vLLM rollout | Megatron training | NVIDIA GPUs
#
# Prerequisites on every node:
#   CUDA_DEVICE_MAX_CONNECTIONS=1
#   NCCL_NVLS_ENABLE=0
#   VLLM_USE_V1=1
#   Megatron-Bridge/mbridge installed, or set MBRIDGE_PATH=/path/to/Megatron-Bridge.
# Minimum 8 nodes x 8x 80GB+ GPUs recommended for the default smoke settings.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}


############################### configs ################################

MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}
NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

ACTOR_LR=${ACTOR_LR:-1e-6}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-8}
ACTOR_VPP=${ACTOR_VPP:-null}
ACTOR_EP=${ACTOR_EP:-8}
ACTOR_ETP=${ACTOR_ETP:-1}
ACTOR_CP=${ACTOR_CP:-1}
PIPELINE_MODEL_PARALLEL_LAYOUT=${PIPELINE_MODEL_PARALLEL_LAYOUT:-"Et*6|t*6|t*6|t*5|t*5|t*5|t*5|t*5L"}

REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_VPP=${REF_VPP:-${ACTOR_VPP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_ETP=${REF_ETP:-${ACTOR_ETP}}
REF_CP=${REF_CP:-${ACTOR_CP}}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.275}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-512}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_v4_flash_grpo_vllm_megatron}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/verl/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/aime-2024.parquet}
CHAT_TEMPLATE_FILE=${CHAT_TEMPLATE_FILE:-"${CKPTS_DIR}/deepseek_v4_flash_chat_template.jinja"}

OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-${MAX_RESPONSE_LENGTH}}
OVERLONG_BUFFER_ENABLE=${OVERLONG_BUFFER_ENABLE:-False}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

########################### parameter arrays ###########################

DEFAULT_CHAT_TEMPLATE='{% for message in messages %}{% if message["content"] is string %}{{ message["content"] }}{% else %}{% for content in message["content"] %}{% if content["type"] == "text" %}{{ content["text"] }}{% endif %}{% endfor %}{% endif %}{% if not loop.last %}{{ "\n\n" }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ "\n" }}{% endif %}'
if [ ! -f "${CHAT_TEMPLATE_FILE}" ]; then
    mkdir -p "$(dirname "${CHAT_TEMPLATE_FILE}")"
    printf "%s\n" "${DEEPSEEK_V4_FLASH_CHAT_TEMPLATE:-${DEFAULT_CHAT_TEMPLATE}}" >"${CHAT_TEMPLATE_FILE}"
fi

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation=left
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.custom_chat_template="@${CHAT_TEMPLATE_FILE}"
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    "++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${ROLLOUT_KV_CACHE_DTYPE}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_fused_mhc=False
    ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_rope_fusion=False
    ++actor_rollout_ref.ref.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    "++actor_rollout_ref.ref.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'"
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.resume_mode=disable
    trainer.val_before_train=False
    trainer.log_val_generations=0
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    actor_rollout_ref.nccl_timeout=3600
    model_engine=megatron
)

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
