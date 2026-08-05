#!/usr/bin/env bash
# GRPO scale demo | DeepSeek-V3 671B | vLLM rollout | Megatron training | NVIDIA GPUs
#
# Prerequisites on every node:
#   CUDA_DEVICE_MAX_CONNECTIONS=1
#   NCCL_NVLS_ENABLE=0
#   VLLM_USE_V1=1
#   pip install git+https://github.com/ISEEKYAN/mbridge
# Also: remove `quantization_config` from DeepSeek-V3 config.json and set
# `num_nextn_predict_layers=0` (MTP not yet supported).
# Minimum 12 nodes x 8x 80GB+ GPUs recommended.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export VLLM_USE_V1=1

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-deepseek-ai/DeepSeek-V3}
NNODES=${NNODES:-12}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-96}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((max_prompt_length + max_response_length))}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

actor_tp=${ACTOR_TP:-8}
actor_pp=${ACTOR_PP:-12}
actor_ep=${ACTOR_EP:-8}
actor_etp=${ACTOR_ETP:-1}
actor_cp=${ACTOR_CP:-1}
last_layer=${LAST_LAYER:-6}
offload=${OFFLOAD:-True}
optim_offload=${OPTIM_OFFLOAD:-True}
optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

rollout_tp=${ROLLOUT_TP:-32}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-8}

overlong_buffer_len=${OVERLONG_BUFFER_LEN:-4096}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_grpo_scale_demo}
experiment_name=${EXPERIMENT_NAME:-deepseek_v3_671b_vllm_megatron}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/verl/ckpts/${project_name}/${experiment_name}"}
# ---- end user-adjustable ----

train_files=$HOME/data/dapo-math-17k.parquet
val_files=$HOME/data/aime-2024.parquet
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation=left
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_fused_kernels=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${optim_offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=${last_layer}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${max_response_length}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.resume_mode=auto
    trainer.log_val_generations=10
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    model_engine=megatron
    actor_rollout_ref.nccl_timeout=1200
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
