#!/usr/bin/env bash
# GSPO | MoE | vLLM rollout | Megatron training | NVIDIA GPUs
# GSPO with Qwen3-30B-A3B (MoE).

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-30720}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

clip_ratio_low=${CLIP_RATIO_LOW:-3e-4}
clip_ratio_high=${CLIP_RATIO_HIGH:-4e-4}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-1}
actor_ep=${ACTOR_EP:-8}
actor_etp=${ACTOR_ETP:-1}

rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}
rollout_n=${ROLLOUT_N:-16}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_gspo_qwen3_moe}
experiment_name=${EXPERIMENT_NAME:-qwen3_30b_a3b_vllm_megatron}

train_file=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
val_file=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}
# ---- end user-adjustable ----
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    model_engine=megatron
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
