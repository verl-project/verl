#!/usr/bin/env bash
# GRPO | text | Megatron training | NVIDIA GPUs
# Router-replay example on Qwen3-30B-A3B (MoE). See README for R2 vs R3 modes.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
NNODES=${NNODES:-}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

ROUTING_REPLAY_MODE=${ROUTING_REPLAY_MODE:-R3}        # R2 | R3
ENABLE_ROLLOUT_ROUTING_REPLAY=${ENABLE_ROLLOUT_ROUTING_REPLAY:-}

TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$HOME/data/gsm8k/train.parquet}
TEST_DATA_PATH=${TEST_DATA_PATH:-$HOME/data/gsm8k/test.parquet}

train_batch_size=${TRAIN_BATCH_SIZE:-}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-}
micro_bs=${MICRO_BS:-3}
max_prompt_length=${MAX_PROMPT_LENGTH:-}
max_response_length=${MAX_RESPONSE_LENGTH:-}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-}

actor_lr=${ACTOR_LR:-1e-6}

actor_pp=${ACTOR_PP:-}
actor_tp=${ACTOR_TP:-}
actor_ep=${ACTOR_EP:-8}
actor_etp=${ACTOR_ETP:-1}
rollout_tp=${ROLLOUT_TP:-}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.65}
rollout_n=${ROLLOUT_N:-8}

offload=${OFFLOAD:-True}

total_training_steps=${TOTAL_TRAINING_STEPS:-50000}
project_name=${PROJECT_NAME:-verl_grpo_router_replay}
experiment_name=${EXPERIMENT_NAME:-}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
NNODES=${NNODES:-6}
train_batch_size=${train_batch_size:-3}
ppo_mini_batch_size=${ppo_mini_batch_size:-3}
max_prompt_length=${max_prompt_length:-512}
max_response_length=${max_response_length:-512}
actor_pp=${actor_pp:-6}
actor_tp=${actor_tp:-1}
rollout_tp=${rollout_tp:-4}
actor_use_dynamic_bsz=False
rollout_log_prob_use_dynamic_bsz=False
ref_log_prob_use_dynamic_bsz=False
moe_permute_fusion=False
trainer_balance_batch=False

if [ -z "$ENABLE_ROLLOUT_ROUTING_REPLAY" ]; then
    if [ "$ROUTING_REPLAY_MODE" = "R3" ]; then
        ENABLE_ROLLOUT_ROUTING_REPLAY=True
    else
        ENABLE_ROLLOUT_ROUTING_REPLAY=False
    fi
fi

ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu:-$(((max_prompt_length + max_response_length) * 2))}
experiment_name=${experiment_name:-qwen3_30b_a3b_router_replay_${ROUTING_REPLAY_MODE}_${INFER_BACKEND}_megatron}

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_DATA_PATH"
    data.val_files="$TEST_DATA_PATH"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.model_engine=megatron
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bs}
    actor_rollout_ref.actor.use_dynamic_bsz=${actor_use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTING_REPLAY_MODE}
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=${moe_permute_fusion}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bs}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${rollout_log_prob_use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bs}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${ref_log_prob_use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.balance_batch=${trainer_balance_batch}
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=-1
    trainer.test_freq=10
    trainer.total_training_steps=${total_training_steps}
    trainer.val_before_train=False
)

# Conservative rollout extras shared by all inference backends.
EXTRA=(
    actor_rollout_ref.rollout.skip_tokenizer_init=True
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
