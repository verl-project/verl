#!/usr/bin/env bash
set -xeuo pipefail

# Test script for fully_async_policy E2E regression testing
# This script runs fully async PPO training with both FSDP2 and Megatron backends
# to ensure the asynchronous training mechanism works correctly

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-"fsdp2"}  # fsdp2 or megatron

# Download model if not exists
MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
# huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"


rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# Response length parameters
max_prompt_length=1024
max_response_length=2048
enable_overlong_buffer=True
overlong_buffer_len=128
overlong_penalty_factor=1.0

# Training parameters
loss_agg_mode="token-mean"

# Temperature parameters
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Fully async specific parameters
n_gpus_rollout=4
n_gpus_training=4

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=16
train_prompt_mini_bsz=16
total_rollout_steps=$(((128)))
test_freq=-1
staleness_threshold=0.1
trigger_parameter_sync_step=4
partial_rollout=True

exp_name="$(basename "${MODEL_ID,,}")-fully-async-policy-${ACTOR_STRATEGY}-minimal"

echo "Running fully_async_policy with ${ACTOR_STRATEGY} strategy"
echo "Total GPUs: ${NUM_GPUS}, Rollout GPUs: ${n_gpus_rollout}, Training GPUs: ${n_gpus_training}"

# Common parameters for both FSDP2 and Megatron
common_params=(
    data.train_files="${HOME}/data/gsm8k/train.parquet"
    data.val_files="${HOME}/data/gsm8k/test.parquet"
    data.prompt_key=prompt
    data.truncation='left'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
    data.gen_batch_size=${gen_prompt_bsz}
    data.return_raw_chat=${return_raw_chat}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.calculate_log_probs=True
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.disable_log_stats=False
    reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
    trainer.logger=['console']
    trainer.project_name='verl-test-fully-async'
    trainer.experiment_name="${exp_name}"
    trainer.val_before_train=True
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${n_gpus_training}
    rollout.nnodes=1
    rollout.n_gpus_per_node=${n_gpus_rollout}
    rollout.total_rollout_steps=${total_rollout_steps}
    rollout.total_epochs=2
    rollout.test_freq=${test_freq}
    # Fully async specific configurations
    async_training.staleness_threshold=${staleness_threshold}
    async_training.partial_rollout="${partial_rollout}"
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}"
)

if [ "${ACTOR_STRATEGY}" == "fsdp2" ]; then
    echo "Running fully async training with FSDP2 strategy..."
    # FSDP2 specific parameters
    gen_tp=1
    sp_size=1
    fsdp_size=1
    ref_offload=True
    actor_offload=False

    python3 -m verl.experimental.fully_async_policy.fully_async_main \
        "${common_params[@]}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.strategy=fsdp2 \
        critic.strategy=fsdp2 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} $@

elif [ "${ACTOR_STRATEGY}" == "megatron" ]; then
    echo "Running fully async training with Megatron strategy..."
    # Megatron specific parameters
    gen_tp=2
    train_tp=1
    train_pp=2
    ref_offload=True
    actor_offload=False

    python3 -m verl.experimental.fully_async_policy.fully_async_main \
        --config-path=config \
        --config-name='fully_async_ppo_megatron_trainer.yaml' \
        "${common_params[@]}" \
        actor_rollout_ref.actor.strategy=megatron \
        critic.strategy=megatron \
        actor_rollout_ref.actor.optim.lr_decay_steps=10000000 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.megatron.param_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.optimizer_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.grad_offload=${actor_offload} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
        actor_rollout_ref.ref.megatron.param_offload=${ref_offload} $@
else
    echo "Error: Unknown strategy ${ACTOR_STRATEGY}. Please use 'fsdp2' or 'megatron'"
    exit 1
fi

echo "Fully async policy E2E test completed successfully with ${ACTOR_STRATEGY} strategy"

