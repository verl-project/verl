#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VERL_USE_GPT_OSS=0
export PYTHONPATH=/llm-align/liuchonghan/verl_lao:${PYTHONPATH:-}

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.main_ppo"}
TRAIN_FILES=${TRAIN_FILES:-/llm-align/liuchonghan/all_data_merged_rlhf.json}
MODEL_ID=${MODEL_ID:-/llm-align/liuchonghan/Qwen3-8B}
PROJECT_NAME=${PROJECT_NAME:-rlvr_8b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rlvr_8b_grpo_megatron_single}

# Single node, 8 GPUs
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=${MASTER_PORT:-23457}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

TP_SIZE=1
PP_SIZE=1

rollout_mode=${ROLLOUT_MODE:-async}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}
RETURN_RAW_CHAT=${RETURN_RAW_CHAT:-True}

python3 $ENTRYPOINT --config-path=/llm-align/liuchonghan/verl_lao/verl/trainer/config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$TRAIN_FILES \
    data.val_max_samples=2048 \
    data.return_raw_chat=$RETURN_RAW_CHAT \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_ID \
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP_SIZE \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP_SIZE \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=300 \
    trainer.test_freq=300 \
    trainer.total_epochs=5 \
    +ray_kwargs.ray_init.num_cpus=32 \
    custom_reward_function.path=/llm-align/liuchonghan/verl_lao/recipes_custom/RLVR_ABCDE_dense/reward_function.py \
    custom_reward_function.name=char_count_reward_function
