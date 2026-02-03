#!/usr/bin/env bash
set -xeuo pipefail


ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.main_ppo"}
TRAIN_FILES=${TRAIN_FILES:-/mnt/data/liuchonghan/vmlu_dataset/all_data_merged_rlhf.json}
VAL_FILES=${VAL_FILES:-}
MODEL_ID=${MODEL_ID:-/mnt/data/liuchonghan/75_0129_ckpt3000}
PROJECT_NAME=${PROJECT_NAME:-rlvr}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rlvr_72b_grpo}

NNODES=${PET_NNODES:-${WORLD_SIZE:-30}}
NODE_RANK=${PET_NODE_RANK:-${RANK:-0}}
MASTER_ADDR=${PET_MASTER_ADDR:-${MASTER_ADDR:-"127.0.0.1"}}
MASTER_PORT=${PET_MASTER_PORT:-${MASTER_PORT:-23457}}
N_GPUS_PER_NODE=${PET_NPROC_PER_NODE:-${NPROC_PER_NODE:-${N_GPUS_PER_NODE:-8}}}

RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_ADDRESS=${RAY_ADDRESS:-$MASTER_ADDR:$RAY_PORT}

echo ">>> 节点信息: RANK $NODE_RANK / WORLD_SIZE $NNODES"
echo ">>> 通信信息: MASTER $MASTER_ADDR : $MASTER_PORT"
echo ">>> Ray 地址: $RAY_ADDRESS"

export WANDB_MODE=offline
export NCCL_DEBUG=WARN

if [ "$NODE_RANK" -eq 0 ]; then
  ray start --head \
    --node-ip-address="$MASTER_ADDR" \
    --port="$RAY_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT"
else
  ray start --address="$RAY_ADDRESS" --block &
fi

# Give Ray a moment to settle
sleep 5

python3 $ENTRYPOINT \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.train_batch_size=2048 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_ID \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2048 + 1024)) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0.05 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=5 \
    trainer.use_legacy_worker_impl=disable \
    ray_kwargs.ray_init.address=$RAY_ADDRESS \
    custom_reward_function.path=./reward_function.py \
    custom_reward_function.name=char_count_reward_function
