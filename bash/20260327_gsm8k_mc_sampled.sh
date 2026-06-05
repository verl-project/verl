#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=${GPU_IDS:-}
GPUS=${GPUS:-}
MODEL_PATH=${MODEL_PATH:-Qwen/qwen2.5-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/local/home/tommaben/data/gsm8k_mc_sampled/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}
PROJECT_NAME=${PROJECT_NAME:-multiple_choice_question_study}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_3B_mc_sampled_gsm8k_n8}
LOG_FILE=${LOG_FILE:-verl_mc_sampled.log}

if [[ -n "$GPU_IDS" ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
    GPUS=${#GPU_ID_ARRAY[@]}
fi

GPUS=${GPUS:-8}

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "Using trainer.n_gpus_per_node=$GPUS"

VLLM_USE_FLASHINFER_SAMPLER=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=0 \
    trainer.test_freq=1 \
    trainer.logger='["console","wandb"]' \
    +trainer.best_ckpt_metric='val-aux/openai/gsm8k/score/mean@1' \
    +trainer.best_ckpt_mode=max \
    +trainer.best_ckpt_keep_only=true \
    trainer.early_stop_metric='val-aux/openai/gsm8k/score/mean@1' \
    trainer.early_stop_patience=10 \
    trainer.early_stop_mode=max \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.total_epochs=5 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_mc.py \
    custom_reward_function.name=compute_score \
    val_custom_reward_function.path=verl/utils/reward_score/gsm8k.py \
    val_custom_reward_function.name=compute_score \
    2>&1 | tee $LOG_FILE
