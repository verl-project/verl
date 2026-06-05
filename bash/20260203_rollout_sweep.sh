#!/usr/bin/env bash
set -euo pipefail

# Rollout sweep for opt1x

gsm8k_train_path=/Users/tbe/data/gsm8k/train.parquet
gsm8k_test_path=/Users/tbe/data/gsm8k/test.parquet
gsm8k_train_path_mc=/Users/tbe/data/gsm8k_mc/train.parquet
gsm8k_test_path_mc=/Users/tbe/data/gsm8k_mc/test.parquet
gsm8k_train_path_mc_opt1x=/local/home/tommaben/data/opt1x_gsm_mc_stage/train.parquet

GPUS=8
PREFIX="rollout_sweep_qwen25_3B_mc_opt1x_gsm8k"
N_LIST="1 2 4 8 16"

for N in ; do
  EXP_NAME="_n"

  VLLM_USE_FLASHINFER_SAMPLER=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files= \
    data.val_files="[,]" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n= \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node= \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=0 \
    trainer.test_freq=1 \
    trainer.logger='["console","wandb"]' \
    +trainer.best_ckpt_metric='val-aux/openai/gsm8k/score/mean@1' \
    +trainer.best_ckpt_mode=max \
    +trainer.best_ckpt_keep_only=true \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.total_epochs=5 \
    trainer.project_name='multiple_choice_question_study' \
    trainer.experiment_name= \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_mixed.py \
    custom_reward_function.name=compute_score \
    2>&1 | tee verl_demo_.log

  python3 scripts/push_to_wandb.py \
    --project multiple_choice_question_study \
    --run-name upload_ \
    --artifact-name  \
    --aliases best latest \
    --paths checkpoints/multiple_choice_question_study//best/actor/huggingface

done
