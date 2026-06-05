#!/usr/bin/env bash
set -euo pipefail
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
gsm8k_train_path_mc=$HOME/data/gsm8k_mc/train.parquet
gsm8k_test_path_mc=$HOME/data/gsm8k_mc/test.parquet
gsm8k_train_path_mc_aug2x=/local/home/tommaben/data/aug2x_gsm_mc_stage_aug/train.parquet
gsm8k_train_path_mc_opt1x=/local/home/tommaben/data/opt1x_gsm_mc_stage/train.parquet
gsm8k_train_path_mc_nocot=/local/home/tommaben/data/nocot_gsm_mc_stage/train.parquet
gsm8k_test_path_mc_nocot=/local/home/tommaben/data/nocot_gsm_mc_stage/test.parquet
gsm8k_train_path_mc_opt10x=/local/home/tommaben/data/opt10x_gsm_mc_stage/train.parquet
gsm8k_train_path_mc_open0p1x=/local/home/tommaben/data/open0p1x_gsm_mc_stage/train.parquet
gsm8k_test_path_mc_open0p1x=/local/home/tommaben/data/open0p1x_gsm_mc_stage/test.parquet
gsm8k_train_path_frac0p1x=/local/home/tommaben/data/frac0p1x_gsm8k/train.parquet
gsm8k_test_path_frac0p1x=/local/home/tommaben/data/frac0p1x_gsm8k/test.parquet
GPUS=${GPUS:-8}

VLLM_USE_FLASHINFER_SAMPLER=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$gsm8k_train_path_mc_opt10x \
    data.val_files="[$gsm8k_test_path_mc,$gsm8k_test_path,$gsm8k_test_path_mc_nocot]" \
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
    trainer.project_name='multiple_choice_question_study' \
    trainer.experiment_name='qwen25_3B_mc_opt10x_gsm8k_n8' \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_mixed.py \
    custom_reward_function.name=compute_score \
    2>&1 | tee verl_demo.log

python3 scripts/push_to_wandb.py \
    --project multiple_choice_question_study \
    --run-name upload_qwen25_3B_mc_opt10x_gsm8k_n8 \
    --artifact-name qwen25_3B_mc_opt10x_gsm8k_n8 \
    --aliases best latest \
    --paths checkpoints/multiple_choice_question_study/qwen25_3B_mc_opt10x_gsm8k_n8/best/actor/huggingface

VLLM_USE_FLASHINFER_SAMPLER=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$gsm8k_train_path_mc_open0p1x \
    data.val_files="[$gsm8k_test_path_mc,$gsm8k_test_path]" \
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
    trainer.project_name='multiple_choice_question_study' \
    trainer.experiment_name='qwen25_3B_mc_open0p1x_gsm8k_n8' \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_mixed.py \
    custom_reward_function.name=compute_score \
    2>&1 | tee verl_demo.log

python3 scripts/push_to_wandb.py \
    --project multiple_choice_question_study \
    --run-name upload_qwen25_3B_mc_open0p1x_gsm8k_n8 \
    --artifact-name qwen25_3B_mc_open0p1x_gsm8k_n8 \
    --aliases best latest \
    --paths checkpoints/multiple_choice_question_study/qwen25_3B_mc_open0p1x_gsm8k_n8/best/actor/huggingface

VLLM_USE_FLASHINFER_SAMPLER=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$gsm8k_train_path_frac0p1x \
    data.val_files="[$gsm8k_test_path,$gsm8k_test_path_mc]" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
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
    trainer.total_epochs=50 \
    trainer.project_name='multiple_choice_question_study' \
    trainer.experiment_name='qwen25_3B_frac0p1x_gsm8k_n8' \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_mixed.py \
    custom_reward_function.name=compute_score \
    2>&1 | tee verl_demo.log

python3 scripts/push_to_wandb.py \
    --project multiple_choice_question_study \
    --run-name upload_qwen25_3B_frac0p1x_gsm8k_n8 \
    --artifact-name qwen25_3B_frac0p1x_gsm8k_n8 \
    --aliases best latest \
    --paths checkpoints/multiple_choice_question_study/qwen25_3B_frac0p1x_gsm8k_n8/best/actor/huggingface
