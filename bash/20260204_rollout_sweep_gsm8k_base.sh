#!/usr/bin/env bash
set -euo pipefail

# conda activate can touch unbound vars; disable -u temporarily
set +u
. /local/home/tommaben/miniconda3/etc/profile.d/conda.sh
conda activate verl_c
set -u

set -a
source /local/home/tommaben/repo/custom/verl/.env
set +a

cd /local/home/tommaben/repo/custom/verl

# Rollout sweep for baseline GSM8K (Qwen2.5-3B base)


gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
gsm8k_train_path_mc=$HOME/data/gsm8k_mc/train.parquet
gsm8k_test_path_mc=$HOME/data/gsm8k_mc/test.parquet

GPUS=${GPUS:-8}
PREFIX="rollout_sweep_qwen25_3B_gsm8k_base"
N_LIST="2 4 8 16"

for N in $N_LIST; do
  EXP_NAME="qwen25_3B_gsm8k_base_n${N}"

  VLLM_USE_FLASHINFER_SAMPLER=1 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$gsm8k_train_path \
    data.val_files="[$gsm8k_test_path_mc,$gsm8k_test_path]" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=disable \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.logger='["console","wandb"]' \
    +trainer.best_ckpt_metric='val-aux/openai/gsm8k/score/mean@1' \
    +trainer.best_ckpt_mode=max \
    +trainer.best_ckpt_keep_only=true \
    trainer.early_stop_metric='val-aux/openai/gsm8k/score/mean@1' \
    trainer.early_stop_patience=10 \
    trainer.early_stop_mode=max \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    trainer.total_epochs=15 \
    trainer.project_name='multiple_choice_question_study' \
    trainer.experiment_name="${EXP_NAME}" \
    custom_reward_function.path=verl/utils/reward_score/gsm8k_mixed.py \
    custom_reward_function.name=compute_score \
    2>&1 | tee "verl_demo_${EXP_NAME}.log"

  python3 scripts/push_to_wandb.py \
    --project multiple_choice_question_study \
    --run-name "upload_${EXP_NAME}" \
    --artifact-name "${EXP_NAME}" \
    --aliases best latest \
    --paths "checkpoints/multiple_choice_question_study/${EXP_NAME}/best/actor/huggingface"

done
