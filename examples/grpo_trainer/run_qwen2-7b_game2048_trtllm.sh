#!/usr/bin/env bash
# Train Qwen2-7B-Instruct to generate 2048 game strategies via GRPO + trtllm.
#
# Closely follows run_qwen2-7b_math_trtllm.sh — only the data paths,
# prompt/response lengths, and rollout.n are changed for the 2048 task.
#
# Why 7B instead of gpt-oss-20b?
#   7B bf16 ≈ 14 GB vs 20B ≈ 40 GB, leaving much more room for trtllm KV cache
#   and FSDP optimizer states on each H100 (140 GB).
#
# Before running, generate the dataset:
#   python examples/data_preprocess/game2048.py --local_save_dir ~/data/game2048
#
# Usage:
#   bash examples/grpo_trainer/run_qwen2-7b_game2048_trtllm.sh         # TP=4
#   bash examples/grpo_trainer/run_qwen2-7b_game2048_trtllm.sh 2       # TP=2

set -x

# -----------------------------------------------------------------------
# Logging — tee output to 2048_log_<date>_<time>.txt next to this script
# -----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/2048_log_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# Clean SLURM / MPI env vars to avoid PMIx mismatch errors
for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

export RAY_DEDUP_LOGS=0

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
TP=${1:-4}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-7B-Instruct"}
DATADIR=${DATADIR:-"$HOME/data/game2048"}

TRAIN_PATH="$DATADIR/train.parquet"
TEST_PATH="$DATADIR/test.parquet"

PROJECT_NAME=${PROJECT_NAME:-"verl_grpo_game2048"}
EXP_NAME="trtllm-qwen2-7b-game2048-tp${TP}-8gpus${EXP_NAME_SUFFIX:+"-"}${EXP_NAME_SUFFIX}"

if [ $TP -eq 4 ]; then
    MAX_BATCH_SIZE=1024
else
    MAX_BATCH_SIZE=384
fi

# -----------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    \
    data.train_files="['$TRAIN_PATH']" \
    data.val_files="['$TEST_PATH']" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=trtllm \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_BATCH_SIZE} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_timeout_iters=32 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_max_tokens_ratio=0.5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.resume_mode=disable \
    trainer.total_epochs=15 \
    "${@:2}"
