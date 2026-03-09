#!/usr/bin/env bash
# Train gpt-oss-20b to generate 2048 game strategies via GRPO.
#
# Config mirrors the Unsloth reinforcement-fine-tuning notebook as closely as
# possible, using trtllm as the rollout engine (from run_qwen2-7b_math_trtllm.sh):
#   - Model:            gpt-oss-20b (bf16, converted from mxfp4)
#   - Rollout engine:   trtllm (async)
#   - reasoning_effort: low  (matches notebook's reasoning_effort="low")
#   - rollout.n:        2    (matches notebook's num_generations=2)
#   - LoRA rank:        4    (matches notebook's lora_rank=4)
#   - lr:               5e-5 (matches notebook's learning_rate=5e-5)
#   - temperature:      1.0  (matches notebook's temperature=1.0)
#   - max_seq_length:   768  → prompt 256 + response 512
#   - total_epochs:     16   ≈ 1000 steps with batch_size=64
#
# Before running, convert the model once:
#   python examples/grpo_trainer/convert_gptoss_to_bf16.py
#
# Then generate the dataset:
#   python examples/data_preprocess/game2048.py --local_save_dir ~/data/game2048
#
# Usage:
#   bash examples/grpo_trainer/run_gptoss_game2048.sh

set -x

# -----------------------------------------------------------------------
# Logging — tee all output to 2048_log_<date>_<time>.txt next to this script
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
# Config  (override via env vars)
# -----------------------------------------------------------------------
TP=${1:-4}
MODEL_PATH=${MODEL_PATH:-"$HOME/models/gpt-oss-20b-bf16"}
DATADIR=${DATADIR:-"$HOME/data/game2048"}

TRAIN_PATH="$DATADIR/train.parquet"
TEST_PATH="$DATADIR/test.parquet"

PROJECT_NAME=${PROJECT_NAME:-"verl_grpo_game2048"}
EXP_NAME=${EXP_NAME:-"gptoss-20b-game2048-trtllm-tp${TP}"}

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
    data.train_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.reasoning_effort=low \
    \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=4 \
    actor_rollout_ref.model.lora_alpha=8 \
    \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
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
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_BATCH_SIZE} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_timeout_iters=32 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_max_tokens_ratio=0.5 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
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
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.total_epochs=16 \
    "${@:2}"
