#!/usr/bin/env bash
# Official Qwen3-VL Geo3K OPD pair on four Ascend NPUs: 2B <- 4B.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-2}
export ACTOR_TP=${ACTOR_TP:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export TEACHER_TP=${TEACHER_TP:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-3072}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-5}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.20}
export ROLLOUT_SEED=${ROLLOUT_SEED:-42}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_2b_from_4b_megatron_vllm_ascend}

exec "$SCRIPT_DIR/run_qwen3_vl_megatron.sh" \
    data.shuffle=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.seed="$ROLLOUT_SEED" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.max_num_seqs="$TRAIN_BATCH_SIZE" \
    actor_rollout_ref.rollout.max_num_batched_tokens=3073 \
    distillation.teacher_models.teacher_model.inference.enforce_eager=False \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=1 \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=3073 \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    "$@"
