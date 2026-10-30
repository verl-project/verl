#!/usr/bin/env bash
# Official Qwen3-VL Geo3K OPD pair | Megatron training | NVIDIA GPUs or Ascend NPUs

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

export DEVICE
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-5}
export ROLLOUT_SEED=${ROLLOUT_SEED:-42}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_2b_from_4b_${DEVICE}_megatron_vllm}

HARDWARE_CONFIG=()
max_num_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
case "${DEVICE}" in
    gpu)
        export NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
        export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}
        export ACTOR_TP=${ACTOR_TP:-2}
        export ROLLOUT_TP=${ROLLOUT_TP:-2}
        export TEACHER_TP=${TEACHER_TP:-2}
        export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
        export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
        export TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.4}
        HARDWARE_CONFIG+=(
            actor_rollout_ref.actor.use_dynamic_bsz=True
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
            actor_rollout_ref.rollout.calculate_log_probs=True
        )
        ;;
    npu)
        export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
        export TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}
        export ACTOR_TP=${ACTOR_TP:-1}
        export ROLLOUT_TP=${ROLLOUT_TP:-1}
        export TEACHER_TP=${TEACHER_TP:-1}
        export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-3072}
        export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.20}
        export TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.20}
        HARDWARE_CONFIG+=(
            actor_rollout_ref.actor.use_torch_compile=True
            actor_rollout_ref.actor.use_dynamic_bsz=False
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.calculate_log_probs=False
            actor_rollout_ref.rollout.max_num_seqs="$TRAIN_BATCH_SIZE"
            actor_rollout_ref.rollout.max_num_batched_tokens="$max_num_tokens"
            distillation.teacher_models.teacher_model.inference.max_num_seqs=1
            distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="$max_num_tokens"
        )
        ;;
    *)
        echo "DEVICE must be gpu or npu, got: ${DEVICE}" >&2
        exit 1
        ;;
esac

exec "$SCRIPT_DIR/run_qwen3_vl_megatron.sh" \
    data.shuffle=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.seed="$ROLLOUT_SEED" \
    actor_rollout_ref.rollout.enforce_eager=False \
    distillation.teacher_models.teacher_model.inference.enforce_eager=False \
    distillation.distillation_loss.topk=64 \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=True \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.val_before_train=True \
    trainer.log_val_generations=5 \
    "${HARDWARE_CONFIG[@]}" \
    "$@"
