#!/usr/bin/env bash
# On-policy distillation | Qwen3-VL | vLLM Ascend | Megatron | Ascend NPU
# Generic model-pair base; use run_qwen3_vl_2b_megatron.sh for the
# official-aligned four-NPU 2B <- 4B recipe.

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-4B-Instruct}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-1}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/geo3k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/geo3k/test.parquet}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-12}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}

OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-true}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-true}

ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_TP=${ACTOR_TP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEACHER_TP=${TEACHER_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.65}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:--1}

PROJECT_NAME=${PROJECT_NAME:-verl_opd_ascend}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_from_8b_megatron_vllm_ascend}
# ---- end user-adjustable ----

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.image_key=images
    data.train_batch_size="$TRAIN_BATCH_SIZE"
    data.max_prompt_length="$MAX_PROMPT_LENGTH"
    data.max_response_length="$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR"
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$ACTOR_TP"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.use_kl_loss=False
)

if [[ "$OPTIMIZER_CPU_OFFLOAD" == "true" ]]; then
    ACTOR+=(
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="$OPTIMIZER_OFFLOAD_FRACTION"
        +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer="$USE_PRECISION_AWARE_OPTIMIZER"
    )
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION"
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len="$max_model_len"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.n_gpus_per_node="$NGPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_epochs="$TOTAL_EPOCHS"
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS"
)

DISTILLATION=(
    model_engine=megatron
    distillation.enabled=True
    distillation.n_gpus_per_node="$TEACHER_NGPUS_PER_NODE"
    distillation.nnodes="$NNODES"
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$TEACHER_TP"
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$TEACHER_GPU_MEMORY_UTILIZATION"
    distillation.teacher_models.teacher_model.inference.max_model_len="$max_model_len"
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${DISTILLATION[@]}" \
    "$@"
