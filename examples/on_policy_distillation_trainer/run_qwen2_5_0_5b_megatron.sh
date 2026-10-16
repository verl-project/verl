#!/usr/bin/env bash
# On-policy distillation | Qwen2.5 | vLLM rollout | Megatron training | NVIDIA GPUs or Ascend NPUs

set -xeuo pipefail

########################### user-adjustable ###########################
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen2.5-0.5B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-3B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/gsm8k/test.parquet}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-}

train_batch_size=${TRAIN_BATCH_SIZE:-12}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-12}
max_prompt_length=${MAX_PROMPT_LENGTH:-256}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
actor_tp=${ACTOR_TP:-}
rollout_tp=${ROLLOUT_TP:-}
teacher_tp=${TEACHER_TP:-}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-}

total_epochs=${TOTAL_EPOCHS:-15}
total_training_steps=${TOTAL_TRAINING_STEPS:-100}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_opd_qwen2_5}
experiment_name=${EXPERIMENT_NAME:-qwen2_5_0_5b_from_3b_${DEVICE}_megatron_vllm}
########################### end user-adjustable ###########################

########################### device configuration ###########################
DEVICE_CONFIG=()
case "${DEVICE}" in
    gpu)
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
        TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
        actor_tp=${actor_tp:-1}
        rollout_tp=${rollout_tp:-1}
        teacher_tp=${teacher_tp:-1}
        rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.6}
        teacher_gpu_mem_util=${teacher_gpu_mem_util:-0.8}
        DEVICE_CONFIG+=(
            actor_rollout_ref.actor.megatron.param_offload=True
            actor_rollout_ref.actor.megatron.optimizer_offload=True
        )
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
        export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
        export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
        TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
        actor_tp=${actor_tp:-1}
        rollout_tp=${rollout_tp:-1}
        teacher_tp=${teacher_tp:-1}
        rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.6}
        teacher_gpu_mem_util=${teacher_gpu_mem_util:-0.8}
        DEVICE_CONFIG+=(
            actor_rollout_ref.actor.megatron.use_mbridge=True
            actor_rollout_ref.actor.megatron.vanilla_mbridge=True
        )
        ;;
    *)
        echo "DEVICE must be gpu or npu, got: ${DEVICE}" >&2
        exit 1
        ;;
esac

max_num_tokens=$((max_prompt_length + max_response_length + 1))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size="$train_batch_size"
    data.max_prompt_length="$max_prompt_length"
    data.max_response_length="$max_response_length"
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="$actor_lr"
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ppo_max_token_len_per_gpu"
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="$actor_tp"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.use_kl_loss=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp"
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_mem_util"
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len="$max_num_tokens"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$ppo_max_token_len_per_gpu"
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name="$project_name"
    trainer.experiment_name="$experiment_name"
    trainer.n_gpus_per_node="$NGPUS_PER_NODE"
    trainer.nnodes="$NNODES"
    trainer.save_freq="$save_freq"
    trainer.test_freq="$test_freq"
    trainer.total_epochs="$total_epochs"
    trainer.total_training_steps="$total_training_steps"
)

DISTILLATION=(
    model_engine=megatron
    distillation.enabled=True
    distillation.n_gpus_per_node="$TEACHER_WORLD_SIZE"
    distillation.nnodes="$NNODES"
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="$teacher_tp"
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="$teacher_gpu_mem_util"
    distillation.teacher_models.teacher_model.inference.max_model_len="$max_num_tokens"
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
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
    "${DEVICE_CONFIG[@]}" \
    "$@"
