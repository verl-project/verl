#!/usr/bin/env bash
set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

FAMILY="Qwen"
STUDENT_MODEL=Qwen3-VL-2B-Instruct
GSM8K_TEACHER_MODEL=Qwen3-4B-Instruct-2507
GEO3K_TEACHER_MODEL=Qwen3-VL-4B-Instruct

USE_POLICY_GRADIENT=True
DISTILLATION_LOSS_MODE="k1"
USE_FUSED_KERNELS=False

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='verl_on_policy_distillation_example_gsm8k_geo3k'
EXP_NAME="${FAMILY}/student-${STUDENT_MODEL}/teacher-gsm8k-${GSM8K_TEACHER_MODEL}/teacher-geo3k-${GEO3K_TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}-pg-${USE_POLICY_GRADIENT}"

MAX_PROMPT=1024
MAX_RESPONSE_LENGTH=2048
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=False

STUDENT_WORLD_SIZE=2

# Number of replicas per teacher. Each replica occupies
# (inference.tensor_model_parallel_size * inference.data_parallel_size *
# inference.pipeline_model_parallel_size) GPUs — with TP=DP=PP=1 below, that's 1 GPU per
# replica, so the teacher pool size must equal the sum of num_replicas.
TEACHER_NUM_REPLICAS_GSM8K=1
TEACHER_NUM_REPLICAS_GEO3K=1
TEACHER_POOL_WORLD_SIZE=$(( TEACHER_NUM_REPLICAS_GSM8K + TEACHER_NUM_REPLICAS_GEO3K ))

SP=1

ENFORCE_EAGER=False # true for faster debugging

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet
geo3k_train_path=$DATA_PATH/geo3k/train.parquet
geo3k_test_path=$DATA_PATH/geo3k/test.parquet

TRAIN_FILES="['$gsm8k_train_path','$geo3k_train_path']"
TEST_FILES="['$gsm8k_test_path','$geo3k_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path="${FAMILY}/${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

# Multi-teacher: one teacher per dataset, routed by the sample's `data_source` value.
# Each teacher has its own model_path and inference config.
DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=$TEACHER_POOL_WORLD_SIZE
    distillation.nnodes=1
    # --- gsm8k teacher ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="${FAMILY}/${GSM8K_TEACHER_MODEL}"
    +distillation.teacher_models.gsm8k.num_replicas=$TEACHER_NUM_REPLICAS_GSM8K
    +distillation.teacher_models.gsm8k.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.8
    +distillation.teacher_models.gsm8k.inference.enforce_eager=$ENFORCE_EAGER
    +distillation.teacher_models.gsm8k.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=$MAX_NUM_TOKENS
    # --- geo3k teacher (VL) ---
    +distillation.teacher_models.geo3k.key="hiyouga/geometry3k"
    +distillation.teacher_models.geo3k.model_path="${FAMILY}/${GEO3K_TEACHER_MODEL}"
    +distillation.teacher_models.geo3k.num_replicas=$TEACHER_NUM_REPLICAS_GEO3K
    +distillation.teacher_models.geo3k.inference.name=$ROLLOUT_NAME
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=0.8
    +distillation.teacher_models.geo3k.inference.enforce_eager=$ENFORCE_EAGER
    +distillation.teacher_models.geo3k.inference.max_model_len=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.max_num_seqs=$MAX_NUM_TOKENS
    +distillation.teacher_models.geo3k.inference.engine_kwargs.vllm.mm_processor_cache_gb=0
    # --- loss ---
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.n=1
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=True
    trainer.resume_mode=disable
    trainer.log_val_generations=5
)



############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
