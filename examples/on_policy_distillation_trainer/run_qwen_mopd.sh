#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache

set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

FAMILY="Qwen"
STUDENT_MODEL=Qwen2.5-1.5B-Instruct
TEACHER_MODEL_MATH=Qwen2.5-Math-1.5B-Instruct
TEACHER_MODEL_CODING=Qwen2.5-Coder-1.5B-Instruct

# DISTILLATION_LOSS_MODE="k3"
DISTILLATION_LOSS_MODE="forward_kl_topk"

DISTILLATION_LOSS_MAX_CLAMP=null
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

PROJECT_NAME='verl_multi_task_on_policy_distillation_example_gsm8k'
EXP_NAME="${FAMILY}/student-${STUDENT_MODEL}/teacher-coding-${TEACHER_MODEL_CODING}/teacher-math-${TEACHER_MODEL_MATH}/loss-${DISTILLATION_LOSS_MODE}-maxclamp-${DISTILLATION_LOSS_MAX_CLAMP}-logprobminclamp-${DISTILLATION_LOG_PROB_MIN_CLAMP}"

MAX_PROMPT=1024
MAX_RESPONSE_LENGTH=1024
TRAIN_PROMPT_BSZ=128
STUDENT_MICRO_BATCH_SIZE_PER_GPU=1
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
TEACHER_MICRO_BATCH_SIZE_PER_GPU=1
TEACHER_MAX_TOKEN_LEN_PER_GPU=$(( TEACHER_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=False

ACTOR_ROLLOUT_REF_WORLD_SIZE=4
TEACHER_WORLD_SIZE=2
SP_SIZE=1

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet

math_train_path=$DATA_PATH/math_dataset/train.parquet
math_test_path=$DATA_PATH/math_dataset/test.parquet

codeforces_train_path=$DATA_PATH/codeforces/train.parquet
codeforces_test_path=$DATA_PATH/codeforces/test.parquet

TRAIN_FILES="['$gsm8k_train_path', '$math_train_path', '$codeforces_train_path']"
TEST_FILES="['$gsm8k_test_path', '$math_test_path', '$codeforces_test_path']"

TRAIN_FILES="['$gsm8k_train_path', '$codeforces_train_path']"
TEST_FILES="['$gsm8k_test_path', '$codeforces_test_path']"

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
    data.seed=42
)

MODEL=(
    actor_rollout_ref.model.path="${FAMILY}/${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.lora_rank=256
    actor_rollout_ref.model.lora_alpha=512
    # actor_rollout_ref.model.lora.lora_A_init_method=kaiming
    # # Optional: Use canonical LoRA
    # actor_rollout_ref.model.lora.type="canonical_lora"
    # actor_rollout_ref.model.lora.target_modules='["linear_q","linear_k","linear_v","linear_proj","linear_fc1_up","linear_fc1_gate","linear_fc2"]'

    # # Optional: Add dropout to LoRA layers
    # actor_rollout_ref.model.lora.dropout=0.05
    # actor_rollout_ref.model.lora.dropout_position=pre
)
# "['openai/gsm8k', 'DigitalLearningGmbH/MATH-lighteval']"
DISTILLATION=(
    actor_rollout_ref.distillation.enabled=True
    actor_rollout_ref.distillation.enable_resource_pool=True
    actor_rollout_ref.distillation.nnodes=1
    actor_rollout_ref.distillation.n_gpus_per_node=$TEACHER_WORLD_SIZE
    actor_rollout_ref.distillation.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.distillation.log_prob_micro_batch_size_per_gpu=$TEACHER_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.distillation.log_prob_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.distillation.ulysses_sequence_parallel_size=$SP_SIZE
    actor_rollout_ref.distillation.fsdp_config.param_offload=True
    actor_rollout_ref.distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    actor_rollout_ref.distillation.distillation_loss.jsd_beta=0.5
    actor_rollout_ref.distillation.distillation_loss.topk=64
    actor_rollout_ref.distillation.distillation_loss.use_policy_loss=False
    actor_rollout_ref.distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    actor_rollout_ref.distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
    actor_rollout_ref.distillation.teacher_models.num_teachers=2
    actor_rollout_ref.distillation.teacher_models.teacher0.path="${FAMILY}/${TEACHER_MODEL_MATH}"
    actor_rollout_ref.distillation.teacher_models.teacher0.use_remove_padding=True
    actor_rollout_ref.distillation.teacher_models.teacher0.domain='openai/gsm8k'
    actor_rollout_ref.distillation.teacher_models.teacher0.num_gpus_per_node=1
    actor_rollout_ref.distillation.teacher_models.teacher1.path="${FAMILY}/${TEACHER_MODEL_CODING}"
    actor_rollout_ref.distillation.teacher_models.teacher1.use_remove_padding=True
    actor_rollout_ref.distillation.teacher_models.teacher1.domain='codeforces'
    actor_rollout_ref.distillation.teacher_models.teacher1.num_gpus_per_node=1
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.rollout.n=1
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$ACTOR_ROLLOUT_REF_WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
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
    "${ACTOR[@]}" \
    "${TRAINER[@]}" \
    "$@"
