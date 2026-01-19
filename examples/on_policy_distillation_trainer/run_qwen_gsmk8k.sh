#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,4
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache

set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

FAMILY="Qwen"
STUDENT_MODEL=Qwen2.5-0.5B-Instruct
TEACHER_MODEL=Qwen2.5-7B-Instruct

# DISTILLATION_LOSS_MODE="jsd_topk"
# DISTILLATION_LOSS_MODE="k3"
DISTILLATION_LOSS_MODE="reverse_kl_topk+"

PROJECT_NAME='verl_on_policy_distillation_example_gsm8k'
EXP_NAME="${FAMILY}/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}"

MAX_PROMPT=256
MAX_RESPONSE_LENGTH=512
TRAIN_PROMPT_BSZ=128
MICRO_BATCH_SIZE=8
MAX_TOKEN_LEN_PER_GPU=$(( MICRO_BATCH_SIZE * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))

WORLD_SIZE=2
SP_SIZE=1

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet

TRAIN_FILES="['$gsm8k_train_path']"
TEST_FILES="['$gsm8k_test_path']"

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${FAMILY}/${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

DISTILLATION=(
    actor_rollout_ref.distillation.enabled=True
    actor_rollout_ref.distillation.loss_mode=$DISTILLATION_LOSS_MODE
    actor_rollout_ref.distillation.jsd_beta=0.5
    actor_rollout_ref.distillation.topk=64
    actor_rollout_ref.distillation.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.distillation.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
    actor_rollout_ref.distillation.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.distillation.fsdp_config.param_offload=True
    actor_rollout_ref.distillation.teacher_model.path="${FAMILY}/${TEACHER_MODEL}"
    actor_rollout_ref.distillation.teacher_model.use_remove_padding=True
    actor_rollout_ref.distillation.ulysses_sequence_parallel_size=$SP_SIZE
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
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
    trainer.n_gpus_per_node=$WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=True
    trainer.use_legacy_worker_impl=disable
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
