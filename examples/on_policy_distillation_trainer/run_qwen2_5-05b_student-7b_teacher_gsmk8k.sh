#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate verl
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6,7
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache

set -xeuo pipefail




############################ Quick Config ############################

rollout_name="vllm" # sglang or vllm
project_name='verl_on_policy_distillation_example_gsm8k'
exp_name='qwen2_5-05b_student-7b_teacher'

max_prompt_length=256
max_response_length=512
train_prompt_bsz=128

############################ Paths ############################

gsm8k_train_path=$DATA_PATH/gsm8k/train.parquet
gsm8k_test_path=$DATA_PATH/gsm8k/test.parquet
math_train_path=$DATA_PATH/math/train.parquet
math_test_path=$DATA_PATH/math/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

############################ Parameter Groups ############################
MICRO_BATCH_SIZE=2

DATA=(
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.train_batch_size=$train_prompt_bsz
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

DISTILLATION=(
    actor_rollout_ref.distillation_config.enabled=True
    actor_rollout_ref.distillation_config.loss_mode=jsd_topk
    actor_rollout_ref.distillation_config.jsd_beta=0.5
    actor_rollout_ref.distillation_config.topk=32
    actor_rollout_ref.distillation_config.teacher_model.path=Qwen/Qwen2.5-7B-Instruct
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_bsz
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$rollout_name
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.rollout.n=1
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE # Teacher batch size
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=40
    trainer.total_epochs=15
    trainer.val_before_train=False
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
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
