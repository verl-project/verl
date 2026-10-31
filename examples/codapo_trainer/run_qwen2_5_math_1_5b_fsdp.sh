#!/usr/bin/env bash
# CoDaPO | Qwen2.5-Math-1.5B | MATH | vLLM + FSDP

set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/math/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/math/test.parquet}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
ROLLOUT_N=${ROLLOUT_N:-8}
TOP_K=${TOP_K:-4}
WEIGHT_OFFSET=${WEIGHT_OFFSET:-0.1}
ACCURACY_KEY=${ACCURACY_KEY:-acc}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-100}
PROJECT_NAME=${PROJECT_NAME:-verl_codapo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_math_1_5b}

DATA=(
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.do_sample=True
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
)

ALGORITHM=(
    algorithm.adv_estimator=codapo
    algorithm.use_kl_in_reward=False
    +algorithm.codapo_top_k=${TOP_K}
    +algorithm.codapo_weight_offset=${WEIGHT_OFFSET}
    +algorithm.codapo_accuracy_key=${ACCURACY_KEY}
)

REWARD=(
    reward.reward_manager.name=dapo
)

TRAINER=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.total_epochs=10
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
