#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate verlMega
export PATH=$CONDA_PREFIX/bin:$PATH
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export DATA_PATH=$PWD/../verlData
export HF_HOME=$DATA_PATH
export VLLM_CACHE_DIR=$DATA_PATH/vllm_cache

set -xeuo pipefail

############################ Quick Config ############################

ROLLOUT_NAME="vllm" # sglang or vllm

FAMILY="Qwen"
STUDENT_MODEL=Qwen2.5-0.5B
TEACHER_MODEL=Qwen2.5-0.5B-Instruct

DISTILLATION_LOSS_MODE="reverse_kl_topk+"
DISTILLATION_LOSS_MODE="jsd_topk"
DISTILLATION_LOSS_MODE="k3"

PROJECT_NAME='verl_on_policy_distillation_example_gsm8k'
EXP_NAME="${FAMILY}/student-${STUDENT_MODEL}/teacher-${TEACHER_MODEL}/loss-${DISTILLATION_LOSS_MODE}"

MAX_PROMPT=256
MAX_RESPONSE_LENGTH=512
TRAIN_PROMPT_BSZ=4

STUDENT_MICRO_BATCH_SIZE=4
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))

TEACHER_MICRO_BATCH_SIZE=4
TEACHER_MAX_TOKEN_LEN_PER_GPU=$(( TEACHER_MICRO_BATCH_SIZE * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))

WORLD_SIZE=1
TP=1
PP=1
CP=1
EP=1
ETP=1

ALL_OFFLOAD=${ALL_OFFLOAD:-True}


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
    actor_rollout_ref.model.use_fused_kernels=True
)



DISTILLATION=(
    actor_rollout_ref.distillation.enabled=True
    actor_rollout_ref.distillation.loss_mode=$DISTILLATION_LOSS_MODE
    actor_rollout_ref.distillation.jsd_beta=0.5
    actor_rollout_ref.distillation.topk=128
    actor_rollout_ref.distillation.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.distillation.log_prob_micro_batch_size_per_gpu=$TEACHER_MICRO_BATCH_SIZE
    actor_rollout_ref.distillation.log_prob_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.distillation.teacher_model.path="${FAMILY}/${TEACHER_MODEL}"
    actor_rollout_ref.distillation.teacher_model.use_remove_padding=True
    actor_rollout_ref.distillation.megatron.use_remove_padding=True
    actor_rollout_ref.distillation.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.distillation.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.distillation.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.distillation.megatron.context_parallel_size=${CP}
    actor_rollout_ref.distillation.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.distillation.megatron.param_offload=${ALL_OFFLOAD}
)

DISTILLATION_DEBUG=(
    actor_rollout_ref.distillation.enabled=True
    actor_rollout_ref.distillation.loss_mode=$DISTILLATION_LOSS_MODE
    actor_rollout_ref.distillation.jsd_beta=0.5
    actor_rollout_ref.distillation.topk=128
    actor_rollout_ref.distillation.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.distillation.log_prob_micro_batch_size_per_gpu=$TEACHER_MICRO_BATCH_SIZE
    actor_rollout_ref.distillation.log_prob_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.distillation.teacher_model.path="${FAMILY}/${TEACHER_MODEL}"
    actor_rollout_ref.distillation.teacher_model.use_remove_padding=True
    actor_rollout_ref.distillation.megatron.use_remove_padding=True
    actor_rollout_ref.distillation.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.distillation.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.distillation.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.distillation.megatron.context_parallel_size=${CP}
    actor_rollout_ref.distillation.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.distillation.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.distillation.use_torch_compile=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=4e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

ACTOR_DEBUG=(
    actor_rollout_ref.actor.optim.lr=4e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.actor.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
)

ROLLOUT_DEBUG=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.agent.num_workers=1
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

TRAINER_DEBUG=(
    trainer.logger='["console"]'
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXP_NAME
    trainer.n_gpus_per_node=$WORLD_SIZE
    trainer.nnodes=1
    trainer.save_freq=200
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.use_legacy_worker_impl=disable
)


############################ Launch ############################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION_DEBUG[@]}" \
    "${ROLLOUT_DEBUG[@]}" \
    "${ACTOR_DEBUG[@]}" \
    "${TRAINER_DEBUG[@]}" \
    "$@"


# python3 -m verl.trainer.main_ppo \
#     --config-path=config \
    # --config-name='ppo_megatron_trainer.yaml' \
#     "${DATA[@]}" \
#     "${ALGORITHM[@]}" \
#     "${MODEL[@]}" \
#     "${DISTILLATION[@]}" \
#     "${ROLLOUT[@]}" \
#     "${ACTOR[@]}" \
#     "${TRAINER[@]}" \
#     "$@"
