#!/usr/bin/env bash
# GRPO | GPT-OSS-20B BF16 | VeOmni actor EP | vLLM rollout EP
#
# This example requires a vLLM revision with streamed GPT-OSS expert loading.
# Use a BF16 checkpoint, for example lmsys/gpt-oss-20b-bf16, rather than the
# original MXFP4 checkpoint.

set -xeuo pipefail

# ---- user-adjustable ----
# DATA_ROOT=${DATA_ROOT:-$HOME/data}
# MODEL_PATH=${MODEL_PATH:-$HOME/models/gpt-oss-20b-bf16}
DATA_ROOT=${DATA_ROOT:-/mnt/hdfs/mlsys/datasets}
MODEL_PATH=${MODEL_PATH:-hdfs://harunava/home/byte_arnold_va_ssd/mlsys/models/gpt-oss-20b-BF16}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

ACTOR_EP_SIZE=${ACTOR_EP_SIZE:-8}
ACTOR_SP_SIZE=${ACTOR_SP_SIZE:-1}

ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-8}
ROLLOUT_DP_SIZE=${ROLLOUT_DP_SIZE:-1}
ROLLOUT_EP_SIZE=${ROLLOUT_EP_SIZE:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gpt_oss_20b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gpt_oss_20b_veomni_ep_vllm_ep}
# ---- end user-adjustable ----

if (( ROLLOUT_EP_SIZE != ROLLOUT_TP_SIZE * ROLLOUT_DP_SIZE )); then
    echo "ROLLOUT_EP_SIZE must equal ROLLOUT_TP_SIZE * ROLLOUT_DP_SIZE for vLLM" >&2
    exit 1
fi

TRAIN_FILES="['${DATA_ROOT}/gsm8k/train.parquet','${DATA_ROOT}/math_dataset/train.parquet']"
VAL_FILES="['${DATA_ROOT}/gsm8k/test.parquet','${DATA_ROOT}/math_dataset/test.parquet']"
MAX_TOKEN_LENGTH=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.return_raw_chat=True
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
    actor_rollout_ref.actor.ppo_epochs=2
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LENGTH}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.veomni.param_offload=True
    actor_rollout_ref.actor.veomni.optimizer_offload=True
    actor_rollout_ref.actor.veomni.enable_full_shard=True
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=${ACTOR_SP_SIZE}
    actor_rollout_ref.actor.veomni.expert_parallel_size=${ACTOR_EP_SIZE}
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_3
    actor_rollout_ref.actor.veomni.moe_implementation=quack
    actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation=liger_kernel
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP_SIZE}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP_SIZE}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.max_num_seqs=128
)

TRAINER=(
    +trainer.use_legacy_worker_impl=disable
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=True
    trainer.save_freq=-1
    trainer.test_freq=5
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=10
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    model_engine=veomni \
    "$@"
