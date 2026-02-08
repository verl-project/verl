#!/usr/bin/env bash
set -xeuo pipefail

project_name="verl_grpo_qwen3-next-80b"
experiment_name="Qwen3_Next_80B_Instruct"

# Paths
WORK_DIR=${WORK_DIR:-"${HOME}/verl"}
MODEL_PATH=${WORK_DIR}/Qwen3-Next-80B-A3B-Instruct
TRAIN_FILE=${WORK_DIR}/datasets/dapo-math-17k/dapo-math-17k.parquet
TEST_FILE=${WORK_DIR}/datasets/aime/aime-2024.parquet

# algorithm
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001

clip_ratio_low=0.2
clip_ratio_high=0.28

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# batch
train_batch_size=16
rollout_n=16
ppo_mini_batch_size=8

# length
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))

# algorithm
learning_rate=1e-6
warmup_steps=0
# enable_filter_groups=True

# performance
sp_size=8
gen_tp=4
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='error'
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.nccl_timeout=14400

    # fsdp
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload}
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16

    # optimizer
    actor_rollout_ref.actor.optim.lr=${learning_rate}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${warmup_steps}

    # ppo config
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}

    # entropy
    actor_rollout_ref.actor.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True

    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0

    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.calculate_log_probs=True

    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1

    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

REF=(
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload}
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${offload}
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=False

    actor_rollout_ref.ref.entropy_checkpointing=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${experiment_name}"
    trainer.n_gpus_per_node=16
    trainer.nnodes=4
    trainer.val_before_train=False
    trainer.save_freq=5
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.device=npu
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_activation_offload=${offload}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

# =========================================================
echo "Starting Training with:"
echo "Project: ${project_name}, Exp: ${experiment_name}"
echo "Rollout N: ${rollout_n}, Batch Size: ${train_batch_size}, LR: ${learning_rate}"


python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
