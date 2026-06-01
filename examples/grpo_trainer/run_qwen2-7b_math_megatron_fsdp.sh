#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

########################### Quick Config ###########################

TP=${TP:-4}
PP=${PP:-1}
GEN_TP=${GEN_TP:-4}

rollout_mode=${rollout_mode:-async}
return_raw_chat=${return_raw_chat:-True}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}

HF_MODEL_PATH=${HF_MODEL_PATH:-Qwen/Qwen2.5-Math-7B}
gsm8k_train_path=${gsm8k_train_path:-$HOME/data/gsm8k/train.parquet}
gsm8k_test_path=${gsm8k_test_path:-$HOME/data/gsm8k/test.parquet}
math_train_path=${math_train_path:-$HOME/data/math/train.parquet}
math_test_path=${math_test_path:-$HOME/data/math/test.parquet}

train_files=${train_files:-"['$gsm8k_train_path', '$math_train_path']"}
test_files=${test_files:-"['$gsm8k_test_path', '$math_test_path']"}

########################### Parameter Arrays ###########################

DATA=(
    "data.train_files=${train_files}"
    "data.val_files=${test_files}"
    "data.return_raw_chat=${return_raw_chat}"
    data.train_batch_size=32
    data.max_prompt_length=512
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    "actor_rollout_ref.model.path=${HF_MODEL_PATH}"
    "actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}"
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.mode=${rollout_mode}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=2
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
    actor_rollout_ref.ref.megatron.use_megatron_fsdp=True
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name='verl_grpo_example_gsm8k_math'
    trainer.experiment_name='qwen2_7b_megatron_fsdp'
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
)

########################### Launch ###########################

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
