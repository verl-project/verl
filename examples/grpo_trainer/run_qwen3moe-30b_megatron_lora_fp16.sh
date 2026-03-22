#!/usr/bin/env bash
set -xeuo pipefail
pwd=`pwd`

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

TP=${TP:-2}
PP=${PP:-2}
CP=${CP:-2}
EP=${EP:-4}
ETP=${ETP:-1}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

optimizer_offload_fraction=1.

dtype="float16" # ["bfloat16", "float16"]
rollout_name="vllm"
project_name='verl_grpo_example_gsm8k_math_fp16'
exp_name='qwen3_30b_a3b_megatron_lora'
adv_estimator=grpo

# Paths
MODEL_PATH=$HOME/Qwen/Qwen3-30B-A3B-Instruct-2507
CKPTS_DIR=${pwd}/ckpt/${exp_name}

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet

########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${gsm8k_train_path}
    data.val_files=${gsm8k_test_path}
    data.train_batch_size=128
    data.max_prompt_length=1024
    data.max_response_length=1024
    data.truncation='error'
    data.filter_overlong_prompts=True
    data.shuffle=False
    data.return_raw_chat=$return_raw_chat
    data.filter_overlong_prompts_workers=128
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.lora.rank=16
    actor_rollout_ref.model.lora.alpha=32
    actor_rollout_ref.model.lora.dtype=${dtype}
    actor_rollout_ref.model.use_fused_kernels=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=3e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=${dtype}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${ALL_OFFLOAD}
)

ROLLOUT=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=8
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.dtype=${dtype}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.megatron.dtype=${dtype}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=5
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.max_actor_ckpt_to_keep=1
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.log_val_generations=10
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
    2>&1 | tee ${pwd}/log/${exp_name}_$(date +'%Y%m%d_%H%M%S').log