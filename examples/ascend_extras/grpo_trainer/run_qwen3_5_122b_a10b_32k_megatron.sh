#!/usr/bin/env bash
set -xeuo pipefail

########################### Environment ###########################

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=${VLLM_ASCEND_ENABLE_PREFETCH_MLP:-1}
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=${VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE:-1}
export VLLM_ASCEND_ENABLE_FLASHCOMM1=${VLLM_ASCEND_ENABLE_FLASHCOMM1:-1}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}

########################### Quick Config ###########################

# Node Info
NNODES=${NNODES:-4}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

# ---- user-adjustable ----

project_name=${project_name:-verl_grpo_qwen3_5_122b_geo3k}
exp_name=${exp_name:-qwen3_5_122b_megatron_npu_4k_32k}
adv_estimator=${adv_estimator:-grpo}
rollout_name=${rollout_name:-vllm}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-122B-A10B"}
train_path=${train_path:-"${RAY_DATA_HOME}/datasets/geo3k/train.parquet"}
test_path=${test_path:-"${RAY_DATA_HOME}/datasets/geo3k/test.parquet"}

TP=${TP:-2}
PP=${PP:-4}
CP=${CP:-4}
EP=${EP:-16}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-16}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

# ---- end user-adjustable ----

########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=16
    data.max_prompt_length=$((1024 * 4))
    data.max_response_length=$((1024 * 32))
    data.truncation='error'
    data.filter_overlong_prompts=True
)

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)

ACTOR=(
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

    actor_rollout_ref.actor.optim.lr=1e-6
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True

    actor_rollout_ref.actor.checkpoint.strict=False
    actor_rollout_ref.actor.checkpoint.save_contents="['model']"

    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16

    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.enforce_eager=False

    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096

    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,12,24,48,64]"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes="${NNODES}"
    trainer.save_freq=15
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.total_training_steps=100
    trainer.total_epochs=15
)

EXTRA=(
    model_engine=megatron
)

########################### Launch ###########################

mkdir -p logs

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
