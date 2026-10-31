#!/bin/bash
set -xeuo pipefail

# Ascend MXFP8 compatibility note (2026-10-31)
# Verified verl baseline: 18eb23dde5ec84edf51a8b780a74a116524aa5cb (0.10.0.dev).
# Use the existing Megatron 0.18 / MindSpeed 0.18 environment .
# Temporary workaround:
#   cd /verl
#   git checkout 18eb23dde5ec84edf51a8b780a74a116524aa5cb
# Commit 8f016846 (PR #8287) removed the previous Ascend MXFP8 weight-reloading support.
# The tested main revision fails with a shape mismatch during the first weight sync.
# We plan to restore and validate Ascend MXFP8 compatibility as soon as possible.
# Once the fix is available and verified, use a main revision containing the fix;
# checking out the pinned commit above will no longer be necessary.

# Project Configuration
project_name='GRPO-Qwen3-30b-A3B-BASE-MATH'
exp_name='GRPO-Qwen3-30B-A3B-BASE-Megatron-vLLM'

# Node Info
NNODES=1
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

# Model Weights Paths
MODEL_PATH=/model/Qwen3-30B-A3B-MXFP8
MCORE_MODEL_PATH=Qwen/Qwen3-30B-A3B-Base-dist
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

# File System Paths
TRAIN_FILE=${TRAIN_FILE:-"/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/gsm8k/test.parquet"}


# Data Configuration
max_prompt_length=1024
max_response_length=8192

# Training Batch Configuration
train_prompt_bsz=64 
train_prompt_mini_bsz=16
n_resp_per_prompt=8

# Algorithm Configuration
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.000

# Performance and Memory Management Configuration
all_offload=True
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 8))
optimizer_offload_fraction=1

# Megatron Configuration
train_tp=4
train_ep=4
train_etp=1
train_pp=2
train_cp=1

# vLLM Configuration
gen_tp=2
gen_dp=1
gen_ep=1
gpu_memory_utilization=0.6
max_model_len=$((max_prompt_length + max_response_length))
max_num_batched_tokens=$(((max_prompt_length + max_response_length) * 1))



# Data Configuration
DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.train_batch_size=${train_prompt_bsz}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='left'
    data.validation_shuffle=False
    data.shuffle=False
)

# Model Configuration
MODEL_CONFIG=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

# Algorithm Configuration
ALGORITHM_CONFIG=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
)

# Actor Model Configuration
ACTOR_CONFIG=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.param_offload=${all_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}
    +actor_rollout_ref.actor.megatron.override_transformer_config.grad_offload=${all_offload}
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.position_embedding_type=rope
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall"
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.hccl_op_mode=\"tp:6,pp:2,tp_dp:2,cp:2,ep:6,tp_ep_mp:2,dp:2,tp_cp:2,dp_cp:2,default_group:2\"
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 
    +actor_rollout_ref.actor.megatron.override_transformer_config.transformer_impl=transformer_engine
    # Low-precision parameters
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=e4m3
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_format=e4m3
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=mxfp8
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_reuse_quantized_weight=False
)

# Reference Model Configuration
REF_CONFIG=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=${all_offload}
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${MCORE_MODEL_PATH}
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
)

# Rollout Configuration
ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.data_parallel_size=${gen_dp}
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    actor_rollout_ref.rollout.ignore_eos=False
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
    # Low-precision parameters
    actor_rollout_ref.rollout.quantization=ascend
)

# Trainer Configuration
TRAINER_CONFIG=(
    trainer.logger='["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NPUS_PER_NODE}"
    trainer.device='npu'
    trainer.total_epochs=15
    trainer.val_before_train=False
    trainer.val_only=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.default_local_dir="${CKPTS_DIR}"
)

# Main GRPO Training Command
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${ALGORITHM_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" | tee train.log
