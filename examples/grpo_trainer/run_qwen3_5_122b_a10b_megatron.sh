#!/usr/bin/env bash
# Qwen3.5-122B-A10B MoE GRPO RL with Megatron (four nodes, 8 GPUs, H20, 96G, geo3k dataset)
# Using verlai/verl:vllm017.latest docker image
# Requirements:
#   - 32 GPUs (96GB each, e.g. 4x8 H20)
#   - Additional packages on top of the base image:
#       pip install --upgrade transformers
#       pip install flash-linear-attention
#       pip install -U git+https://github.com/ISEEKYAN/mbridge.git
#   - Megatron-LM==0.16.0
#
# Requirements on Ascend:
#   - 4 nodes, 16 trainer devices per node
#   - Additional packages on base image(quay.io/ascend/verl:v0.8.0-cann9.0.0-torch2.9.0post2-a3-ubuntu22.04-py3.11-vllm):
#       pip install viztracer flash-linear-attention nvidia-modelopt nvidia-ml-py nvidia-resiliency-ext megatron-energon
#   - Megatron-LM==0.16.0
#   - MindSpeed==0.16.0
#   - Megatron-Bridge==de93536e
#
# Qwen3.5 architecture notes:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention which currently does
#   NOT support packed sequences (THD format) in Megatron-LM. Therefore:
#     - model.use_remove_padding=False           (deprecated option, will be removed in the future forces bshd compute format)
#     - actor.megatron.use_remove_padding=False  (forces bshd compute format)
#     - actor.use_dynamic_bsz=False              (required for bshd mode)
#
#   Once Megatron-LM adds THD support for Qwen3.5 GDN, use_remove_padding
#   can be set to True for better performance.
#
# Tested parallelism config:
#   GPU (32 GPUs / 4 node): TP=2 PP=2 CP=1 EP=8 ETP=1 GEN_TP=8
#   NPU (4 nodes):          TP=2 PP=4 CP=1 EP=16 ETP=1 GEN_TP=16
#

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

set -xeuo pipefail
unset http_proxy
unset https_proxy
# download geo3k dataset
hf download tyzhu/geo3k --repo-type dataset --local-dir $HOME/data/geo3k

# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

case "${DEVICE}" in
    gpu)
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
        ;;
    npu)
        export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
        export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-garbage_collection_threshold:0.8}
        export USE_OPTIMIZED_MODEL=${USE_OPTIMIZED_MODEL:-0}
        export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-5400}
        export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-300}
        export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
        export COMBINED_ENABLE=${COMBINED_ENABLE:-1}
        export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
        export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
        export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

# ---- user-adjustable ----
test_files=${test_files:-$HOME/data/geo3k/test.parquet}
train_files=${train_files:-$HOME/data/geo3k/train.parquet}
HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen3.5-122B-A10B"}

save_contents="['model', 'extra', 'optimizer']"

project_name=${project_name:-'verl_grpo_qwen3_5_122b_geo3k'}
exp_name=${exp_name:-'qwen3_5_122b_megatron'}

rollout_backend="vllm"

save_path=${save_path:-"Qwen/Qwen3.5-122B/verl_checkpoint"}
save_freq=50

train_batch_size=128
max_prompt_length=3240
max_response_length=4096
adv_estimator=${adv_estimator:-grpo}

TP=${TP:-2}
CP=${CP:-1}
ETP=${ETP:-1}
nnodes=${nnodes:-4}

case "${DEVICE}" in
    gpu)
        PP=${PP:-2}
        EP=${EP:-8}
        GEN_TP=${GEN_TP:-8}
        n_devices_per_node=${NDEVICES_PER_NODE:-8}
        rollout_gpu_memory_utilization=${rollout_gpu_memory_utilization:-0.66}
        rollout_log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu:-1}
        ref_log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu:-1}
        vllm_max_model_len=${vllm_max_model_len:-15768}
        ;;
    npu)
        PP=${PP:-4}
        EP=${EP:-16}
        GEN_TP=${GEN_TP:-16}
        n_devices_per_node=${NDEVICES_PER_NODE:-16}
        rollout_gpu_memory_utilization=${rollout_gpu_memory_utilization:-0.6}
        rollout_log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu:-4}
        ref_log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu:-4}
        vllm_max_model_len=${vllm_max_model_len:-8192}
        ;;
esac

ACTOR_VPP=${ACTOR_VPP:-null}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
########################### Parameter Arrays ###########################

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$ACTOR_VPP
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.checkpoint.save_contents="${save_contents}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_backend}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}
    actor_rollout_ref.rollout.n=6
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=${vllm_max_model_len}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

MODEL=(
    actor_rollout_ref.model.path=$HF_MODEL_PATH
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR_ROLLOUT_REF_COMMON=(
    actor_rollout_ref.nccl_timeout=10800
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)

DATA=(
    data.train_files=$train_files
    data.val_files=$test_files
    data.train_batch_size=$train_batch_size
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.truncation='right'
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=64
)

TRAINER=(
    trainer.logger=['console','wandb']
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=$n_devices_per_node
    trainer.nnodes=$nnodes
    trainer.save_freq=$save_freq
    trainer.default_local_dir=${save_path}
    trainer.test_freq=10
    trainer.val_before_train=False
    trainer.total_epochs=20
)

EXTRA=(
    model_engine=megatron
)

case "${DEVICE}" in
    gpu)
        ACTOR+=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=\"none\"
            +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
        )
        ;;
    npu)
        ACTOR+=(
            actor_rollout_ref.actor.megatron.vanilla_mbridge=False
            actor_rollout_ref.actor.checkpoint.strict=False
            ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
        )
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
        )
        ;;
esac

########################### Launch ###########################
export HYDRA_FULL_ERROR=1
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR_ROLLOUT_REF_COMMON[@]}" \
    "${TRAINER[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${EXTRA[@]}" \
    "$@"
