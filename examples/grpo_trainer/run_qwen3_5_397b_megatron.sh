#!/usr/bin/env bash
# Qwen3.5-397 MoE GRPO RL with Megatron
#
# notes on vllm:
#     by 20260225, the latest vllm nightly does not support qwen3.5 rollout, to use this script, you need to
#         1. wait until vllm supports qwen3.5 officially, and build a verl docker with that version of vllm
#         2. self build a verl docker image with vllm from source code with qwen3.5 support (main branch 20260225 is OK)
#     I succeeded in running this script with the main branch of vllm on 20260225, yet there are still some minor issues
#     the vllm qwen3.5 during initialization, need to be fixed. Also, the cuda_graph is somehow not working, need to be
#     fixed, either by verl team with supoorts to vllm0.16, or by vllm team.
#
# Requirements on Ascend:
#   - 16 nodes * A3 (16*8*2=256 die)
#   - Additional packages on base image(verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5):
#       pip install viztracer flash-linear-attention nvidia-modelopt nvidia-ml-py nvidia-resiliency-ext megatron-energon
#   - Megatron-LM==0.16.0
#   - MindSpeed==0.16.0
#   - Megatron-Bridge==de93536e
#   - verl==0.8.0
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
# Tested parallelism config (16 NPUs * 16 node A3):
#   TP=2 PP=4 CP=1 EP=64 ETP=1 GEN_TP=16 GEN_DP=16 GEN_EP=256

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_USE_V1=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
mkdir -p logs
set -xeuo pipefail
unset http_proxy
unset https_proxy

########################### Quick Config ###########################

# ---- user-adjustable ----
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

nnodes=${nnodes:-16}
save_path=${save_path:-"Qwen/Qwen3.5-397B/verl_checkpoint"}
save_freq=50

case "${DEVICE}" in
    gpu)
        TP=${TP:-2}
        PP=${PP:-2}
        CP=${CP:-1}
        EP=${EP:-64}
        ETP=${ETP:-1}
        GEN_DP=${GEN_DP:-16}
        GEN_TP=${GEN_TP:-8}
        GEN_EP=${GEN_EP:-128}
        n_devices_per_node=${NDEVICES_PER_NODE:-8}
        ;;
    npu)
        TP=${TP:-2}
        PP=${PP:-4}
        CP=${CP:-1}
        EP=${EP:-64}
        ETP=${ETP:-1}
        GEN_DP=${GEN_DP:-16}
        GEN_TP=${GEN_TP:-16}
        GEN_EP=${GEN_EP:-256}
        n_devices_per_node=${NDEVICES_PER_NODE:-16}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

ALL_OFFLOAD=${ALL_OFFLOAD:-True}

rollout_name="vllm"
project_name='verl_grpo_qwen3_5_397b_geo3k'
exp_name='qwen3_5_397b_megatron'
adv_estimator=grpo

HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen3.5-397B-A17B"}
train_path=${train_path:-$HOME/data/geo3k/train.parquet}
test_path=${test_path:-$HOME/data/geo3k/test.parquet}
start_time=$(date +%Y%m%d)_$(date +%H%M%S)
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=64
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.truncation='error'
    data.filter_overlong_prompts=True
)

MODEL=(
    actor_rollout_ref.model.path=${HF_MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
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
    # +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.data_parallel_size=${GEN_DP}
    actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,12,24,48,64]"
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${nnodes}
    trainer.save_freq=${save_freq}
    trainer.default_local_dir=${save_path}
    trainer.val_before_train=True
    trainer.test_freq=5
    trainer.total_epochs=15
)

EXTRA=(
    model_engine=megatron
)

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        ACTOR+=(
            actor_rollout_ref.actor.megatron.vanilla_mbridge=False
            actor_rollout_ref.actor.checkpoint.strict=False
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
        )
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
        )
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### Launch ###########################

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee logs/qwen3.5-397b_grpo_megatron-${start_time}.log
