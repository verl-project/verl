#!/usr/bin/env bash
# Qwen3.5-35B-A3B MoE GRPO RL with Megatron (single node, 8 GPUs, geo3k dataset)
# Using verlai/verl:uv.cu130.dev3 docker image
# Requirements:
#   - 8 GPUs (80GB each, e.g. 1x8 H100/H200)
#   - Image dependency cache: Megatron-Core 0.18.0 / Megatron-Bridge 0.5.2.
#   - flash-linear-attention is installed by the launcher's megatron extra.
#     The launcher uses dependencies from the current uv.lock.
#
# CUDA dependencies from the current uv.lock (Python 3.12):
#   Megatron-Core 0.19.2 / Megatron-Bridge 0.6.2; flash-linear-attention 0.5.2.
#
# Requirements on Ascend (source install per the MindSpeed-Bridge install guide:
# https://gitcode.com/Ascend/MindSpeed-Bridge/blob/master/docs/zh/install_guide.md):
#   - 16 NPUs (2*64GB each, e.g. a single A3 node)
#   - vllm==0.30.0, vllm-ascend@<g15ec56012>, transformers==5.21.0
#   - flash-linear-attention-npu==v26.9.1 (version pinned to match vllm-ascend)
#   - Megatron-LM==core_v0.18.0
#   - Megatron-Bridge==v0.5.0
#   - MindSpeed==core_r0.18.0
#   - MegatronAdaptor==core_r0.18.0
#   - TransformerEngineNPU==main
#   - MindSpeed-Ops==master
#   - MindSpeed-Bridge==master
#   - The NPU path uses MindSpeed-Bridge by default, with ascend GDN enabled:
#       +actor_rollout_ref.actor.megatron.override_transformer_config.use_triton_gdn=False
#       +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_gdn=True
#     GDN models with use_ascend_gdn=True also need the AscendC op package `fla_npu`:
#     https://gitcode.com/Ascend/MindSpeed-Bridge/blob/master/docs/zh/feature/ascend_gdn.md
#
# NPU path notes (default targets a single-node 16-NPU Atlas A3):
#   - MTP speculative decoding (opt-in, MTP_ENABLE=True) requires
#     vllm-ascend >= v0.27.1rc1 (the earliest tag containing commit 40348c31,
#     on main since 2026-08-07, the first build with usable Qwen3.5-family
#     MTP spec decode) and a checkpoint with MTP layers
#     (mtp_num_hidden_layers > 0, e.g. Qwen3.5-35B-A3B-40L)
#   - rl_config.enabled=True (set in the npu ROLLOUT below) is required for
#     RL weight sync with vllm-ascend >= v0.27.1rc1: those builds default to
#     the NZ weight layout (weight_nz_mode=1), which breaks the in-place
#     ND-format weight updates Megatron pushes to vLLM each training step.
#
# Qwen3.5 architecture notes:
#   This example uses BSHD compute format on CUDA:
#     - model.use_remove_padding=False
#     - actor.megatron.use_remove_padding=False
#     - actor.use_dynamic_bsz=False
#   Megatron-Core 0.18.0 and 0.19.2 also support THD for GDN.
#   The settings above retain BSHD for this example's CUDA path.
#   Ascend overrides below enable packing with MindSpeed-Bridge.
#
# Tested parallelism config (8 GPUs / 1 node):
#   TP=2 PP=1 CP=1 EP=8 ETP=1 GEN_TP=8
#
export VLLM_USE_V1=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
set -xeuo pipefail

########################### Quick Config ###########################

# ---- user-adjustable ----
# DEVICE is auto-detected by probing torch_npu; override only for special cases.
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
case "${DEVICE}" in
    gpu)
        TP=${TP:-2}
        PP=${PP:-1}
        CP=${CP:-1}
        EP=${EP:-8}
        ETP=${ETP:-1}
        GEN_TP=${GEN_TP:-8}
        n_devices_per_node=${NDEVICES_PER_NODE:-8}
        ;;
    npu)
        TP=${TP:-2}
        PP=${PP:-2}
        CP=${CP:-1}
        # Add hyperparameters to enable CP when CP != 1
        # +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=megatron_cp_algo
        EP=${EP:-8}
        ETP=${ETP:-1}
        GEN_TP=${GEN_TP:-8}
        n_devices_per_node=${NDEVICES_PER_NODE:-16}
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
# MTP speculative decoding (opt-in): requires vllm-ascend >= v0.27.1rc1
# (see NPU path notes above) and a checkpoint with MTP layers.
MTP_ENABLE=${MTP_ENABLE:-False}

rollout_name="vllm"
project_name='verl_grpo_qwen3_5_35b_geo3k'
exp_name='qwen3_5_35b_megatron'
adv_estimator=grpo

HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen3.5-35B-A3B"}
train_path=${train_path:-$HOME/data/geo3k/train.parquet}
test_path=${test_path:-$HOME/data/geo3k/test.parquet}
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
########################### Parameter Arrays ###########################

DATA=(
    data.train_files=${train_path}
    data.val_files=${test_path}
    data.train_batch_size=32
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
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
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
    actor_rollout_ref.rollout.enable_prefix_caching=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    actor_rollout_ref.rollout.enforce_eager=False
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
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
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.val_before_train=False
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
        export CPU_AFFINITY_CONF=1
        export MULTI_STREAM_MEMORY_REUSE=1
        export TASK_QUEUE_ENABLE=1
        export HCCL_OP_EXPANSION_MODE="AIV"
        export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
        export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
        export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
        # 1x8-NPU (A5) variant: NDEVICES_PER_NODE=8 PP=1, rollout dp/tp=4/2,
        # gpu_memory_utilization=0.55, MTP_ENABLE=True, plus vLLM sleep mode
        # (rl_config.sleep_mode_extra_cleanup).
        ACTOR+=(
            actor_rollout_ref.actor.use_dynamic_bsz=True
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
            actor_rollout_ref.actor.checkpoint.strict=False
            actor_rollout_ref.actor.megatron.use_remove_padding=True
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_triton_gdn=False
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_gdn=True
            +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
            +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
        )
        ROLLOUT+=(
            ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=True
            # vllm-ascend >= v0.27.1rc1 defaults to the NZ weight layout
            # (weight_nz_mode=1); RL weight sync from Megatron is in-place
            # ND-format, so force weight_nz_mode=0 via rl_config.enabled=True.
            +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=True
        )
        MODEL+=(
            actor_rollout_ref.model.use_remove_padding=True
        )
        # MTP speculative decoding (opt-in); needs vllm-ascend >= v0.27.1rc1
        # and a checkpoint with MTP layers.
        if [ "${MTP_ENABLE}" = "True" ]; then
            # MTP decode graph capture needs a larger MoE alltoall HCCL window buffer.
            export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
            # MTP decode emits (1 + 3 speculative) * num_reqs tokens per step; capture
            # sizes must cover that in multiples of 4, else decode falls back to eager.
            ROLLOUT+=(
                # Tuned rollout parallelism from the A3 MTP graph-mode validation;
                # the no-MTP default keeps the original GEN_TP=8 rollout layout.
                actor_rollout_ref.rollout.data_parallel_size=2
                actor_rollout_ref.rollout.tensor_model_parallel_size=4
                actor_rollout_ref.rollout.expert_parallel_size=8
                actor_rollout_ref.rollout.max_num_seqs=160
                +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,16,64,128,256,512,640]"
                +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=640
            )
            MODEL+=(
                actor_rollout_ref.model.mtp.enable=True
                actor_rollout_ref.model.mtp.enable_train=True
                actor_rollout_ref.model.mtp.enable_rollout=True
                actor_rollout_ref.model.mtp.detach_encoder=True
                actor_rollout_ref.model.mtp.num_speculative_tokens=3
            )
        fi
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

########################### Launch ###########################

# uv (set VERL_USE_UV=0 for system python): GPU vllm/sglang × megatron run the driver and every Ray worker
# (runtime_env.py_executable) through `uv run` on the matching extras of the committed uv.lock;
# other backends / NPU fall back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ] && { [ "${rollout_name}" = vllm ] || [ "${rollout_name}" = sglang ]; }; then
    LAUNCH=(uv run --frozen --all-packages --extra "${rollout_name}" --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra ${rollout_name} --extra megatron")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
