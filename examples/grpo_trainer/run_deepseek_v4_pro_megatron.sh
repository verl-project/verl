#!/usr/bin/env bash
# GRPO | DeepSeek-V4-Pro | vLLM rollout | Megatron training | NVIDIA GPUs (H100 and GB200 profiles)
#
# Derived from run_deepseek_v4_flash_megatron.sh: the algorithm, data and reward settings are the
# same; only the scale-dependent knobs differ and are selected with HW=h100 or HW=gb200.
#
#   HW=h100   64 nodes x 8 H100 80GB.  Training TP1 x PP32 x EP8 (DP2), rollout TP8 x DP8 x EP64
#             (one vLLM replica spans 8 nodes).  Validated with response length 4096; for 8192 use
#             ACTOR_EP=16 MAX_RESPONSE_LENGTH=8192 (DP1), see the notes below.
#   HW=gb200  64 nodes x 4 GB200.  Training TP1 x PP16 x EP16 (DP1), rollout TP1 x DP32 x EP32
#             (one replica spans 8 nodes), response length 8192.  NNODES=128 ACTOR_EP=32 ROLLOUT_DP=64
#             halves the per-rank expert parameters when the host memory is tight.
#
# Why Pro is not a scaled-up Flash:
# - A Pro vLLM replica does not fit in one node, so the rollout uses cross-node data-parallel
#   replicas.  This needs verl main at or after #8234 (merged 2026-10-26, 3dd97a0e): before it,
#   vLLM >= 0.29 rejected the headless engines verl starts on the other nodes of a replica.
# - DSV4 forces tensor_model_parallel_size=1 and verl requires rollout expert_parallel_size = TP x DP.
# - Stage 0 of the pipeline holds the embedding and the three hash-routed layers; on 80 GB GPUs the
#   bf16 grad buffer (GRAD_REDUCE_IN_FP32=False) and the precision-aware optimizer are both required,
#   otherwise the fp32 master shards are materialized on the GPU and stage 0 runs out of memory.
# - The precision-aware optimizer produces mixed bf16/fp32 gradient groups on DSV4 (hc alpha/bias,
#   ape, attn_sink are fp32) and the quantized params need no-grad fp32 main-param shards.  verl applies
#   runtime equivalents of https://github.com/NVIDIA/Megatron-LM/pull/7279 and
#   https://github.com/NVIDIA/Megatron-LM/pull/8245 in the Megatron engine until they are merged
#   (VERL_MEGATRON_OPTIMIZER_COMPAT=0 disables them).
# - Response length 8192 on 80 GB GPUs needs ACTOR_EP=16 (DP1): with EP8 stage 0 holds ~11.6 B params
#   (46 GiB of bf16 params + grads) next to the ~17 GiB that vLLM keeps resident after sleep, and the
#   first update_actor runs out of memory.  The same applies when loading from a Megatron
#   dist-checkpoint (bf16 params) with DIST_CKPT_PATH.
# - With all-CPU optimizer offload the host memory per node is set by the HybridDeviceOptimizer, not
#   by DP: ~12 bytes per parameter of fp32 master + Adam state per rank.  Keep the Ray object store
#   small (32 GiB) when starting the cluster, or it competes with the optimizer for host memory.
# - Checkpoints default to save_contents=[model,extra] (load_contents follows it by default): the
#   optimizer state of the 1.6 T model is ~22 TB, and resuming an offloaded optimizer needs verl main
#   at or after #8235 (merged 2026-10-26, 0bd668c1).
#
# Megatron-Bridge must be installed and importable on every node.
#
# Validated with:
# - Megatron-Bridge 0.6.2 (https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/c0e164ed)
# - Megatron-LM https://github.com/NVIDIA/Megatron-LM/commit/43e45e13 (the uv.lock still pins
#   megatron-core core_v0.19.2, which has no DeepSeek-V4; see https://github.com/verl-project/verl/issues/6864)
# - vLLM 0.29.0

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}


############################### hardware profile ###############################

HW=${HW:-h100}
case "${HW}" in
    h100)
        _NGPUS_PER_NODE=8
        _ACTOR_PP=32
        _ACTOR_EP=8
        # 61 layers (verl disables MTP) over PP32: stage 0 = embedding + 3 hash layers, 27 x 2 layers,
        # 4 x 1 layer, the last with the loss.
        _PIPELINE_MODEL_PARALLEL_LAYOUT="Et*3|$(printf 't*2|%.0s' $(seq 1 27))t*1|t*1|t*1|t*1L"
        _ROLLOUT_TP=8
        _ROLLOUT_DP=8
        _ROLLOUT_GPU_MEM_UTIL=0.6
        _MAX_RESPONSE_LENGTH=4096
        _PRECISION_AWARE_OPTIMIZER=True
        _NCCL_TIMEOUT=7200           # the first step of a PP32 model compiles kernels for a long time
        _ROLLOUT_MOE_BACKEND=marlin  # vLLM 0.29's DeepGEMM MXFP4 kernel is SM100+ only; verl's default (deep_gemm) fails at load_model on Hopper
        ;;
    gb200)
        _NGPUS_PER_NODE=4
        _ACTOR_PP=16
        _ACTOR_EP=16
        _PIPELINE_MODEL_PARALLEL_LAYOUT=""   # Megatron-Bridge generates the DSV4 layout for PP16 itself
        _ROLLOUT_TP=1
        _ROLLOUT_DP=32
        _ROLLOUT_GPU_MEM_UTIL=0.7
        _MAX_RESPONSE_LENGTH=8192
        _PRECISION_AWARE_OPTIMIZER=False  # optional on 189 GB GPUs; True frees ~20 % host memory and lets 64 nodes save
        _NCCL_TIMEOUT=3600
        _ROLLOUT_MOE_BACKEND=marlin  # DeepGEMM MXFP4 hit an illegal memory access in vLLM's profiling run at 10240 batched tokens; Marlin is also 5-9 % faster here
        ;;
    *)
        echo "ERROR: HW must be h100 or gb200, got '${HW}'" >&2
        exit 1
        ;;
esac

############################### configs ################################

MODEL_PATH=${MODEL_PATH:-$HDFS_ROOT/model/DeepSeek-V4-Pro}
NNODES=${NNODES:-64}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${_NGPUS_PER_NODE}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-${_MAX_RESPONSE_LENGTH}}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

ACTOR_LR=${ACTOR_LR:-1e-6}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
# bf16 instead of fp32 DDP grad buffers: -11 % actor GPU memory on GB200, and on H100 the only way
# stage 0 (~11 B parameters per rank) fits at all.  Alignment is unchanged.
GRAD_REDUCE_IN_FP32=${GRAD_REDUCE_IN_FP32:-False}
# Megatron precision-aware optimizer: fp32 master shards live in the (offloaded) optimizer state
# instead of on the GPU, main grads in bf16, Adam moments in fp32 (see the header for the two
# Megatron-LM fixes verl applies at runtime).
PRECISION_AWARE_OPTIMIZER=${PRECISION_AWARE_OPTIMIZER:-${_PRECISION_AWARE_OPTIMIZER}}
MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-bf16}

ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-${_ACTOR_PP}}
ACTOR_VPP=${ACTOR_VPP:-null}
ACTOR_EP=${ACTOR_EP:-${_ACTOR_EP}}
ACTOR_ETP=${ACTOR_ETP:-1}
ACTOR_CP=${ACTOR_CP:-1}
PIPELINE_MODEL_PARALLEL_LAYOUT=${PIPELINE_MODEL_PARALLEL_LAYOUT:-${_PIPELINE_MODEL_PARALLEL_LAYOUT}}

REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_VPP=${REF_VPP:-${ACTOR_VPP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_ETP=${REF_ETP:-${ACTOR_ETP}}
REF_CP=${REF_CP:-${ACTOR_CP}}

# Optional: load actor and ref from a Megatron dist-checkpoint instead of the HF checkpoint.  The
# online HF -> Megatron load walks the whole task list on every rank; on nodes with few GPUs and a
# large host-memory share per rank (GB200) the converted checkpoint is the safer start.  The
# checkpoint holds bf16 parameters, so on 80 GB GPUs combine it with ACTOR_EP=16 (see the header).
DIST_CKPT_PATH=${DIST_CKPT_PATH:-}

ROLLOUT_TP=${ROLLOUT_TP:-${_ROLLOUT_TP}}
ROLLOUT_DP=${ROLLOUT_DP:-${_ROLLOUT_DP}}
ROLLOUT_EP=${ROLLOUT_EP:-$((ROLLOUT_TP * ROLLOUT_DP))}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-${_ROLLOUT_GPU_MEM_UTIL}}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-512}
# MoE kernel for the MXFP4 routed experts.  Both profiles use Marlin (W4A16): verl's default for
# low-precision rollouts is DeepGEMM, but vLLM 0.29's DeepGEMM MXFP4 kernel is SM100+ only (every
# vLLM worker fails at load_model on H100) and on GB200 it died with an illegal memory access in the
# profiling run at 10240 batched tokens; where it ran (3072 batched tokens) it was 5-9 % slower than
# Marlin.  FlashInfer TRT-LLM cannot be refitted and the FlashInfer CUTLASS kernel is not selectable
# for a native DSV4 checkpoint.  Set ROLLOUT_MOE_BACKEND=none to fall back to verl's default.
ROLLOUT_MOE_BACKEND=${ROLLOUT_MOE_BACKEND:-${_ROLLOUT_MOE_BACKEND}}
ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-R3}   # disabled | R2 | R3; rollout routing replay is enabled only for R3

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
SAVE_FREQ=${SAVE_FREQ:--1}
# [model,extra] only: the optimizer state is ~22 TB and resuming it with optimizer offload needs #8235 (merged).
SAVE_CONTENTS=${SAVE_CONTENTS:-[model,extra]}
TEST_FREQ=${TEST_FREQ:--1}
NCCL_TIMEOUT=${NCCL_TIMEOUT:-${_NCCL_TIMEOUT}}

LOGGER=${LOGGER:-'["console","wandb"]'}
PROJECT_NAME=${PROJECT_NAME:-verl_dapo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_v4_pro_grpo_vllm_megatron_${HW}}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/verl/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

TRAIN_FILE=${TRAIN_FILE:-$DATA_ROOT/dataset/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-$DATA_ROOT/dataset/aime25_test.parquet}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-${MAX_RESPONSE_LENGTH}}
OVERLONG_BUFFER_ENABLE=${OVERLONG_BUFFER_ENABLE:-False}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

########################### parameter arrays ###########################

ENABLE_THINKING=${ENABLE_THINKING:-True}

ALGORITHM=(
    algorithm.adv_estimator=grpo
)

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation=left
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}
    ++actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=${GRAD_REDUCE_IN_FP32}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=False
)
if [ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT}" ]; then
    ACTOR+=("++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'")
fi
if [ "${PRECISION_AWARE_OPTIMIZER}" = "True" ]; then
    ACTOR+=(
        +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.main_grads_dtype=${MAIN_GRADS_DTYPE}
        +actor_rollout_ref.actor.optim.override_optimizer_config.exp_avg_dtype=fp32
        +actor_rollout_ref.actor.optim.override_optimizer_config.exp_avg_sq_dtype=fp32
    )
fi

# The ref worker inherits the actor's Megatron settings; they are spelled out so a profile change
# cannot leave the two out of step.
REF=(
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    ++actor_rollout_ref.ref.megatron.override_transformer_config.apply_dsa_kernel_fusion=True
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_fused_mhc=False
)
if [ -n "${PIPELINE_MODEL_PARALLEL_LAYOUT}" ]; then
    REF+=("++actor_rollout_ref.ref.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'")
fi
if [ -n "${DIST_CKPT_PATH}" ]; then
    ACTOR+=(
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=True
        actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    )
    REF+=(
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=True
        actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}"
    )
fi

# Context parallelism needs three extra transformer-config settings for DeepSeek-V4 on top of
# `context_parallel_size` (see run_deepseek_v4_flash_megatron.sh).  Appended only when CP > 1.
CP_ARGS=()
if [ "${ACTOR_CP}" -gt 1 ]; then
    CP_ARGS=(
        ++actor_rollout_ref.actor.megatron.override_transformer_config.cp_partition_mode=contiguous
        ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_packing_scheduler=dp_balanced
        ++actor_rollout_ref.actor.megatron.override_transformer_config.max_seqlen_per_dp_cp_rank=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / ACTOR_CP))
    )
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=$([ "${ROUTER_REPLAY_MODE}" = "R3" ] && echo True || echo False)
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${ROLLOUT_KV_CACHE_DTYPE}
)
if [ -n "${ROLLOUT_MOE_BACKEND}" ] && [ "${ROLLOUT_MOE_BACKEND}" != "none" ]; then
    ROLLOUT+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=${ROLLOUT_MOE_BACKEND})
fi

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger="${LOGGER}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.resume_mode=auto
    trainer.val_before_train=False
    trainer.log_val_generations=0
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT}
    model_engine=megatron
)

echo "DeepSeek-V4-Pro GRPO | HW=${HW} | ${NNODES} nodes x ${NGPUS_PER_NODE} GPUs | actor TP${ACTOR_TP}/PP${ACTOR_PP}/EP${ACTOR_EP}/CP${ACTOR_CP} | rollout TP${ROLLOUT_TP} x DP${ROLLOUT_DP} x EP${ROLLOUT_EP} util ${ROLLOUT_GPU_MEM_UTIL} | response ${MAX_RESPONSE_LENGTH} | precision_aware=${PRECISION_AWARE_OPTIMIZER} | replay ${ROUTER_REPLAY_MODE}"

########################### launch ###########################

# uv (set VERL_USE_UV=0 for system python): the GPU vllm × megatron driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the matching extras of the committed uv.lock.
# Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra megatron")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${CP_ARGS[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
