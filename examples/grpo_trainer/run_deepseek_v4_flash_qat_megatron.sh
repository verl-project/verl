#!/usr/bin/env bash
# GRPO | DeepSeek-V4-Flash | vLLM fp8 rollout | Megatron fp8 training + MXFP4 expert QAT | NVIDIA GPUs
#
# What this adds over run_deepseek_v4_flash_megatron.sh
# -----------------------------------------------------
# The DeepSeek-V4 technical report states that, during post-training, "we incorporate FP4
# quantization-aware training for MoE expert weights". This script reproduces that: a weight-only
# fake-quant onto the MXFP4 grid (E2M1 values, one E8M0 power-of-two scale per 32 contiguous K
# elements) is applied to the ROUTED experts only, while the GEMMs keep running in real FP8 through
# TransformerEngine's blockwise autocast.
#
#   weight -> MXFP4 fake-quant (bf16 container on the FP4 grid) -> TE cast to FP8 E4M3 128x128
#   input  -> TE cast to FP8 E4M3 1x128
#                                   -> real FP8 tensor-core GEMM
#
# The FP4 step is simulated because Hopper has no FP4 tensor cores; the FP8 GEMM is not simulated.
# On Blackwell the same configuration is the natural starting point for a native FP4 kernel.
#
# Why this matches the checkpoint. DeepSeek-V4-Flash ships its routed experts already quantized to
# MXFP4 (packed E2M1 + F8_E8M0 scales, block 32) while every other quantized tensor is FP8 E4M3 with
# 128x128 scales. Megatron-Bridge re-quantizes exactly those expert tensors back to MXFP4 on every
# weight sync to vLLM, so the training forward and the rollout weights share one grid.
#
# Knobs
#   QAT_ENABLE=True               turn the fake-quant on (default False, i.e. plain fp8 training)
#   QAT_MODE=mxfp4_experts        verl/utils/modelopt/quantize.py; scopes to *mlp.experts*
#   QAT_BYPASS_TE_FP8_ASSERT=True ModelOpt refuses to run any quantizer under TE's fp8_autocast.
#                                 That guard exists to keep pure simulation faithful; here the
#                                 stacking is intentional (FP4 weights feeding a real FP8 GEMM), so
#                                 it is lifted at runtime. Activations are quantized once, by TE --
#                                 do not also enable ModelOpt's input_quantizer.
#
# Requires nvidia-modelopt (its MX fake-quant CUDA extension is JIT-built on first use, so nvcc must
# be available on the workers) and Megatron-Bridge importable on every node.
#
# With:
# - Megatron-Bridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/c7774d44d4b3101dc6bdf8c8d38a32e909e1ea11
# - Megatron-LM: https://github.com/NVIDIA/Megatron-LM/commit/1ff25ca7e339fe521165da7f4373d9f52e7af436

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Keep TE and vLLM on FP32 block scales rather than ue8m0, so both sides agree on the FP8 grid.
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1
export VLLM_USE_DEEP_GEMM_E8M0=0
# The ModelOpt MX kernel is JIT-compiled for the visible device; pin it so a future image listing
# many architectures does not multiply the one-off nvcc time.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

############################### configs ################################

MODEL_PATH=${MODEL_PATH:-$HDFS_ROOT/model/DeepSeek-V4-Flash}
NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
KL_COEF=${KL_COEF:-0.001}

ACTOR_LR=${ACTOR_LR:-1e-6}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-8}
ACTOR_VPP=${ACTOR_VPP:-null}
ACTOR_EP=${ACTOR_EP:-16}
ACTOR_ETP=${ACTOR_ETP:-1}
ACTOR_CP=${ACTOR_CP:-1}
PIPELINE_MODEL_PARALLEL_LAYOUT=${PIPELINE_MODEL_PARALLEL_LAYOUT:-"Et*6|t*6|t*6|t*5|t*5|t*5|t*5|t*5L"}

REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_VPP=${REF_VPP:-${ACTOR_VPP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_ETP=${REF_ETP:-${ACTOR_ETP}}
REF_CP=${REF_CP:-${ACTOR_CP}}

ROLLOUT_EP=${ROLLOUT_EP:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.40}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-512}
ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-R3}

# Truncated importance sampling over the rollout/trainer logprob ratio. Set to "off" to disable.
ROLLOUT_IS_MODE=${ROLLOUT_IS_MODE:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}

# MXFP4 quantization-aware training on the routed MoE experts. See the header.
QAT_ENABLE=${QAT_ENABLE:-False}
QAT_MODE=${QAT_MODE:-mxfp4_experts}
QAT_BYPASS_TE_FP8_ASSERT=${QAT_BYPASS_TE_FP8_ASSERT:-True}

ALL_OFFLOAD=${ALL_OFFLOAD:-True}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-30}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
# Directory for raw rollout dumps (prompt/response/score as jsonl); "null" disables them.
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}

PROJECT_NAME=${PROJECT_NAME:-verl_dsv4_flash}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_v4_flash_grpo_fp8_mxfp4qat}
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
    algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
    algorithm.kl_ctrl.kl_coef=${KL_COEF}
)

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$TEST_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    # DeepSeek-V4 renders prompts through verl's own continuous-token builder rather than a jinja
    # chat template, and the official checkpoints ship none. Length filtering is the one remaining
    # path that would still call apply_chat_template, so keep it off.
    data.filter_overlong_prompts=False
    data.truncation=error
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
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ACTOR_ETP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=False
    # TE fp8 GEMM autocast: the "real FP8" half of the recipe. It stays on together with QAT.
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="hybrid"
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise"
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise"
    "++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'"
)

# Context parallelism needs three extra transformer-config settings for DeepSeek-V4 on top of
# `context_parallel_size`. Each of them is enforced by Megatron-Core, so without them the run
# aborts at model build or in the first attention forward. They are appended only when CP > 1.
#
#   cp_partition_mode=contiguous
#     DSv4 attention requires every CP rank to own ONE consecutive interval of the packed THD
#     buffer. Megatron-Core defaults to "zigzag" and raises
#     "DSv4 Hybrid with CP requires cp_partition_mode='contiguous'."
#   sequence_packing_scheduler=dp_balanced
#     Megatron-Core asserts "DSv4 Hybrid with CP requires a sequence_packing_scheduler for THD
#     inputs." Needs Transformer Engine >= 2.9.
#   max_seqlen_per_dp_cp_rank
#     Documented as max sequence length / cp_size; it drives how sub-samples are assigned to
#     each DPxCP rank.
CP_ARGS=()
if [ "${ACTOR_CP}" -gt 1 ]; then
    CP_ARGS=(
        ++actor_rollout_ref.actor.megatron.override_transformer_config.cp_partition_mode=contiguous
        ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_packing_scheduler=dp_balanced
        ++actor_rollout_ref.actor.megatron.override_transformer_config.max_seqlen_per_dp_cp_rank=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / ACTOR_CP))
    )
fi

QAT_ARGS=()
if [ "${QAT_ENABLE}" = "True" ]; then
    QAT_ARGS=(
        ++actor_rollout_ref.actor.megatron.qat.enable=True
        ++actor_rollout_ref.actor.megatron.qat.mode=${QAT_MODE}
        ++actor_rollout_ref.actor.megatron.qat.bypass_te_fp8_assert=${QAT_BYPASS_TE_FP8_ASSERT}
        # rollout.qat is an oc.select alias of actor.megatron.qat in the generated config. The vLLM
        # worker must not build quantizers: it consumes the MXFP4 weights the bridge exports.
        ++actor_rollout_ref.rollout.qat.enable=False
    )
fi

TIS_ARGS=()
if [ "${ROLLOUT_IS_MODE}" != "off" ]; then
    TIS_ARGS=(
        algorithm.rollout_correction.rollout_is=${ROLLOUT_IS_MODE}
        algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}
    )
fi

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${ROLLOUT_KV_CACHE_DTYPE}
    +actor_rollout_ref.rollout.quantization=fp8
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_BUFFER_ENABLE}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
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
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    actor_rollout_ref.nccl_timeout=3600
    model_engine=megatron
)

########################### launch ###########################

# uv (set VERL_USE_UV=0 for system python): the GPU vllm x megatron driver and every Ray worker
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
    "${CP_ARGS[@]}" \
    "${QAT_ARGS[@]}" \
    "${TIS_ARGS[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
