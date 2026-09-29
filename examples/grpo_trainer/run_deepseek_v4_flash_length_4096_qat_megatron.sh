#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731: MXFP4 QAT, response 4096, 40 steps, validation every 10.
# Standalone MXFP4 expert QAT recipe with TE FP8 training.
# Plan A: prompt/response 1024/4096, overlong buffer 512 with penalty 1.0,
# clip high 0.28 / dual clip 10.0, no thinking; expert MXFP4 QAT plus TE FP8 training.
# Set MODEL_PATH, TRAIN_FILE and TEST_FILE in the launch environment.
# Optional: set OUTPUT_DIR or override each checkpoint/rollout/validation directory.
# For two segments, use TOTAL_TRAINING_STEPS=20, then 40 with the same checkpoint directory.
# Requires DeepSeek-V4 support in Megatron-Bridge, Megatron-Core and vLLM.

# Megatron prerequisite: HDO with native FP32 parameters
#
# This DeepSeek-V4-Flash example enables `optimizer_cpu_offload=True` and retains some model
# parameters in FP32. For this combination, Megatron's `DistributedOptimizer` must detach FP32
# model parameters before creating their optimizer shards. Otherwise, reconstructing
# `HybridDeviceOptimizer` can fail during optimizer initialization with:
#
# ```text
# ValueError: can't optimize a non-leaf Tensor
# ```
#
# The required fix is in `megatron/core/optimizer/distrib_optimizer.py`, inside
# `DistributedOptimizer._build_model_and_main_param_groups`, in the FP32-parameter branch:
#
# ```diff
# - shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
# + shard_model_param = model_param.detach().view(-1)[param_range.start : param_range.end]
# ```
#
# Use a Megatron version containing this fix, or backport it to the selected dependency checkout.
# The upstream fix is [NVIDIA/Megatron-LM#6982](https://github.com/NVIDIA/Megatron-LM/pull/6982),
# commit
# [`d5ff7ea72cffe1eb1daedaa2c7b2858694ab0d3c`](https://github.com/NVIDIA/Megatron-LM/commit/d5ff7ea72cffe1eb1daedaa2c7b2858694ab0d3c).
#
# **Validation provenance:** the completed 40-step MXFP4 QAT experiment used Megatron
# `1ff25ca7e339fe521165da7f4373d9f52e7af436` with an equivalent local runtime patch,
# `_patch_nonleaf_fp32_shards`. Both the initial 20-step segment and the resumed segment to step
# 40 logged that the patch was applied. Validation ran at steps 10/20/30/40, with checkpoints
# saved at steps 20/40 and optimizer state restored between segments.
#
# The runtime patch helper is not included in the published verl changes. Reproducing the
# experiment with Megatron `1ff25ca7e` therefore requires the dependency fix above in addition to
# the verl changes. This optimizer-initialization prerequisite is separate from the MXFP4
# checkpoint serialization and HDO checkpoint-resume fixes in verl.

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1  # Use FP32 scales for TE block-wise FP8
export VLLM_USE_DEEP_GEMM_E8M0=0             # Use FP32 scales for vLLM with DeepGEMM

############################### configs ################################

: "${MODEL_PATH:?Set MODEL_PATH to the model checkpoint directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet file}"
: "${TEST_FILE:?Set TEST_FILE to the validation parquet file}"
OUTPUT_DIR=${OUTPUT_DIR:-./outputs}
NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-5120}

USE_KL_IN_REWARD=${USE_KL_IN_REWARD:-False}
KL_COEF=${KL_COEF:-0.001}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
ACTOR_LR=${ACTOR_LR:-1e-6}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}

ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-8}
ACTOR_VPP=${ACTOR_VPP:-null}
ACTOR_EP=${ACTOR_EP:-16}
ACTOR_ETP=${ACTOR_ETP:-1}
ACTOR_CP=${ACTOR_CP:-1}
PIPELINE_MODEL_PARALLEL_LAYOUT=${PIPELINE_MODEL_PARALLEL_LAYOUT:-"Et*6|t*6|t*6|t*5|t*5|t*5|t*5|t*5L"}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

ROLLOUT_EP=${ROLLOUT_EP:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_N_VAL=${ROLLOUT_N_VAL:-16}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-5120}
# Keep the proven prefill budget independent of the longer response/context cap.
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-3072}
ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MB:-512}
ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-R3}
ROLLOUT_IS_MODE=${ROLLOUT_IS_MODE:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
# The submitter sets this to 20 for segment 1 and 40 for the resumed segment.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
PROJECT_NAME=${PROJECT_NAME:-verl_deepseek_v4_flash}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-dsv4-0731-mxfp4qat-len4096-40step-val10}
CKPTS_DIR=${CKPTS_DIR:-$OUTPUT_DIR/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$OUTPUT_DIR/rollout_dumps/$EXPERIMENT_NAME}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-$OUTPUT_DIR/validation_dumps/$EXPERIMENT_NAME}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-512}
OVERLONG_BUFFER_ENABLE=${OVERLONG_BUFFER_ENABLE:-True}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
ENABLE_THINKING=${ENABLE_THINKING:-False}

########################### parameter arrays ###########################

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
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=64
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
    # actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE}
    # actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    # actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    # actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
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
    # grad_offload removed upstream by verl 559c337a (#7544, BREAKING) as unused --
    # setting it now raises ConfigAttributeError: Key 'grad_offload' is not in struct.
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTER_REPLAY_MODE}
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_dsa_kernel_fusion=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True
    ++actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=False
    # moe_router_fusion: TE fused-router backward conflicts with router_replay R3 (inplace-modified
    # routing map in autograd) — RuntimeError at te/pytorch/router.py:72. Keep OFF with R3.
    # ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_fusion=True
    # ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_padding_for_quantization=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="hybrid"
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise"
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise"
    # +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_param=True
    # +actor_rollout_ref.actor.megatron.override_ddp_config.fp8_param_gather=True
    # +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_grad_reduce=True
    # +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_param_gather=True
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
    "++actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout='${PIPELINE_MODEL_PARALLEL_LAYOUT}'"
)

if [ "${RECOMPUTE_GRANULARITY}" = "selective" ]; then
    RECOMPUTE_ARGS=(
        ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=selective
        "++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules=[mla_up_proj]"
        ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=null
        ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=null
    )
else
    RECOMPUTE_ARGS=(
        ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
        ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
        ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    )
fi

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

# 截断重要性采样, 对 rollout/trainer 的 logprob 比值做裁剪。
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
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_N_VAL}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
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
    trainer.max_actor_ckpt_to_keep=2
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.resume_mode=auto
    trainer.val_before_train=False
    trainer.log_val_generations=100
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
    trainer.default_local_dir="${CKPTS_DIR}"
)

EXTRA=(
    actor_rollout_ref.nccl_timeout=3600
    model_engine=megatron
)

# Simulate MXFP4 routed-expert weights while retaining the TE FP8 GEMM path.
# vLLM receives Bridge-exported weights and must not construct QAT quantizers.
QAT_ARGS=(
    ++actor_rollout_ref.actor.megatron.qat.enable=True
    ++actor_rollout_ref.actor.megatron.qat.mode=mxfp4_experts
    ++actor_rollout_ref.actor.megatron.qat.bypass_te_fp8_assert=True
    ++actor_rollout_ref.rollout.qat.enable=False
)

########################### launch ###########################

python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${RECOMPUTE_ARGS[@]}" \
    "${CP_ARGS[@]}" \
    "${TIS_ARGS[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${QAT_ARGS[@]}" \
    "$@"
