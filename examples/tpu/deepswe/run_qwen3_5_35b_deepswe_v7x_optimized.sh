#!/usr/bin/env bash
# DeepSWE | Qwen3.5-35B-A3B | optimized for TPU v7x: 2x2x2 trainer + 3 x 2x2x1 rollout replicas
#
# Wraps run_qwen3_5_35b_deepswe.sh with the settings measured fastest on TPU v7x (32k context, 16 issues x 16 attempts
# = 256 trajectories per step). Steady state ~320 s/step vs ~490 s/step for the plain recipe on the same hardware; the
# rollout (vLLM-TorchTPU) is the critical path, the trainer phase is ~115 s/step.
#
#   - Sampler: MAX_NUM_SEQS 128, CONCURRENCY 384, one full prefetched batch (NUM_WARMUP_BATCHES=1, staleness 1),
#     TPU_MOE_SKIP_PADDED_TOKENS=1, TPU_ENABLE_GDN_DYNAMIC_TILING=1, prefix caching, 512-token KV blocks.
#   - Rollout MoE: vLLM expert parallelism across each replica's 8 chips (ROLLOUT_EXPERT_PARALLEL_SIZE=8, attention
#     stays TP 8) with the fused EP MoE kernel (USE_MOE_FUSED_EP_KERNEL=1).
#   - Trainer: FSDP16 without EP, on-device fp32 AdamW, one 32768-token GatedDeltaNet row slot per micro-batch, MoE
#     router gate frozen (needs verl-hardware-plugin `gdn_row_slot_len`, `splash_max_block_size`, `frozen_param_regex`).
#   - Image: examples/tpu/grpo/Dockerfile built with --build-arg TORCHTITAN_COMMIT=4e16e67aed8c1e702e6f440b1ff81619bb6873d4
#     (google-pytorch/torchtitan#142, as for Qwen3.5-397B): gdn_row_slot_len needs its gated_delta_document_resets,
#     which the Dockerfile's default torchtitan commit does not have.
#   - Rollout correction: the base recipe's sequence rejection (rollout_rs=seq_mean_k1, geometric-mean
#     pi_train / pi_rollout band) with the lower edge widened from 0.999 to 0.998: this stack's trainer/rollout
#     log-prob bias (~+0.0009) rejects ~45% of sequences at 0.999 and ~5% at 0.998, with no step-time cost.
#
# Set the dataset/model/sandbox variables the base recipe reads (MODEL_PATH, TRAIN_FILE, TEST_FILE, UNI_AGENT_HOME,
# TASK_CONFIG, SANDBOX_*, NODE_SELECTOR_*) before running.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_35b_deepswe_r2e_gym_v7x_optimized}"

# Topology: trainer 2 hosts x 8 TensorCores (2x2x2), FSDP16 only; 3 rollout replicas of 8 chips (2x2x1).
export NNODES_TRAINER="${NNODES_TRAINER:-2}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-3}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-8}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-8}"
export ROLLOUT_EXPERT_PARALLEL_SIZE="${ROLLOUT_EXPERT_PARALLEL_SIZE:-8}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-dummy}"
export TENSOR_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_STRATEGY=all_gather
export DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-16}"
export EXPERT_PARALLEL_SIZE=1
export FUSE_QKV=True
export USE_KL_LOSS=False

# Trainer memory/compute: on-device fp32 parameters + fused AdamW, no host offload.
export PARAM_DTYPE=float32
export FP32_MASTER=False
export OPTIMIZER_HOST_OFFLOAD=False
export OPTIM_IMPL=fused
export SPARSECORE_OFFLOAD=False
export TRAINER_SCOPED_VMEM_LIMIT_KIB=65472
# SparseCore async FSDP collective offload only (keep the TensorCore/SparseCore gather defaults for the EP=1 dispatcher).
export TRAINER_EXTRA_LIBTPU_INIT_ARGS="${TRAINER_EXTRA_LIBTPU_INIT_ARGS:---xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_enable_async_all_gather=true --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true}"
export AC_SAVE_SPLASH_AND_ROUTING=True
export USE_DYNAMIC_BSZ=False

# 32k context; one 32768-token micro-batch (one GDN row slot) per rank, padded to one compiled shape.
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-24576}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export MAX_TOKEN_LEN_PER_DEVICE=$((MICRO_BATCH_SIZE * MAX_MODEL_LEN))
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_DEVICE}"
export STATIC_MICRO_BATCH_LEN="${MAX_TOKEN_LEN_PER_DEVICE}"
export PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-2048}"

# Sampler. 16 issues x 16 attempts = 256 trajectories per step (MLPerf BATCH_SIZE=16 x NUM_GENERATIONS=16), one
# mini-batch per step. A full prefetched batch (staleness 1) keeps the replicas busy through the trainer phase.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
export MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-2}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-True}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
# One agent session per decode slot (3 replicas x 128); 512 sessions was slower (more sandbox/prefill load).
export CONCURRENCY="${CONCURRENCY:-384}"
export MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"

exec bash "${SCRIPT_DIR}/run_qwen3_5_35b_deepswe.sh" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TPU_MOE_SKIP_PADDED_TOKENS='1'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.TPU_ENABLE_GDN_DYNAMIC_TILING='1'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.USE_MOE_FUSED_EP_KERNEL='1'" \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.splash_max_block_size=1024 \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.gdn_row_slot_len="${MAX_MODEL_LEN}" \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.frozen_param_regex=moe.router.gate \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.verify_parity=off \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.release_buffers_after_sync=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    algorithm.rollout_correction.bypass_mode=False \
    "algorithm.rollout_correction.rollout_rs_threshold='${ROLLOUT_RS_THRESHOLD:-0.998_1.002}'" \
    "$@"
