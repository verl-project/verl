#!/usr/bin/env bash

# Usage:
# 0. This script is designed to run on Ascend NPUs.
# 1. Start a Ray cluster of 32 nodes with 16 NPUs per node:
#    16 nodes for training and 16 disjoint nodes for rollout.
# 2. Set WORK_DIR (or MODEL_PATH, TRAIN_FILE and TEST_FILE), then run
#    this script from the verl repository root. Extra arguments are Hydra overrides.

# | Dependencies                   | Branch           | Commit ID                                  |
# | ------------------------------ | ---------------- | ------------------------------------------ |
# | vllm-project/vllm              | releases/v0.23.0 | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
# | vllm-project/vllm-ascend       | releases/v0.23.0 | `dc55ef82f8236585c8f2ee237f5b43507cae8686` |
# | NVIDIA/Megatron-LM             | core_v0.18.0     | `ba7b5ebce12af60627a80985792a1449ce45f46c` |
# | Ascend/MindSpeed               | core_r0.18.0     | `1881e01a996074b30d424349237d55a2c608e6b7` |
# | Ascend/MegatronAdaptor         | core_r0.18.0     | `eb0e5043438dd753099432bc3b3d8e5916ed1844` |
# | Ascend/TransformerEngineNPU    | main             | `e40ec34036eb7e04ad11b983f8c6b0dc9fcd88f9` |
# | Ascend/MindSpeed-Ops           | master           | `dbee3f41b156bb051fc046facf69548400be2be1` |
# | Ascend/MindSpeed-Bridge        | master           | `d82eec3ce2b31deb82cc5474b6813498efded908` |
# | NVIDIA-NeMo/Megatron-Bridge    | v0.5.0           | `fcbb6031103d0ca845c1a54d4fee55ecfcca17b6` |

set -xeuo pipefail

# Stop before configuring training if this machine has no available Ascend NPUs.
if ! python3 -c 'import torch; import torch_npu; raise SystemExit(not torch.npu.is_available())'; then
    echo "Error: This script requires available Ascend NPUs." >&2
    exit 1
fi

# Project Configuration
project_name='verl_megatron_dapo_math_17k_examples'
experiment_name='glm5_2-16train-16rollout-fully-async'

# Model Weights and File System Paths
WORK_DIR=${WORK_DIR:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${WORK_DIR}/glm52_weights"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
TRAIN_FILE=${TRAIN_FILE:-"${WORK_DIR}/datasets/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${WORK_DIR}/datasets/dapo-math-17k.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${RAY_DATA_HOME}/rollout_data/${project_name}/${experiment_name}/$(date +%Y%m%d_%H%M%S)"}

# Fully-async uses disjoint Ray placement groups for training and rollout.
NNODES_TRAIN=${NNODES_TRAIN:-16}
NNODES_ROLLOUT=${NNODES_ROLLOUT:-16}
NGPUS_PER_NODE_TRAIN=${NGPUS_PER_NODE_TRAIN:-16}
NGPUS_PER_NODE_ROLLOUT=${NGPUS_PER_NODE_ROLLOUT:-16}
# Train config (16 * 16 = 256 devices; DP = 256/PP/TP/CP = 2)
PP=8
TP=4
EP=32
ETP=1
CP=4

INFER_TP=8
INFER_DP=8
INFER_EP=64

train_world_size=$((NNODES_TRAIN * NGPUS_PER_NODE_TRAIN))
train_model_parallel_size=$((TP * PP * CP))
rollout_world_size=$((NNODES_ROLLOUT * NGPUS_PER_NODE_ROLLOUT))
if (( train_model_parallel_size <= 0 || train_world_size % train_model_parallel_size != 0 )); then
    echo "train world_size (${train_world_size}) must be divisible by TP*PP*CP (${train_model_parallel_size})" >&2
    exit 1
fi
if (( INFER_TP <= 0 || rollout_world_size % INFER_TP != 0 )); then
    echo "rollout world_size (${rollout_world_size}) must be divisible by INFER_TP (${INFER_TP})" >&2
    exit 1
fi
echo "[config] train: nodes=${NNODES_TRAIN} world=${train_world_size} TP=${TP} PP=${PP} CP=${CP} DP=$((train_world_size / train_model_parallel_size))"
echo "[config] rollout: nodes=${NNODES_ROLLOUT} world=${rollout_world_size} TP=${INFER_TP} DP=${INFER_DP} EP=${INFER_EP}"

max_num_seqs=64

# Batch Configuration
# Fully-async trainers consume batches from the rollout queue.
train_batch_size=0
gen_batch_size=${GEN_BATCH_SIZE:-1}
ppo_mini_batch_size=16
n_resp_per_prompt=8

require_batches=${REQUIRE_BATCHES:-1}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-4}
staleness_threshold=${STALENESS_THRESHOLD:-0.3}
partial_rollout=${PARTIAL_ROLLOUT:-True}
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-$((trigger_parameter_sync_step * require_batches * ppo_mini_batch_size * 100))}

RAY_ADDRESS_OVERRIDE=${RAY_ADDRESS_OVERRIDE:-auto}
RAY_INIT=()
if [[ -n "${RAY_ADDRESS_OVERRIDE}" && "$*" != *"ray_kwargs.ray_init.address"* ]]; then
    RAY_INIT+=(+ray_kwargs.ray_init.address="${RAY_ADDRESS_OVERRIDE}")
fi

balance_batch=False
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 32))

max_num_batched_tokens=4096
total_length=$(($max_prompt_length+$max_response_length))

ROLLOUT_IS=${ROLLOUT_IS:-token}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-false}
ROLLOUT_RS=${ROLLOUT_RS:-seq_mean_k1}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-0.98_1.02}

use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

loss_mode=${LOSS_MODE:-gspo}
loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-mean}
clip_ratio_low=${CLIP_RATIO_LOW:-0.0003}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.0004}

# Data Configuration
DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=$train_batch_size
    data.gen_batch_size=$gen_batch_size
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=False
    data.truncation='left'
    +data.apply_chat_template_kwargs.enable_thinking=False
)

# Model Configuration
MODEL_ARGS=(
    actor_rollout_ref.nccl_timeout=7200
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
)

# Algorithm Configuration
ALGORITHM_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=$use_kl_in_reward
    algorithm.kl_ctrl.kl_coef=$kl_coef
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD}
    algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE}
    algorithm.rollout_correction.rollout_rs=${ROLLOUT_RS}
    algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD}
)

# Actor Configuration
ACTOR_ARGS=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    actor_rollout_ref.actor.checkpoint.strict=False
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='fused'
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_distributed_optimizer=True
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=False
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP
    +actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_softmax_in_fp32=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_grouped_recompute=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.experimental_attention_variant="dsa"
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_dsa_absorb=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_lightning_indexer=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_sparse_flash_attention=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_lightning_indexer_kl_loss=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${CP}
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_algo=kvallgather_cp_algo
    +actor_rollout_ref.actor.megatron.override_transformer_config.reset_position_ids=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_ascend_mc2=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_model_parallel_layout="Et*10|t*12|t*8|t*8|t*12|t*12|t*12|t*4L"
    actor_rollout_ref.actor.megatron.router_replay.mode=R3
)

# Ref Configuration
REF_ARGS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$EP
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ETP
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
)

# Rollout Configuration
ROLLOUT_ARGS=(
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.num_nextn_predict_layers=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    +actor_rollout_ref.rollout.engine_kwargs.vllm.safetensors_load_strategy=lazy
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP
    actor_rollout_ref.rollout.data_parallel_size=$INFER_DP
    actor_rollout_ref.rollout.expert_parallel_size=$INFER_EP
    actor_rollout_ref.rollout.load_format='dummy'
    actor_rollout_ref.rollout.max_num_seqs=$max_num_seqs
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1824
    actor_rollout_ref.rollout.max_model_len=$total_length
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=True
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[2, 4, 8, 16, 24, 32, 64]"
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
)

# Async Training Configuration
ASYNC_TRAINING_ARGS=(
    rollout.n_gpus_per_node=$NGPUS_PER_NODE_ROLLOUT
    rollout.nnodes=$NNODES_ROLLOUT
    rollout.total_rollout_steps=$total_rollout_steps
    async_training.staleness_threshold=$staleness_threshold
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step
    async_training.require_batches=$require_batches
    async_training.partial_rollout=$partial_rollout
)

# Trainer Configuration
TRAINER_ARGS=(
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name=$experiment_name
    trainer.n_gpus_per_node=$NGPUS_PER_NODE_TRAIN
    trainer.nnodes=$NNODES_TRAIN
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.resume_mode="auto"
    trainer.balance_batch=${balance_batch}
    trainer.device=npu
    trainer.val_before_train=False
    trainer.total_epochs=100
)

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name=fully_async_ppo_megatron_trainer.yaml \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_ASCEND_GLM52_PATCH='1'" \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${ASYNC_TRAINING_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "${RAY_INIT[@]}" \
    "$@" 2>&1
