#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Score Centering (SC / SC + TIS / SC + MIS) | Qwen3-30B-A3B-Base | 32K math reasoning
# Reference: Score Centering Stabilizes Off-policy Reinforcement Learning (arXiv:2609.20807)
#
# Score Centering modes (select via SC_MODE):
#   - SC_MODE=sc      : Score Centering without IS weights (validated at 32K scale)
#   - SC_MODE=sc_tis  : Score Centering + Token-level TIS, cap=2.0 (validated at 32K scale)
#   - SC_MODE=sc_mis  : Score Centering + Token-level MIS in [0.5, 5.0] (supported in code; not experimentally validated at 32K scale)
#
# Data preparation (see docs/data/dapo_math_variants.md and docs/data/aime_variants.md):
#   python3 examples/data_preprocess/aime.py --output-dir ${HOME}/data/aime

set -xeuo pipefail

# ---- user-adjustable ----
DEVICE=${DEVICE:-gpu}
SC_MODE=${SC_MODE:-sc}
TRAIN_BACKEND=${TRAIN_BACKEND:-megatron}

MODEL_PATH=${MODEL_PATH:-${HOME}/models/Qwen3-30B-A3B-Base}
TRAIN_FILE=${TRAIN_FILE:-${HOME}/data/dapo-math-17k/train.parquet}
VAL_FILE=${VAL_FILE:-${HOME}/data/aime/aime2024-2026.parquet}

NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-16}

# Native 32K context: 1024 prompt + 31743 response + 1 = 32768.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-31743}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
LOSS_SCALE_FACTOR=${LOSS_SCALE_FACTOR:-16384}

ACTOR_LR=${ACTOR_LR:-1e-6}
TOPK_LOG_PROBS=${TOPK_LOG_PROBS:-128}
# TileLang fused linear cross-entropy evaluates policy + top-k log-probs without materializing [T, V] logits.
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}
FUSED_KERNEL_BACKEND=${FUSED_KERNEL_BACKEND:-tilelang}

ROLLOUT_TP=${ROLLOUT_TP:-4}
# Enable rollout expert parallelism by default (EP=TP=4) so each rank holds full 768-wide experts
# divisible by the 128x128 FP8_BLOCK tile size while preserving TP=4 KV-cache headroom.
ROLLOUT_EP=${ROLLOUT_EP:-${ROLLOUT_TP}}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
# Low-precision rollout defaults on Megatron / VeOmni: FP8 block-quantized routed experts + FP8 KV cache
# (dense attention/router/lm_head stay in BF16; set ROLLOUT_QUANT_MOE=null for full BF16 weights).
ROLLOUT_QUANT_LINEAR=${ROLLOUT_QUANT_LINEAR:-null}
ROLLOUT_QUANT_MOE=${ROLLOUT_QUANT_MOE:-FP8_BLOCK}
ROLLOUT_KV_CACHE_DTYPE=${ROLLOUT_KV_CACHE_DTYPE:-fp8}

NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-4}
MAX_OFF_POLICY_THRESHOLD=${MAX_OFF_POLICY_THRESHOLD:-null}
REWARD_PRESET=${REWARD_PRESET:-math_dapo}
# Reward assigned by EOSNaiveRewardManager when valid_response_ids does not end on a native EOS token
# (-1.0 matches math_dapo's [-1, +1] incorrect score; set 0.0 for binary verifiers or null to disable).
MISSING_EOS_REWARD=${MISSING_EOS_REWARD:--1.0}
REWARD_MANAGER_PATH=${REWARD_MANAGER_PATH:-examples/rollout_correction/eos_reward_manager.py}

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1760}
SAVE_FREQ=${SAVE_FREQ:-80}
TEST_FREQ=${TEST_FREQ:-80}
SEED=${SEED:-42}

PROJECT_NAME=${PROJECT_NAME:-verl_score_centering_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_${SC_MODE}_32k}
# ---- end user-adjustable ----

case "${SC_MODE}" in
    sc)
        rollout_is=null
        rollout_is_threshold=2.0
        ;;
    sc_tis)
        rollout_is=token
        rollout_is_threshold=2.0
        ;;
    sc_mis)
        # Note: SC + MIS is supported by the implementation, but has not been experimentally validated at 32K scale.
        rollout_is=token
        rollout_is_threshold="0.5_5.0"
        ;;
    *)
        echo "Unsupported SC_MODE='${SC_MODE}'. Choose from: sc | sc_tis | sc_mis" >&2
        exit 1
        ;;
esac

case "${TRAIN_BACKEND}" in
    megatron)
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        ACTOR_TP=${ACTOR_TP:-4}
        ACTOR_PP=${ACTOR_PP:-1}
        ACTOR_CP=${ACTOR_CP:-2}
        ACTOR_EP=${ACTOR_EP:-8}
        PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
        UV_BACKEND_EXTRA=megatron
        TF=actor_rollout_ref.actor.megatron.override_transformer_config
        BACKEND_ARGS=(
            model_engine=megatron
            actor_rollout_ref.actor.optim.lr_decay_style=constant
            actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP}
            actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP}
            actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP}
            actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP}
            actor_rollout_ref.actor.megatron.param_offload=True
            actor_rollout_ref.actor.megatron.optimizer_offload=True
            actor_rollout_ref.actor.megatron.use_mbridge=True
            actor_rollout_ref.actor.megatron.seed=${SEED}
            ++${TF}.recompute_granularity=full
            ++${TF}.recompute_method=uniform
            ++${TF}.recompute_num_layers=1
            ++${TF}.moe_grouped_gemm=True
            ++${TF}.moe_permute_fusion=True
            ++${TF}.moe_router_dtype=fp32
            actor_rollout_ref.rollout.quantization.linear=${ROLLOUT_QUANT_LINEAR}
            actor_rollout_ref.rollout.quantization.moe=${ROLLOUT_QUANT_MOE}
        )
        ;;
    veomni)
        ACTOR_SP=${ACTOR_SP:-4}
        ACTOR_EP=${ACTOR_EP:-8}
        PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
        UV_BACKEND_EXTRA=veomni
        BACKEND_ARGS=(
            model_engine=veomni
            actor_rollout_ref.actor.optim.lr_scheduler_type=constant
            actor_rollout_ref.actor.veomni.param_offload=True
            actor_rollout_ref.actor.veomni.optimizer_offload=True
            actor_rollout_ref.actor.veomni.enable_full_shard=False
            actor_rollout_ref.actor.veomni.fsdp_size=-1
            actor_rollout_ref.actor.veomni.ulysses_parallel_size=${ACTOR_SP}
            actor_rollout_ref.actor.veomni.expert_parallel_size=${ACTOR_EP}
            actor_rollout_ref.actor.veomni.seed=${SEED}
            actor_rollout_ref.rollout.quantization.linear=${ROLLOUT_QUANT_LINEAR}
            actor_rollout_ref.rollout.quantization.moe=${ROLLOUT_QUANT_MOE}
        )
        ;;
    fsdp2)
        ACTOR_SP=${ACTOR_SP:-8}
        PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
        UV_BACKEND_EXTRA=fsdp
        BACKEND_ARGS=(
            actor_rollout_ref.actor.strategy=fsdp2
            actor_rollout_ref.actor.optim.lr_scheduler_type=constant
            actor_rollout_ref.actor.fsdp_config.param_offload=True
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
            actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${ACTOR_SP}
            actor_rollout_ref.actor.fsdp_config.seed=${SEED}
        )
        ;;
    *)
        echo "Unsupported TRAIN_BACKEND='${TRAIN_BACKEND}'. Choose from: megatron | veomni | fsdp2" >&2
        exit 1
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=rloo
    algorithm.gamma=1.0
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=True
    algorithm.rollout_correction.loss_type=reinforce
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.score_centering=True
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.seed=${SEED}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}
    actor_rollout_ref.model.fused_kernel_options.impl_backend=${FUSED_KERNEL_BACKEND}
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.data_loader_seed=${SEED}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
    actor_rollout_ref.actor.loss_scale_factor=${LOSS_SCALE_FACTOR}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.calculate_entropy=True
    actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode
    +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True
    +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is}
    +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    +actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.topk_log_probs=${TOPK_LOG_PROBS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.seed=${SEED}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${ROLLOUT_KV_CACHE_DTYPE}
    actor_rollout_ref.rollout.val_kwargs.n=32
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
)

REWARD=(
    reward.preset=${REWARD_PRESET}
    reward.reward_manager.source=importlib
    reward.reward_manager.name=EOSNaiveRewardManager
    reward.reward_manager.module.path="${REWARD_MANAGER_PATH}"
    +reward.reward_kwargs.missing_eos_reward=${MISSING_EOS_REWARD}
)

TRAINER=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}
    trainer.v1.sampler.max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD}
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=10000
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

########################### launch ###########################
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra "${UV_BACKEND_EXTRA}" --extra math python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra ${UV_BACKEND_EXTRA} --extra math")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${RAY[@]}" \
    "$@"
