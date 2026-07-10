#!/usr/bin/env bash
# GRPO scale demo | DeepSeek-V4 | vLLM rollout | Megatron Lite training | GPU
#
# Megatron Lite is Megatron's agentic experimental path. Its upstream home is
# Megatron-LM's dev branch:
# https://github.com/NVIDIA/Megatron-LM/tree/dev/experimental/lite
#
# This launcher currently tracks the submitter's active branch until the latest
# mlite changes merge upstream. That checkout provides both megatron.lite and
# the verl_mlite backend glue:
#
#   git clone -b lite https://github.com/verl-project/Megatron-LM mlite
#   pip install -e mlite/experimental/lite/examples/verl
#
# DeepSeek-V4 uses fused DSA kernels on Hopper and Blackwell GPUs. The critical
# DSA-only dependencies are nvidia-cutlass-dsl==4.5.2 and nvidia-cudnn-frontend.
# cudnn-frontend release 1.24.1 is sufficient for Blackwell, while Hopper still
# needs a develop-branch build with IndexerForwardSm90 support.
#
# MODEL_VARIANT selects the DeepSeek-V4 target and its default mlite mesh:
#   - flash: 16 nodes, PP4 EP8  CP4
#   - pro:   64 nodes, PP8 EP16 CP4
#
# DS4 is fixed to TP1/ETP1. The architecture does not support TP/ETP
# sharding, and there is no plan to support it.
#
# OPTIMIZER selects the Megatron Lite optimizer path:
#   - fsdp2:    Megatron Lite FSDP2 wrapper, lower memory pressure, default
#   - dist_opt: original Megatron distributed optimizer
# When using dist_opt, prefer a larger PP*EP mesh to reduce per-rank model and
# optimizer memory pressure and avoid OOM.

set -xeuo pipefail

########################### mlite backend knobs ###########################
MODEL_VARIANT=${MODEL_VARIANT:-flash}
MLITE_ROOT=${MLITE_ROOT:-$HOME/mlite}
MLITE_VERL_ROOT=${MLITE_VERL_ROOT:-${MLITE_ROOT}/experimental/lite/examples/verl}
MLITE_LITE_ROOT=${MLITE_LITE_ROOT:-${MLITE_ROOT}/experimental/lite}

OPTIMIZER=${OPTIMIZER:-fsdp2} # dist_opt
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

case "${MODEL_VARIANT}" in
    flash)
        MODEL_PATH=${MODEL_PATH:-${FLASH_MODEL_PATH:-}}
        NNODES=${NNODES:-16}
        PP=${PP:-4}
        EP=${EP:-8}
        CP=${CP:-4}
        ;;
    pro)
        MODEL_PATH=${MODEL_PATH:-${PRO_MODEL_PATH:-}}
        NNODES=${NNODES:-64}
        PP=${PP:-8}
        EP=${EP:-16}
        CP=${CP:-4}
        ;;
    *)
        echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}. Expected flash or pro." >&2
        exit 1
        ;;
esac

: "${MODEL_PATH:?set MODEL_PATH, or set FLASH_MODEL_PATH/PRO_MODEL_PATH for MODEL_VARIANT=${MODEL_VARIANT}}"
########################### end mlite backend knobs ###########################

########################### user-adjustable ###########################
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}

TRAIN_FILE=${TRAIN_FILE:-$HOME/data/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/gsm8k/test.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}
ROLLOUT_N=${ROLLOUT_N:-16}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROJECT_NAME=${PROJECT_NAME:-verl-mlite-deepseek_v4_${MODEL_VARIANT}-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_v4_${MODEL_VARIANT}_grpo_${OPTIMIZER}}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
export PYTHONPATH="${MLITE_VERL_ROOT}:${MLITE_LITE_ROOT}:${MLITE_ROOT}:${VERL_ROOT:-}:${MEGATRON_ROOT:-}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES
fi

########################### parameter arrays ###########################
ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.truncation=error
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor@actor_rollout_ref.actor=mlite_actor
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.engine.tp=1
    actor_rollout_ref.actor.engine.pp=${PP}
    actor_rollout_ref.actor.engine.vpp=1
    actor_rollout_ref.actor.engine.ep=${EP}
    actor_rollout_ref.actor.engine.cp=${CP}
    actor_rollout_ref.actor.engine.etp=1
    actor_rollout_ref.actor.engine.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.engine.attention_backend_override=flash
    actor_rollout_ref.actor.engine.impl_cfg.use_thd=True
    +actor_rollout_ref.actor.engine.impl_cfg.optimizer=${OPTIMIZER}
    +actor_rollout_ref.actor.engine.impl_cfg.recompute=[full]
    +actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=1.0
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.free_cache_engine=True
)

TRAINER=(
    critic.enable=False
    trainer.logger=[console]
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.val_before_train=False
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

EXTRA=(
    hydra.searchpath=[pkg://verl_mlite.config]
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${EXTRA[@]}" \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@"
