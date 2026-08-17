#!/usr/bin/env bash
# GRPO scale demo | Qwen3.5-35B-A3B | vLLM rollout | Megatron Lite training | GPU
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
# Qwen3.5 uses the Megatron Lite allgather CP path. CP is intentionally kept as
# an explicit mlite knob below; the default single-node run uses CP8. This path
# depends on FLA (flash-linear-attention) 5.0 in the runtime environment:
# https://github.com/fla-org/flash-linear-attention
#
# Default mlite mesh:
#   - 8 GPUs, PP1 EP8 CP8, TP1 ETP1, fsdp2
#
# Mesh accounting follows Megatron Lite's per-pipeline-stage layout:
#   ngpu / pp = tp * ep * dp = etp * ep * edp
# With the default 8 GPUs, PP1, EP8, TP1, and ETP1, this gives DP=1 and EDP=1.
#
# OPTIMIZER selects the Megatron Lite optimizer path:
#   - fsdp2:    Megatron Lite FSDP2 wrapper, lower memory pressure, default
#   - dist_opt: original Megatron distributed optimizer
# When using dist_opt, prefer a larger PP*EP mesh to reduce per-rank model and
# optimizer memory pressure and avoid OOM.

set -xeuo pipefail

########################### mlite backend knobs ###########################
MLITE_ROOT=${MLITE_ROOT:-$HOME/mlite}
MLITE_VERL_ROOT=${MLITE_VERL_ROOT:-${MLITE_ROOT}/experimental/lite/examples/verl}
MLITE_LITE_ROOT=${MLITE_LITE_ROOT:-${MLITE_ROOT}/experimental/lite}

MODEL_PATH=${MODEL_PATH:-${QWEN3_5_35B_MODEL_PATH:-Qwen3.5-35B-A3B}}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
TP=${TP:-1}
PP=${PP:-1}
EP=${EP:-8}
CP=${CP:-8}
ETP=${ETP:-1}
OPTIMIZER=${OPTIMIZER:-fsdp2} # dist_opt
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

NGPU=$((NNODES * NDEVICES_PER_NODE))
if (( NGPU % PP != 0 )); then
    echo "Invalid mesh: NGPU=${NGPU} must be divisible by PP=${PP}." >&2
    exit 1
fi

NGPU_PER_PP=$((NGPU / PP))
if (( NGPU_PER_PP % (TP * EP) != 0 )); then
    echo "Invalid mesh: NGPU/PP=${NGPU_PER_PP} must be divisible by TP*EP=$((TP * EP))." >&2
    exit 1
fi
if (( NGPU_PER_PP % (ETP * EP) != 0 )); then
    echo "Invalid mesh: NGPU/PP=${NGPU_PER_PP} must be divisible by ETP*EP=$((ETP * EP))." >&2
    exit 1
fi

DP=$((NGPU_PER_PP / (TP * EP)))
EDP=$((NGPU_PER_PP / (ETP * EP)))
echo "MLITE_MESH model=qwen3_5_35b ngpu=${NGPU} pp=${PP} tp=${TP} ep=${EP} etp=${ETP} cp=${CP} dp=${DP} edp=${EDP} optimizer=${OPTIMIZER}"
########################### end mlite backend knobs ###########################

########################### user-adjustable ###########################
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-$HOME/data/geo3k/test.parquet}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-5}

SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
PROJECT_NAME=${PROJECT_NAME:-verl-mlite-qwen3_5_35b-geo3k-grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_35b_grpo_${OPTIMIZER}}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
export PYTHONPATH="${MLITE_VERL_ROOT}:${MLITE_LITE_ROOT}:${MLITE_ROOT}:${VERL_ROOT:-}:${MEGATRON_ROOT:-}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset ROCR_VISIBLE_DEVICES
    unset HIP_VISIBLE_DEVICES
fi

########################### parameter arrays ###########################
ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.engine.tp=${TP}
    actor_rollout_ref.actor.engine.pp=${PP}
    actor_rollout_ref.actor.engine.vpp=1
    actor_rollout_ref.actor.engine.ep=${EP}
    actor_rollout_ref.actor.engine.cp=${CP}
    actor_rollout_ref.actor.engine.etp=${ETP}
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
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
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
