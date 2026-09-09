#!/usr/bin/env bash
# SAPO | text | vLLM rollout | Megatron training | Ascend NPU
# SAPO (Smooth Advantage PO) replaces ratio clipping with a smooth tau-parameterized
# surrogate (arXiv:2511.20347).
#
# Qwen3.5 architecture notes:
#   Qwen3.5 uses Gated Delta Net (GDN) linear attention which currently does
#   NOT support packed sequences (THD format) in Megatron-LM. Therefore:
#     - model.use_remove_padding=False           (forces bshd compute format)
#     - actor.megatron.use_remove_padding=False  (forces bshd compute format)
#     - actor.use_dynamic_bsz=False              (required for bshd mode)
#
# Tested parallelism config (8 nodes, 16 NPUs/node, 128 NPUs total):
#   TP=4 PP=2 CP=1 EP=1 ETP=1 GEN_TP=4  →  DP=16
#
# Requirements on Ascend:
#   - Docker image: quay.io/ascend/verl:verl-9.0.0-a3-ubuntu22.04-py3.11-latest
#   - Megatron-LM==0.16.1
#   - MindSpeed==0.16.0
#   - Megatron-Bridge==de93536e
#   - pip install viztracer flash-linear-attention nvidia-modelopt nvidia-ml-py nvidia-resiliency-ext megatron-energon

set -xeuo pipefail

# This script runs on Ascend NPU only.
python3 -c 'import torch_npu' 2>/dev/null || { echo "torch_npu not available. This script requires Ascend NPU." >&2; exit 1; }

ulimit -n 32768
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export TASK_QUEUE_ENABLE=1
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=5400
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_BUFFSIZE=300
export CPU_AFFINITY_CONF=1
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_NZ=0
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.8
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### Quick Config ###########################

# ---- user-adjustable ----
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/dapo-math-17k/train.parquet"}
VAL_FILE=${VAL_FILE:-"${RAY_DATA_HOME}/datasets/aime-2024/test.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME:-verl_sapo_qwen3_5}/${EXPERIMENT_NAME:-qwen3_5_4b_vllm_sapo_megatron}"}
ROLLOUTS_DIR=${ROLLOUTS_DIR:-"${RAY_DATA_HOME}/rollouts/${PROJECT_NAME:-verl_sapo_qwen3_5}/${EXPERIMENT_NAME:-qwen3_5_4b_vllm_sapo_megatron}"}

NNODES=${NNODES:-8}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}

TP=${TP:-4}
PP=${PP:-2}
CP=${CP:-1}
EP=${EP:-1}
ETP=${ETP:-1}
GEN_TP=${GEN_TP:-4}
ACTOR_VPP=${ACTOR_VPP:-null}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

tau_pos=${TAU_POS:-1.0}
tau_neg=${TAU_NEG:-1.05}

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}

# DP = NNODES * NGPUS_PER_NODE / (TP * PP * CP)
train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-20480}
max_response_length=${MAX_RESPONSE_LENGTH:-20480}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-10240}
rollout_n=${ROLLOUT_N:-16}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
vllm_max_model_len=${VLLM_MAX_MODEL_LEN:-40960}

total_epochs=${TOTAL_EPOCHS:-32}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-1000}

project_name=${PROJECT_NAME:-verl_sapo_qwen3_5}
experiment_name=${EXPERIMENT_NAME:-qwen3_5_4b_vllm_sapo_megatron}

use_dynamic_bsz=False
use_remove_padding=False
update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-4096}
# ---- end user-adjustable ----

########################### Parameter Arrays ###########################

DATA=(
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='error'
    data.filter_overlong_prompts=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding}
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo
    +actor_rollout_ref.actor.policy_loss.tau_pos=${tau_pos}
    +actor_rollout_ref.actor.policy_loss.tau_neg=${tau_neg}
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=${vllm_max_model_len}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TP}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${PP}
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.val_before_train=False
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.rollout_data_dir="${ROLLOUTS_DIR}"
    trainer.device=npu
)

EXTRA=(
    model_engine=megatron
    actor_rollout_ref.nccl_timeout=10800
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.actor.checkpoint.strict=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True
)

########################### Launch ###########################

echo "=============================="
echo "Launching SAPO training on $NNODES nodes x $NGPUS_PER_NODE NPUs = $((NNODES * NGPUS_PER_NODE)) NPUs"
echo "  TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP GEN_TP=$GEN_TP"
echo "  DP=$((NNODES * NGPUS_PER_NODE / (TP * PP * CP)))"
echo "  train_batch_size=$train_batch_size, rollout.n=$rollout_n"
echo "  MODEL_PATH=$MODEL_PATH"
echo "=============================="

mkdir -p logs

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee logs/run_qwen3_5_4b_sapo_megatron_npu.log
