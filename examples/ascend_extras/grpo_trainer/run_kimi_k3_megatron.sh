#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

# Kimi-K3 CountBench GRPO: vLLM-Ascend rollout + Megatron/MindSpeed actor/ref.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND_ROOT=${BACKEND_ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)}
RUN_MODE=${KIMI_RUN_MODE:-single}
RUN_TIMESTAMP=${VERL_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
MODEL_PATH=${MODEL_PATH:-/mnt/share/w00848461/weights/K3-top16-pruned_bf16_countbench_128_n4}
NUM_HIDDEN_LAYERS=$(python3 - "$MODEL_PATH/config.json" <<'PYCONFIG'
import json
import sys
with open(sys.argv[1]) as stream:
    config = json.load(stream)
print(config.get("text_config", config)["num_hidden_layers"])
PYCONFIG
)
TRAIN_FILE=${TRAIN_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train_448.parquet}
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-}
MODELOPT_SITE=${MODELOPT_SITE:-${BACKEND_ROOT}/.python_deps/modelopt-0.46.0}
VLLM_METADATA_SITE=${VLLM_METADATA_SITE:-}
VLLM_SOURCE=${VLLM_SOURCE:-${BACKEND_ROOT}/vllm}
VLLM_ASCEND_SOURCE=${VLLM_ASCEND_SOURCE:-${BACKEND_ROOT}/vllm-ascend}
CANN_HOME=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
CANN_PYTHON_SITE=${CANN_PYTHON_SITE:-${CANN_HOME}/python/site-packages}
CANN_TBE_SITE=${CANN_TBE_SITE:-${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe}

SAVE_FREQ=${SAVE_FREQ:--1}
OPTIMIZER_CPU_OFFLOAD=${OPTIMIZER_CPU_OFFLOAD:-True}
OPTIMIZER_STORE_PARAM_REMAINDERS=${OPTIMIZER_STORE_PARAM_REMAINDERS:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
ACTOR_RECOMPUTE=${ACTOR_RECOMPUTE:-True}
ACTOR_TP=${ACTOR_TP:-4}
ACTOR_CP=${ACTOR_CP:-1}

case "$RUN_MODE" in
    single)
        NNODES=1
        NPUS_PER_NODE=${NPUS_PER_NODE:-8}
        ACTOR_PP=${ACTOR_PP:-1}
        ACTOR_EP=${ACTOR_EP:-2}
        ACTOR_ETP=${ACTOR_ETP:-1}
        ROLLOUT_TP=${ROLLOUT_TP:-8}
        ROLLOUT_EP=${ROLLOUT_EP:-8}
        VISIBLE_DEVICES_DEFAULT=0,1,2,3,4,5,6,7
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-kimi_k3_megatron_1node_${RUN_TIMESTAMP}}
        ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
        LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
        KIMI_KDA_DISABLE_INTERNAL_RECOMPUTE=${KIMI_KDA_DISABLE_INTERNAL_RECOMPUTE:-False}
        ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
        ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
        ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
        ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
        TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-8}
        ;;
    multinode)
        : "${NNODES:?NNODES must be set by the multi-node Ray entry}"
        if ((NNODES < 2)); then
            echo "multinode mode requires NNODES >= 2, got $NNODES" >&2
            exit 2
        fi
        NPUS_PER_NODE=${NPUS_PER_NODE:-16}
        # PP4 limits parameter residency; EP8/ETP2 avoids one-group GroupedLinear.
        ACTOR_PP=${ACTOR_PP:-4}
        ACTOR_EP=${ACTOR_EP:-8}
        ACTOR_ETP=${ACTOR_ETP:-2}
        ROLLOUT_TP=${ROLLOUT_TP:-16}
        ROLLOUT_EP=${ROLLOUT_EP:-16}
        VISIBLE_DEVICES_DEFAULT=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-kimi_k3_megatron_${NNODES}nodes_${RUN_TIMESTAMP}}
        ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-2}
        LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
        KIMI_KDA_DISABLE_INTERNAL_RECOMPUTE=${KIMI_KDA_DISABLE_INTERNAL_RECOMPUTE:-True}
        ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
        ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
        ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
        if ((ACTOR_PP >= 4)); then
            ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}
        else
            ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.38}
        fi
        TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-$((NNODES * 8))}
        ;;
    *)
        echo "unsupported KIMI_RUN_MODE: $RUN_MODE (expected single or multinode)" >&2
        exit 2
        ;;
esac

TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=${TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT:-600}
TQ_NUM_THREADS=${TQ_NUM_THREADS:-8}
# Keep the 2240-MiB embedding/head tensor in one reusable IPC bucket.
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES:-4096}
for positive_integer in ACTOR_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU TQ_STORAGE_UNITS TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT TQ_NUM_THREADS ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES; do
    if [[ ! "${!positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$positive_integer must be a positive integer, got ${!positive_integer}" >&2
        exit 2
    fi
done
if [[ "${OPTIMIZER_CPU_OFFLOAD,,}" != true && "${OPTIMIZER_STORE_PARAM_REMAINDERS,,}" == true ]]; then
    echo "NPU FusedAdam requires OPTIMIZER_STORE_PARAM_REMAINDERS=False" >&2
    exit 2
fi
if [[ "${OPTIMIZER_CPU_OFFLOAD,,}" == true ]]; then
    OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
else
    OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-0.0}
fi
ROLLOUT_CUDAGRAPH_MODE=${ROLLOUT_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
ROLLOUT_CUDAGRAPH_CAPTURE_SIZES=${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-'[1,2,4,8,16]'}
ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE=${ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
ROLLOUT_SLEEP_EXTRA_CLEANUP=${ROLLOUT_SLEEP_EXTRA_CLEANUP:-False}
if ((ACTOR_PP > 1)); then
    pipeline_base_layers=$((NUM_HIDDEN_LAYERS / ACTOR_PP))
    PIPELINE_FIRST_LAYERS=${PIPELINE_FIRST_LAYERS:-$pipeline_base_layers}
    PIPELINE_LAST_LAYERS=${PIPELINE_LAST_LAYERS:-$((
        NUM_HIDDEN_LAYERS - pipeline_base_layers * (ACTOR_PP - 1)
    ))}
fi

ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_VISUAL_TOKENS=${MAX_VISUAL_TOKENS:-1024}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-4096}

PROJECT_NAME=${PROJECT_NAME:-verl_kimi_k3}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BACKEND_ROOT}/verl/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${BACKEND_ROOT}/verl/logs}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}

required_mm_length=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH - 1 + MAX_VISUAL_TOKENS))
if ((MAX_MODEL_LENGTH < required_mm_length)); then
    echo "MAX_MODEL_LENGTH=$MAX_MODEL_LENGTH is smaller than the multimodal budget $required_mm_length" >&2
    exit 2
fi

world_size=$((NNODES * NPUS_PER_NODE))
dense_parallel_size=$((ACTOR_TP * ACTOR_PP * ACTOR_CP))
expert_parallel_size=$((ACTOR_ETP * ACTOR_EP * ACTOR_PP))
if ((world_size % dense_parallel_size != 0 || world_size % expert_parallel_size != 0)); then
    echo "invalid Megatron topology: world=$world_size dense=$dense_parallel_size expert=$expert_parallel_size" >&2
    exit 2
fi
if ((ROLLOUT_EP % ROLLOUT_TP != 0)); then
    echo "rollout EP=$ROLLOUT_EP must be divisible by TP=$ROLLOUT_TP" >&2
    exit 2
fi
ROLLOUT_DP=${ROLLOUT_DP:-$((ROLLOUT_EP / ROLLOUT_TP))}
rollout_world_size=$((ROLLOUT_TP * ROLLOUT_DP))
if ((ROLLOUT_EP != rollout_world_size || world_size % rollout_world_size != 0)); then
    echo "invalid rollout topology: world=$world_size TP=$ROLLOUT_TP DP=$ROLLOUT_DP EP=$ROLLOUT_EP" >&2
    exit 2
fi
if ((ACTOR_PP > 1)); then
    middle_pipeline_stages=$((ACTOR_PP - 2))
    middle_pipeline_layers=$((NUM_HIDDEN_LAYERS - PIPELINE_FIRST_LAYERS - PIPELINE_LAST_LAYERS))
    if ((middle_pipeline_layers < 0)); then
        echo "pipeline first/last layers exceed $NUM_HIDDEN_LAYERS" >&2
        exit 2
    fi
    if ((middle_pipeline_stages == 0)); then
        if ((middle_pipeline_layers != 0)); then
            echo "PP=2 pipeline layers must add up to $NUM_HIDDEN_LAYERS" >&2
            exit 2
        fi
    elif ((middle_pipeline_layers % middle_pipeline_stages != 0)); then
        echo "the $middle_pipeline_layers middle layers cannot be divided across $middle_pipeline_stages pipeline stages" >&2
        exit 2
    fi
fi
dense_dp=$((world_size / dense_parallel_size))
if ((TRAIN_BATCH_SIZE % dense_dp != 0 || PPO_MINI_BATCH_SIZE % dense_dp != 0)); then
    echo "batch sizes must be divisible by dense DP=$dense_dp" >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$VISIBLE_DEVICES_DEFAULT}
unset NON_MEGATRON || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT
export TQ_NUM_THREADS
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export MINDSPEED_BRIDGE_AUTOREG_MODE=${MINDSPEED_BRIDGE_AUTOREG_MODE:-off}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp48s3u1u1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-17340}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_VERSION=${VLLM_VERSION:-0.31.1.dev0}
export VLLM_ASCEND_ENABLE_FLASHCOMM=${VLLM_ASCEND_ENABLE_FLASHCOMM:-1}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
KIMI_USE_FUSED_KDA=${KIMI_USE_FUSED_KDA:-True}
KIMI_VLLM_USE_TRAINING_CAUSAL_CONV1D=${KIMI_VLLM_USE_TRAINING_CAUSAL_CONV1D:-True}
KIMI_KDA_OPROJ_FP32_REDUCE=${KIMI_KDA_OPROJ_FP32_REDUCE:-True}
export KIMI_RUN_TOKEN=${KIMI_RUN_TOKEN:-$(cat /proc/sys/kernel/random/uuid)}

source_roots=(
    "$TRANSFORMERS_SITE"
    "$MODELOPT_SITE"
    "$VLLM_METADATA_SITE"
    "$VLLM_SOURCE"
    "$VLLM_ASCEND_SOURCE"
    "$BACKEND_ROOT/verl"
    "$BACKEND_ROOT/MS-Bridge-KIMI-K3"
    "$BACKEND_ROOT/Megatron-Bridge/src"
    "$BACKEND_ROOT/Megatron-LM-KIMI-K3"
    "$BACKEND_ROOT/MegatronAdaptor"
    "$BACKEND_ROOT/TransformerEngineNPU"
    "$BACKEND_ROOT/MindSpeed-Ops"
    "$BACKEND_ROOT/.runtime-site"
    "$CANN_PYTHON_SITE"
    "$CANN_TBE_SITE"
)
joined_pythonpath=""
for source_root in "${source_roots[@]}"; do
    if [[ -n "$source_root" ]]; then
        joined_pythonpath+="${joined_pythonpath:+:}$source_root"
    fi
done
export PYTHONPATH="$joined_pythonpath"

pipeline_summary=""
if ((ACTOR_PP > 1)); then
    if ((middle_pipeline_stages == 0)); then
        pipeline_summary=" pipeline_layers=${PIPELINE_FIRST_LAYERS}+${PIPELINE_LAST_LAYERS}"
    else
        pipeline_layers_per_middle_stage=$((middle_pipeline_layers / middle_pipeline_stages))
        pipeline_summary=" pipeline_layers=${PIPELINE_FIRST_LAYERS}+${pipeline_layers_per_middle_stage}x${middle_pipeline_stages}+${PIPELINE_LAST_LAYERS}"
    fi
fi
echo "run=$RUN_MODE model=$MODEL_PATH world=$world_size TP=$ACTOR_TP PP=$ACTOR_PP EP=$ACTOR_EP ETP=$ACTOR_ETP rollout_TP=$ROLLOUT_TP rollout_DP=$ROLLOUT_DP rollout_EP=$ROLLOUT_EP${pipeline_summary} steps=$TOTAL_TRAINING_STEPS"
echo "optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION store_param_remainders=$OPTIMIZER_STORE_PARAM_REMAINDERS actor_param_offload=$ACTOR_PARAM_OFFLOAD actor_optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD ref_param_offload=$REF_PARAM_OFFLOAD recompute=$ACTOR_RECOMPUTE enforce_eager=$ROLLOUT_ENFORCE_EAGER rollout_gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION update_weights_bucket_mb=$ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES cudagraph_mode=$ROLLOUT_CUDAGRAPH_MODE cudagraph_sizes=$ROLLOUT_CUDAGRAPH_CAPTURE_SIZES cudagraph_max=$ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE sleep_extra_cleanup=$ROLLOUT_SLEEP_EXTRA_CLEANUP"
echo "micro_batch_actor=$ACTOR_MICRO_BATCH_SIZE_PER_GPU micro_batch_log_prob=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
echo "training_kda=fused:$KIMI_USE_FUSED_KDA internal_recompute_disabled=$KIMI_KDA_DISABLE_INTERNAL_RECOMPUTE dist_checkpoint=${ACTOR_USE_DIST_CHECKPOINTING:-True} resume_mode=${RESUME_MODE:-disable}"
echo "transfer_queue_storage_units=$TQ_STORAGE_UNITS transfer_queue_timeout_s=$TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT transfer_queue_threads_per_unit=$TQ_NUM_THREADS"
DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    # Kimi-K3's remote-code tokenizer retains an SSLContext, so even
    # datasets' one-worker pool cannot pickle it.
    # A null num_proc keeps this lightweight filter in the Ray actor process.
    data.filter_overlong_prompts_workers=null
    data.shuffle=False
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.external_lib=mindspeed_bridge.runtime.kimi_k3_autoload
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
)

ACTOR=(
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}"
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16
    actor_rollout_ref.actor.optim.exp_avg_dtype=bf16
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=bf16
    "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=$OPTIMIZER_CPU_OFFLOAD"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.store_param_remainders=$OPTIMIZER_STORE_PARAM_REMAINDERS"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.megatron.router_replay.mode=R3
    "actor_rollout_ref.actor.megatron.use_dist_checkpointing=${ACTOR_USE_DIST_CHECKPOINTING:-False}"
    "actor_rollout_ref.actor.megatron.param_offload=$ACTOR_PARAM_OFFLOAD"
    "actor_rollout_ref.actor.megatron.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD"
)

if [[ "${ACTOR_RECOMPUTE,,}" == true ]]; then
    ACTOR+=(
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
        actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    )
fi

REF=(
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.ref.use_torch_compile=False
    "actor_rollout_ref.ref.megatron.param_offload=$REF_PARAM_OFFLOAD"
)

if ((ACTOR_PP > 1)); then
    ACTOR+=(
        "+actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=$PIPELINE_FIRST_LAYERS"
        "+actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=$PIPELINE_LAST_LAYERS"
    )
    REF+=(
        "++actor_rollout_ref.ref.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=$PIPELINE_FIRST_LAYERS"
        "++actor_rollout_ref.ref.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=$PIPELINE_LAST_LAYERS"
    )
fi

BRIDGE=()
for role in actor ref; do
    BRIDGE+=(
        "actor_rollout_ref.$role.megatron.use_mbridge=True"
        "actor_rollout_ref.$role.megatron.use_remove_padding=False"
        "actor_rollout_ref.$role.megatron.tensor_model_parallel_size=$ACTOR_TP"
        "actor_rollout_ref.$role.megatron.pipeline_model_parallel_size=$ACTOR_PP"
        "actor_rollout_ref.$role.megatron.context_parallel_size=$ACTOR_CP"
        "actor_rollout_ref.$role.megatron.expert_model_parallel_size=$ACTOR_EP"
        "actor_rollout_ref.$role.megatron.expert_tensor_parallel_size=$ACTOR_ETP"
        "actor_rollout_ref.$role.megatron.sequence_parallel=True"
        "++actor_rollout_ref.$role.megatron.override_transformer_config.attention_backend=auto"
        "++actor_rollout_ref.$role.megatron.override_transformer_config.seq_length=$MAX_MODEL_LENGTH"
        "++actor_rollout_ref.$role.megatron.override_transformer_config.kimi_use_fused_kda=$KIMI_USE_FUSED_KDA"
        "++actor_rollout_ref.$role.megatron.override_transformer_config.kimi_kda_disable_internal_recompute=$KIMI_KDA_DISABLE_INTERNAL_RECOMPUTE"
        "++actor_rollout_ref.$role.megatron.override_transformer_config.kimi_kda_oproj_fp32_reduce=$KIMI_KDA_OPROJ_FP32_REDUCE"
        "++actor_rollout_ref.$role.megatron.override_transformer_config.kimi_use_fused_short_conv=True"
    )
done

ROLLOUT=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.enabled=True
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
    "actor_rollout_ref.rollout.data_parallel_size=$ROLLOUT_DP"
    "actor_rollout_ref.rollout.expert_parallel_size=$ROLLOUT_EP"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    actor_rollout_ref.rollout.top_p=1.0
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LENGTH"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS:-2048}"
    actor_rollout_ref.rollout.enable_prefix_caching=False
    "actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER"
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=$ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="$ROLLOUT_CUDAGRAPH_MODE"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="$ROLLOUT_CUDAGRAPH_CAPTURE_SIZES"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=$ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.rl_config.sleep_mode_extra_cleanup=$ROLLOUT_SLEEP_EXTRA_CLEANUP"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_training_causal_conv1d=$KIMI_VLLM_USE_TRAINING_CAUSAL_CONV1D"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_kda_oproj_fp32_reduce=$KIMI_KDA_OPROJ_FP32_REDUCE"
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    "trainer.nnodes=$NNODES"
    "trainer.n_gpus_per_node=$NPUS_PER_NODE"
    trainer.balance_batch=False
    trainer.val_before_train=False
    "trainer.resume_mode=${RESUME_MODE:-disable}"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    "trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-1}"
    "trainer.test_freq=${TEST_FREQ:--1}"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
    trainer.default_local_dir="$OUTPUT_ROOT"
)
if [[ -n "${RESUME_FROM_PATH:-}" ]]; then
    TRAINER+=(trainer.resume_from_path="$RESUME_FROM_PATH")
fi

TRANSFER_QUEUE=(
    "transfer_queue.backend.SimpleStorage.num_data_storage_units=$TQ_STORAGE_UNITS"
)

RAY=(++ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=mindspeed_bridge.runtime.kimi_k3.setup_kimi_k3_runtime)
RAY+=('++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_AUTO_DETECT_QUANTIZATION="1"')
ray_env_vars=(
    TMPDIR XDG_CACHE_HOME HF_HOME HF_MODULES_CACHE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR PYTHONDONTWRITEBYTECODE
    ASCEND_RT_VISIBLE_DEVICES PYTHONPATH OMP_NUM_THREADS PYTHONHASHSEED
    TOKENIZERS_PARALLELISM TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT TQ_NUM_THREADS
    MINDSPEED_BRIDGE_AUTOREG_MODE
    VLLM_USE_V1 VLLM_VERSION VLLM_ASCEND_ENABLE_FLASHCOMM
    VLLM_ASCEND_ENABLE_NZ VLLM_DISABLE_COMPILE_CACHE
    CUDA_DEVICE_MAX_CONNECTIONS KIMI_RUN_TOKEN
    HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME HCCL_HOST_SOCKET_PORT_RANGE
    HCCL_NPU_SOCKET_PORT_RANGE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT
)
for name in "${ray_env_vars[@]}"; do
    [[ -v "$name" ]] || continue
    RAY+=("++ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\"")
done

KIMI_RAY_ARGS=()
if [[ "$RUN_MODE" == single ]]; then
    KIMI_RAY_ARGS=(
        '++ray_kwargs.ray_init.address=local'
        "++ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-64}"
        "++ray_kwargs.ray_init.resources.NPU=$NPUS_PER_NODE"
        "++ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY:-8589934592}"
    )
else
    : "${RAY_ADDRESS:?multinode mode must be started by ray_start_multi_nodes.sh}"
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
log_file="$LOG_DIR/${EXPERIMENT_NAME}.log"
echo "log=$log_file output=$OUTPUT_ROOT"
cd "$BACKEND_ROOT/verl"

python3 -c 'import runpy; from mindspeed_bridge.runtime.kimi_k3 import setup_kimi_k3_runtime; setup_kimi_k3_runtime(driver=True); runpy.run_module("verl.trainer.main_ppo", run_name="__main__", alter_sys=True)' \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${BRIDGE[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${TRANSFER_QUEUE[@]}" \
    "${RAY[@]}" \
    "${KIMI_RAY_ARGS[@]}" \
    "$@" 2>&1 | tee "$log_file"
