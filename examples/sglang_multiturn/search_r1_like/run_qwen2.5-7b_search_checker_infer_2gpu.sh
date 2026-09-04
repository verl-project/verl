#!/bin/bash
set -euo pipefail
set -x

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: $0 <mode> [hydra overrides...]"
    echo "Modes: search_only | checker_guarded | triage_guarded | triage_relaxed_guarded"
    exit 1
fi
shift

# Accept trailing shell-style KEY=VALUE assignments for the common runtime knobs
# so users can write:
#   bash script.sh checker_guarded INPUT_DATA=... OUTPUT_ROOT=... ...
# instead of exporting them before the command.
HYDRA_EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        INPUT_DATA=*|OUTPUT_ROOT=*|MODEL_PATH=*|PROMPT_KEY=*|N_SAMPLES=*|TEMPERATURE=*|TOP_P=*|MAX_PROMPT_LENGTH=*|MAX_RESPONSE_LENGTH=*|MAX_MODEL_LEN=*|GPU_MEM_UTIL=*|MAX_ASSISTANT_TURNS=*|MAX_TOOL_RESPONSE_LENGTH=*|TP_SIZE=*)
            export "$arg"
            ;;
        *)
            HYDRA_EXTRA_ARGS+=("$arg")
            ;;
    esac
done

ulimit -n 65535

module load cuda
unset ROCR_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF
unset TORCH_ALLOC_CONF

export XDG_CACHE_HOME=/ocean/projects/med230010p/yji3/.cache
export HF_HOME=/ocean/projects/med230010p/yji3/.cache/huggingface
export HF_DATASETS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/transformers
export HF_HUB_CACHE=/ocean/projects/med230010p/yji3/.cache/huggingface/hub
export TMPDIR=/ocean/projects/med230010p/yji3/.tmp
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PROJECT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

INPUT_DATA="${INPUT_DATA:-/ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/inference_outputs}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
N_SAMPLES="${N_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2304}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.40}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-4}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-96}"
TP_SIZE="${TP_SIZE:-1}"

mkdir -p "$OUTPUT_ROOT"

function now() {
    date '+%m%d_%H%M%S'
}

STAMP="$(now)"

COMMON_ARGS=(
    trainer.nnodes=1
    trainer.n_gpus_per_node=2
    data.train_files="$INPUT_DATA"
    data.prompt_key="$PROMPT_KEY"
    data.return_raw_chat=True
    actor_rollout_ref.model.path="$MODEL_PATH"
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.skip_tokenizer_init=False
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE"
    actor_rollout_ref.rollout.n="$N_SAMPLES"
    actor_rollout_ref.rollout.temperature="$TEMPERATURE"
    actor_rollout_ref.rollout.top_p="$TOP_P"
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH"
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH"
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
)

case "$MODE" in
    search_only)
        CONFIG_NAME="search_multiturn_grpo"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_tool_config.yaml"
        OUTPUT_PATH="$OUTPUT_ROOT/search_only__$(basename "$INPUT_DATA" .parquet)__${STAMP}.parquet"
        MODE_ARGS=(
            +data.output_path="$OUTPUT_PATH"
            actor_rollout_ref.rollout.multi_turn.format=search_r1
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS"
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
        )
        ;;
    checker_guarded)
        CONFIG_NAME="search_multiturn_grpo_explicitcheck"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_checker_tool_config.yaml"
        OUTPUT_PATH="$OUTPUT_ROOT/checker_guarded__$(basename "$INPUT_DATA" .parquet)__${STAMP}.parquet"
        MODE_ARGS=(
            +data.output_path="$OUTPUT_PATH"
            actor_rollout_ref.rollout.multi_turn.format=search_r1_with_checker
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS"
            actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH"
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
            +actor_rollout_ref.rollout.multi_turn.triage.enable=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.require_search=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.allow_plain_answer=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.min_answer_chars=80
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
        )
        ;;
    triage_guarded)
        CONFIG_NAME="search_multiturn_grpo_explicitcheck"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_checker_tool_config.yaml"
        OUTPUT_PATH="$OUTPUT_ROOT/triage_guarded__$(basename "$INPUT_DATA" .parquet)__${STAMP}.parquet"
        MODE_ARGS=(
            +data.output_path="$OUTPUT_PATH"
            actor_rollout_ref.rollout.multi_turn.format=search_r1_with_checker
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS"
            actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH"
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
            +actor_rollout_ref.rollout.multi_turn.triage.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.online_escalation=True
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_search=1
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_check=1
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_turn=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_search=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_check=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_turn=5
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_search=4
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_check=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_turn=7
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.contradiction_threshold=0.30
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.support_threshold=0.40
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.require_search=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.allow_plain_answer=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.min_answer_chars=80
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
        )
        ;;
    triage_relaxed_guarded)
        CONFIG_NAME="search_multiturn_grpo_explicitcheck"
        TOOL_CONFIG="$CONFIG_PATH/tool_config/medical_search_checker_tool_config.yaml"
        OUTPUT_PATH="$OUTPUT_ROOT/triage_relaxed_guarded__$(basename "$INPUT_DATA" .parquet)__${STAMP}.parquet"
        MODE_ARGS=(
            +data.output_path="$OUTPUT_PATH"
            actor_rollout_ref.rollout.multi_turn.format=search_r1_with_checker
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS"
            actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH"
            actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True
            +actor_rollout_ref.rollout.multi_turn.triage.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.online_escalation=True
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_search=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_check=1
            +actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_turn=4
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_search=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_check=2
            +actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_turn=6
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_search=4
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_check=3
            +actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_turn=7
            +actor_rollout_ref.rollout.multi_turn.triage.heuristic.easy_threshold=0.20
            +actor_rollout_ref.rollout.multi_turn.triage.heuristic.hard_threshold=0.50
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.contradiction_threshold=0.30
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.support_threshold=0.40
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.on_checker_http_error=False
            +actor_rollout_ref.rollout.multi_turn.triage.escalation.reset_counters_on_checker=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.enable=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.require_search=True
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.allow_plain_answer=False
            +actor_rollout_ref.rollout.multi_turn.triage.auto_check.min_answer_chars=80
            actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
        )
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

cd "$PROJECT_DIR"

python3 -m verl.trainer.main_generation_server \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    "${COMMON_ARGS[@]}" \
    "${MODE_ARGS[@]}" \
    "${HYDRA_EXTRA_ARGS[@]}"

echo "Saved generations to: $OUTPUT_PATH"
