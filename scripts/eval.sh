#!/bin/bash
# set -x

# ============================ Configurations ===========================
PROJECT_DIR="$HOME/CoT-Data-verl"   # !!!!! Change this to where you want to save checkpoints & logs!!!!!
MODEL_NAME="Qwen2.5-Coder-7B"
CONFIG_DIR=$(realpath "../config")

# Model Path Config:
# 1. Use specific checkpoints if specified by CLI param / environment variable.
# 2. Otherwise use the cached base model.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
BASE_MODEL="Qwen/Qwen2.5-Coder-7B"
IS_BASE_MODEL=False                       # Whether the model is a base model (without a chat template)

# Inference Params
N_SAMPLES="${N_SAMPLES:-1}"           # 采样次数（pass@n 评估时可设为 n）
TEMPERATURE="${TEMPERATURE:-0.6}"     # 采样温度
TOP_P="${TOP_P:-0.95}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Set this to 1 and specify output directory to skip generation (Testing)
SKIP_GEN="${SKIP_GEN:-0}"
TARGET_EVAL_DIR="${TARGET_EVAL_DIR:-}"

# Control seed for generation
SEED="${SEED:-42}"
# =======================================================================


# ===================== Function: Latest Gen Lookup =====================
find_latest_eval_dir() {
    local search_pattern="${MODEL_NAME}--${DATA_NAME}--eval--*"
    local latest_dir
    
    # 按目录名字符串排序（时间戳在末尾，字典序即时间序）
    latest_dir=$(find "${PROJECT_DIR}/evals" -maxdepth 1 -type d -name "${search_pattern}" 2>/dev/null | sort | tail -n 1)
    
    if [ -z "$latest_dir" ]; then
        echo "[ERROR] No existing evaluation found for ${MODEL_NAME}/${DATA_NAME}" >&2
        return 1
    fi
    
    # 验证 parquet 文件存在
    if [ ! -f "${latest_dir}/generated/responses.parquet" ]; then
        echo "[ERROR] Found ${latest_dir} but missing generated/responses.parquet" >&2
        return 1
    fi
    
    echo "$latest_dir"
}
# =======================================================================


# ============================== 参数解析 ================================
if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 <dataset_name> <gpu_ids> [checkpoint_path] [other configs...]"
    echo "Supported datasets: arc-challenge, aqua_rat, gsm8k, livecodebench, math, math-500, numinamath, strategyQA, theoremQA"
    echo ""
    echo "Examples:"
    echo "  # Evaluate the base model on gsm8k"
    echo "  bash $0 gsm8k 0,1,2,3"
    echo ""
    echo "  # Evaluate with a specific checkpoint (after PPO RL training)"
    echo "  bash $0 livecodebench 0,1,2,3,4,5,6,7 $PROJECT_DIR/outputs/Qwen2.5-0.5B-gsm8k-ppo/checkpoint-last"
    echo ""
    echo "  # Pass@16 evaluation"
    echo "  N_SAMPLES=16 bash $0 MATH 0,1,2,3 /path/to/checkpoint"
    exit 1
fi

DATASET=$1
shift 1
shopt -s nocasematch    # Enable caseless match
case $DATASET in
    "ai2_arc" | "ai2-arc" | "arc-challenge")
        DATA_NAME="arc-challenge"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/multiple_choice.py")
        EVAL_DATA="/data/open_datasets/ai2_arc/ARC-Challenge/test-processed.parquet"
        
        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for ARC-Challenge: Success!"
        ;;
    "aqua_rat" | "aqua-rat")
        DATA_NAME="aqua_rat"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/multiple_choice.py")
        EVAL_DATA="/data/open_datasets/aqua_rat/processed/test-processed.parquet"
        
        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for AQuA-RAT: Success!"
        ;;
    "gsm8k")
        DATA_NAME="gsm8k"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/gsm8k.py")
        EVAL_DATA="/data/open_datasets/GSM8K/test.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for GSM8K: Success!"
        ;;
    "livecodebench")
        DATA_NAME="gsm8k"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/gsm8k.py")
        EVAL_DATA="/data/open_datasets/livecodebench/..."

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for LiveCodeBench: Success!"
        ;;
    "math")
        DATA_NAME="math"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/math_verify.py")
        EVAL_DATA="/data/open_datasets/MATH/train_processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for MATH: Success!"
        ;;
    "math-500")
        DATA_NAME="math-500"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/math_verify.py")
        EVAL_DATA="/data/open_datasets/MATH-500/test-processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for MATH-500: Success!"
        ;;
    "numinamath" | "numinamath-CoT")
        DATA_NAME="numinamath"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/math.py")
        EVAL_DATA="/data/open_datasets/NuminaMath-CoT/test-processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for NuminaMath-CoT: Success!"
        ;;
    "strategyQA")
        DATA_NAME="strategyQA"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/truefalse.py")
        EVAL_DATA="/data/open_datasets/StrategyQA/data/test-processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for StrategyQA: Success!"
        ;;
    "theoremQA")
        DATA_NAME="gsm8k"
        REWARD_FUNCTION_PATH=$(realpath "../verl/utils/reward_score/gsm8k.py")
        EVAL_DATA="/data/open_datasets/TheoremQA/..."

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for TheoremQA: Success!"
        ;;
    *)
        # Default: unknown dataset
        echo "[ERROR] Unsupported dataset $DATASET."
        echo "Supported datasets: ai2_arc, aqua_rat, gsm8k, livecodebench, math, math-500, numinamath, strategyQA, theoremQA"
        exit 1
        ;;
esac
shopt -u nocasematch    # Disable caseless match

gpu_ids=$1
shift 1

# 确定实际使用的模型路径（优先级：命令行参数 > 环境变量 > Base model）
if [ $# -gt 0 ] && [[ "$1" == */* ]]; then
    MODEL_PATH="$1"
    echo "[INFO] Loading model from CLI param: $MODEL_PATH"
    shift 1
elif [ -n "$CHECKPOINT_PATH" ] && [ -d "$CHECKPOINT_PATH" ]; then
    MODEL_PATH="$CHECKPOINT_PATH"
    echo "[INFO] Loading model checkpoint from set environment variable: $MODEL_PATH"
else
    MODEL_PATH="$BASE_MODEL"
    echo "[INFO] Loading base model: $MODEL_PATH"
fi

export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpus_per_node=$(echo $gpu_ids | tr ',' ' ' | wc -w)

# 检查 tensor_model_parallel_size 兼容性
tp_size=3
if [ $((n_gpus_per_node % tp_size)) -ne 0 ]; then
    echo "[WARNING]: n_gpus_per_node($n_gpus_per_node) is not divisible by tensor_model_parallel_size($tp_size). It is advised to set tensor_model_parallel_size to a devisor of $n_gpus_per_node (e.g. 1, $n_gpus_per_node)."
    echo "[INFO] Fallback: Setting tp_size to 1..."
    tp_size=1  # 自动回退到 1 避免错误
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline


# ============================== Generation Skip =============================
if [ "$SKIP_GEN" = "1" ]; then
    if [ -n "$TARGET_EVAL_DIR" ]; then
        echo "[INFO] SKIP_GEN=1, using designated TARGET_EVAL_DIR: ${TARGET_EVAL_DIR}..."
        EVAL_OUTPUT_DIR=${TARGET_EVAL_DIR}
        if [ ! -f "${EVAL_OUTPUT_DIR}/generated/responses.parquet" ]; then
            echo "[ERROR] Designated dir missing responses.parquet: ${EVAL_OUTPUT_DIR}"
            exit 1
        fi
    else
        echo "[INFO] SKIP_GEN=1, searching for latest existing results..."
        EVAL_OUTPUT_DIR=$(find_latest_eval_dir) || exit 1
    fi
    
    echo "[INFO] Reusing: ${EVAL_OUTPUT_DIR}"
    GENERATION_OUTPUT="${EVAL_OUTPUT_DIR}/generated/responses.parquet"
else
    # Output directory for eval
    EVAL_OUTPUT_DIR="${PROJECT_DIR}/evals/${MODEL_NAME}--${DATA_NAME}--eval--$(date +%m%d-%H%M%S)"
    mkdir -p ${EVAL_OUTPUT_DIR}/{generated,logs}


    # ========================== Stage 1: Generation =========================
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo "Stage 1: Generation (Generate Responses)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

    GENERATION_OUTPUT="${EVAL_OUTPUT_DIR}/generated/responses.parquet"

    GEN_ARGS="\
    model.path=${MODEL_PATH} \
    model.no_chat=${IS_BASE_MODEL} \
    data.path=${EVAL_DATA} \
    data.output_path=${GENERATION_OUTPUT} \
    data.prompt_key=${PROMPT_KEY} \
    data.n_samples=${N_SAMPLES} \
    data.batch_size=${BATCH_SIZE} \
    rollout.temperature=${TEMPERATURE} \
    rollout.seed=${SEED} \
    rollout.prompt_length=${MAX_PROMPT_LEN} \
    rollout.response_length=${MAX_RESPONSE_LEN} \
    rollout.gpu_memory_utilization=0.8 \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=1 \
    trainer.device=cuda \
    ray_init.num_cpus=48"

    # 添加额外参数（如 temperature=0.0 等）
    if [ $# -gt 0 ]; then
        echo "[INFO] additional gen args: $@"
        GEN_ARGS="${GEN_ARGS} $@"
    fi

    echo "=========================================="
    echo "Generation Setup:"
    echo "  GPUs: $gpu_ids (count: $n_gpus_per_node)"
    echo "  Model Path: $MODEL_PATH"
    echo "  Evaluation Data: $EVAL_DATA"
    echo "  Output Path: $EVAL_OUTPUT_DIR"
    echo "  Sampling Params: n=$N_SAMPLES, temperature=$TEMPERATURE"
    echo "=========================================="

    echo "[INFO] Begin generation..."
    python3 -m verl.trainer.main_generation \
        --config-path=${CONFIG_DIR} \
        --config-name=generation \
        ${GEN_ARGS} \
        2>&1 | tee ${EVAL_OUTPUT_DIR}/logs/generation.log

    if [ ${PIPESTATUS[0]} -ne 0 ] || [ ! -f "${GENERATION_OUTPUT}" ]; then
        echo "[ERROR] Generation Failed!"
        exit 1
    fi

    echo "[INFO] Generation Done!"

fi

# ============================ Stage 2: Evaluation ===========================
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Stage 2: Evaluation (Verify Responses)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

EVALUATION_OUTPUT="${EVAL_OUTPUT_DIR}/generated/responses_labeled.json"

EVAL_ARGS="\
data.path=${GENERATION_OUTPUT} \
data.output_path=${EVALUATION_OUTPUT} \
data.response_key=responses \
data.data_source_key=${DATA_SOURCE_KEY} \
data.reward_model_key=${REWARD_MODEL_KEY} \
custom_reward_function.path=${REWARD_FUNCTION_PATH} \
ray_init.num_cpus=48"

echo "=========================================="
echo "Evaluation Setup:"
echo "  Input File: ${GENERATION_OUTPUT}"
echo "  Response Column: responses"
echo "  Ground Truth Column: ${REWARD_MODEL_KEY}.ground_truth"
echo "=========================================="

echo "[INFO] Begin evaluation..."
python3 -m verl.trainer.main_eval \
    --config-path=${CONFIG_DIR} \
    --config-name=evaluation \
    ${EVAL_ARGS} \
    2>&1 | tee ${EVAL_OUTPUT_DIR}/logs/evaluation.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Evaluation Failed!"
    exit 1
fi

echo "[INFO] Evaluation Done!"

# ============================ 结果汇总 ===========================
echo ""
echo "=========================================="
echo "Evaluation Successful!"
echo "=========================================="
echo "Output Path: ${EVAL_OUTPUT_DIR}"
echo ""
echo "Folder Structure:"
echo "  ${EVAL_OUTPUT_DIR}/"
echo "  ├── generated/"
echo "  |   ├── responses.parquet        # 包含 responses 列"
echo "  |   └── responses_labeled.json   # 包含 response 的对错评估结果"
echo "  └── logs/"
echo "      ├── generation.log           # 生成日志"
echo "      └── evaluation.log           # 评估日志与指标"
echo ""
echo "查看指标:"
grep -E "(test_score|pass@|accuracy|reward)" ${EVAL_OUTPUT_DIR}/logs/evaluation.log 2>/dev/null | tail -5 || echo "请查看 ${EVAL_OUTPUT_DIR}/logs/evaluation.log"