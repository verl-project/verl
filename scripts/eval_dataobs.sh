#!/bin/bash
# DataObs 专用评估脚本
# 用法: bash eval_dataobs.sh <checkpoint_path> <base_model> <data_path> <output_path> <gpu_id>

if [ "$#" -lt 5 ]; then
    echo "Usage: bash $0 <checkpoint_path> <base_model> <data_path> <output_path> <gpu_id>"
    echo "Example: bash $0 /data/hrh/COT/GSM8K/training/split_0/global_step_2 /data/pretrain_models/Qwen2.5-0.5B-Instruct /data/open_datasets/GSM8K/test.parquet /data/hrh/COT/GSM8K/eval/split_0 0"
    exit 1
fi

CHECKPOINT_PATH="$1"
BASE_MODEL="$2"
EVAL_DATA="$3"
EVAL_OUTPUT_DIR="$4"
GPU_ID="${5:-0}"

# =========================== Load User Configs =========================
# Find & Load Config File
# Precedence: 1. Env CONFIG_FILE  2. ../config/bash_config.env

# CONFIG_DIR: 使用绝对路径或从环境变量读取
if [ -z "$CONFIG_DIR" ]; then
    CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config"
fi

CONFIG_FILE="${CONFIG_FILE:-${CONFIG_DIR}/bash_config.env}"

if [ -f "$CONFIG_FILE" ]; then
    echo "[INFO] Loading config from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "[WARNING] Config file not found. Using default values."
fi

# fallback
PROJECT_DIR="${PROJECT_DIR:-$HOME/CoT-Data-verl}"
STORE_DIR="${STORE_DIR:-/data/hjw}"
# =======================================================================

# 配置
export CUDA_VISIBLE_DEVICES=$GPU_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

n_gpus_per_node=$(echo $GPU_ID | tr ',' ' ' | wc -w)
tp_size=$n_gpus_per_node

echo "=========================================="
echo "DataObs Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base Model: $BASE_MODEL"
echo "GPU: $GPU_ID"
echo "Config dir: $CONFIG_DIR"
echo "=========================================="

# 检查 checkpoint 是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "[ERROR] Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# 检查 config 目录是否存在
if [ ! -d "$CONFIG_DIR" ]; then
    echo "[ERROR] Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Stage 1: Generation
mkdir -p ${EVAL_OUTPUT_DIR}/{generated,logs}
GENERATION_OUTPUT="${EVAL_OUTPUT_DIR}/generated/responses.parquet"

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Stage 1: Generation"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

# 如果 checkpoint 中有 adapter_model.safetensors，说明是 LoRA 模型，需要指定基础模型
if [ -f "$CHECKPOINT_PATH/adapter_model.safetensors" ]; then
    echo "[INFO] Detected LoRA adapter, using base model: $BASE_MODEL"
    MODEL_PATH="$BASE_MODEL"

    python3 -m verl.trainer.main_generation \
        --config-path=${CONFIG_DIR} \
        --config-name=generation \
        model.path=${MODEL_PATH} \
        +model.lora_path=${CHECKPOINT_PATH} \
        model.no_chat=false \
        data.path=${EVAL_DATA} \
        data.output_path=${GENERATION_OUTPUT} \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.batch_size=32 \
        rollout.temperature=0.6 \
        rollout.seed=42 \
        rollout.prompt_length=512 \
        rollout.response_length=1024 \
        rollout.gpu_memory_utilization=0.8 \
        trainer.n_gpus_per_node=${n_gpus_per_node} \
        trainer.nnodes=1 \
        trainer.device=cuda \
        ray_init.num_cpus=48 \
        2>&1 | tee ${EVAL_OUTPUT_DIR}/logs/generation.log
else
    echo "[INFO] Using checkpoint as full model: $CHECKPOINT_PATH"
    python3 -m verl.trainer.main_generation \
        --config-path=${CONFIG_DIR} \
        --config-name=generation \
        model.path=${CHECKPOINT_PATH} \
        model.no_chat=false \
        data.path=${EVAL_DATA} \
        data.output_path=${GENERATION_OUTPUT} \
        data.prompt_key=prompt \
        data.n_samples=1 \
        data.batch_size=32 \
        rollout.temperature=0.6 \
        rollout.seed=42 \
        rollout.prompt_length=512 \
        rollout.response_length=1024 \
        rollout.gpu_memory_utilization=0.8 \
        trainer.n_gpus_per_node=${n_gpus_per_node} \
        trainer.nnodes=1 \
        trainer.device=cuda \
        ray_init.num_cpus=48 \
        2>&1 | tee ${EVAL_OUTPUT_DIR}/logs/generation.log
fi

if [ ${PIPESTATUS[0]} -ne 0 ] || [ ! -f "${GENERATION_OUTPUT}" ]; then
    echo "[ERROR] Generation Failed!"
    exit 1
fi

echo "[INFO] Generation Done!"

# Stage 2: Evaluation
EVALUATION_OUTPUT="${EVAL_OUTPUT_DIR}/generated/responses_labeled.json"
REWARD_FUNCTION_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/verl/utils/reward_score/gsm8k.py"

echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo "Stage 2: Evaluation"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

python3 -m verl.trainer.main_eval \
    --config-path=${CONFIG_DIR} \
    --config-name=evaluation \
    data.path=${GENERATION_OUTPUT} \
    data.output_path=${EVALUATION_OUTPUT} \
    data.response_key=responses \
    data.data_source_key=data_source \
    data.reward_model_key=reward_model \
    custom_reward_function.path=${REWARD_FUNCTION_PATH} \
    ray_init.num_cpus=48 \
    2>&1 | tee ${EVAL_OUTPUT_DIR}/logs/evaluation.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] Evaluation Failed!"
    exit 1
fi

echo "[INFO] Evaluation Done!"

# 提取准确率
echo ""
echo "=========================================="
echo "Evaluation Results:"
echo "=========================================="
grep -E "(test_score|pass@|accuracy|reward)" ${EVAL_OUTPUT_DIR}/logs/evaluation.log | tail -10 || echo "See ${EVAL_OUTPUT_DIR}/logs/evaluation.log for details"

echo ""
echo "Output directory: ${EVAL_OUTPUT_DIR}"

