#!/bin/bash
# Quick verification script - check all paths before running experiments

set -e

# Load config from run_experiment.sh (source the defaults)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  SPEC-RL Dynamic Spec Setup Verification"
echo "============================================================"
echo ""

# Use the same defaults as run_experiment.sh
WORKING_DIR="${WORKING_DIR:-/home/lingquh1xx/L2598/Temp/Spec-RL}"
MODEL_PATH="${MODEL_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/model}"
MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B-base}"
DATA_PATH="${DATA_PATH:-/home/lingquh1xx/L2598/Temp/Spec-RL/data}"
DATASET_NAME="${DATASET_NAME:-deepmath}"
TRAIN_FILE="${TRAIN_FILE:-train_sample_6144}"
NUM_GPU="${NUM_GPU:-8}"

echo "Configuration:"
echo "  WORKING_DIR: ${WORKING_DIR}"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  MODEL_NAME: ${MODEL_NAME}"
echo "  DATA_PATH: ${DATA_PATH}"
echo "  DATASET_NAME: ${DATASET_NAME}"
echo "  TRAIN_FILE: ${TRAIN_FILE}"
echo "  NUM_GPU: ${NUM_GPU}"
echo ""

# Check each path
errors=0

check_path() {
    local label="$1"
    local path="$2"
    local required="$3"  # "file" or "dir" or "any"
    
    if [ -e "${path}" ]; then
        if [ -f "${path}" ] && [ "${required}" = "dir" ]; then
            echo "  [WARN] ${label}: ${path} exists but is a file (expected directory)"
        elif [ -d "${path}" ] && [ "${required}" = "file" ]; then
            echo "  [WARN] ${label}: ${path} exists but is a directory (expected file)"
        else
            echo "  [OK] ${label}: ${path}"
        fi
    else
        echo "  [FAIL] ${label}: ${path} NOT FOUND"
        ((errors++))
    fi
}

echo "Checking paths..."
echo ""

# Model
check_path "Model dir" "${MODEL_PATH}/${MODEL_NAME}" "dir"

# Check model files exist
if [ -d "${MODEL_PATH}/${MODEL_NAME}" ]; then
    echo ""
    echo "  Model files:"
    for f in config.json tokenizer.json tokenizer_config.json model.safetensors index.json; do
        if [ -f "${MODEL_PATH}/${MODEL_NAME}/${f}" ]; then
            echo "    [OK] ${f}"
        elif [ -f "${MODEL_PATH}/${MODEL_NAME}/model-${f}" ]; then
            echo "    [OK] model-${f}"
        else
            echo "    [INFO] ${f}: not found (may use different name)"
        fi
    done
    echo ""
fi

# Data
check_path "Train data" "${DATA_PATH}/${DATASET_NAME}/${TRAIN_FILE}.parquet" "file"

# Reward function
check_path "Reward fn" "${WORKING_DIR}/custom_reward/verl_math_verify.py" "file"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,index,memory.total --format=csv,noheader 2>/dev/null | while IFS=, read -r name index mem; do
        echo "  GPU${index}: ${name} (${mem})"
    done
    
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1 | tr -d ' ')
    echo "  Available GPUs: ${gpu_count} (requested: ${NUM_GPU})"
    if [ "${gpu_count}" -lt "${NUM_GPU}" ]; then
        echo "  [WARN] Requested ${NUM_GPU} GPUs but only ${gpu_count} available"
    fi
else
    echo ""
    echo "[WARN] nvidia-smi not found - cannot verify GPU availability"
fi

# Check Ray
if command -v ray &> /dev/null; then
    echo ""
    echo "Ray Status:"
    ray --version 2>/dev/null || true
else
    echo ""
    echo "[WARN] Ray CLI not found"
fi

# Summary
echo ""
echo "============================================================"
if [ ${errors} -eq 0 ]; then
    echo "  All required paths found! Ready to run experiments."
    echo ""
    echo "  Start with:"
    echo "    bash run_experiment.sh phase1"
    echo "============================================================"
    exit 0
else
    echo "  ${errors} required path(s) missing. Please fix before running."
    echo "============================================================"
    exit 1
fi
