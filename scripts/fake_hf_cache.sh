#!/bin/bash
# set -x

# ============================ Configurations ===========================
MODEL_NAME="Qwen2.5-0.5B-Instruct"
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

MODEL_PATH="/data/pretrain_models/${MODEL_NAME}"

HF_CACHE="$HOME/.cache/huggingface"
REPO_DIR="${HF_CACHE}/hub/models--${MODEL_ID//\//--}"

# Use a fake commit hash (40 bits in hexadecimal), can be replaced with a real commit hash from HF
FAKE_HASH="0000000000000000000000000000000000000000"
# =======================================================================

# Create HF cache structure
mkdir -p "${REPO_DIR}/refs"
mkdir -p "${REPO_DIR}/snapshots/${FAKE_HASH}"

# Create refs/main which points towards the fake hash
echo -n "${FAKE_HASH}" > "${REPO_DIR}/refs/main"

# Copy the model files to snapshots/<hash>/
cp -r "${MODEL_PATH}/"* "${REPO_DIR}/snapshots/${FAKE_HASH}/"

echo "Model stored in HF cache format: ${REPO_DIR}"
echo "Commit hash: ${FAKE_HASH}"
ls -la "${REPO_DIR}/snapshots/${FAKE_HASH}/"