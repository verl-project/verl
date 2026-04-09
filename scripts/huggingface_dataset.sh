#!/bin/bash
# set -x

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <username/dataset_name> [dataset_name (for saving)]"
    exit 1
fi

REPO_NAME=$1

if [ $# -gt 1 ]; then
    DATASET_NAME=$2
else
    DATASET_NAME="${REPO_NAME##*/}"
fi

SAVE_PATH="/data/open_datasets/$DATASET_NAME"

# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_DOWNLOAD_TIMEOUT=120
# export HF_HUB_MAX_WORKERS=1
# export HF_HUB_ENABLE_HF_TRANSFER=0

export HF_TOKEN= # export your HF token as environment var. to run this script

huggingface-cli download \
    $REPO_NAME \
    --repo-type dataset \
    --local-dir $SAVE_PATH \
    --resume-download \
    # --include "*.parquet" "*.json" "*.csv"

echo "数据集已下载到: $SAVE_PATH"