#!/usr/bin/env bash

set -euo pipefail

IMAGE_TAG=${IMAGE_TAG:-verl-intel-gpu:latest}
NUM_GPUS=${NUM_GPUS:-2}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
DATA_DIR=${DATA_DIR:-$HOME/data}
MODEL_PATH=${MODEL_PATH:-}
RENDER_GID=${RENDER_GID:-$(getent group render | cut -d: -f3)}

docker run --rm -it \
    --device /dev/dri \
    --group-add "${RENDER_GID}" \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --shm-size 16g \
    --network host \
    --tmpfs /tmp:exec,size=8g \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${DATA_DIR}:/root/data" \
    -e NUM_GPUS="${NUM_GPUS}" \
    -e MODEL_PATH="${MODEL_PATH}" \
    "${IMAGE_TAG}" \
    bash tests/special_intel_gpu/run_ppo_intel_gpu.sh