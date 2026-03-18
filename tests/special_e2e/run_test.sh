#!/bin/bash
set -xeuo pipefail

# Get the configuration name and engine name from arguments
CONFIG_NAME="$1"
ENGINE="${2:-vllm}"

# Download model if needed
#hf download Qwen/Qwen2.5-0.5B --local-dir "$HOME/models/Qwen/Qwen2.5-0.5B"

# Run the training with the specified configuration
python3 -m verl.trainer.main_ppo \
    --config-name "$CONFIG_NAME" "$@" 