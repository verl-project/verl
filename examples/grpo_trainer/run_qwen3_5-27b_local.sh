#!/bin/bash
# Qwen3.5-27B GRPO training script for FSDP backend using local transformers
# This script uses local transformers code and skips weight loading

set -x

# 1. Activate conda environment
echo "Activating conda environment verl_qwen3.5_fsdp..."
conda activate verl_qwen3.5_fsdp

# 2. Setup NPU/CANN environment
echo "Setting up NPU/CANN environment..."
source /home/CANN/CANN8.5.0/ascend-toolkit/set_env.sh

# Set PYTHONPATH to use local transformers
export PYTHONPATH="/home/t00906153/project/verl-qwen3.5/transformers/src:$PYTHONPATH"
export PYTHONPATH="/home/t00906153/project/verl-qwen3.5/verl:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH"

# Model path - using local model directory
MODEL_PATH="/home/t00906153/project/verl-qwen3.5/model/Qwen3.5-27B"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Model directory not found: $MODEL_PATH"
    echo "Will use dummy model path for testing with weight skipping"
    MODEL_PATH="/tmp/dummy_qwen3_5_model"
    mkdir -p "$MODEL_PATH"
    
    # Create minimal config for testing
    python3 -c "
import json, os
model_path = '$MODEL_PATH'
config = {
    'model_type': 'qwen3_5',
    'vocab_size': 32000,
    'hidden_size': 5120,
    'num_hidden_layers': 4,
    'num_attention_heads': 40,
    'intermediate_size': 17408,
    'max_position_embeddings': 32768,
    'rope_theta': 10000.0,
    'attention_bias': False,
    'architectures': ['Qwen3_5ForConditionalGeneration']
}
os.makedirs(model_path, exist_ok=True)
config_path = os.path.join(model_path, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f'Created dummy config at: {config_path}')
"
fi

# Check if config.json exists, create if not
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Creating minimal config.json for testing..."
    python3 -c "
import json, os
model_path = '$MODEL_PATH'
config = {
    'model_type': 'qwen3_5',
    'vocab_size': 32000,
    'hidden_size': 5120,
    'num_hidden_layers': 4,
    'num_attention_heads': 40,
    'intermediate_size': 17408,
    'max_position_embeddings': 32768,
    'rope_theta': 10000.0,
    'attention_bias': False,
    'architectures': ['Qwen3_5ForConditionalGeneration']
}
os.makedirs(model_path, exist_ok=True)
config_path = os.path.join(model_path, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f'Created config at: {config_path}')
"
fi

echo "Using model from: $MODEL_PATH"
echo "Note: Will skip weight loading and use reduced layers (4 layers instead of 40)"

# For testing with small dataset
DATA_PATH="/home/t00906153/project/verl-qwen3.5/data"
if [ ! -d "$DATA_PATH" ]; then
    echo "Creating data directory..."
    mkdir -p "$DATA_PATH"
fi

# Create dummy data if needed
DUMMY_TRAIN_FILE="$DATA_PATH/train.parquet"
DUMMY_VAL_FILE="$DATA_PATH/test.parquet"

if [ ! -f "$DUMMY_TRAIN_FILE" ]; then
    echo "Creating dummy training data..."
    python3 -c "
import pandas as pd
import numpy as np

# Create dummy data
data = {
    'prompt': ['What is 2+2?'] * 10,
    'response': ['4'] * 10,
    'reward': [1.0] * 10
}

df = pd.DataFrame(data)
df.to_parquet('$DUMMY_TRAIN_FILE')
print(f'Created dummy training data with {len(df)} samples')
"
fi

if [ ! -f "$DUMMY_VAL_FILE" ]; then
    echo "Creating dummy validation data..."
    python3 -c "
import pandas as pd
import numpy as np

# Create dummy data
data = {
    'prompt': ['What is 3+3?'] * 5,
    'response': ['6'] * 5,
    'reward': [1.0] * 5
}

df = pd.DataFrame(data)
df.to_parquet('$DUMMY_VAL_FILE')
print(f'Created dummy validation data with {len(df)} samples')
"
fi

# Set environment variable to skip weight loading
export VERL_SKIP_WEIGHT_LOADING="true"
export VERL_DEBUG_MODE="true"

echo "Starting training with local transformers and skipping weight loading..."

# First, apply patches to skip weight loading
echo "Applying patches to skip weight loading and NPU optimization..."
python3 -c "
import sys
import os
sys.path.insert(0, '/home/t00906153/project/verl-qwen3.5/verl')
sys.path.insert(0, '/home/t00906153/project/verl-qwen3.5/transformers/src')

print('Python version:', sys.version)
print('Current directory:', os.getcwd())

# Apply weight loading patches
os.environ['VERL_SKIP_WEIGHT_LOADING'] = 'true'
from verl.models.transformers.skip_weights import apply_all_patches
apply_all_patches()
print('✅ Weight skipping patches applied')

# Load NPU patches
import verl.models.transformers.npu_patch
print('✅ NPU patches loaded')

# Test verl import
import verl
print('✅ verl imported successfully')

# Test Qwen3.5 adapter
from verl.models.transformers import qwen3_5
print('✅ Qwen3.5 adapter imported')

print('All patches applied successfully')
"

# Run training with FSDP backend
# Note: We're using minimal settings for testing with reduced layers
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$DUMMY_TRAIN_FILE" \
    data.val_files="$DUMMY_VAL_FILE" \
    data.train_batch_size=4 \
    data.max_prompt_length=32 \
    data.max_response_length=64 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.model_type=qwen3_5 \
    +actor_rollout_ref.model.num_hidden_layers=4 \
    +actor_rollout_ref.model.hidden_size=5120 \
    +actor_rollout_ref.model.num_attention_heads=40 \
    +actor_rollout_ref.model.intermediate_size=17408 \
    +actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    +actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.kl_loss_coef=0.001 \
    +actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    +actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.model.enable_gradient_checkpointing=False \
    +actor_rollout_ref.actor.fsdp_config.param_offload=False \
    +actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    +actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +algorithm.use_kl_in_reward=False \
    +trainer.critic_warmup=0 \
    +trainer.logger='["console"]' \
    +trainer.project_name='verl_qwen3_5_27b_local_test' \
    +trainer.experiment_name='qwen3_5_27b_grpo_fsdp_local_reduced' \
    +trainer.n_gpus_per_node=1 \
    +trainer.nnodes=1 \
    +trainer.save_freq=1 \
    +trainer.test_freq=1 \
    +trainer.total_epochs=1 \
    +trainer.max_steps=1 \
    +trainer.debug_mode=true \
    +trainer.device=npu $@

echo "Training script completed"

# Notes:
# 1. Using minimal batch sizes and steps for testing
# 2. tensor_model_parallel_size=1 for single GPU testing
# 3. Skipping weight loading via environment variable
# 4. Using local transformers code