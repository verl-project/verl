#!/bin/bash
# set -x

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


# ============================ Configurations ===========================
MODEL_NAME="Qwen2.5-0.5B-Instruct"
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"      # Only need to change this if you wish to ppo with base model.

# Generation Params
TEMPERATURE="${TEMPERATURE:-1.0}"       # 采样温度
TOP_P="${TOP_P:-1.0}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"

# Validation Specific Params
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0}"
VAL_TOP_P="${VAL_TOP_P:-1.0}"

# Epochs
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"

# Control seed for generation (Currently unused!!)
SEED="${SEED:-42}"
# =======================================================================


# =========================== Param Parsing =============================
if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <dataset_name> <gpu_ids> [checkpoint_path] [other_configs...]"
    echo "Supported datasets: arc-challenge, aqua_rat, gsm8k, livecodebench, math, math-500, numinamath, strategyQA, theoremQA"
    
    echo ""
    echo "Examples:"
    echo "  # Do PPO training with the base model on gsm8k"
    echo "  bash $0 gsm8k 0,1,2,3"
    echo ""
    echo "  # Train with a specific checkpoint (after SFT training)"
    echo "  bash $0 gsm8k 0,1,2,3,4,5,6,7 $PROJECT_DIR/outputs/Qwen2.5-0.5B--gsm8k--sft/checkpoint-last"
    echo ""
    echo "  # Use GPU 4-7 and train with a specific checkpoint"
    echo "  bash $0 MATH 4,5,6,7 /path/to/checkpoint"
    exit 1
fi

# +++++ Dataset Info +++++
DATASET=$1
shift 1
shopt -s nocasematch    # Enable caseless match
case $DATASET in
    "ai2_arc" | "ai2-arc" | "arc-challenge")
        DATA_NAME="arc-challenge"
        TRAIN_DATA_DIR="/data/open_datasets/ai2_arc/ARC-Challenge/test-processed.parquet"
        TEST_DATA_DIR="/data/open_datasets/ai2_arc/ARC-Challenge/test-processed.parquet"
        
        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for ARC-Challenge: Success!"
        exit 1
        ;;
    "aqua_rat" | "aqua-rat")
        DATA_NAME="aqua_rat"
        TRAIN_DATA_DIR="/data/open_datasets/aqua_rat/processed/test-processed.parquet"
        TEST_DATA_DIR="/data/open_datasets/aqua_rat/processed/test-processed.parquet"
        
        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for AQuA-RAT: Success!"
        exit 1
        ;;
    "gsm8k")
        DATA_NAME="gsm8k"
        TRAIN_DATA_DIR="/data/open_datasets/GSM8K/train.parquet"
        TEST_DATA_DIR="/data/open_datasets/GSM8K/test.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for GSM8K: Success!"
        ;;
    "livecodebench")
        DATA_NAME="gsm8k"
        TRAIN_DATA_DIR="/data/open_datasets/livecodebench/..."
        TEST_DATA_DIR="/data/open_datasets/livecodebench/..."

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for LiveCodeBench: Success!"
        exit 1
        ;;
    "math")
        DATA_NAME="math"
        TRAIN_DATA_DIR="/data/open_datasets/MATH/train_processed.parquet"
        TEST_DATA_DIR="/data/open_datasets/MATH-500/test-processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for MATH: Success!"
        ;;
    "math-500")
        DATA_NAME="math-500"
        TRAIN_DATA_DIR="/data/open_datasets/MATH-500/test-processed.parquet"
        TEST_DATA_DIR="/data/open_datasets/MATH-500/test-processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for MATH-500: Success!"
        exit 1
        ;;
    "numinamath" | "numinamath-CoT")
        DATA_NAME="numinamath"
        TRAIN_DATA_DIR="/data/open_datasets/NuminaMath-CoT/train-processed-0.parquet"
        TEST_DATA_DIR="/data/open_datasets/NuminaMath-CoT/test-processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for NuminaMath-CoT: Success!"
        ;;
    "strategyQA")
        DATA_NAME="strategyQA"
        TRAIN_DATA_DIR="/data/open_datasets/StrategyQA/data/test-processed.parquet"
        TEST_DATA_DIR="/data/open_datasets/StrategyQA/data/test-processed.parquet"

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)

        echo "[INFO] Load config for StrategyQA: Success!"
        exit 1
        ;;
    "theoremQA")
        DATA_NAME="gsm8k"
        TRAIN_DATA_DIR="/data/open_datasets/TheoremQA/..."
        TEST_DATA_DIR="/data/open_datasets/TheoremQA/..."

        PROMPT_KEY="prompt"              # Question
        DATA_SOURCE_KEY="data_source"    # Data Source
        REWARD_MODEL_KEY="reward_model"  # Dict Containing GT (ground_truth)
        
        echo "[INFO] Load config for TheoremQA: Success!"
        exit 1
        ;;
    *)
        # Default: unknown dataset
        echo "[ERROR] Unsupported dataset $DATASET."
        echo "Supported datasets: ai2_arc, aqua_rat, gsm8k, livecodebench, math, math-500, numinamath, strategyQA, theoremQA"
        exit 1
        ;;
esac
shopt -u nocasematch    # Disable caseless match

# +++++ GPU Info +++++
gpu_ids=$1
shift 1

# +++++ Model Path +++++    (Precedence: CLI > SFT > Base Model)
SFT_CHECKPOINT="${STORE_DIR}/outputs/${MODEL_NAME}--${DATA_NAME}--sft/checkpoint-last"  # Checkpoint from SFT

if [ $# -gt 0 ] && [[ "$1" == */* ]]; then
    MODEL_PATH="$1"
    echo "[INFO] Loading model from CLI param: $MODEL_PATH"
    shift 1
elif [ -n "$SFT_CHECKPOINT" ] && [ -d "$SFT_CHECKPOINT" ] && [ -f "${SFT_CHECKPOINT}/model.safetensors" ]; then
    MODEL_PATH="$SFT_CHECKPOINT"
    echo "[INFO] Loading model checkpoint from set environment variable: $MODEL_PATH"
else
    MODEL_PATH="$BASE_MODEL"
    echo "[INFO] Loading base model: $MODEL_PATH"
fi


# =========================== Param Settings =============================

# Set Visible GPUs & Auto Calculate # of GPUs
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpus_per_node=$(echo $gpu_ids | tr ',' ' ' | wc -w)

# Check compatibility of tensor_model_parallel_size
tp_size=2
if [ $((n_gpus_per_node % tp_size)) -ne 0 ]; then
    echo "[WARNING]: n_gpus_per_node($n_gpus_per_node) is not divisible by tensor_model_parallel_size($tp_size). It is advised to set tensor_model_parallel_size to a devisor of $n_gpus_per_node (e.g. 1, $n_gpus_per_node)."
    echo "[INFO] Fallback: Setting tp_size to $n_gpus_per_node..."
    tp_size=$n_gpus_per_node  # Automatic fallback to avoid errors
fi

# Output & Log Settings
PPO_OUTPUT_DIR="${STORE_DIR}/outputs/${MODEL_NAME}--${DATA_NAME}--ppo"
LOG_PATH="${PROJECT_DIR}/wandb_logs"
LOG_NAME="${MODEL_NAME}--${DATA_NAME}--ppo--$(date +%Y%m%d-%H%M%S)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

export WANDB_DIR=${LOG_PATH}
export WANDB_RUN_ID="exp-${MODEL_NAME}-${DATA_NAME}-ppo"

echo "=========================================="
echo "PPO Training Setup:"
echo "  GPUs: $gpu_ids (count: $n_gpus_per_node)"
echo "  Model Path: $MODEL_PATH"
echo "  Data: $DATA_NAME"
echo "  Training Params: top_p=$TOP_P, temperature=$TEMPERATURE"
echo "  Validation Params: top_p=$VAL_TOP_P, temperature=$VAL_TEMPERATURE"
echo "  Num of Epochs: $TOTAL_EPOCHS"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
    --config-path=${CONFIG_DIR} \
    --config-name=ppo_trainer \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=False \
    data.train_files=${TRAIN_DATA_DIR} \
    data.val_files=${TEST_DATA_DIR} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key=${PROMPT_KEY} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P} \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules="all-linear" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${DATA_NAME}_ppo \
    trainer.experiment_name=${LOG_NAME} \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=1 \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${PPO_OUTPUT_DIR} \
    custom_reward_function.path=$(realpath "../verl/utils/reward_score/reward_fn_router.py") \
    custom_reward_function.name="compute_score_router" \
    $@
