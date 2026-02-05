#!/usr/bin/env bash
set -xeuo pipefail
 
# ============================================================
# GKD Training Script (FSDP2 Backend)
# ============================================================
#
# Prerequisites:
# Start teacher server first:
#   bash /workspace/mlf2/verl/recipe/gkd/teacher/start_server.sh
#
# Usage:
#   bash train_gkd_fsdp2.sh
#
# To override parameters:
#   bash train_gkd_fsdp2.sh trainer.n_gpus_per_node=4

# To stop the background run
# kill -9 $(cat run.pid) || true
# ray stop -f || true
 
# ============================================================
# Environment Setup
# ============================================================
ulimit -u 65535
export TOKENIZERS_PARALLELISM=false
export TORCH_CUDA_ARCH_LIST="9.0"
export RAY_TMPDIR=/workspace/ray_tmp
mkdir -p "$RAY_TMPDIR"
 
# Cleanup previous runs
ray stop -f || true
 
# ============================================================
# GPUs to use (modify as needed)
# ============================================================
export CUDA_VISIBLE_DEVICES=4,5
 
# ============================================================
# Run Training
# ============================================================
# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/config"
 
# VERL root directory (adjust if needed)
# VERL_ROOT="/workspace/mlf2/verl"
# cd "$VERL_ROOT"

# ============================================================
# Logging
# ============================================================
rm -f "$SCRIPT_DIR"/run*
# rm -rf "$SCRIPT_DIR"/outputs
# rm -rf "$SCRIPT_DIR"/checkpoints
# rm -rf "$SCRIPT_DIR"/wandb
LOG="$SCRIPT_DIR/run_gkd_fsdp2_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="$SCRIPT_DIR/run.pid"
 
# Note: hydra.searchpath adds verl's config directory to search path
# so that defaults: - /ppo_trainer@_here_ can find the base config
# gsm8k_qwen2p5_14b_to_0p5b, openthoughts_qwen3_8b_to_0p6b
nohup \
env HYDRA_FULL_ERROR=1 \
python3 -u -m verl.trainer.main_ppo \
  --config-path "$CONFIG_DIR" \
  --config-name gsm8k_qwen2p5_14b_to_0p5b \
  "hydra.searchpath=[pkg://verl.trainer.config]" \
  "$@" \
> "$LOG" 2>&1 < /dev/null &
 
echo $! > "$PIDFILE"