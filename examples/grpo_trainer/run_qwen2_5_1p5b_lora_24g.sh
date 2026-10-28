#!/usr/bin/env bash
# Single-GPU GRPO + LoRA example for a 24 GB card (RTX 4090 / L4 / A10).
#
# The official GRPO examples assume 8x80 GB. On one 24 GB card the rollout
# (vLLM colocate) and the actor update both need care; this recipe is a
# verified set of knobs for Qwen2.5-Math-1.5B-Instruct.
#
# Tested stack (2026-10): torch 2.8.0+cu128, vllm 0.11.0, verl v0.8.0,
# LoRA r=32, FSDP. RTX 4090 24 GB: 20-step smoke and a 600-step run both fit
# (~18-20 GB peak). Newer verl main may pin newer torch/vLLM versions and rename
# config keys; treat these values as a starting point and re-check them.
#
# DATA format: the script uses main_ppo's default NaiveRewardManager, so each
# parquet row needs `prompt`, `data_source` (e.g. "math"), and
# `reward_model.ground_truth` (a math_verify-compatible answer string).
#
# Usage:
#   MODEL=/path/to/Qwen2.5-Math-1.5B-Instruct \
#   DATA=/path/to/math/train.parquet \
#   bash examples/grpo_trainer/run_qwen2_5_1p5b_lora_24g.sh
#
# Optional: STEPS=300 (default), TOTAL_EPOCHS=100, GPU_MEM=0.45, SAVE_FREQ=100.
# `TOTAL_EPOCHS` is only a guard: training stops at STEPS first as long as DATA
# has at least STEPS * train_batch_size rows (default 300 * 4 = 1,200).
set -euo pipefail

MODEL=${MODEL:?set MODEL to the Qwen2.5-Math-1.5B-Instruct path}
DATA=${DATA:?set DATA to a verl-format math parquet (prompt + data_source + reward_model.ground_truth)}
STEPS=${STEPS:-300}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}
SAVE_FREQ=${SAVE_FREQ:-100}
GPU_MEM=${GPU_MEM:-0.45}

python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA" \
  data.val_files="$DATA" \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
  data.max_response_length=3072 \
  data.seed=42 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_shm=False \
  actor_rollout_ref.rollout.load_format=dummy \
  actor_rollout_ref.rollout.layered_summon=False \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.data_loader_seed=42 \
  actor_rollout_ref.actor.fsdp_config.seed=42 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.adv_estimator=grpo \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.test_freq=-1 \
  trainer.project_name=verl_grpo_single_24g \
  trainer.experiment_name=qwen2_5_1p5b_lora_24g
