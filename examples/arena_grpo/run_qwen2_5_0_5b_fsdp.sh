#!/usr/bin/env bash
# GRPO + OpenAgora sandbox agent loop | Qwen2.5-0.5B-Instruct | FSDP training
#
# The agent runs inside OpenAgora sandboxes (Docker). LLM calls made by the
# agent are proxied by the OpenAgora server to an external OpenAI-compatible
# backend, and rewards are computed by OpenAgora's verification plane instead
# of a reward function in the trainer.
#
# Prerequisites:
#   1. openagora-server running on ARENA_ENDPOINT (default localhost:9090), see
#      https://github.com/albert-lv/OpenAgora
#   2. An OpenAI-compatible LLM backend reachable at ARENA_LLM_BACKEND, e.g.:
#        vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001 --dtype bfloat16 --enforce-eager
#   3. The agent sandbox image available on the host
#      (default openagora-agent-minimal:latest)
#   4. An RL dataset in parquet format whose extra_info struct column may carry
#      a per-sample `openagora_verify` command (see README.md in this directory)
#   5. openagora-sdk installed in the verl environment (`pip install openagora-sdk`)

set -xeuo pipefail

########################### user-adjustable ###########################
# OpenAgora server / sandbox configuration.
export ARENA_ENDPOINT=${ARENA_ENDPOINT:-localhost:9090}
export ARENA_AGENT_IMAGE=${ARENA_AGENT_IMAGE:-openagora-agent-minimal:latest}
export ARENA_LLM_BACKEND=${ARENA_LLM_BACKEND:-http://localhost:8001/v1}
# Fallback verify command; per-sample extra_info.openagora_verify takes precedence.
export ARENA_VERIFY_COMMAND=${ARENA_VERIFY_COMMAND:-true}
export ARENA_TIMEOUT_SECONDS=${ARENA_TIMEOUT_SECONDS:-600}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-./data/train.parquet}
VAL_FILES=${VAL_FILES:-./data/test.parquet}

train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
max_prompt_length=${MAX_PROMPT_LENGTH:-256}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.01}
rollout_n=${ROLLOUT_N:-4}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-1}

PROJECT_NAME=${PROJECT_NAME:-verl_arena_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_0_5b_arena_grpo_fsdp_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/train_grpo_arena.py" \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size=${train_batch_size} \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=${actor_lr} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.n=${rollout_n} \
  actor_rollout_ref.rollout.agent.default_agent_loop=arena_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${SCRIPT_DIR}/arena_agent_loop.yaml" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
  trainer.nnodes=${NNODES} \
  trainer.save_freq=${save_freq} \
  trainer.test_freq=${test_freq} \
  trainer.total_epochs=${total_epochs} \
  "$@"
