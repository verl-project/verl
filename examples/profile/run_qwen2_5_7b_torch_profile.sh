#!/usr/bin/env bash
# GRPO profiling (torch profiler, scheduled) | text | vLLM rollout | FSDP training | NVIDIA GPUs
#
# Captures PyTorch profiler chrome traces of BOTH the actor update loop (training)
# and the vLLM rollout engine (inference). Traces (.json.gz) are written under
# global_profiler.save_path and can be opened in chrome://tracing or Perfetto:
#   <save_path>/                              -> actor (training) traces
#   <save_path>/agent_loop_rollout_replica_* -> rollout (inference) traces
#
# Training (actor) demonstrates torch.profiler.schedule: instead of tracing every
# mini-batch, the profiler advances one step per mini-batch (via profiler.step())
# and only records a wait/warmup/active window, repeated `repeat` times. Set
# PROFILE_SCHEDULE_ACTIVE=0 to disable scheduling and trace the whole window
# continuously instead.
#
# Inference (rollout) is profiled by vLLM's own engine-side torch profiler, which
# ONLY runs in "discrete" mode and has no notion of torch.profiler.schedule/step().
# It is therefore forced to discrete=True for the rollout (independent of the
# actor's schedule) and captures the full generate_sequences window on each profiled
# step. Set PROFILE_ROLLOUT=False to profile training only.

set -xeuo pipefail

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Which training steps to profile, where to save, and which ranks to trace.
profile_steps=${PROFILE_STEPS:-"[1,2]"}
profile_save_path=${PROFILE_SAVE_PATH:-$HOME/profile_data}
profile_ranks=${PROFILE_RANKS:-"[0]"}
profile_ranks_all=${PROFILE_RANKS_ALL:-False}
profile_discrete=${PROFILE_DISCRETE:-False}
profile_contents=${PROFILE_CONTENTS:-"['cpu','cuda']"}

# torch.profiler.schedule (advances once per mini-batch of the actor update loop).
# Each cycle records `warmup` (discarded) + `active` (kept) mini-batches, repeated
# `repeat` times, after skipping `skip_first`. active<=0 disables scheduling.
profile_schedule_skip_first=${PROFILE_SCHEDULE_SKIP_FIRST:-0}
profile_schedule_wait=${PROFILE_SCHEDULE_WAIT:-0}
profile_schedule_warmup=${PROFILE_SCHEDULE_WARMUP:-1}
profile_schedule_active=${PROFILE_SCHEDULE_ACTIVE:-2}
profile_schedule_repeat=${PROFILE_SCHEDULE_REPEAT:-1}

# Inference (rollout) profiling. The vLLM engine profiler runs in discrete mode only,
# so it ignores the schedule above and traces the whole generate_sequences window on
# each profiled step. `ranks` here are rollout *replica* indices (not training ranks).
# Optionally restrict to a response-token window (null = from first token / until end).
profile_rollout=${PROFILE_ROLLOUT:-True}
profile_rollout_token_start=${PROFILE_ROLLOUT_TOKEN_START:-null}
profile_rollout_token_end=${PROFILE_ROLLOUT_TOKEN_END:-null}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-4}

save_freq=${SAVE_FREQ:--1}
test_freq=${TEST_FREQ:-5}
total_epochs=${TOTAL_EPOCHS:-5}

project_name=${PROJECT_NAME:-verl_grpo_profile}
experiment_name=${EXPERIMENT_NAME:-qwen2_5_7b_torch_profile}
# ---- end user-adjustable ----
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=$HOME/data/gsm8k/train.parquet
    data.val_files=$HOME/data/gsm8k/test.parquet
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    # Enable the torch profiler on the actor update loop with a schedule.
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks=${profile_ranks}
    actor_rollout_ref.actor.profiler.all_ranks=${profile_ranks_all}
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=${profile_discrete}
    actor_rollout_ref.actor.profiler.tool_config.torch.contents=${profile_contents}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.skip_first=${profile_schedule_skip_first}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.wait=${profile_schedule_wait}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.warmup=${profile_schedule_warmup}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=${profile_schedule_active}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.repeat=${profile_schedule_repeat}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    # Enable the torch profiler on the vLLM rollout engine (inference). This is
    # collected by vLLM's engine-side profiler and REQUIRES discrete mode, so we
    # force discrete=True here independently of the actor's schedule/discrete above.
    actor_rollout_ref.rollout.profiler.enable=${profile_rollout}
    actor_rollout_ref.rollout.profiler.ranks=${profile_ranks}
    actor_rollout_ref.rollout.profiler.all_ranks=${profile_ranks_all}
    actor_rollout_ref.rollout.profiler.tool_config.torch.discrete=True
    actor_rollout_ref.rollout.profiler.tool_config.torch.contents=${profile_contents}
    actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_start=${profile_rollout_token_start}
    actor_rollout_ref.rollout.profiler.tool_config.torch.profile_token_end=${profile_rollout_token_end}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    global_profiler.tool=torch
    global_profiler.steps=${profile_steps}
    global_profiler.save_path=${profile_save_path}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
