#!/usr/bin/env bash
# GRPO profiling (torch profiler) | text | vLLM rollout | FSDP training | NVIDIA GPUs
#
# Captures PyTorch profiler chrome traces of BOTH the training side and the vLLM
# rollout engine (inference). Traces (.json.gz) are written under
# global_profiler.save_path and can be opened in chrome://tracing or Perfetto. The
# training traces land there directly; the rollout ones are moved there from the
# engine's own directory by relocate_results (see PROFILE_RELOCATE below), so a
# single command over save_path covers both.
#
# Training is collected continuously (discrete=False), so one file per profiled step
# per rank holds the whole step as that worker ran it: compute_log_prob (old log
# probs), compute_ref_log_prob and update_actor are record_function rows named after
# their stage, with the update loop's mini_batch<i> rows nested under update_actor.
# To keep that file small when the update loop has many mini-batches, set
# PROFILE_SCHED_ACTIVE=N to keep only the first N update mini-batches (every other
# stage stays in full); see the schedule vars below.
#
# Inference (rollout) is profiled by vLLM's own engine-side torch profiler, which
# ONLY runs in "discrete" mode. It is therefore forced to discrete=True for the
# rollout (independent of the actor's discrete setting) and captures the full
# generate_sequences window on each profiled step. Set PROFILE_ROLLOUT=False to
# profile training only.

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

# Optional torch.profiler.schedule over the update loop's mini-batches (the log-prob and rollout
# stages are always kept in full). Enabled only when active > 0. active <= 0 (the default) records
# the whole update loop.
#   discrete=False -> only `active` is honored: every stage in full + the first `active` update
#                     mini-batches. skip_first/wait/warmup/repeat are ignored (dropping the leading
#                     mini-batches would also drop the log-prob stages that ran before them).
#   discrete=True  -> the full schedule applies to the update stage's own trace: skip_first/wait/
#                     warmup drop leading mini-batches and only the `active` window is kept.
profile_sched_active=${PROFILE_SCHED_ACTIVE:-0}
profile_sched_skip_first=${PROFILE_SCHED_SKIP_FIRST:-0}
profile_sched_wait=${PROFILE_SCHED_WAIT:-0}
profile_sched_warmup=${PROFILE_SCHED_WARMUP:-0}
profile_sched_repeat=${PROFILE_SCHED_REPEAT:-0}

# The vLLM engine is given an output directory rather than a file name, so each replica writes
# into <save_path>/agent_loop_rollout_replica_<n>/. Relocation moves those traces up into
# save_path when the step's profiling finishes, renamed rollout-replica<n>_..., which keeps every
# trace of a step in one directory -- what post-processing that does not walk sub-directories
# needs. Set PROFILE_RELOCATE=False to keep the engine's own layout.
profile_relocate=${PROFILE_RELOCATE:-True}

# Optional command run ONCE, after the LAST profiled step, on each selected rank, e.g. to upload the
# traces (they are node-local otherwise). Backend stop + relocate_results still run every profiled
# step, so all steps' traces have accumulated in save_path by then; because the command runs a single
# time (not once per step), uploading the whole directory sends each trace exactly once:
#   PROFILE_FINISH_HOOK_CMD='mlx asset upload "$VERL_PROFILE_SAVE_PATH"'
# save_path is usually node-local, so the command runs on every selected rank/node -- set
# PROFILE_RANKS to one rank per node so each node's directory is uploaded once. Env vars exported:
# VERL_PROFILE_SAVE_PATH, VERL_PROFILE_TOOL, VERL_PROFILE_RANK, VERL_PROFILE_PID, VERL_PROFILE_ROLE,
# VERL_PROFILE_RAY_NSIGHT_DIR (nsys only).
profile_finish_hook_cmd=${PROFILE_FINISH_HOOK_CMD:-null}

# Inference (rollout) profiling. The vLLM engine profiler runs in discrete mode only and
# traces the whole generate_sequences window on each profiled step. `ranks` here are
# rollout *replica* indices (not training ranks).
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
    # Enable the torch profiler on the training side; each profiled step is collected whole.
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks=${profile_ranks}
    actor_rollout_ref.actor.profiler.all_ranks=${profile_ranks_all}
    actor_rollout_ref.actor.profiler.tool_config.torch.discrete=${profile_discrete}
    actor_rollout_ref.actor.profiler.tool_config.torch.contents=${profile_contents}
    # Sub-sample the update loop's mini-batches (active<=0 records them all).
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.active=${profile_sched_active}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.skip_first=${profile_sched_skip_first}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.wait=${profile_sched_wait}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.warmup=${profile_sched_warmup}
    actor_rollout_ref.actor.profiler.tool_config.torch.schedule.repeat=${profile_sched_repeat}
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
    # force discrete=True here independently of the actor's discrete setting above.
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
    # Inherited by every role, rollout replicas included.
    global_profiler.relocate_results=${profile_relocate}
    global_profiler.finish_hook_cmd="${profile_finish_hook_cmd}"
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
