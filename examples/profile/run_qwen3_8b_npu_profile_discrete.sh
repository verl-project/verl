#!/usr/bin/env bash
# GRPO profiling (discrete per-rank) | text | vLLM rollout | FSDP training | Ascend NPU
#
# Captures NPU traces on a subset of ranks in "discrete" mode
# (one trace per worker module). Useful for targeted hot-path analysis.

set -xeuo pipefail

export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}

profile_steps=${PROFILE_STEPS:-"[2,4]"}
profile_ranks=${PROFILE_RANKS:-"[1,2]"}
profile_ranks_all=${PROFILE_RANKS_ALL:-False}
profile_discrete=${PROFILE_DISCRETE:-True}
profile_save_path=${PROFILE_SAVE_PATH:-$HOME/profile_data}
profile_level=${PROFILE_LEVEL:-level0}
profile_contents=${PROFILE_CONTENTS:-"['npu','cpu']"}
profile_analysis=${PROFILE_ANALYSIS:-True}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

actor_lr=${ACTOR_LR:-5e-8}

rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-4}

project_name=${PROJECT_NAME:-verl_grpo_profile}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_npu_profile_discrete}
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks=${profile_ranks}
    actor_rollout_ref.actor.profiler.all_ranks=${profile_ranks_all}
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=${profile_discrete}
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=${profile_contents}
    actor_rollout_ref.actor.profiler.tool_config.npu.level=${profile_level}
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=${profile_analysis}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.profiler.enable=True
    actor_rollout_ref.ref.profiler.ranks=${profile_ranks}
    actor_rollout_ref.ref.profiler.all_ranks=${profile_ranks_all}
    actor_rollout_ref.ref.profiler.tool_config.npu.discrete=${profile_discrete}
    actor_rollout_ref.ref.profiler.tool_config.npu.contents=${profile_contents}
    actor_rollout_ref.ref.profiler.tool_config.npu.level=${profile_level}
    actor_rollout_ref.ref.profiler.tool_config.npu.analysis=${profile_analysis}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=-1
    trainer.test_freq=5
    trainer.total_epochs=5
)

EXTRA=(
    global_profiler.tool=npu
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
