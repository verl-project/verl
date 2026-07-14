#!/usr/bin/env bash
# On-policy distillation | single-teacher | reverse-KL on student-top-K | FSDP teacher | NVIDIA GPUs

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-1.7B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-8B}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Teacher runs as its own FSDP worker group; total teacher GPUs = nnodes * gpus_per_node.
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-4}

# Reverse KL hyperparams.
distillation_topk=${DISTILLATION_TOPK:-32}
teacher_chunk_size=${TEACHER_CHUNK_SIZE:-1024}

train_batch_size=${TRAIN_BATCH_SIZE:-32}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-50}

project_name=${PROJECT_NAME:-verl_distill_reverse_kl_fsdp}
experiment_name=${EXPERIMENT_NAME:-qwen3_1.7b_from_qwen3_8b_reverse_kl_topk_fsdp}
# ---- end user-adjustable ----

gsm8k_train=$HOME/data/gsm8k/train.parquet
gsm8k_test=$HOME/data/gsm8k/test.parquet

train_files="['$gsm8k_train']"
val_files="['$gsm8k_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

# Reverse-KL OPD with an FSDP-served teacher: teacher_backend=fsdp hosts the teacher as
# a frozen FSDP worker instead of a vLLM rollout. loss_mode=reverse_kl_topk uses topk as
# the *student* top-K width (not the teacher top-K, as in forward_kl_topk).
EXTRA=(
    distillation.enabled=True
    distillation.teacher_backend=fsdp
    distillation.teacher_chunk_size=${teacher_chunk_size}
    distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    distillation.teacher_models.teacher_model.key="openai/gsm8k"
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    # --- teacher FSDP engine ---
    +distillation.teacher_fsdp_config.strategy=fsdp
    +distillation.teacher_fsdp_config.dtype=bfloat16
    +distillation.teacher_fsdp_config.param_offload=True
    +distillation.teacher_fsdp_config.optimizer_offload=False
    +distillation.teacher_fsdp_config.fsdp_size=-1
    # --- loss ---
    distillation.distillation_loss.loss_mode=reverse_kl_topk
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
