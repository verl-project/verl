#!/usr/bin/env bash
# On-policy distillation | single teacher | vLLM rollout | VeOmni training | NVIDIA GPUs
#
# VeOmni engine variant of the FSDP OPD script. Key differences:
#   - model_engine=veomni for FSDP2-based training
#   - use_fused_kernels=True enables veomni's fused-linear kernels:
#     * For RL/SFT batches: fused log-prob + entropy (no logits materialization)
#     * For top-K distillation batches (loss_mode=forward_kl_topk):
#       veomni's chunk_topk_distill kernel computes the top-K forward-KL
#       distillation loss without materializing [B, L, V] logits — saving
#       significant GPU memory for large-vocabulary models.
#   Without use_fused_kernels, the vanilla FSDP actor materializes full
#   [B, L, V] logits and computes distillation loss eagerly. The VeOmni fused
#   chunk_topk_distill path avoids this by streaming the lm_head
#   projection chunk-by-chunk.

set -xeuo pipefail

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-0.6B}
TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-1.7B}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-7}
TEACHER_NGPUS=${TEACHER_NGPUS:-1}
teacher_tp=${TEACHER_TP:-1}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-32}

train_batch_size=${TRAIN_BATCH_SIZE:-14}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-14}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

actor_lr=${ACTOR_LR:-1e-5}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.3}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.3}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-200}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_distill_opd_veomni}
experiment_name=${EXPERIMENT_NAME:-qwen3_0.6b_from_1.7b_opd_veomni_fused}
# ---- end user-adjustable ----

train_files=$HOME/data/gsm8k/train.parquet
val_files=$HOME/data/gsm8k/test.parquet

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
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.veomni.param_offload=False
    actor_rollout_ref.actor.veomni.optimizer_offload=False
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
    trainer.logger=console
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    model_engine=veomni
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NGPUS}
    distillation.nnodes=1
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="$TEACHER_MODEL"
    +distillation.teacher_models.gsm8k.num_replicas=1
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=${teacher_tp}
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    +distillation.teacher_models.gsm8k.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
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
