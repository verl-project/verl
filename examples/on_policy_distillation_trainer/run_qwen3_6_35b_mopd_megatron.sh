#!/usr/bin/env bash
# On-policy distillation | multi-teacher (reasoning_gym caesar_cipher + simple_geometry) | vLLM rollout with MTP
# | Megatron EP4 training | fully async, at most one step stale, no partial rollout | 8 NVIDIA GPUs
#
# Qwen3.6-35B-A3B is distilled from two GRPO teachers, one per task, routed by `data_source`.
# Layout (8 GPUs): trainer EP4 on 4, two TP1 rollout engines on 2, one TP1 engine per teacher on 2.
# Rollout runs ahead of training by at most one batch (staleness_threshold=1); a weight sync waits for
# in-flight requests (weight_sync_pause_mode=wait), so every response comes from one policy version.
# The resource pools fit one 8-GPU node or 2 nodes x 4 GPUs; for more than one node, start a Ray
# cluster first and run this script on the head node.
# Data, models and results: qwen3_6_35b_mopd/README.md.

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3.6-35B-A3B}
CAESAR_TEACHER_MODEL=${CAESAR_TEACHER_MODEL:-semianalysisai/Qwen3.6-35B-A3B-caesar-cipher-GRPO-20260924}
GEOMETRY_TEACHER_MODEL=${GEOMETRY_TEACHER_MODEL:-semianalysisai/Qwen3.6-35B-A3B-simple-geometry-GRPO-20260925}
DATA_DIR=${DATA_DIR:-$HOME/data/mopd_reasoning_gym}

TRAINER_NNODES=${TRAINER_NNODES:-1}
TRAINER_GPUS_PER_NODE=${TRAINER_GPUS_PER_NODE:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
rollout_tp=${ROLLOUT_TP:-1}
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-2}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-30720}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-32768}

actor_lr=${ACTOR_LR:-1e-6}
total_training_steps=${TOTAL_TRAINING_STEPS:-20}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:--1}

project_name=${PROJECT_NAME:-verl_distill_mopd_reasoning_gym}
experiment_name=${EXPERIMENT_NAME:-qwen3_6_35b_a3b_mopd_megatron_ep4_fully_async}
# ---- end user-adjustable ----

max_model_len=$(( max_prompt_length + max_response_length + 1 ))
reward_fn=$(dirname "$(realpath "$0")")/qwen3_6_35b_mopd/reward.py

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # fully async streams single prompts; a training step takes ppo_mini_batch_size samples.
    data.train_files="$DATA_DIR/train.parquet"
    data.val_files="['$DATA_DIR/caesar_cipher-dev.parquet', '$DATA_DIR/simple_geometry-dev.parquet']"
    data.train_batch_size=0
    data.gen_batch_size=1
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation='error'
    data.shuffle=True
    +data.apply_chat_template_kwargs.enable_thinking=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    # Fused log-probs avoid materializing [30k, 248k] fp32 logits. verl turns them off when
    # model.mtp is enabled, so MTP is enabled on the rollout side only (see ROLLOUT).
    actor_rollout_ref.model.use_fused_kernels=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.weight_decay=0.0
    actor_rollout_ref.actor.optim.betas='[0.9,0.999]'
    # EP4 on 4 GPUs leaves expert data parallel size 1, so the distributed optimizer cannot shard
    # expert states: fp32 Adam moments OOM and CPU Adam is ~4x slower. bf16 moments fit.
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
    actor_rollout_ref.actor.optim.exp_avg_dtype=bf16
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=bf16
    actor_rollout_ref.actor.optim.main_grads_dtype=fp32
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.freeze_vision_tower=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${TRAINER_GPUS_PER_NODE}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.dtype=bfloat16
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
)

ROLLOUT=(
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.max_model_len=${max_model_len}
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=128
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_token_len_per_gpu}
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    # Weight syncs wait for in-flight requests instead of aborting them, so every response comes
    # from one policy version and long responses are not cut at a sync.
    actor_rollout_ref.rollout.weight_sync_pause_mode=wait
    # MTP speculative decoding on the rollout only. The drafter is not in the trained model and is
    # never synced, so it must be loaded from the checkpoint (load_format=auto, not dummy).
    actor_rollout_ref.rollout.mtp="{_target_:verl.workers.config.MtpConfig,enable:true,enable_rollout:true,method:mtp,num_speculative_tokens:${num_speculative_tokens}}"
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.agent.num_workers=8
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
)

# Rollout log-probs serve as old log-probs (no recompute pass); the PPO ratio is train/rollout.
ALGO=(
    algorithm.rollout_correction.bypass_mode=True
)

# One weight sync per training step; rollout may generate up to one extra batch ahead of the trainer.
# partial_rollout=False: nothing is resumed across a sync (with `wait`, nothing is aborted either).
ASYNC=(
    async_training.trigger_parameter_sync_step=1
    async_training.require_batches=1
    async_training.staleness_threshold=1
    async_training.partial_rollout=False
    # Two TP1 replicas x 64 keeps one 128-prompt batch in flight.
    async_training.concurrent_samples_per_replica=$(( train_batch_size * rollout_tp / ROLLOUT_GPUS ))
    rollout.nnodes=1
    rollout.n_gpus_per_node=${ROLLOUT_GPUS}
    rollout.total_rollout_steps=$(( total_training_steps * train_batch_size ))
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.nnodes=${TRAINER_NNODES}
    trainer.n_gpus_per_node=${TRAINER_GPUS_PER_NODE}
    trainer.val_before_train=True
    trainer.test_freq=${test_freq}
    trainer.save_freq=${save_freq}
    trainer.total_epochs=1
    trainer.resume_mode=disable
)

teacher() {
    local name=$1 key=$2 path=$3
    echo "+distillation.teacher_models.${name}.key=${key}"
    echo "+distillation.teacher_models.${name}.model_path=${path}"
    echo "+distillation.teacher_models.${name}.num_replicas=1"
    for kv in name=vllm load_format=auto tensor_model_parallel_size=1 gpu_memory_utilization=0.8 \
        max_model_len=${max_model_len} prompt_length=${max_prompt_length} response_length=${max_response_length} \
        max_num_batched_tokens=32768 max_num_seqs=128 enable_chunked_prefill=True enable_prefix_caching=False \
        enforce_eager=False; do
        echo "+distillation.teacher_models.${name}.inference.${kv}"
    done
}

# k1 estimator as the per-token policy-gradient reward, no task reward; reward.py only scores accuracy.
mapfile -t TEACHERS < <(teacher caesar caesar_cipher "$CAESAR_TEACHER_MODEL"; \
    teacher geometry simple_geometry "$GEOMETRY_TEACHER_MODEL")
EXTRA=(
    distillation.enabled=True
    distillation.nnodes=1
    distillation.n_gpus_per_node=2
    distillation.teacher_key=data_source
    "${TEACHERS[@]}"
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    reward.custom_reward_function.path="$reward_fn"
    reward.custom_reward_function.name=compute_score
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × megatron extras of the committed uv.lock.
# Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra megatron")
fi
"${LAUNCH[@]}" -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name=fully_async_ppo_megatron_trainer \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${ALGO[@]}" \
    "${ASYNC[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
