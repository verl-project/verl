#!/usr/bin/env bash
# MTP + DAPO | MiMo-7B (speculative MTP head) | SGLang rollout | Megatron training | NVIDIA GPUs
# Fully-async (one-step-off) split-placement training demo, multi-node capable.
#
# Layout defaults to the "4+4" split (4 trainer GPUs + 4 rollout GPUs on a single node).
# Scale via TRAIN_NNODES / ROLLOUT_NNODES for a true multi-node run (e.g. 4 trainer nodes +
# 4 rollout nodes, hence the historical name "dapo_mimo_7b_with_mtp_math_megatron_4_4").

set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---- user-adjustable ----
# NOTE: remember to set max_position_embeddings=32768 in the model's config.json after downloading.
MODEL_PATH=${MODEL_PATH:-XiaomiMiMo/MiMo-7B-RL}

# Fully-async split-placement layout: trainer group + rollout group.
TRAIN_NNODES=${TRAIN_NNODES:-1}
TRAIN_NGPUS_PER_NODE=${TRAIN_NGPUS_PER_NODE:-4}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-4}

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
gen_batch_size=${GEN_BATCH_SIZE:-1}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR:-0.1}

actor_tp=${ACTOR_TP:-2}
actor_pp=${ACTOR_PP:-1}
actor_cp=${ACTOR_CP:-1}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}

staleness_threshold=${STALENESS_THRESHOLD:-0.5}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-4}
require_batches=${REQUIRE_BATCHES:-1}

total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-51200}
total_epochs=${TOTAL_EPOCHS:-10}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:--1}

project_name=${PROJECT_NAME:-verl_mtp_fully_async}
experiment_name=${EXPERIMENT_NAME:-dapo_mimo_7b_mtp_fully_async}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${experiment_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
# ---- end user-adjustable ----

actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=0 \
    data.gen_batch_size=${gen_batch_size} \
    data.trust_remote_code=True \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.mtp.enable=True \
    actor_rollout_ref.model.mtp.enable_train=True \
    actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=${mtp_loss_scaling_factor} \
    actor_rollout_ref.model.mtp.detach_encoder=True \
    actor_rollout_ref.model.mtp.enable_rollout=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.grad_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${actor_cp} \
    actor_rollout_ref.ref.megatron.param_offload=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=$((1024 * 4)) \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    rollout.nnodes=${ROLLOUT_NNODES} \
    rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=True \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.nnodes=${TRAIN_NNODES} \
    trainer.n_gpus_per_node=${TRAIN_NGPUS_PER_NODE} \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 "$@"
