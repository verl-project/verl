#!/usr/bin/env bash
set -xeuo pipefail

# Workaround for NVIDIA driver bug (r560-r575) causing SIGSEGV in ncclCuMemHostEnable()
# on PCIe machines without P2P access. See: https://github.com/NVIDIA/nccl/issues/1838
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

# Fully async OPD on GSM8K.
# Student rollout uses single-turn generation; teacher forward and student
# update share the fused Megatron trainer resource pool.

############################ Quick Config ############################

ROLLOUT_NAME="sglang"

# Keep MTP/speculative decoding disabled while validating fused-node OPD.
mtp_params=(
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable_rollout=False
)

STUDENT_MODEL=${STUDENT_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
GSM8K_TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-7B}

DISTILLATION_LOSS_MODE="k1"
USE_POLICY_GRADIENT=True

MAX_PROMPT=${MAX_PROMPT:-1600}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

# Fully async specific
ROLLOUT_NNODES=2
N_GPUS_ROLLOUT=8
TRAINER_NNODES=2
N_GPUS_TRAINING=8
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-4096}

# Megatron parallelism
GEN_TP=4
TRAIN_TP=4
TRAIN_PP=2

STALENESS_THRESHOLD=0.5
TRIGGER_PARAMETER_SYNC_STEP=4
SAVE_EVERY_TRAIN_STEPS=${SAVE_EVERY_TRAIN_STEPS:-16}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/your_local_dir/"}

if (( SAVE_EVERY_TRAIN_STEPS % TRIGGER_PARAMETER_SYNC_STEP != 0 )); then
    echo "SAVE_EVERY_TRAIN_STEPS must be divisible by TRIGGER_PARAMETER_SYNC_STEP" >&2
    exit 1
fi
SAVE_FREQ_PARAM_VERSIONS=$(( SAVE_EVERY_TRAIN_STEPS / TRIGGER_PARAMETER_SYNC_STEP ))

############################ Data ############################

GSM8K_TRAIN="/your_local_dir/to_gsm8k/train.parquet"
GSM8K_TEST="/your_local_dir/to_gsm8k/test.parquet"

TRAIN_FILES="['${GSM8K_TRAIN}']"
TEST_FILES="['${GSM8K_TEST}']"

ACTOR_OFFLOAD=True

############################ Parameter Groups ############################

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.prompt_key=prompt
    data.truncation='left'
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=0
    data.gen_batch_size=1
    data.return_raw_chat=True
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
)

STUDENT=(
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.lr_decay_steps=10000000
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode="token-mean"
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_NUM_TOKENS
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_NUM_TOKENS
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH
    actor_rollout_ref.rollout.single_turn_response_length=$MAX_RESPONSE_LENGTH
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=16
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.multi_turn.enable=False
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl'
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True
)

# Single-teacher OPD: teacher Megatron forward shares the trainer resource pool.
DISTILLATION=(
    distillation.enabled=True
    distillation.teacher_execution=trainer
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=0
    distillation.nnodes=0
    # --- single teacher ---
    +distillation.teacher_models.gsm8k.key="openai/gsm8k"
    +distillation.teacher_models.gsm8k.model_path="${GSM8K_TEACHER_MODEL}"
    +distillation.teacher_models.gsm8k.num_replicas=0
    +distillation.teacher_models.gsm8k.inference.name=$ROLLOUT_NAME
    # --- loss ---
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=1
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=False
)

REWARD=(
    reward.reward_manager.name=dapo_judge
    ++reward.custom_reward_function.path='verl/utils/reward_score/zero.py'
    ++reward.custom_reward_function.name='compute_score'
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console","tensorboard"]'
    trainer.project_name='verl-test-fully-async-opd'
    trainer.experiment_name="qwen2.5-7b-fully-async-fused-node-opd"
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ_PARAM_VERSIONS}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode=disable
    trainer.nnodes=${TRAINER_NNODES}
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.log_val_generations=10
    +trainer.use_legacy_worker_impl=disable
    trainer.total_epochs=2
    trainer.test_freq=-1
)

ASYNC_TRAINING=(
    rollout.nnodes=${ROLLOUT_NNODES}
    rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
    rollout.total_rollout_steps=${TOTAL_ROLLOUT_STEPS}
    async_training.staleness_threshold=${STALENESS_THRESHOLD}
    async_training.partial_rollout=True
    async_training.trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP}
    async_training.require_batches=1
    async_training.use_trainer_do_validate=False
)

############################ Launch ############################

echo "Running fully_async_policy + Single-Teacher OPD"
echo "Student: ${STUDENT_MODEL}"
echo "Teacher: ${GSM8K_TEACHER_MODEL}"
echo "Dataset: ${GSM8K_TRAIN}"
echo "Single-turn: prompt=${MAX_PROMPT}, response=${MAX_RESPONSE_LENGTH}, total_tokens=${MAX_NUM_TOKENS}"
echo "MTP/speculative decoding: disabled"
echo "GPUs: ${N_GPUS_ROLLOUT}x${ROLLOUT_NNODES} rollout + ${N_GPUS_TRAINING}x${TRAINER_NNODES} fused teacher/training"
echo "Checkpoints: every ${SAVE_EVERY_TRAIN_STEPS} trainer steps -> ${CHECKPOINT_DIR}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    actor_rollout_ref.hybrid_engine=False \
    critic.strategy=megatron \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${STUDENT[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "${mtp_params[@]}" \
    "$@"

echo "Fully async fused-node OPD completed successfully"
