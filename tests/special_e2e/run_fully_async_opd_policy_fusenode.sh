#!/usr/bin/env bash
set -xeuo pipefail

# Workaround for NVIDIA driver bug (r560-r575) causing SIGSEGV in ncclCuMemHostEnable()
# on PCIe machines without P2P access. See: https://github.com/NVIDIA/nccl/issues/1838
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

# Fully async OPD with nonrouter multi-teacher scoring.
# The two teachers below are an example; add more teacher_models entries and
# matching datasets as needed. In fused mode every trajectory is scored by every
# teacher, matching the V1 fused path. Isomorphic teacher snapshots and the
# student share one Megatron resource pool.

############################ Quick Config ############################

ROLLOUT_NAME="sglang"

# true: trainer-colocated fused teachers (default and recommended by this script)
# false: teachers use dedicated nodes and standalone SGLang inference engines
#   NOTE: standalone mode uses router=True (each sample is routed to the teacher
#   matching its data_source); fused mode keeps nonrouter=True (all teachers
#   score each sample, the trainer later selects by data_source).
FUSE_TEACHER=${FUSE_TEACHER:-true}
case "${FUSE_TEACHER,,}" in
    true|false) ;;
    *) echo "FUSE_TEACHER must be true or false, got: ${FUSE_TEACHER}" >&2; exit 1 ;;
esac
FUSE_TEACHER=${FUSE_TEACHER,,}


STUDENT_MODEL=${STUDENT_MODEL:-"Qwen/Qwen3.5-35B-A3B"}
# Example teacher models. Extend this list together with the DISTILLATION entries below.
MATH_DAPO_TEACHER_MODEL=${MATH_DAPO_TEACHER_MODEL:-"Qwen/Qwen3.5-35B-A3B"}
AIME_2024_TEACHER_MODEL=${AIME_2024_TEACHER_MODEL:-"Qwen/Qwen3.5-35B-A3B"}

mtp_params=(
    actor_rollout_ref.model.mtp.enable=False
    actor_rollout_ref.model.mtp.enable_train=False
    actor_rollout_ref.model.mtp.enable_rollout=False
)

DISTILLATION_LOSS_MODE="k1"
USE_POLICY_GRADIENT=True

MAX_PROMPT=${MAX_PROMPT:-1600}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))

# Student resources are unchanged between modes.
ROLLOUT_NNODES=2
N_GPUS_ROLLOUT=8
TRAINER_NNODES=2
N_GPUS_TRAINING=8
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-40960}

# Standalone mode needs one dedicated TP=8 replica per configured teacher.
# This two-teacher example therefore defaults to two teacher nodes.
TEACHER_NNODES=${TEACHER_NNODES:-2}
N_GPUS_TEACHER_PER_NODE=${N_GPUS_TEACHER_PER_NODE:-8}
TEACHER_TP=${TEACHER_TP:-8}

# Megatron parallelism (35B-A3B MoE)
GEN_TP=8
TRAIN_TP=8
TRAIN_PP=2

STALENESS_THRESHOLD=0.5
TRIGGER_PARAMETER_SYNC_STEP=4
SAVE_EVERY_TRAIN_STEPS=${SAVE_EVERY_TRAIN_STEPS:-32}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${HOME}/ckpt_dir/}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${CHECKPOINT_ROOT}/fully_async_opd_multiteacher"}

if (( SAVE_EVERY_TRAIN_STEPS % TRIGGER_PARAMETER_SYNC_STEP != 0 )); then
    echo "SAVE_EVERY_TRAIN_STEPS must be divisible by TRIGGER_PARAMETER_SYNC_STEP" >&2
    exit 1
fi
SAVE_FREQ_PARAM_VERSIONS=$(( SAVE_EVERY_TRAIN_STEPS / TRIGGER_PARAMETER_SYNC_STEP ))

############################ Data ############################

DAPO_TRAIN=${DAPO_TRAIN:-"${HOME}/data/dapo/train.parquet"}
AIME_TRAIN=${AIME_TRAIN:-"${HOME}/data/aime/train.parquet"}

DAPO_TEST=${DAPO_TEST:-"${HOME}/data/dapo/test.parquet"}
AIME_TEST=${AIME_TEST:-"${HOME}/data/aime/test.parquet"}

TRAIN_FILES="['${DAPO_TRAIN}','${AIME_TRAIN}']"
TEST_FILES="['${DAPO_TEST}','${AIME_TEST}']"

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
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.grad_offload=False
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_NUM_TOKENS
)

TEACHER_RUNTIME=(
    actor_rollout_ref.teacher.strategy=megatron
    actor_rollout_ref.teacher.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.teacher.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.teacher.log_prob_max_token_len_per_gpu=$MAX_NUM_TOKENS
    actor_rollout_ref.teacher.megatron.param_offload=True
    actor_rollout_ref.teacher.megatron.optimizer_offload=False
    actor_rollout_ref.teacher.megatron.grad_offload=False
    actor_rollout_ref.teacher.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.teacher.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.teacher.megatron.expert_model_parallel_size=8
    actor_rollout_ref.teacher.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.teacher.megatron.context_parallel_size=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=128
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

DISTILLATION_BASE=(
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.topk=1
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

if [[ "$FUSE_TEACHER" == "true" ]]; then
    # Nonrouter fused: all teachers score each sample; the trainer later selects
    # by data_source, matching the run_v1_separate_async script.
    DISTILLATION=(
        "${DISTILLATION_BASE[@]}"
        distillation.nonrouter=True
        distillation.teacher_execution=trainer
        distillation.n_gpus_per_node=0
        distillation.nnodes=0
        +distillation.teacher_models.t1.key="math_dapo"
        +distillation.teacher_models.t1.model_path="${MATH_DAPO_TEACHER_MODEL}"
        +distillation.teacher_models.t2.key="aime_2024"
        +distillation.teacher_models.t2.model_path="${AIME_2024_TEACHER_MODEL}"
    )
    TEACHER_RUNTIME_ARGS=("${TEACHER_RUNTIME[@]}")
    TEACHER_MODE_DESCRIPTION="fused trainer-colocated teachers (nonrouter)"
else
    # Standalone: rollout-served teachers reside on dedicated nodes. The agent
    # loop routes each sample to the teacher matching its data_source, so
    # nonrouter=False. Switching the Python teacher manager to nonrouter all-
    # teacher scoring would require a separate code change.
    echo "WARNING: this script is tuned and validated for fused nonrouter teachers by default; standalone teachers are routed by data_source and require one dedicated TP-sized replica per configured teacher." >&2
    DISTILLATION=(
        "${DISTILLATION_BASE[@]}"
        distillation.nonrouter=False
        distillation.teacher_execution=rollout
        distillation.n_gpus_per_node=${N_GPUS_TEACHER_PER_NODE}
        distillation.nnodes=${TEACHER_NNODES}
        +distillation.teacher_models.t1.key="math_dapo"
        +distillation.teacher_models.t1.model_path="${MATH_DAPO_TEACHER_MODEL}"
        +distillation.teacher_models.t1.num_replicas=1
        +distillation.teacher_models.t1.inference.name=${ROLLOUT_NAME}
        +distillation.teacher_models.t1.inference.tensor_model_parallel_size=${TEACHER_TP}
        +distillation.teacher_models.t1.inference.gpu_memory_utilization=0.7
        +distillation.teacher_models.t1.inference.enforce_eager=False
        +distillation.teacher_models.t1.inference.max_model_len=${MAX_NUM_TOKENS}
        +distillation.teacher_models.t1.inference.max_num_batched_tokens=${MAX_NUM_TOKENS}
        +distillation.teacher_models.t1.inference.max_num_seqs=16
        +distillation.teacher_models.t2.key="aime_2024"
        +distillation.teacher_models.t2.model_path="${AIME_2024_TEACHER_MODEL}"
        +distillation.teacher_models.t2.num_replicas=1
        +distillation.teacher_models.t2.inference.name=${ROLLOUT_NAME}
        +distillation.teacher_models.t2.inference.tensor_model_parallel_size=${TEACHER_TP}
        +distillation.teacher_models.t2.inference.gpu_memory_utilization=0.7
        +distillation.teacher_models.t2.inference.enforce_eager=False
        +distillation.teacher_models.t2.inference.max_model_len=${MAX_NUM_TOKENS}
        +distillation.teacher_models.t2.inference.max_num_batched_tokens=${MAX_NUM_TOKENS}
        +distillation.teacher_models.t2.inference.max_num_seqs=16
    )
    TEACHER_RUNTIME_ARGS=()
    TEACHER_MODE_DESCRIPTION="standalone teachers on ${TEACHER_NNODES} dedicated nodes (routed by data_source)"
fi

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=False
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name='verl-test-fully-async-opd'
    trainer.experiment_name="fully-async-opd-multiteacher"
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ_PARAM_VERSIONS}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.resume_mode=disable
    trainer.nnodes=${TRAINER_NNODES}
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.log_val_generations=0
    +trainer.use_legacy_worker_impl=disable
    trainer.total_epochs=1
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

echo "Running fully_async_policy + nonrouter Multi-Teacher OPD: ${TEACHER_MODE_DESCRIPTION}"
echo "Student: ${STUDENT_MODEL}"
echo "Example teachers: math_dapo -> ${MATH_DAPO_TEACHER_MODEL}, aime_2024 -> ${AIME_2024_TEACHER_MODEL}"
echo "Train datasets: ${DAPO_TRAIN} (math_dapo), ${AIME_TRAIN} (aime_2024)"
echo "Test datasets: ${DAPO_TEST} (math_dapo), ${AIME_TEST} (aime_2024)"
echo "Single-turn: prompt=${MAX_PROMPT}, response=${MAX_RESPONSE_LENGTH}, total_tokens=${MAX_NUM_TOKENS}"
echo "MTP/speculative decoding: disabled"
if [[ "$FUSE_TEACHER" == "true" ]]; then
    echo "GPUs: ${N_GPUS_ROLLOUT}x${ROLLOUT_NNODES} rollout + ${N_GPUS_TRAINING}x${TRAINER_NNODES} fused training/teacher"
else
    echo "GPUs: ${N_GPUS_ROLLOUT}x${ROLLOUT_NNODES} rollout + ${N_GPUS_TRAINING}x${TRAINER_NNODES} training + ${N_GPUS_TEACHER_PER_NODE}x${TEACHER_NNODES} standalone teachers"
fi
echo "Checkpoints: every ${SAVE_EVERY_TRAIN_STEPS} trainer steps -> ${CHECKPOINT_DIR}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    actor_rollout_ref.hybrid_engine=False \
    critic.strategy=megatron \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${STUDENT[@]}" \
    ${TEACHER_RUNTIME_ARGS[@]+"${TEACHER_RUNTIME_ARGS[@]}"} \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${ASYNC_TRAINING[@]}" \
    "${mtp_params[@]}" \
    "$@"

echo "Fully async multi-teacher OPD completed successfully (${TEACHER_MODE_DESCRIPTION})"
