#!/usr/bin/env bash
set -xeuo pipefail

# Megatron-LM is baked into the CI image, but Ray workers do not inherit its PYTHONPATH.
export PYTHONPATH="/workspace/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_USE_V1=1

NUM_GPUS=${NUM_GPUS:-8}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-$((NUM_GPUS / 2))}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-$((NUM_GPUS - N_GPUS_TRAINING))}
ACTOR_STRATEGY=${ACTOR_STRATEGY:-fsdp2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
VANILLA_MBRIDGE=${VANILLA_MBRIDGE:-False}

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python, as the ascend image does): on GPU this
# runs every python entrypoint here — including the Ray workers, via
# runtime_env.py_executable — through `uv run` on the matching extras of the
# committed uv.lock, so the job needs no install step. The rollout engine is vllm
# throughout; the training extra follows ACTOR_STRATEGY (fsdp2 rides the `fsdp`
# extra). NPU falls back to ambient python.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    case "${ACTOR_STRATEGY}" in
        megatron) TRAIN_EXTRA=megatron ;;
        *)        TRAIN_EXTRA=fsdp ;;
    esac
    UV_EXTRAS=(--extra vllm --extra "${TRAIN_EXTRA}")
    LAUNCH=(uv run --frozen --all-packages "${UV_EXTRAS[@]}" python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages ${UV_EXTRAS[*]}")
fi

if ((N_GPUS_TRAINING <= 0 || N_GPUS_ROLLOUT <= 0 || N_GPUS_TRAINING + N_GPUS_ROLLOUT != NUM_GPUS)); then
    echo "Invalid GPU split: total=${NUM_GPUS}, training=${N_GPUS_TRAINING}, rollout=${N_GPUS_ROLLOUT}"
    exit 1
fi

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
TRAIN_BATCH_SIZE=$((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE))
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}

if ((ROLLOUT_TP <= 0 || N_GPUS_TRAINING % 2 != 0 || N_GPUS_ROLLOUT % ROLLOUT_TP != 0)); then
    echo "Training GPUs must be even and rollout GPUs must be divisible by ROLLOUT_TP"
    exit 1
fi

experiment_name="$(basename "${MODEL_ID,,}")-v1-separate-async-${ACTOR_STRATEGY}-minimal"

common_params=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=separate_async
    trainer.v1.separate_async.num_warmup_batches=1
    trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP}
    trainer.v1.sampler.max_off_policy_threshold=8
    trainer.v1.sampler.max_off_policy_strategy=drop
    transfer_queue.enable=True
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.prompt_key=prompt
    data.truncation=left
    data.filter_overlong_prompts=True
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.gen_batch_size=1
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.rollout_correction.bypass_mode=False
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.nnodes=1
    actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    reward.reward_manager.name=dapo
    trainer.logger='["console"]'
    trainer.project_name=verl-test-v1-separate-async
    trainer.experiment_name="${experiment_name}"
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

echo "Running V1 separate_async with ${ACTOR_STRATEGY}: ${N_GPUS_TRAINING} training + ${N_GPUS_ROLLOUT} rollout GPUs"

if [[ "${ACTOR_STRATEGY}" == "fsdp2" ]]; then
    "${LAUNCH[@]}" -m verl.trainer.main_ppo \
        "${common_params[@]}" \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        "${RAY[@]}" "$@"
elif [[ "${ACTOR_STRATEGY}" == "megatron" ]]; then
    TRAIN_TP=${TRAIN_TP:-2}
    if ((TRAIN_TP <= 0 || N_GPUS_TRAINING % TRAIN_TP != 0)); then
        echo "N_GPUS_TRAINING must be divisible by TRAIN_TP"
        exit 1
    fi
    TRAIN_PP=$((N_GPUS_TRAINING / TRAIN_TP))

    "${LAUNCH[@]}" -m verl.trainer.main_ppo \
        model_engine=megatron \
        "${common_params[@]}" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.megatron.vanilla_mbridge=${VANILLA_MBRIDGE} \
        actor_rollout_ref.actor.megatron.param_offload=False \
        actor_rollout_ref.actor.megatron.optimizer_offload=False \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP} \
        "${RAY[@]}" "$@"
else
    echo "Unknown ACTOR_STRATEGY=${ACTOR_STRATEGY}; expected fsdp2 or megatron"
    exit 1
fi

echo "V1 separate_async E2E completed successfully with ${ACTOR_STRATEGY}"
