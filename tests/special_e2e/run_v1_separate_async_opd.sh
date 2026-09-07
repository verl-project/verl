#!/usr/bin/env bash
set -xeuo pipefail

# Megatron-LM is baked into the CI image, but Ray workers do not inherit its PYTHONPATH.
export PYTHONPATH="/workspace/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VLLM_USE_V1=1

# Avoid ncclCuMemHostEnable crashes on PCIe runners without P2P access.
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python, as the non-uv images do): on GPU this
# runs every python entrypoint here — including the Ray workers, via
# runtime_env.py_executable — through `uv run` on the matching extras of the
# committed uv.lock, so the job needs no install step. This script is vllm x
# megatron throughout. NPU falls back to ambient python.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    UV_EXTRAS=(--extra vllm --extra megatron)
    LAUNCH=(uv run --frozen --all-packages "${UV_EXTRAS[@]}" python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages ${UV_EXTRAS[*]}")
fi

NUM_GPUS=${NUM_GPUS:-8}
N_GPUS_TRAINING=${N_GPUS_TRAINING:-4}
N_GPUS_ROLLOUT=${N_GPUS_ROLLOUT:-2}
N_GPUS_TEACHERS=${N_GPUS_TEACHERS:-2}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TRAIN_TP=${TRAIN_TP:-2}

if ((
    NUM_GPUS <= 0 || N_GPUS_TRAINING <= 0 || N_GPUS_ROLLOUT <= 0 || N_GPUS_TEACHERS <= 0
    || N_GPUS_TRAINING + N_GPUS_ROLLOUT + N_GPUS_TEACHERS != NUM_GPUS
)); then
    echo "Invalid GPU split: total=${NUM_GPUS}, training=${N_GPUS_TRAINING}, rollout=${N_GPUS_ROLLOUT}, teachers=${N_GPUS_TEACHERS}"
    exit 1
fi
if ((TRAIN_TP <= 0 || ROLLOUT_TP <= 0 || N_GPUS_TRAINING % TRAIN_TP != 0 || N_GPUS_ROLLOUT % ROLLOUT_TP != 0)); then
    echo "Training and rollout GPUs must be divisible by their tensor parallel sizes"
    exit 1
fi
if ((N_GPUS_TEACHERS != 2)); then
    echo "This test requires one GPU for each of its two teachers"
    exit 1
fi
TRAIN_PP=$((N_GPUS_TRAINING / TRAIN_TP))

STUDENT_MODEL_ID=${STUDENT_MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
GSM8K_TEACHER_MODEL_ID=${GSM8K_TEACHER_MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}
GEO3K_TEACHER_MODEL_ID=${GEO3K_TEACHER_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}
STUDENT_MODEL=${STUDENT_MODEL:-${HOME}/models/${STUDENT_MODEL_ID}}
GSM8K_TEACHER_MODEL=${GSM8K_TEACHER_MODEL:-${HOME}/models/${GSM8K_TEACHER_MODEL_ID}}
GEO3K_TEACHER_MODEL=${GEO3K_TEACHER_MODEL:-${HOME}/models/${GEO3K_TEACHER_MODEL_ID}}

GSM8K_TRAIN=${GSM8K_TRAIN:-${HOME}/data/gsm8k/train.parquet}
GSM8K_TEST=${GSM8K_TEST:-${HOME}/data/gsm8k/test.parquet}
GEO3K_TRAIN=${GEO3K_TRAIN:-${HOME}/data/geo3k/train.parquet}
GEO3K_TEST=${GEO3K_TEST:-${HOME}/data/geo3k/test.parquet}
TRAIN_FILES="['${GSM8K_TRAIN}','${GEO3K_TRAIN}']"
VAL_FILES="['${GSM8K_TEST}','${GEO3K_TEST}']"

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_NUM_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
TRAIN_BATCH_SIZE=$((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE))
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}

if ((
    MAX_PROMPT_LENGTH <= 0 || MAX_RESPONSE_LENGTH <= 0 || MAX_NUM_SEQS <= 0
    || PPO_MAX_TOKEN_LEN_PER_GPU <= 0 || N_RESP_PER_PROMPT <= 0
    || PARAMETER_SYNC_STEP <= 0 || PPO_MINI_BATCH_SIZE <= 0 || TOTAL_TRAINING_STEPS <= 0
)); then
    echo "Batch and training step settings must be positive"
    exit 1
fi

params=(
    model_engine=megatron
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
    data.image_key=images
    data.shuffle=True
    data.seed=1
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.gen_batch_size=1
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=True
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1
    actor_rollout_ref.actor.optim.lr_decay_steps=10000000
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.param_offload=False
    actor_rollout_ref.actor.megatron.optimizer_offload=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TRAIN_TP}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${TRAIN_PP}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.nnodes=1
    actor_rollout_ref.rollout.n_gpus_per_node=${N_GPUS_ROLLOUT}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
    distillation.enabled=True
    distillation.teacher_key=data_source
    distillation.n_gpus_per_node=${N_GPUS_TEACHERS}
    distillation.nnodes=1
    +distillation.teacher_models.gsm8k.key=openai/gsm8k
    +distillation.teacher_models.gsm8k.model_path="${GSM8K_TEACHER_MODEL}"
    +distillation.teacher_models.gsm8k.num_replicas=1
    +distillation.teacher_models.gsm8k.inference.name=vllm
    +distillation.teacher_models.gsm8k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.gsm8k.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.gsm8k.inference.enforce_eager=False
    +distillation.teacher_models.gsm8k.inference.max_model_len=${MAX_NUM_TOKENS}
    +distillation.teacher_models.gsm8k.inference.max_num_batched_tokens=${MAX_NUM_TOKENS}
    +distillation.teacher_models.gsm8k.inference.max_num_seqs=${MAX_NUM_SEQS}
    +distillation.teacher_models.geo3k.key=hiyouga/geometry3k
    +distillation.teacher_models.geo3k.model_path="${GEO3K_TEACHER_MODEL}"
    +distillation.teacher_models.geo3k.num_replicas=1
    +distillation.teacher_models.geo3k.inference.name=vllm
    +distillation.teacher_models.geo3k.inference.tensor_model_parallel_size=1
    +distillation.teacher_models.geo3k.inference.gpu_memory_utilization=0.7
    +distillation.teacher_models.geo3k.inference.enforce_eager=False
    +distillation.teacher_models.geo3k.inference.max_model_len=${MAX_NUM_TOKENS}
    +distillation.teacher_models.geo3k.inference.max_num_batched_tokens=${MAX_NUM_TOKENS}
    +distillation.teacher_models.geo3k.inference.max_num_seqs=${MAX_NUM_SEQS}
    +distillation.teacher_models.geo3k.inference.engine_kwargs.vllm.mm_processor_cache_gb=0
    distillation.distillation_loss.loss_mode=k1
    distillation.distillation_loss.topk=64
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=True
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=128
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=verl-test-v1-separate-async-opd
    trainer.experiment_name=qwen3-vl-2b-v1-separate-async-multi-teacher-opd
    trainer.val_before_train=False
    trainer.test_freq=-1
    trainer.save_freq=-1
    trainer.resume_mode=disable
    trainer.nnodes=1
    trainer.n_gpus_per_node=${N_GPUS_TRAINING}
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

echo "Running V1 separate_async Multi-Teacher OPD"
echo "Student: ${STUDENT_MODEL}"
echo "Teachers: ${GSM8K_TEACHER_MODEL}, ${GEO3K_TEACHER_MODEL}"
echo "GPUs: ${N_GPUS_TRAINING} training + ${N_GPUS_ROLLOUT} rollout + ${N_GPUS_TEACHERS} teachers"

"${LAUNCH[@]}" -m verl.trainer.main_ppo "${params[@]}" "${RAY[@]}" "$@"

echo "V1 separate_async Multi-Teacher OPD E2E completed successfully"
