#!/usr/bin/env bash
# MTP | MiMo-7B | SGLang rollout | Megatron training | NVIDIA GPUs or Ascend NPUs

set -xeuo pipefail

########################### platform and defaults ###########################

DEVICE=${DEVICE:-cuda}
MODEL_PATH=${MODEL_PATH:-XiaomiMiMo/MiMo-7B-RL}
NNODES=${NNODES:-1}

if [[ "$DEVICE" == "npu" ]]; then
    devices_per_node=${NPUS_PER_NODE:-4}
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
    export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
    export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
    export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
    export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
    export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
    export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
    export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
    export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
    export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}

    train_batch_size=${TRAIN_BATCH_SIZE:-16}
    val_batch_size=${VAL_BATCH_SIZE:-16}
    ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}
    ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE:-1}
    max_prompt_length=${MAX_PROMPT_LENGTH:-512}
    max_response_length=${MAX_RESPONSE_LENGTH:-1024}
    ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
    dataloader_num_workers=${DATALOADER_NUM_WORKERS:-0}

    actor_tp=${ACTOR_TP:-2}
    actor_pp=${ACTOR_PP:-1}
    actor_cp=${ACTOR_CP:-1}
    rollout_tp=${ROLLOUT_TP:-1}
    rollout_n=${ROLLOUT_N:-2}
    rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.45}
    rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-8}
    rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
    rollout_cuda_graph_max_bs=${ROLLOUT_CUDA_GRAPH_MAX_BS:-8}
    checkpoint_bucket_mb=${CHECKPOINT_BUCKET_MB:-1280}

    total_epochs=${TOTAL_EPOCHS:-10}
    total_training_steps=${TOTAL_TRAINING_STEPS:-100}
    save_freq=${SAVE_FREQ:-100}
    test_freq=${TEST_FREQ:--1}
    lr_warmup_steps=${LR_WARMUP_STEPS:-10}

    data_root=${DATA_ROOT:-$HOME/data/math}
    train_file=${TRAIN_FILE:-$data_root/train.parquet}
    val_file=${VAL_FILE:-$data_root/test.parquet}
    project_name=${PROJECT_NAME:-verl_mtp}
    experiment_name=${EXPERIMENT_NAME:-mimo_7b_mtp_sglang_megatron_ascend}
    logger='["console"]'
else
    devices_per_node=${NGPUS_PER_NODE:-8}
    export CUDA_DEVICE_MAX_CONNECTIONS=1

    train_batch_size=${TRAIN_BATCH_SIZE:-128}
    ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
    max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
    max_response_length=${MAX_RESPONSE_LENGTH:-8192}
    ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}

    actor_tp=${ACTOR_TP:-2}
    actor_pp=${ACTOR_PP:-2}
    actor_cp=${ACTOR_CP:-2}
    rollout_tp=${ROLLOUT_TP:-4}
    rollout_n=${ROLLOUT_N:-16}
    rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.8}

    total_epochs=${TOTAL_EPOCHS:-10}
    total_training_steps=${TOTAL_TRAINING_STEPS:-400}
    save_freq=${SAVE_FREQ:--1}
    test_freq=${TEST_FREQ:-10}
    lr_warmup_steps=${LR_WARMUP_STEPS:-10}

    train_file=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
    val_file=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}
    project_name=${PROJECT_NAME:-verl_mtp}
    experiment_name=${EXPERIMENT_NAME:-mimo_7b_mtp_sglang_megatron}
    logger='["console","wandb"]'
fi

actor_lr=${ACTOR_LR:-1e-6}
entropy_coeff=${ENTROPY_COEFF:-0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
mtp_loss_scaling_factor=${MTP_LOSS_SCALING_FACTOR:-0.1}
offload=${OFFLOAD:-True}

if [[ "$DEVICE" == "npu" ]]; then
    world_size=$((NNODES * devices_per_node))
    if (( world_size < 2 )); then
        echo "SGLang HCCL synchronization requires at least two trainer ranks." >&2
        exit 2
    fi
    if (( rollout_tp != 1 )); then
        echo "The SGLang HCCL backend requires ROLLOUT_TP=1." >&2
        exit 2
    fi
fi

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

DATA=(
    "data.train_files=['$train_file']"
    "data.val_files=['$val_file']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.mtp.enable=True
    actor_rollout_ref.model.mtp.enable_train=True
    actor_rollout_ref.model.mtp.enable_rollout=False
    actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=${mtp_loss_scaling_factor}
    actor_rollout_ref.model.mtp.detach_encoder=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps}
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.max_resp_len=${max_response_length}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    "trainer.logger=$logger"
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=${total_training_steps}
)

if [[ "$DEVICE" == "npu" ]]; then
    output_dir=${OUTPUT_DIR:-$PWD/checkpoints/$experiment_name}
    log_dir=${LOG_DIR:-$PWD/logs}
    resume_mode=${RESUME_MODE:-auto}
    max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1}
    mkdir -p "$output_dir" "$log_dir"

    ALGORITHM+=(
        algorithm.rollout_correction.bypass_mode=True
        algorithm.rollout_correction.loss_type=ppo_clip
    )
    DATA+=(
        data.val_batch_size=${val_batch_size}
        data.dataloader_num_workers=${dataloader_num_workers}
        data.filter_overlong_prompts=True
        data.truncation=error
    )
    ACTOR+=(
        actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=True
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size}
        actor_rollout_ref.actor.megatron.vanilla_mbridge=False
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
        +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.position_embedding_type=rope
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.normalization=RMSNorm
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rmsnorm=True
        actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash
        +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    )
    ROLLOUT+=(
        actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
        actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
        actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size}
        actor_rollout_ref.rollout.calculate_log_probs=True
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        actor_rollout_ref.rollout.enable_prefix_caching=True
        actor_rollout_ref.rollout.enforce_eager=False
        actor_rollout_ref.rollout.free_cache_engine=True
        actor_rollout_ref.rollout.checkpoint_engine.backend=sglang_hccl
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${checkpoint_bucket_mb}
        +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.sglang_hccl.process_group_backend=hccl
        +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.sglang_hccl.request_timeout=3600
        actor_rollout_ref.rollout.checkpoint_manager_class=verl.checkpoint_engine.sglang_hccl_checkpoint_engine.SGLangHCCLCheckpointEngineManager
        +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=ascend
        +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=${rollout_cuda_graph_max_bs}
        actor_rollout_ref.rollout.val_kwargs.n=1
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0
        actor_rollout_ref.rollout.val_kwargs.top_p=0.7
        actor_rollout_ref.rollout.val_kwargs.top_k=-1
        actor_rollout_ref.rollout.val_kwargs.do_sample=True
    )
    REF+=(
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ppo_micro_batch_size}
        actor_rollout_ref.ref.megatron.vanilla_mbridge=False
    )
    REWARD+=(
        +reward.reward_kwargs.overlong_buffer_cfg.len=256
        +reward.reward_kwargs.overlong_buffer_cfg.log=False
    )
    TRAINER+=(
        trainer.device=npu
        trainer.resume_mode=${resume_mode}
        trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep}
        "trainer.default_local_dir=$output_dir"
    )
else
    DATA+=(data.truncation=left)
    REWARD+=(+reward.reward_kwargs.overlong_buffer_cfg.len=4096)
fi

########################### launch ###########################

EXTRA=(
    model_engine=megatron
)

# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the sglang × megatron extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [[ "${VERL_USE_UV:-1}" != 0 && "$DEVICE" != "npu" ]]; then
    LAUNCH=(uv run --frozen --all-packages --extra sglang --extra megatron python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra sglang --extra megatron")
fi

COMMAND=(
    "${LAUNCH[@]}"
    -m verl.trainer.main_ppo
    "${DATA[@]}"
    "${ALGORITHM[@]}"
    "${MODEL[@]}"
    "${ACTOR[@]}"
    "${ROLLOUT[@]}"
    "${REF[@]}"
    "${REWARD[@]}"
    "${TRAINER[@]}"
    "${EXTRA[@]}"
    "${RAY[@]}"
    "$@"
)

if [[ "$DEVICE" == "npu" ]]; then
    log_file="$log_dir/${experiment_name}_$(date +%Y%m%d_%H%M%S).log"
    PYTHONUNBUFFERED=1 "${COMMAND[@]}" 2>&1 | tee "$log_file"
else
    "${COMMAND[@]}"
fi
