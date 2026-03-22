#!/bin/bash


# ReposiREPO_ROOT="/users/${USER}/scratch/rl-as-a-service"tory and paths

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen2.5-Math-7B"}
TRAIN_FILE=${TRAIN_FILE:-"${HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${HOME}/data/aime-2024.parquet"}

# Naming
project_name=${PROJECT_NAME:-'LORA'}
exp_name=${EXP_NAME:-'DAPO-Qwen2.5-7b-vllm-fully-async-2-2-lora'}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${project_name}/${exp_name}"}

# LoRA parameters (override via env)
lora_rank=${LORA_RANK:-32}
lora_alpha=${LORA_ALPHA:-32}
lora_target_modules=${LORA_TARGET_MODULES:-"all-linear"}

rollout_mode=${ROLLOUT_MODE:-"async"}
rollout_name=${ROLLOUT_NAME:-"vllm"}

if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# Algorithm parameters
adv_estimator=${ADV_ESTIMATOR:-grpo}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

# Length parameters
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-True}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

# Training parameters
loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_top_p=${VAL_TOP_P:-0.7}

# Performance parameters
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 2))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 3))}
ref_offload=${REF_OFFLOAD:-True}
actor_offload=${ACTOR_OFFLOAD:-False}
gen_tp=${GEN_TP:-1}
sp_size=${SP_SIZE:-1}
fsdp_size=${FSDP_SIZE:-2}

# Fully async specific parameters
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
n_gpus_rollout=${N_GPUS_ROLLOUT:-2}
n_gpus_training=${N_GPUS_TRAINING:-2}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-0}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-1}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}
total_rollout_steps=${TOTAL_ROLLOUT_STEPS:-$(((512 * 100)))}
test_freq=${TEST_FREQ:-10}
staleness_threshold=${STALENESS_THRESHOLD:-0.1}
trigger_parameter_sync_step=${TRIGGER_PARAMETER_SYNC_STEP:-4}
require_batches=${REQUIRE_BATCHES:-4}
partial_rollout=${PARTIAL_ROLLOUT:-True}

cd src/verl

# 
pip install "numpy==2.1.*"
# pip install "cupy-cuda12x==13.0.*"

# prevents checkout
rm recipe/README.md
git fetch upstream
# older commit has a checkpoint management error with Lora
# ValueError: There is no module or parameter named 'base_model' in Qwen2ForCausalLM
git checkout 016c1d5a7a3f2973d68fda2f7abe5e7df9e05e00

# Patch vLLM for torch versions where assume_32bit_indexing is absent.
python3 "${REPO_ROOT}/lora/patch_vllm_decorators.py"

PYTHONUNBUFFERED=1 python -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=${lora_target_modules} \
    actor_rollout_ref.hybrid_engine=False \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.checkpoint_engine.backend='nccl' \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.total_epochs=10 \
    trainer.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" \
    actor_rollout_ref.rollout.load_format=safetensors
