#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export VERL_LOGGING_LEVEL=INFO
export VERL_PPO_LOGGING_LEVEL=INFO

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
RM_MODEL_PATH=${RM_MODEL_PATH:-${HOME}/models/Skywork/Skywork-Reward-V2-Llama-3.2-1B}
#hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

USE_DUMMY_MODEL=${USE_DUMMY_MODEL:-False}
DUMMY_MODEL_PATH=${DUMMY_MODEL_PATH:-${HOME}/dummy_models/${MODEL_ID}}
if [ "$USE_DUMMY_MODEL" = "True" ]; then
    if [ -z "${DUMMY_MODEL_CONFIG_PATH}"  ]; then
        echo "[ERROR] DUMMY_MODEL_CONFIG_PATH not set"
        exit 1
    fi

    python scripts/init_random_model.py \
        --hf_model_path "${MODEL_PATH}" \
        --new_config_path "${DUMMY_MODEL_CONFIG_PATH}" \
        --output_path "${DUMMY_MODEL_PATH}"

    MODEL_PATH="${DUMMY_MODEL_PATH}"
fi

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-gae}
# Validation
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TEST_FREQ=${TEST_FREQ:--1}
# Save & Resume
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:--1}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN:-2400}
forward_max_token_len_per_gpu=${FWD_MAX_TOKEN_LEN:-4800}
train_traj_micro_bsz_per_gpu=${MICRO_BSZ:-2} # b
n_resp_per_prompt=4 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

LORA_RANK=${LORA_RANK:-0}
CRITIC_LORA_RANK=${CRITIC_LORA_RANK:-$LORA_RANK}
LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"['linear_qkv','linear_proj','linear_fc1','linear_fc2']"}
LORA_MERGE=${LORA_MERGE:-False}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_RM_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

COMMON_PP=${COMMON_PP:-2}
COMMON_VPP=${COMMON_VPP:-2}
COMMON_CP=${COMMON_CP:-2}
COMMON_TP=${COMMON_TP:-2}
COMMON_EP=${COMMON_EP:-1}
COMMON_ETP=${COMMON_ETP:-1}

TRAIN_TP=${TRAIN_TP:-$COMMON_TP}
INFER_TP=${INFER_TP:-$COMMON_TP}

ACTOR_PP=${ACTOR_PP:-$COMMON_PP}
ACTOR_VPP=${ACTOR_VPP:-$COMMON_VPP}
ACTOR_CP=${ACTOR_CP:-$COMMON_CP}
ACTOR_TP=${ACTOR_TP:-$TRAIN_TP}
ACTOR_EP=${ACTOR_EP:-$COMMON_EP}
ACTOR_ETP=${ACTOR_ETP:-$COMMON_ETP}
ROLLOUT_TP=${ROLLOUT_TP:-$INFER_TP}
REF_PP=${REF_PP:-$COMMON_PP}
REF_VPP=${REF_VPP:-$COMMON_VPP}
REF_CP=${REF_CP:-$COMMON_CP}
REF_TP=${REF_TP:-$TRAIN_TP}
REF_EP=${REF_EP:-$COMMON_EP}
REF_ETP=${REF_ETP:-$COMMON_ETP}
CRITIC_PP=${CRITIC_PP:-$COMMON_PP}
CRITIC_VPP=${CRITIC_VPP:-$COMMON_VPP}
CRITIC_CP=${CRITIC_CP:-$COMMON_CP}
CRITIC_TP=${CRITIC_TP:-$TRAIN_TP}
CRITIC_EP=${CRITIC_EP:-$COMMON_EP}
CRITIC_ETP=${CRITIC_ETP:-$COMMON_ETP}

ALL_OFFLOAD=${ALL_OFFLOAD:-False}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
USE_MBRIDGE=${USE_MBRIDGE:-False}
VANILLA_MBRIDGE=${VANILLA_MBRIDGE:-True}
VALUE_VANILLA_MBRIDGE=${VALUE_VANILLA_MBRIDGE:-$VANILLA_MBRIDGE}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}

LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-null}

CHECKPOINT_CONTENTS=['model','hf_model','optimizer','extra']
SKIP_SAVE_HF_MODEL=${SKIP_SAVE_HF_MODEL:-0}
if [ $SKIP_SAVE_HF_MODEL -eq 1 ]; then
    CHECKPOINT_CONTENTS=['model','optimizer','extra']
fi

USE_DIST_CKPT=${USE_DIST_CKPT:-False}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-${HOME}/dist_ckpt/${MODEL_ID}}
if [ "$USE_DIST_CKPT" = "True" ]; then
    if [ "$USE_DUMMY_MODEL" = "True" ]; then
        DIST_CKPT_PATH=${HOME}/dist_ckpt_dummy/${MODEL_ID}
    fi
    python scripts/converter_hf_to_mcore.py \
        --hf_model_path "${MODEL_PATH}" \
        --output_path "${DIST_CKPT_PATH}"
fi

ENGINE=${ENGINE:-"vllm"}
if [ "$ENGINE" = "vllm" ]; then
    export VLLM_USE_V1=1
fi

exp_name="$(basename "${MODEL_ID,,}")-megatron-gsm8k-minimal"
ROLLOUT_MODE="async"
ROLLOUT_QUANTIZATION=${ROLLOUT_QUANTIZATION:-null}

RETURN_RAW_CHAT="True"
SKIP_TOKENIZER_INIT="True"

OPTIM_MEMORY_EFFICIENT=${OPTIM_MEMORY_EFFICIENT:-False}

PROFILE_ENABLE=${PROFILE_ENABLE:-False}
PROFILE_STEPS=${PROFILE_STEPS:-[1]}
PROFILE_RANKS_ALL=${PROFILE_RANKS_ALL:-True}
PROFILE_RANKS=${PROFILE_RANKS:-[0,1,2,3]}
DISCRETE=${DISCRETE:-True}  # or True

python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.return_raw_chat=${RETURN_RAW_CHAT} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS} \
    actor_rollout_ref.model.lora.rank=${LORA_RANK} \
    actor_rollout_ref.model.lora.alpha=${LORA_ALPHA} \
    actor_rollout_ref.model.lora.target_modules=${LORA_TARGET_MODULES} \
    actor_rollout_ref.model.lora.merge=${LORA_MERGE} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=$OPTIM_MEMORY_EFFICIENT \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=$OPTIM_MEMORY_EFFICIENT \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=$OPTIM_MEMORY_EFFICIENT \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=${VANILLA_MBRIDGE} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$ACTOR_PP \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$ACTOR_VPP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$ACTOR_CP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$ACTOR_TP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$ACTOR_EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ACTOR_ETP \
    actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.checkpoint.save_contents=$CHECKPOINT_CONTENTS \
    actor_rollout_ref.actor.profiler.enable=$PROFILE_ENABLE \
    actor_rollout_ref.actor.profiler.ranks=$PROFILE_RANKS \
    actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.mode="${ROLLOUT_MODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    ++actor_rollout_ref.rollout.quantization=${ROLLOUT_QUANTIZATION} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    actor_rollout_ref.ref.megatron.use_mbridge=${USE_MBRIDGE} \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=${VANILLA_MBRIDGE} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$REF_PP \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=$REF_VPP \
    actor_rollout_ref.ref.megatron.context_parallel_size=$REF_CP \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$REF_TP \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$REF_EP \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$REF_ETP \
    actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
    critic.optim.lr=2e-5 \
    critic.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    +critic.optim.override_optimizer_config.optimizer_cpu_offload=$OPTIM_MEMORY_EFFICIENT \
    +critic.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=$OPTIM_MEMORY_EFFICIENT \
    +critic.optim.override_optimizer_config.use_precision_aware_optimizer=$OPTIM_MEMORY_EFFICIENT \
    critic.model.path="${MODEL_PATH}" \
    critic.model.lora.rank=${CRITIC_LORA_RANK} \
    critic.model.lora.alpha=${LORA_ALPHA} \
    critic.model.lora.target_modules=${LORA_TARGET_MODULES} \
    critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
    critic.ppo_max_token_len_per_gpu=${forward_max_token_len_per_gpu} \
    critic.megatron.use_mbridge=${USE_MBRIDGE} \
    critic.megatron.vanilla_mbridge=${VALUE_VANILLA_MBRIDGE} \
    critic.megatron.pipeline_model_parallel_size=$CRITIC_PP \
    critic.megatron.virtual_pipeline_model_parallel_size=$CRITIC_VPP \
    critic.megatron.context_parallel_size=$CRITIC_CP \
    critic.megatron.tensor_model_parallel_size=$CRITIC_TP \
    critic.megatron.expert_model_parallel_size=$CRITIC_EP \
    critic.megatron.expert_tensor_parallel_size=$CRITIC_ETP \
    critic.megatron.param_offload=${CRITIC_PARAM_OFFLOAD} \
    critic.megatron.optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD} \
    critic.megatron.grad_offload=${CRITIC_GRAD_OFFLOAD} \
    critic.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    critic.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
    critic.checkpoint.save_contents=$CHECKPOINT_CONTENTS \
    critic.profiler.enable=$PROFILE_ENABLE \
    critic.profiler.ranks=$PROFILE_RANKS \
    critic.profiler.all_ranks=$PROFILE_RANKS_ALL \
    reward_model.enable=True \
    reward_model.model.path="${RM_MODEL_PATH}" \
    reward_model.use_reward_loop=True \
    reward_model.rollout.name=${ENGINE} \
    reward_model.rollout.gpu_memory_utilization=0.6 \
    reward_model.rollout.tensor_model_parallel_size=${INFER_TP} \
    reward_model.rollout.prompt_length=${MAX_RM_LENGTH} \
    reward_model.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    reward_model.num_workers=8 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl-test' \
    trainer.experiment_name="${exp_name}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.total_epochs=2 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    global_profiler.profile_continuous_steps=True \
    global_profiler.tool=nsys \
    global_profiler.steps=$PROFILE_STEPS \
    global_profiler.global_tool_config.nsys.discrete=$DISCRETE $@
