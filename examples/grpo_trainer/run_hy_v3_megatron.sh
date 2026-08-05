#!/usr/bin/env bash
# GRPO | Hy V3 (MoE) | Megatron training | NVIDIA GPUs
# DAPO-style recipe on DAPO-Math-17k / AIME-2024.
#
# Knobs:
#   INFER_BACKEND          rollout backend: vllm | sglang | trtllm   (default: vllm)
#                          NOTE: rollout must have hy3 model support.

set -xeuo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1

########################### user-adjustable ###########################
INFER_BACKEND=${INFER_BACKEND:-vllm}

DATA_DIR=${DATA_DIR:-"$PWD"}
MODEL_PATH=${MODEL_PATH:-tencent/Hy3}
NNODES=${NNODES:-16}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-${GPUS_PER_NODE:-8}}

train_files=${TRAIN_FILES:-${DAPO_MATH_TRAIN:-${DATA_DIR}/DAPO-Math-17k/dapo-math-17k.parquet}}
val_files=${VAL_FILES:-${AIME_VAL:-${DATA_DIR}/AIME-2024/aime-2024.parquet}}

train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-30720}

actor_lr=${ACTOR_LR:-1e-6}
min_lr=${MIN_LR:-1e-6}

rollout_is=${ROLLOUT_IS:-token}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-0.5_4.0}
entropy_coeff=${ENTROPY_COEFF:-0}
actor_clip_ratio_low=${ACTOR_CLIP_RATIO_LOW:-0.2}
actor_clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH:-0.28}
actor_clip_ratio_c=${ACTOR_CLIP_RATIO_C:-10.0}
actor_ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}

use_kl_loss=${USE_KL_LOSS:-True}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

actor_tp=${ACTOR_TP:-1}
actor_pp=${ACTOR_PP:-16}
actor_vpp=${ACTOR_VPP:-null}
actor_ep=${ACTOR_EP:-8}
actor_cp=${ACTOR_CP:-8}
actor_etp=${ACTOR_ETP:-1}
ref_tp=${REF_TP:-${actor_tp}}
ref_pp=${REF_PP:-${actor_pp}}
ref_vpp=${REF_VPP:-${actor_vpp}}
ref_ep=${REF_EP:-${actor_ep}}
ref_cp=${REF_CP:-${actor_cp}}
ref_etp=${REF_ETP:-${actor_etp}}
all_offload=${ALL_OFFLOAD:-True}

rollout_tp=${ROLLOUT_TP:-16}
infer_tp=${INFER_TP:-${rollout_tp}}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.65}
rollout_n=${ROLLOUT_N:-16}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-10240}
rollout_max_model_len=${ROLLOUT_MAX_MODEL_LEN:-10240}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.9}
rollout_top_p=${ROLLOUT_TOP_P:-1}

ref_log_prob_max_token_len_per_gpu=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}
ref_log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
rollout_log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-40960}
rollout_log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}

val_do_sample=${VAL_DO_SAMPLE:-True}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-1.0}
val_n=${VAL_N:-1}
log_val_generations=${LOG_VAL_GENERATIONS:-10}

total_epochs=${TOTAL_EPOCHS:-1000}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_grpo_hy_v3}
experiment_name=${EXPERIMENT_NAME:-hy_v3_a3b_${INFER_BACKEND}_megatron}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
[ "${actor_pp}" -gt 1 ] && actor_vpp_override=${actor_vpp} || actor_vpp_override=null
[ "${ref_pp}" -gt 1 ] && ref_vpp_override=${ref_vpp} || ref_vpp_override=null

########################### parameter arrays ###########################

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
)

REWARD=(
    reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
)

DATA=(
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=False
    data.truncation=left
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.optim.min_lr=${min_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${actor_ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.clip_ratio_low=${actor_clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${actor_clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=${actor_clip_ratio_c}
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    # Parallelism.
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${actor_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${actor_pp}
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${actor_vpp_override}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${actor_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${actor_etp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${actor_cp}
    actor_rollout_ref.actor.megatron.param_offload=${all_offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${all_offload}
    actor_rollout_ref.actor.megatron.grad_offload=${all_offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_enable_expert_bias=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=none
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${infer_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${rollout_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${rollout_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length}
    actor_rollout_ref.rollout.response_length=${max_response_length}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.n=${val_n}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ref_log_prob_max_token_len_per_gpu}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${ref_log_prob_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_pp}
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${ref_vpp_override}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ref_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ref_etp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${ref_cp}
    actor_rollout_ref.ref.megatron.param_offload=${all_offload}
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.resume_mode=auto
    trainer.val_before_train=True
    trainer.log_val_generations=${log_val_generations}
)

EXTRA=(
    model_engine=megatron
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
