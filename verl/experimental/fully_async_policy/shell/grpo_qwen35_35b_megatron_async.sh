#!/usr/bin/env bash
# Qwen3.5-35B-A3B GRPO with Megatron backend + MTP + Fully Async Policy
#
# Requirements:
#     pip install --upgrade transformers==5.3.0
#     mbridge: make sure https://github.com/ISEEKYAN/mbridge/pull/98 this pr has merged
#
# MTP (Multi-Token Prediction) notes:
#   - actor_rollout_ref.model.mtp.enable=True        enables MTP module
#   - actor_rollout_ref.model.mtp.enable_train=True  enables MTP training loss
#   - actor_rollout_ref.model.mtp.enable_rollout=True enables speculative decoding in SGLang
#
# Example parallelism configs for Qwen3.5-35B-A3B:
#   16 GPUs (2 nodes): train_tp=4  train_pp=2  EP=4  gen_tp=8
#
# Run:
#     NNODES_TRAIN=1 NNODES_ROLLOUT=1 bash grpo_qwen35_35b_megatron_async.sh

set -xeuo pipefail

# ================= data / model =================
MODEL_PATH=${MODEL_PATH:-~/models/Qwen3.5-35B-A3B}
TRAIN_FILE=${TRAIN_FILE:-~/data/train.parquet}
TEST_FILE=${TEST_FILE:-~/data/test.parquet}

project_name=${PROJECT_NAME:-'Qwen3.5-35B-A3B-grpo-mtp-megatron'}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H)_exp"}
CKPTS_DIR=${CKPTS_DIR:-~/checkpoints/${project_name}/${exp_name}}
mkdir -p "${CKPTS_DIR}"

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=True
kl_coef=0.01
use_kl_loss=True
kl_loss_coef=0.02

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=6
max_prompt_length=$((1024 * 12))
max_response_length=$((1024 * 4))

loss_mode="gspo"
loss_agg_mode="seq-mean-token-mean"

enable_filter_groups=True
filter_groups_metric=response_filter
max_num_gen_batches=10

temperature=0.7
top_p=0.75
val_top_p=0.7

# ================= performance =================
# Qwen3.5-35B-A3B: train_tp=4 train_pp=2 EP=4 for 1 node (8 GPUs)
gen_tp=${GEN_TP:-8}
train_tp=${TP:-4}
train_pp=${PP:-2}
EP=${EP:-4}
ETP=1
CP=1

offload=True
OPTIM_OFFLOAD=${OPTIM_OFFLOAD:-False}
optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}

use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * max_turns))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * max_turns * 2))

# ================= async policy =================
rollout_name="sglang"
rollout_mode="async"
if [ "$rollout_mode" = "async" ]; then
    return_raw_chat="True"
fi

NNODES_ROLLOUT=${NNODES_ROLLOUT:-1}
NNODES_TRAIN=${NNODES_TRAIN:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=0
gen_prompt_bsz=1
n_resp_per_prompt=8
ppo_mini_batch_size=16
total_rollout_steps=$(((64 * 32 * 10)))
test_freq=10
staleness_threshold=0.5
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True
val_before_train=False

# ================= MTP params =================
mtp_params=(
  actor_rollout_ref.model.mtp.enable=True
  actor_rollout_ref.model.mtp.enable_train=True
  actor_rollout_ref.model.mtp.mtp_loss_scaling_factor=0.1
  actor_rollout_ref.model.mtp.detach_encoder=True
  actor_rollout_ref.model.mtp.enable_rollout=True
)

if [ "$rollout_name" = "sglang" ]; then
  mtp_params+=(
    actor_rollout_ref.model.mtp.speculative_algorithm="NEXTN"
    actor_rollout_ref.model.mtp.speculative_num_steps=3
    actor_rollout_ref.model.mtp.speculative_eagle_topk=1
    actor_rollout_ref.model.mtp.speculative_num_draft_tokens=4
  )
fi

CHECKPOINT_CONTENTS=['model','hf_model','extra']

python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=$(((max_prompt_length + max_response_length) * max_turns - max_prompt_length)) \
    data.train_batch_size="${train_batch_size}" \
    data.return_raw_chat=True \
    data.gen_batch_size=${gen_prompt_bsz} \
    +data.apply_chat_template_kwargs.thinking=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=$(((max_prompt_length + max_response_length) * max_turns - max_prompt_length)) \
    actor_rollout_ref.rollout.single_turn_response_length=${max_response_length} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.megatron.param_offload=False \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    actor_rollout_ref.actor.checkpoint.save_contents=${CHECKPOINT_CONTENTS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_model_len=$(((max_prompt_length + max_response_length) * max_turns)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ETP} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${CP} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${max_turns} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_prompt_length} \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=1024 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mamba_scheduler_strategy=no_buffer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_memory_saver=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_weights_cpu_backup=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_draft_weights_cpu_backup=False \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=15 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.nnodes=${NNODES_TRAIN} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.nnodes=${NNODES_ROLLOUT} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.total_rollout_steps=${total_rollout_steps} \
    trainer.total_epochs=10 \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=${val_before_train} \
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout} \
    "${mtp_params[@]}" \
    "$@"
