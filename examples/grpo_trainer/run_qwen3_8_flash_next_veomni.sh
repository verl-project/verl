#!/usr/bin/env bash
# GRPO | Qwen3-30B-A3B | Megatron training | vLLM rollout
#
# Rollout: FP8 MoE weights and FP8 KV cache
# Training: TileLang fused kernels and R3 router replay
# Algorithm: score centering (SC) with token-level importance sampling (TIS)
# Trainer: colocate_async
#
# Score centering uses bypass-mode REINFORCE with token-level importance
# sampling to remove the training-inference drift term from the policy gradient.

set -xeuo pipefail

# ---- user-adjustable ----
n_gpus_per_node=${NGPUS_PER_NODE:-8}
trainer_nnodes=${NNODES:-1}

project_name='wuxibin_dapo'
exp_name="qwen3_8_flash_next_score_centering_veomni_1125"

# ===================================== Algorithm =====================================
adv_estimator=rloo
loss_mode=bypass_mode
rollout_is=token
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
bypass_mode=true
loss_type=reinforce
loss_agg_mode=token-mean
loss_scale_factor=null
score_centering=true

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
entropy_coeff=0.0

clip_ratio_low=3e-4
clip_ratio_high=4e-4

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# ===================================== Data/Model =====================================
model_path=$HDFS_ROOT/model/Qwen/Qwen3.8-Flash-Next
train_files=$DATA_ROOT/dataset/geo3k/train.parquet
test_files=$DATA_ROOT/dataset/geo3k/test.parquet
checkpoint_dir=$DATA_ROOT/checkpoint/$project_name/$exp_name

train_batch_size=64
ppo_mini_batch_size=64
n_resp_per_prompt=8
n_resp_per_prompt_val=16
enable_thinking=${enable_thinking:-True}

NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-4}
MAX_OFF_POLICY_THRESHOLD=${MAX_OFF_POLICY_THRESHOLD:-16}

# Training config
EP=${EP:-8}
SP=${SP:-2}
ENGRAM_TP=${ENGRAM_TP:-8}

use_remove_padding=True
use_dynamic_bsz=True
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 10))
actor_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 2))

# Inference config
rollout_name=vllm
infer_tp=4
infer_dp=1
infer_ep=1
gpu_memory_utilization=0.8
# ---- end user-adjustable ----

# ---- no user adjustment needed below ----
ENGINE_CONFIG=(
    model_engine=veomni
    # actor_rollout_ref.actor.veomni.enable_fsdp_offload=True
    actor_rollout_ref.actor.veomni.param_offload=True
    actor_rollout_ref.actor.veomni.optimizer_offload=True
    actor_rollout_ref.actor.veomni.enable_async_activation_offload=True
    actor_rollout_ref.actor.veomni.enable_full_shard=True
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=${SP}
    actor_rollout_ref.actor.veomni.expert_parallel_size=${EP}
    actor_rollout_ref.actor.veomni.ple_parallel_size=${ENGRAM_TP}
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2
    actor_rollout_ref.actor.veomni.moe_implementation=fused_triton
    actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation=liger_kernel
    actor_rollout_ref.actor.veomni.rms_norm_implementation=liger_kernel
    actor_rollout_ref.actor.veomni.rms_norm_gated_implementation=fla
    actor_rollout_ref.actor.veomni.causal_conv1d_implementation=fla
    actor_rollout_ref.actor.veomni.chunk_gated_delta_rule_implementation=fla
    actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation=liger_kernel
    actor_rollout_ref.actor.veomni.rotary_pos_emb_vision_implementation=liger_kernel
    actor_rollout_ref.actor.veomni.qsa_attention_implementation=tilelang
    actor_rollout_ref.actor.veomni.gated_mlp_implementation=liger_kernel
    actor_rollout_ref.actor.veomni.load_balancing_loss_implementation=triton
    actor_rollout_ref.actor.veomni.router_replay.mode=R3
)

# Actor model config
ACTOR_CONFIG=(
    actor_rollout_ref.model.path=$model_path
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.fused_kernel_options.impl_backend=tilelang
    actor_rollout_ref.actor.optim.lr=$actor_lr
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff
    actor_rollout_ref.actor.calculate_entropy=True
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}
    '+actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}'
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode
    actor_rollout_ref.actor.loss_scale_factor=$loss_scale_factor
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz
    actor_rollout_ref.actor.pad_to_length=True
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${actor_max_token_len_per_gpu}
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.name=$rollout_name
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization
    actor_rollout_ref.rollout.n=$n_resp_per_prompt
    actor_rollout_ref.rollout.topk_log_probs=128
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    actor_rollout_ref.rollout.quantization.moe=FP8_BLOCK
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8
)

DATA=(
    algorithm.adv_estimator=$adv_estimator
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=$use_kl_in_reward
    algorithm.kl_ctrl.kl_coef=$kl_coef
    algorithm.gamma=$gae_gamma
    algorithm.lam=$gae_lam
    algorithm.rollout_correction.rollout_is=$rollout_is
    algorithm.rollout_correction.rollout_is_threshold=$rollout_is_threshold
    algorithm.rollout_correction.bypass_mode=$bypass_mode
    algorithm.rollout_correction.loss_type=$loss_type
    algorithm.rollout_correction.score_centering=$score_centering
    data.train_files="$train_files"
    data.val_files="$test_files"
    data.return_raw_chat=True
    data.train_batch_size=$train_batch_size
    data.max_prompt_length=$max_prompt_length
    data.max_response_length=$max_response_length
    data.filter_overlong_prompts=False
    data.filter_overlong_prompts_workers=64
    data.truncation='error'
)

TRAINER=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches=${NUM_WARMUP_BATCHES}
    trainer.v1.sampler.max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD}
    trainer.critic_warmup=$critic_warmup
    trainer.logger=['console','swanlab']
    trainer.project_name=$project_name
    trainer.experiment_name=$exp_name
    trainer.n_gpus_per_node=${n_gpus_per_node}
    trainer.nnodes=$trainer_nnodes
    trainer.val_before_train=False
    trainer.log_val_generations=100
    trainer.save_freq=-1
    trainer.test_freq=80
    trainer.max_actor_ckpt_to_keep=2
    trainer.default_local_dir=$checkpoint_dir
    trainer.total_epochs=10
    trainer.total_training_steps=500
)

########################### launch ###########################
python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${TRAINER[@]}" \
    "${ENGINE_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${RUNTIME_ENV[@]}" \
    "$@"
