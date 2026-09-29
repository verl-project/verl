#!/usr/bin/env bash
set -xeuo pipefail

# DeepSeek-V4.1-Flash VLM BF16 LoRA merge=False — Geo3K alignment test.
# Follow the GLM-5.3-Flash Megatron LoRA/R3 recipe, with V4.1 model constraints.

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

project_name='DAPO'
exp_name=${EXPERIMENT_NAME:-'DAPO-deepseek-v41-flash-megatron-lora-vl-geo3k'}

adv_estimator=grpo
gamma=1.0

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
enable_overlong_buffer=False
overlong_buffer_len=${max_response_length}
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=${TRAIN_BATCH_SIZE:-64}
n_resp_per_prompt=${ROLLOUT_N:-4}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-32}

rollout_is="token"
rollout_is_threshold="0.5_5.0"
rollout_is_batch_normalize="true"

bypass_mode="false"
loss_type="ppo_clip"

ROUTING_REPLAY_MODE="R3"

NNODES=${NNODES:-4}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/DeepSeek-V4.1-Flash"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}

temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

use_dynamic_bsz=True
actor_ppo_max_token_len=${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}
infer_ppo_max_token_len=${actor_ppo_max_token_len}
gen_tp=${ROLLOUT_TP:-8}
gen_ep=${gen_tp}
gen_pp=1
gen_dcp=1
train_tp=1
train_pp=1
train_ep=$((NNODES * 8))
train_etp=1
train_cp=1

lora_rank=16
lora_alpha=32

ROLLOUT_RESYNC_BASE=${ROLLOUT_RESYNC_BASE:-True}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-2048}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${ROLLOUT_MAX_MODEL_LEN}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
OPTIMIZER_OFFLOAD_FRACTION=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-3}
ENGRAM_CPU_LOOKUP=${ENGRAM_CPU_LOOKUP:-True}

if ((gen_tp < 1 || 24 % gen_tp != 0 || train_ep % gen_tp != 0)); then
    echo "DeepSeek-V4.1 Engram requires rollout TP/EP to divide 24 (got ${gen_tp})." >&2
    exit 1
fi

run() {
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

run python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.image_key=images \
    data.trust_remote_code=True \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.gamma=${gamma} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.loss_type=${loss_type} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.model.lora.experts_shared_outer_loras=True \
    actor_rollout_ref.model.lora.lora_plus_ratio=16.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.mtp.enable=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    +actor_rollout_ref.model.override_config.num_nextn_predict_layers=0 \
    actor_rollout_ref.model.lora.rank=${lora_rank} \
    actor_rollout_ref.model.lora.alpha=${lora_alpha} \
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming \
    actor_rollout_ref.model.lora.merge=False \
    actor_rollout_ref.model.lora.resync_base=${ROLLOUT_RESYNC_BASE} \
    actor_rollout_ref.model.lora.dtype=bf16 \
    actor_rollout_ref.model.lora.target_modules='["linear_wkv","linear_wgate","linear_wq_b","linear_weights_proj","linear_wk","linear_kv_proj","linear_q_down_proj","linear_q_up_proj","linear_proj","linear_fc1","linear_fc2","vision.patch_embed.proj","vision.blocks.*.attn.wqkv","vision.blocks.*.attn.wo","vision.blocks.*.mlp.w1","vision.blocks.*.mlp.w2","aligner.w1","aligner.w2"]' \
    actor_rollout_ref.model.lora.enable_tower_connector_lora=True \
    actor_rollout_ref.model.lora.freeze_vision_model=True \
    actor_rollout_ref.model.lora.freeze_vision_projection=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=5e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.sequence_parallel=False \
    actor_rollout_ref.actor.megatron.use_megatron_fsdp=False \
    actor_rollout_ref.actor.megatron.param_offload=${PARAM_OFFLOAD} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.pipeline_parallel_size=${gen_pp} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode_context_parallel_size=${gen_dcp} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.megatron.router_replay.mode=${ROUTING_REPLAY_MODE} \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_bias_update_rate=0.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_shared_expert_overlap=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_kernel_backend=cudnn \
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_use_sparse_loss=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.dsa_indexer_loss_coeff=0.0 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_mhc=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=selective \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_modules='["moe","shared_experts","moe_act","mla_up_proj","mhc"]' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.mhc_recompute_layer_num=2 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.engram_cpu_lookup=${ENGRAM_CPU_LOOKUP} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=False \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${OPTIMIZER_OFFLOAD} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=False \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep} \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=${gen_pp} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_moe_shared_loras=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides.num_nextn_predict_layers=0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
    actor_rollout_ref.ref.megatron.sequence_parallel=False \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.param_offload=${PARAM_OFFLOAD} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=5 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    "$@"
