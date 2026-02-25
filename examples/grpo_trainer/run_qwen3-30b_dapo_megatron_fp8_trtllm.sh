set -x

# Clean all slurm / MPI / PMIx env to avoid pmix mismatch error
for v in $(env | awk -F= '/^(PMI|PMIX|MPI|OMPI|SLURM)_/{print $1}'); do
    unset "$v"
done

export RAY_DEDUP_LOGS=0

# -----
# Config
# -----
TP=${1:-4}
ACTOR_TP=${ACTOR_TP:-4}
ACTOR_PP=${ACTOR_PP:-2}
ACTOR_VPP=${ACTOR_VPP:-2}
ACTOR_EP=${ACTOR_EP:-2}
REF_PP=${REF_PP:-2}
REF_VPP=${REF_VPP:-2}
REF_EP=${REF_EP:-2}
GEN_MOE_TP=${GEN_MOE_TP:-2}
GEN_MOE_EP=${GEN_MOE_EP:-2}
PROJECT_NAME=${PROJECT_NAME:-"Qwen3-30B-A3B-DAPO-GB200"}
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# MOE backend for TRTLLM when using FP8 quantization:
#   - Blackwell: use DEEPGEMM
#   - Hopper: use CUTLASS
TRTLLM_MOE_BACKEND=${TRTLLM_MOE_BACKEND:-"DEEPGEMM"}
EXP_NAME=qwen3-30b-dapo-megatron-fp8-trtllm-n${NNODES}-tp${TP}-moe-tp${GEN_MOE_TP}-moe-ep${GEN_MOE_EP}${EXP_NAME_SUFFIX:+"-"}${EXP_NAME_SUFFIX}

if [ $TP -eq 4 ] || [ $TP -eq 2 ]; then
    MAX_NUM_SEQS=1024
else
    MAX_NUM_SEQS=384
fi

# -----
# Data
# -----
DATA_DIR=${DATA_DIR:-"$PWD"}

DAPO_MATH_TRAIN=${DAPO_MATH_TRAIN:-"${DATA_DIR}/data/DAPO-Math-17k/data/dapo-math-17k.parquet"}
AIME_VAL=${AIME_VAL:-"${DATA_DIR}/data/AIME-2024/data/aime-2024.parquet"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-30B-A3B-Base"}

# When PP=1, Megatron interleaved schedule is invalid; pass null so PP=1 works (e.g. 2-node)
[ "${ACTOR_PP}" -gt 1 ] && ACTOR_VPP_OVERRIDE=${ACTOR_VPP} || ACTOR_VPP_OVERRIDE=null
[ "${REF_PP}" -gt 1 ] && REF_VPP_OVERRIDE=${REF_VPP} || REF_VPP_OVERRIDE=null

# -----
# Launch
# -----
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=8192 \
    data.train_files="${DAPO_MATH_TRAIN}" \
    data.val_files="${AIME_VAL}" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.truncation=left \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.train_batch_size=512 \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=30720 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP_OVERRIDE} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.param_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=40960 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP_OVERRIDE} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.rollout.name=trtllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_timeout_iters=32 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_max_tokens_ratio=0.5 \
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.moe_config.backend=${TRTLLM_MOE_BACKEND} \
    +actor_rollout_ref.rollout.moe_tp_size=${GEN_MOE_TP} \
    +actor_rollout_ref.rollout.moe_ep_size=${GEN_MOE_EP} \
    +actor_rollout_ref.rollout.quantization=fp8 \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.resume_mode=auto \
    trainer.total_epochs=1000 \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    "${@:2}"
