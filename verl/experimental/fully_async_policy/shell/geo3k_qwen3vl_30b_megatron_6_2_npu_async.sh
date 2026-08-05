set -x
ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_V1=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0

HF_MODEL_PATH=/home/model/Qwen3-VL-30B-A3B-Instruct

train_path=/data/geo3k/train.parquet
test_path=/data/geo3k/test.parquet

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4
total_rollout_steps=$(((64*160)))
test_freq=10
require_batches=1
partial_rollout=True
total_epochs=200
PROJECT_NAME='ASYNC_training'

####################### with negative KL uncomment
EXP_NAME='qwen3_vl_3b_megatron_async'
enable_chunked_prefill=False
train_prompt_mini_bsz=16
staleness_threshold=0
trigger_parameter_sync_step=4
max_prompt_length=$((1024))
max_response_length=$((1024 * 16))
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
NNODES=${NNODES:-2}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
n_gpus_rollout=12
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))
GEN_TP=${GEN_TP:-4}
CP=${CP:-1}
TP=${TP:-4}
PP=${PP:-2}
EP=${EP:-4}
ETP=${ETP:-1}
use_kl_loss=False
loss_mode=geo_mean
clip_ratio=0.4



python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_steps=51200 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size=${CP} \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GEN_TP \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.ref.megatron.param_offload=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=$EXP_NAME \
    trainer.test_freq="${test_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.test_freq="${test_freq}" \
    trainer.device=npu \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}" 2>&1 | tee "./logs/${PROJECT_NAME}/run_${EXP_NAME}.log"