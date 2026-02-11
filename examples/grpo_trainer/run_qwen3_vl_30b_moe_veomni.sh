set -xeuo pipefail

NUM_GPUS=8
NNODES=1
HOME=$pwd

MODEL_ID=Qwen3_VL-30B-MOE
MODEL_PATH=${HOME}/Qwen3-VL-30B-A3B-Instruct
TRAIN_FILES=${HOME}/geo3k/train.parquet
VAL_FILES=${HOME}/geo3k/test.parquet

PROJECT_NAME=${MODEL_ID}-veomni
EXPERIMENT_NAME=grpo-${NUM_GPUS}gpu

DP_SIZE=${DP_SIZE:-8}
SP_SIZE=${SP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}

train_prompt_bsz=8
max_prompt_length=1024
max_response_length=1024

actor_ppo_max_token_len=$(((max_prompt_length + max_response_length)))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length)))

# 执行 GRPO 训练
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_veomni_trainer.yaml' \
    trainer.use_legacy_worker_impl=disable \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.shuffle=False \
    data.truncation='left' \
    data.image_key=images \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.veomni.enable_full_shard=True \
    actor_rollout_ref.actor.veomni.data_parallel_size=${DP_SIZE} \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=${SP_SIZE} \
    actor_rollout_ref.actor.veomni.expert_parallel_size=${EP_SIZE} \
    actor_rollout_ref.actor.veomni.moe_implementation=fused \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.veomni.param_offload=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=3 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=1024 \
    trainer.use_legacy_worker_impl=disable \
    trainer.resume_mode=disable \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10 $@