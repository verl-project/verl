set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

HF_MODEL_PATH=moonshotai/Moonlight-16B-A3B
DIST_CKPT_PATH=${DIST_CKPT_PATH}

train_path=$HOME/data/gsm8k/train.parquet
test_path=$HOME/data/gsm8k/test.parquet

python3 -m verl.trainer.main_ppo \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_path" \
    data.val_files="$test_path" \
    data.train_batch_size=192 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=3 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=3 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='moonlight_megatron_ep' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=3 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
