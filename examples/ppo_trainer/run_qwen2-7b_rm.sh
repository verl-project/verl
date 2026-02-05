# Discliamer: the model used in the script is only for academic purpose.
set -x

# Data preparation scripts are available in ``examples/data_preprocess``.
# Example usage:
#
#   python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
#   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"


# prepare model ckpt
hf download Qwen/Qwen2-7B-Instruct --local-dir $HOME/models/Qwen2-7B-Instruct &
hf download sfairXC/FsfairX-LLaMA3-RM-v0.1 --local-dir $HOME/models/FsfairX-LLaMA3-RM-v0.1 &
wait

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$HOME/models/Qwen2-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path="$HOME/models/Qwen2-7B-Instruct" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.model.path="$HOME/models/FsfairX-LLaMA3-RM-v0.1" \
    reward_model.use_reward_loop=True \
    reward_model.rollout.name=vllm \
    reward_model.rollout.gpu_memory_utilization=0.8 \
    reward_model.rollout.tensor_model_parallel_size=1 \
    reward_model.rollout.prompt_length=2048 \
    reward_model.rollout.response_length=1024 \
    reward_model.num_workers=8 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_example' \
    trainer.val_before_train=False \
    trainer.experiment_name='Qwen2-7B-Instruct_hybrid_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
