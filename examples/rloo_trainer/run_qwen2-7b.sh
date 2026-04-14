#!/usr/bin/env bash
set -x

device_name=${device_name:-cuda}

common_params=(
    algorithm.adv_estimator=rloo
    data.train_files="$HOME/data/gsm8k/train.parquet"
    data.val_files="$HOME/data/gsm8k/test.parquet"
    data.train_batch_size=1024
    data.max_prompt_length=512
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation='error'
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=256
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=160
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.n=5
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=160
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    algorithm.use_kl_in_reward=True
    algorithm.kl_penalty=kl
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name='verl_rloo_example_gsm8k'
    trainer.experiment_name='qwen2_7b_function_rm'
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    trainer.device="${device_name}"
    trainer.save_freq=-1
    trainer.test_freq=5
    trainer.total_epochs=15
)

if [ "$device_name" ] && [ "$device_name" == "npu" ]; then
    common_params+=(
        actor_rollout_ref.actor.ppo_mini_batch_size=128
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=80
        trainer.n_gpus_per_node=16
    )
fi

python3 -m verl.trainer.main_ppo "${common_params[@]}" "$@"
