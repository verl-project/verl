#!/bin/bash
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    atropos.host=localhost \
    atropos.port=8000 \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    trainer.project_name='verl_atropos_grpo' \
    trainer.experiment_name='atropos_grpo' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.total_epochs=15 \
    trainer.trainer_cls=verl_atropos_reflex.AtroposVerlTrainer $@
