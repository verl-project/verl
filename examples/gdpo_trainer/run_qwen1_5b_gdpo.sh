export DATA_DIR="verl/dataset/rlla_4k"
export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export EXPERIMENT_NAME="qwen2.5-1.5B-GDPO"
export CKPT_DIR="verl/results/gdpo"

# Env variables for computing score in rlla.py
export REFINEDREWARD=0
export COARSEREWARD=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

PROJECT_DIR="$(pwd)"

trainer_n_gpus_per_node=8
trainer_nnodes=1

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gdpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward.custom_reward_function.path="$PROJECT_DIR/verl/utils/reward_score/rlla.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=Var_inspect \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$trainer_nnodes \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_epochs=15 \
    trainer.val_before_train=False 2>&1 | tee ${LOG_PATH}