set -x

project_name='GRPO'
exp_name='GRPO-Qwen3-0.6b-gsm8k-fsdp2-sglang-one-step-off-2-6'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-0.6B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

n_gpus_rollout=2
n_gpus_training=$((NGPUS_PER_NODE - n_gpus_rollout))


python3 -m verl.experimental.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=1152 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=192 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${n_gpus_training}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${n_gpus_rollout}" $@