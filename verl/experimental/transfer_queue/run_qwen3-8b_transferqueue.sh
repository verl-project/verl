set -x

MODEL_PATH="/workspace/models/Qwen3-8B"
TRAIN_FILE="/workspace/datasets/preprocessed/gsm8k/train.parquet"
TEST_FILE="/workspace/datasets/preprocessed/gsm8k/test.parquet"

log_dir="./logs"
mkdir -p ${log_dir}
timestamp=$(date +"%Y%m%d%H%M%S")
log_file="${log_dir}/qwen3-8b_tq_${timestamp}.log"

# You may try to enable zero-copy serialization for TransferQueue when using SimpleStorageUnit backend.
export TQ_ZERO_COPY_SERIALIZATION=False

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# You may also refer to tests/special_e2e/run_transferqueue.sh for more demo scripts

python3 -m verl.experimental.transfer_queue.main_ppo \
    --config-name='transfer_queue_ppo_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.return_raw_chat=$return_raw_chat \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts_workers=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_8b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1000 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=2 \
    trainer.val_before_train=False \
    2>&1 | tee "$log_file"
echo "Finished, log is saved in: $log_file"