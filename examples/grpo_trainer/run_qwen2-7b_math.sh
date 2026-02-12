set -x

# NCCL Configuration for GB200 NVL with Multiple NICs
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# Network configuration for multi-NIC setup
# Use all available high-speed interfaces (you have 6 NICs)
# Exclude loopback, docker, and veth interfaces
export NCCL_SOCKET_IFNAME=^lo,docker,veth

# GB200 NVLink configuration
export NCCL_IB_DISABLE=1           # Disable InfiniBand (not available)
export NCCL_P2P_DISABLE=0          # Enable P2P (required for NVLink)
export NCCL_P2P_LEVEL=NVL          # Use NVLink (confirmed available: NV18)
export NCCL_SHM_DISABLE=0          # Enable shared memory
export NCCL_NET_GDR_LEVEL=PHB      # GPUDirect RDMA

# Network tuning for multi-NIC
export NCCL_SOCKET_NTHREADS=8      # More threads for 6 NICs
export NCCL_NSOCKS_PERTHREAD=8     # More sockets
export NCCL_MIN_NCHANNELS=8        # Utilize multiple NICs

# Timeout
export NCCL_TIMEOUT=1800           # 30 min timeout

# Cross-NUMA node optimization (GPU0-1 on NUMA0, GPU2-3 on NUMA1)
export NCCL_CROSS_NIC=1            # Allow cross-NIC communication

gsm8k_train_path=$PWD/data/gsm8k/train.parquet
gsm8k_test_path=$PWD/data/gsm8k/test.parquet
math_train_path=$PWD/data/math/train.parquet
math_test_path=$PWD/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/lustre/fsw/portfolios/coreai/users/erinh/llm-models/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k_math' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_training_steps=3 \
    trainer.total_epochs=15 $@
