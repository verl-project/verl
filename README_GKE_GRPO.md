# Running GRPO on GKE A3/H200

This guide outlines the steps required to deploy a Ray cluster and successfully run a `verl` Group Relative Policy Optimization (GRPO) training job on Google Kubernetes Engine (GKE) A3 Ultra instances equipped with H200 GPUs.

## 1. Cluster Setup & Topology Constraints

Because GKE A3 Ultra instances utilize an advanced 8-card RDMA optical network, the `networking.gke.io/interfaces` annotation physically binds a worker pod to all 8 hardware NICs simultaneously. 
**You can only schedule 1 Ray worker per physical node.**

Assuming your cluster has 2 physical A3 instances:
*   Apply the pre-configured RayCluster template which requests exactly two workers with 8 GPUs each:
```bash
kubectl apply -f verl-inference-scheduler.yaml
```

## 2. Establish Head Node Connection

Before you can submit a job, you must open a local tunnel to the Ray Head Node dashboard. This connection must remain alive while the job is being submitted.

1.  Establish port forward:
```bash
kubectl port-forward svc/verl-inference-scheduler-head-svc 8265:8265 -n default &
```
2.  Export the Ray address so the CLI knows where to target:
```bash
export RAY_ADDRESS="http://127.0.0.1:8265"
```

## 3. Submit the GRPO Job

The architecture requires configuration to be split into two tiers to prevent operating system crashes:
*   **Hardware bindings** (like `LD_LIBRARY_PATH`) live in the Kubernetes YAML.
*   **Networking behavior** (like `GLOO_SOCKET_IFNAME`, `NCCL_NET_GDR_LEVEL`, and `pip` dependencies) live in `runtime-env.yaml`.

Run the official example script, injecting the runtime environment and overriding the hardware dimensions to match our 2-node, 16-GPU cluster.

> [!WARNING]  
> You **must** override `model_dtype=bfloat16`. 
> Qwen models download as `float32` by default, which physically cannot be computed by the optimized Flash Attention 2 kernels running on Hopper/H200 GPUs. Failing to set this will result in immediate Tensor crashes.

```bash
ray job submit \
    --working-dir . \
    --runtime-env runtime-env.yaml \
    -- bash examples/grpo_trainer/run_qwen2-7b_math.sh \
    data.train_files="['gsm8k/train.parquet']" \
    data.val_files="['gsm8k/test.parquet']" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=wpi \
    actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.wpi.buffer_id=verl-weight-buffer' \
    actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.wpi.socket_dir=/run/wpi/sockets' \
    trainer.logger="['console']"
```

*(Note: Disable `wandb` logging by forcing `trainer.logger="['console']"` to prevent authentication exceptions.)*

For WPI integration:

RAY_ADDRESS="http://127.0.0.1:8265" ray job submit \
    --working-dir . \
    --runtime-env runtime-env.yaml \
    --no-wait \
    -- python3 -m verl.experimental.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    "data.train_files=['/home/ray/data/gsm8k/train.parquet']" \
    "data.val_files=['/home/ray/data/gsm8k/test.parquet']" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    "data.truncation=error" \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=wpi \
    '+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.wpi.buffer_id=verl-weight-buffer' \
    '+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.wpi.socket_dir=/run/wpi/sockets' \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.use_legacy_worker_impl=disable \
    trainer.logger="['console']" \
    trainer.total_training_steps=10 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=4