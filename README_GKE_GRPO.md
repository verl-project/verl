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
    trainer.logger="['console']"
```

*(Note: Disable `wandb` logging by forcing `trainer.logger="['console']"` to prevent authentication exceptions.)*