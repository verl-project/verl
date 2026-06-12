# Standalone Model Engine Server

**Author:** `https://github.com/meituan-search`

Last updated: 05/13/2026.

## Background

During RL training, issues such as training-inference inconsistency and off-policy are inevitable. When utilizing `Decoupled PPO` or `Rollout Correction`, the training engine must perform an additional forward pass to compute `old_log_prob`.

Currently, `fully_async_policy` temporarily stores the previous version of the parameters on the CPU at the trainer side. During each training step, these parameters are loaded back to the GPU to execute the forward inference for `old_log_prob`. However, this `load/offload scheme` introduces two main issues:

- **Training Performance Degradation**: Executing the additional forward computation for `old_log_prob` on the trainer side significantly reduces training speed in practice.
- **Staleness Version Error**: When staleness is enabled, the `old_log_prob` for stale samples should be calculated using the parameter version present at the time of sample generation. The current load/offload scheme only retains one extra copy of the parameters and cannot distinguish between different parameter versions, leading to inconsistencies.

`ModelEngineServer` addresses these issues by independently deploying a group of GPU workers that continuously hold the model weights:

- After the rollout phase, requests are sent to `ModelEngineServer` to calculate log_probs, eliminating the need for additional computations on the trainer side.
- Weights are synchronized from the training side via `CheckpointEngine`, ensuring that samples are processed using the correct parameter versions.
- `ModelEngineServer` can also be used to deploy independent reference models that does not require parameter synchronization.

## Architecture

### Pipeline

By deploying `ModelEngineServer`, a pipeline can be established among rollout, old/ref log_probs, and actor update — all running concurrently.

![model_engine_server_pipeline](https://github.com/ArronHZG/verl-community/blob/main/docs/model_engine_server/model_engine_server_pipeline.svg?raw=true)

### Component Design

Depending on the configuration, `ModelEngineServerManager` consists of `OldInstance` for calculating `old_log_prob` and `RefInstance` for calculating `ref_log_prob`. `OldInstance` receives and synchronizes actor weights through `CheckpointEngine`, while `RefInstance` has fixed weights and does not need synchronization.

![model_engine_server_arch0](https://github.com/ArronHZG/verl-community/blob/main/docs/model_engine_server/model_engine_server_arch0.svg?raw=true)

Each instance is an independent `ModelEngineReplica`, responsible for resource allocation (Ray resource pool) and internal component lifecycle management.

```
ModelEngineServerManager
    ├── _old_instance: ModelEngineReplica("old")
    │       ├── RayWorkerGroup
    │       │       └── ModelEngineWorker × world_size
    │       │               └── ModelEngineServerAdapter
    │       │                       └── TrainingWorker  ← holds model weights
    │       └── ModelEngineServer  (Ray actor)
    │               └── asyncio batch consumer loop
    │
    └── _ref_instance: ModelEngineReplica("ref")
            ├── RayWorkerGroup
            │       └── ModelEngineWorker × world_size  ← static weights
            └── ModelEngineServer  (Ray actor)
```

![model_engine_server_arch1](https://github.com/ArronHZG/verl-community/blob/main/docs/model_engine_server/model_engine_server_arch1.svg?raw=true)

Furthermore, `ModelEngineReplica` is composed of `ModelEngineServer` and `RayWorkerGroup`, which consists of `ModelEngineWorker`:

- **ModelEngineServer**: The request entry and batch scheduling core. Receives requests and aggregates them into batches for `RayWorkerGroup`.
- **RayWorkerGroup**: Manages N `ModelEngineWorker`s, distributes batch data to each GPU according to the DP dimension, and collects results.
- **ModelEngineWorker**: The execution unit holding model weights. Inherits from `CheckpointEngineWorker` and participates in weight synchronization.
- **ModelEngineServerAdapter**: Wraps `TrainingWorker` as an inference interface to execute the actual log prob calculation in forward-only mode.

## Configuration

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | `False` | Enable the standalone model engine server |
| `enable_old_mode` | `False` | Old-policy instance, weights sync'd from actor each step |
| `enable_ref_mode` | `False` | Ref-policy instance, static weights, no sync |
| `nnodes` | — | Number of nodes for the engine server |
| `n_gpus_per_node` | — | GPUs per node for the engine server |
| `batch_size` | `-1` | Requests per batch; `-1` → `dp_size × micro_batch_size` |
| `timeout` | `5.0` | Max seconds to wait before dispatching a partial batch |
| `micro_batch_size_per_gpu` | `1` | Forward-pass micro-batch size per GPU |
| `use_dynamic_bsz` | `False` | whether use dynamic bsz |
| `max_token_len_per_gpu` | `16384` | Max tokens per GPU when `use_dynamic_bsz=True` |
| `strategy` | — | `megatron` (fsdp not yet supported for old_mode) |
| `megatron.{tensor,pipeline,context,expert}_model_parallel_size` | same as actor | parallel config, can be set independently from actor |

The engine server supports **independent parallelism configuration** from the actor. For example, the actor may use TP=4 PP=2, while the engine server uses TP=2 PP=1

**Quick Start**: see [`shell/dapo_7b_math_megatron_16_16_8.sh`](shell/dapo_7b_math_megatron_16_16_8.sh) for an example.

## Experiment

Based on experimental verification with the `qwen2.5-7b-math` model in `fully_async_policy` mode, the main conclusions are as follows:

**Performance overhead**: Using `ModelEngineServer` to calculate log probs increases the end-to-end time by about 5%–10% compared to bypass mode, while the traditional `load/offload scheme` causes an increase of about 60%.  Compared to `load/offload scheme`, `ModelEngineServer` achieves about **1.48x** end-to-end acceleration (cost-adjusted comprehensive benefit ~1.19x).

| Scheme | Resource Allocation<br>(Rollout : Train : ModelEngineServer) | 100-Step e2e Time | Speedup (Cost-Adjusted) |
|--------|--------------------------------------------------------------|-------------------|-------------------------|
| `load/offload scheme` | 16 : 16 : 0 | 16h 4min | — |
| `ModelEngineServer`   | 16 : 16 : 8 | 10h 50min | 1.48x (1.19x) |

**Accuracy and stability**: Combining `ModelEngineServer` to recalculate log probs and enabling
Correction can effectively improve the smoothness of the training curve and avoid accuracy degradation
in the later stages.

> source data: https://wandb.ai/hou-zg-meituan/model-engine-server