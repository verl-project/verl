# NVIDIA Nsight Systems profiling in verl

Last updated: 06/20/2025.

This guide explains how to use NVIDIA Nsight Systems for profiling verl training runs.

## Configuration

Profiling in verl can be configured through several parameters in the trainer configuration file (ppo_trainer.yaml or other files like dapo_trainer.yaml):

### Prerequisites

Nsight Systems version is important, please reference `docker/Dockerfile.vllm.sglang.megatron` for the version we used.

### Global profiling control

verl has one single controller process and multiple worker processes. Both controller and worker processes can be profiled. Since the controller process can be executed in any nodes in the cluster, there is a message printed in the logging to indicate the controller process node hostname and process id.

In `global_profiler`, three new config entries control the profiler behaviors:

* **`global_profiler.steps`**. List of step numbers at which profiling should be performed. For example: [1, 2, 5] will profile steps 1, 2, and 5. And ``null`` means no profiling.

* **`global_profiler.profile_continuous_steps`**. If true, and the following `global_profiler.discrete==False`, then the continuous steps in `global_profiler.steps` will be combined into one database. For example the above step 1 and 2 are in one database, and 5 in another. If false, every step occupies at least one database. The reason for this config is to observe the program behaviors between steps.

Nsys options in controller nodes and worker nodes are configured in `global_profiler.global_tool_config.nsys`:

* **`global_profiler.global_tool_config.nsys.controller_nsight_options`**. This config group is for the single controller. All fields in this config group will be just sent to Nsight Systems when Ray starts the controller process. `ppo_trainer.yaml` provides a workable example. Users can reference [Nsight Systems manual](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) and [Ray user guide](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html) for more details.
* **`global_profiler.global_tool_config.nsys.worker_nsight_options`**. This config group is for the worker processes. Similarly all fields in this config group will be just sent to Nsight Systems when Ray starts the controller process. Capture range is used to control the profiler when to start and stop. So `capture-range: "cudaProfilerApi"` is fixed and does not change it. Users can change `capture-range-end` with some accurate calculation or just leave it `null`.

### Worker process profiling

Verl manages mulitiple RL roles, _Actor_, _Ref_, _Rollout_, _Critic_, _Reward_, which are implemented in different Worker classes. And these workers can be combined into one Ray Actor, running in a process group. Each RL role has its own profiling config group, `profiler`, which consists of three fields:

* **`all_ranks` and `ranks`**. When `all_ranks` is set `True` then all ranks will be profiled; when set `False`, `ranks` will be profiled. By default, verl profiles the whole training process in a series ` worker_process_<PID>.<RID>.nsys-rep` files for each process rank. PID is the process ID; RID is the capture range ID.
* **`discrete`**. When set `False`, all the roles actions in one training step will be dumped in one database. When set `True`, the actions annotated by `DistProfiler.annotate` will be dumped into a discrete database. In this case, each role's action occupies one `<RID>`.
* **Verl collocate mode**. Verl can combine two Worker sub classes to one Worker Actor. In this case, the user should take care that the combined Workers have consistent `discrete`. The Nsight Systems profiler uses a `torch.cuda.profiler.start()` and `stop()` pair to dump a `<step>` database anyway.

### where to find the profiling data

By default the `*.nsys-rep` files are saved in the directory `/tmp/ray/session_latest/logs/nsight/` at each node. According to the Ray manual, this default directory is not changeable. [&#34;however, Ray preserves the `--output` option of the default config&#34;](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html).

Some users may think it is not convenient, but it is understandable that Ray may start hundreds of processes and it would be a big network file system pressure if we save the files in one central place.

## Usage Example

To enable profiling for specific components and steps, modify your ppo_trainer.yaml like this:

### Disable profiler

```yaml
    profiler:
        steps: null # disable profile
```

### Enable profiler and one database for one training step

```yaml
    global_profiler:
        steps: [1, 2, 5]
        discrete: False
    actor_rollout_ref:
        actor:
            profiler:
                enable: True
                all_ranks: True
        # rollout & ref follow actor settings
    critic:
            profiler:
                enable: True
                all_ranks: True
    reward_model:
            profiler:
                enable: True
                all_ranks: True
```

### Enable profiler and multiple databases for one training step

```yaml
    profiler:
        steps: [1, 2, 5]
        discrete: True
```

## Profiling Output

When profiling is enabled, verl will generate Nsight Systems profiles for the specified components and steps. The profiles will include:

- CUDA kernel execution
- Memory operations
- CPU-GPU synchronization
- NVTX markers for key operations

Nsight Systems supports multi-report view, to open multiple databases together. In this mode, different processes and steps can be aligned in one time line for better analysis.
