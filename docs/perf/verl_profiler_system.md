# verl Profiler System

Last updated: 09/07/2026.

## Architecture

The architecture of verl profiler system is like below:

![verl-profiler-arch](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/2bc7ed0ba2f37f21707bfac3b241eca4b86d1bc6/docs/verl_profiler_arch.png)

There is a global profiler and tool configuration to set some common config in single controller level, deciding

- `tool`: which tool to use
- `steps`: which steps to profile
- `save_path`: results saving path

When some tool need to profile behavior of each role, configurations in role-level is needed:

- `tool`: which tool to use
- `enable`: whether enable profiling on this role
- rank info: `all_ranks` and `rank` to decide which rank to profile or log output

For tool config in role-level, there are some detailed behavior needed to control, like the `discrete` mode in nsys profiler.

Every role has a profiler config, and by default, rollout/ref/reward models follow the Actor's behavior.

## CUDA/NPU memory snapshots

Selecting `global_profiler.tool=torch_memory` automatically enables best-effort OOM
snapshot dumping on the selected profiler ranks when the CUDA/NPU allocator observer
is available. The callback logs the Python stack and allocator memory summary, then writes a snapshot under
`<save_path>/oom_<timestamp>/` without synchronizing the device. It remains active outside
the scheduled profiling steps once registered; it cannot run after `SIGKILL`.
Unsupported devices or PyTorch builds emit a warning and continue without the OOM callback.

NPU support requires `torch_npu._C._npu_attach_out_of_memory_observer` and the memory
history/snapshot APIs (present in `torch-npu==2.10.0.post4`). Both regular and OOM NPU
snapshots directly serialize `memory._snapshot()` to a pickle file, skipping the
extra device memory CSV/profiler operations in `memory._dump_snapshot()`. Regular
step-boundary dumps still synchronize the device; OOM dumps do not. CUDA snapshots
continue to use the native dump helper. The callback logs NPU's third argument as
`total_or_limit`, since it represents the process memory limit when configured,
otherwise the device total.

Normal step-boundary snapshots remain controlled by `global_profiler.steps` and
`global_profiler.global_tool_config.torch_memory.memory_snapshot_num_steps`
(default: `1`). Set `global_profiler.profile_continuous_steps=False` when using this
step count so each profiled step contributes one start/stop cycle to the window.
Training that does not select the `torch_memory` tool is unaffected.

## To Add a new profiling tool

New added profiling tool shall reuse the current APIs as much as possible.

1. The logic of **whether to use the tool**: `tool == [new tool]`.
2. Add the global and local tool config to `ppo_trainer.yaml`/`ppo_megatron_trainer.yaml` and each `[role].yaml`, under `global_tool_config.[new tool]` and `tool_config.[new tool]`
3. The tool config should be implemented in `verl/utils/profiler/config.py`, inherit the `BaseConfig` class.
4. Implement profiling tool initialization logic using configurations in `global_profiler.global_tool_config.[new tool]` and the results saving logics (can also save in role-level profile)
5. For role function-level profiling, please follow the nsys profiler way in `nvtx_profiler.py`, implement a profiler class inherit `DistProfiler` and import new profiler in `verl/utils/profiler/__init__.py`
6. Add unit test and examples for others to use in convinience.
