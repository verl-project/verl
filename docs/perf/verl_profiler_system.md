# verl Profiler System

Last updated: 08/18/2025.

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

## To Add a new profiling tool

New added profiling tool shall reuse the current APIs as much as possible.

1. The logic of **whether to use the tool**: `tool == [new tool]`.
2. Add the global and local tool config to `ppo_trainer.yaml`/`ppo_megatron_trainer.yaml` and each `[role].yaml`, under `global_tool_config.[new tool]` and `tool_config.[new tool]`
3. The tool config should be implemented in `verl/utils/profiler/config.py`, inherit the `BaseConfig` class.
4. Implement profiling tool initialization logic using configurations in `global_profiler.global_tool_config.[new tool]` and the results saving logics (can also save in role-level profile)
5. For role function-level profiling, please follow the nsys profiler way in `nvtx_profiler.py`, implement a profiler class inherit `DistProfiler` and import new profiler in `verl/utils/profiler/__init__.py`
6. Add unit test and examples for others to use in convinience.