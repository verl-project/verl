# PyTorch Profiling in verl

Last updated: 01/13/2026.

This guide explains how to use the native [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) for profiling verl training runs.

## Configuration

Profiling in verl can be configured through parameters in the trainer configuration file (e.g., `ppo_trainer.yaml`).

### Global Profiling Control

In `global_profiler`, you can control when and how profiling occurs globally:

* **`global_profiler.steps`**: List of step numbers to profile. E.g., `[1, 2, 5]` profiles steps 1, 2, and 5. Set to `null` to disable.
* **`global_profiler.save_path`**: Directory to save the profiling results. Default is `outputs/profile`.

### Role Profiling Control

Each RL role (Actor, Critic, etc.) has its own `profiler` configuration:

* **`enable`**: Whether to enable profiling for this role.
* **`all_ranks`**: If `True`, profiles all ranks.
* **`ranks`**: List of specific ranks to profile if `all_ranks` is `False`.
* **`tool_config.torch`**: Configuration specific to the PyTorch Profiler.

#### PyTorch Profiler Options (`tool_config.torch`)

You can customize the PyTorch Profiler behavior using the following fields under `tool_config.torch`:

* **`contents`**: List of contents to profile.
    *   **`cpu`**: Profile CPU activities.
    *   **`cuda`**: Profile CUDA activities.
    *   **`memory`**: Track tensor memory allocation/free.
    *   **`shapes`**: Record shapes of operator inputs.
    *   **`stack`**: Record source code file and line number.
* **`profile_token_start`**: Effective only for the rollout role; defines the start response-token index for rollout decoding collection. It is applied only when valid (0-based, `profile_token_end > profile_token_start`, and within response length).
* **`profile_token_end`**: Effective only for the rollout role; defines the stop response-token index (exclusive) for rollout decoding collection. It is applied only when valid (0-based, `profile_token_end > profile_token_start`, and within response length).
* **`schedule`**: (Advanced) Enables [`torch.profiler.schedule`](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule) so that only part of each profiling window is recorded. It only takes effect when `active > 0`; otherwise the profiler collects continuously (the default). verl advances the schedule by calling `profiler.step()` once per mini-batch in the Actor update loop (and once per step in SFT), so a scheduled cycle is measured in mini-batches. The fields mirror the official PyTorch API:
    *   **`skip_first`**: Number of initial steps to ignore before the first cycle begins.
    *   **`wait`**: Steps to idle (no collection) at the start of each cycle.
    *   **`warmup`**: Steps to trace but discard, letting the profiler stabilize, each cycle.
    *   **`active`**: Steps to actively record each cycle. Set `<= 0` (default) to disable scheduling.
    *   **`repeat`**: Number of cycles to record. `0` (default) repeats until profiling stops.


## Examples

### 1. End-to-End Collection

Collects performance data for all steps in a single trace file.

```yaml
global_profiler:
  steps: [1, 2, 5]
  save_path: ./outputs/profile

actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: True
      tool_config:
        torch:
          discrete: False
          contents: [cpu, cuda]
  # rollout & ref follow actor settings
```

### 2. Discrete Mode Collection

Discrete mode saves separate trace files for each step. This is useful for detailed analysis and is **mandatory** when using Agent Loop.

**Configuration Example**

This configuration supports profiling both Training (Actor) and Inference (Rollout). You can enable/disable them independently.

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True # Set to True to profile training
      all_ranks: False
      ranks: [0] # Global Rank 0
      tool_config:
        torch:
          discrete: True
          contents: [cpu, cuda]
  rollout:
    profiler:
      enable: True # Set to True to profile inference
      all_ranks: False
      ranks: [0] # In Agent Loop, this is the Replica Rank (e.g. 0-th instance)
      tool_config:
        torch:
          discrete: True # REQUIRED 
          # Optional response-token window for rollout engine side collection.
          # If start/stop are not set, the entire rollout stage is collected.
          # Collect tokens in [12, 46), i.e. token index 12~45.
          profile_token_start: 12
          profile_token_end: 46
  # ref follow actor settings
```

**Agent Loop Mode Description**

When Rollout runs in [Agent Loop](../advance/agent_loop.rst) mode, performance data for the Rollout phase **must be collected using discrete mode**. In this case, the Profiler is triggered by the inference engine backend.

1. Rank Definition: ranks in the Rollout configuration refers to Replica Rank (inference instance index), not Global Rank.

2. Inference Engine Support: Currently, vLLM and SGLang engines are supported without additional settings. Specific details are as follows:

   *   **vLLM Engine**: Automatically collects AsyncLLM scheduling stacks and inference process performance data.
   *   **SGLang Engine**: Automatically collects inference process performance data. Does not support the memory option in contents.

### 3. Scheduled Collection (`wait`/`warmup`/`active`/`repeat`)

For long update loops with many mini-batches (e.g. large gradient accumulation), you usually don't need to trace every mini-batch. A `schedule` records only a few mini-batches per cycle, keeping traces small while still capturing steady-state behavior. verl calls `profiler.step()` once per mini-batch so the schedule advances automatically.

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: True
      tool_config:
        torch:
          discrete: False
          contents: [cpu, cuda]
          schedule:
            skip_first: 1  # ignore the very first mini-batch
            wait: 1        # then idle 1 mini-batch at the start of each cycle
            warmup: 1      # warm up 1 mini-batch (traced but discarded)
            active: 3      # record 3 mini-batches
            repeat: 2      # capture 2 such cycles, then stop collecting
  # rollout & ref follow actor settings
```

With the configuration above, within each profiled training step verl skips the first mini-batch, then runs two cycles of `wait(1) -> warmup(1) -> active(3)`, producing two trace files (the second suffixed with `_cycle1`). If a training step has fewer mini-batches than the schedule needs, only the mini-batches that were reached are recorded.

`schedule` only applies to the training update loop (Actor RL update and SFT). It is a no-op for the rollout engine side, which uses `profile_token_start`/`profile_token_end` instead.

## Output file naming

Because profiling runs in every training process, each trace file is named so it can be
attributed to a specific process without opening it. The stem is:

```
[<role>_][<scope>_]rank<r>[-of-<world>][_tp<..>-pp<..>-dp<..>-cp<..>]_pid<pid>_<timestamp>[_cycle<N>].json.gz
```

* **`role`**: the worker role (e.g. `actor`, `ref`, `value-model` for the critic), so results
  from different roles at the same rank are distinguishable.
* **`scope`**: the profiled region passed to `start_profile`/`annotate` (e.g. `e2e`); it is also
  used as a sub-directory under `save_path`.
* **`rank`/`world`**: the global `torch.distributed` rank and world size.
* **`tp/pp/dp/cp`**: tensor/pipeline/data/context parallel ranks, included when Megatron's
  parallel state is initialized (plain FSDP data parallelism only reports `rank`).
* **`cycle<N>`**: added for the 2nd+ cycle of a scheduled run (see above).

## Visualization

Collected trace files (usually `.json` or `.json.gz`) are stored in the configured `save_path`.

You can visualize them using:

1.  **Chrome Tracing**: Open `chrome://tracing` in a Chrome browser and load the JSON file.
2.  **Perfetto**: Open [ui.perfetto.dev](https://ui.perfetto.dev/) and load the file (recommended for large traces).
3.  **TensorBoard**: If using the TensorBoard plugin for PyTorch Profiler.
