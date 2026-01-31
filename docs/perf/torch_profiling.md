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
* **`schedule`**: (Advanced) configuration for `wait`, `warmup`, `active`, `repeat` cycles.

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
  # ref follow actor settings
```

> **Note for Agent Loop Mode**:
> When using Agent Loop, `ranks` in rollout config refers to the **Replica Rank** (instance index), not the global rank.

**Inference Backend Setup (for Agent Loop)**

*   **vLLM Engine**:
    *   **Environment Variables Required**:
        *   `VLLM_TORCH_PROFILER_DIR`: **(Required)** Directory to save traces (e.g., `/mnt/traces`).
        *   `VLLM_TORCH_PROFILER_WITH_STACK`: `1` to enable stack tracing (default).
        *   `VLLM_TORCH_PROFILER_RECORD_SHAPES`: `1` to record shapes of operator inputs.
        *   `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY`: `1` to track tensor memory allocation/free.
        *   `VLLM_TORCH_PROFILER_WITH_FLOPS`: `1` to estimate FLOPS.
    *   *Note: vLLM ignores the `save_path` and `contents` in `ppo_trainer.yaml`.*

*   **SGLang Engine**:
    *   **Zero Configuration**: Automatically uses the settings from `ppo_trainer.yaml`.

## Visualization

Collected trace files (usually `.json` or `.json.gz`) are stored in the configured `save_path`.

You can visualize them using:

1.  **Chrome Tracing**: Open `chrome://tracing` in a Chrome browser and load the JSON file.
2.  **Perfetto**: Open [ui.perfetto.dev](https://ui.perfetto.dev/) and load the file (recommended for large traces).
3.  **TensorBoard**: If using the TensorBoard plugin for PyTorch Profiler.
