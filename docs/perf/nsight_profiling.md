# NVIDIA Nsight Systems profiling in verl

Last updated: 07/29/2026.

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

Verl manages multiple RL roles, _Actor_, _Ref_, _Rollout_, _Critic_, _Reward_, which are implemented in different Worker classes. And these workers can be combined into one Ray Actor, running in a process group. Each RL role has its own profiling config group, `profiler`, which consists of three fields:

* **`all_ranks` and `ranks`**. When `all_ranks` is set `True` then all ranks will be profiled; when set `False`, `ranks` will be profiled. By default, verl profiles the whole training process in a series ` worker_process_<PID>.<RID>.nsys-rep` files for each process rank. PID is the process ID; RID is the capture range ID.
* **`discrete`**. When set `False`, all the roles actions in one training step will be dumped in one database. When set `True`, the actions annotated by `DistProfiler.annotate` will be dumped into a discrete database. In this case, each role's action occupies one `<RID>`.
* **Verl collocate mode**. Verl can combine two Worker sub classes to one Worker Actor. In this case, the user should take care that the combined Workers have consistent `discrete`. The Nsight Systems profiler uses a `torch.cuda.profiler.start()` and `stop()` pair to dump a `<step>` database anyway.

### where to find the profiling data

By default the `*.nsys-rep` files are saved in the directory `/tmp/ray/session_latest/logs/nsight/` at each node. According to the Ray manual, this default directory is not changeable. [&#34;however, Ray preserves the `--output` option of the default config&#34;](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html).

Some users may think it is not convenient, but it is understandable that Ray may start hundreds of processes and it would be a big network file system pressure if we save the files in one central place.

### Finish hook: relocate results and run a custom command

Because Ray hardcodes the Nsight output directory, verl provides a *finish hook*. It has two independent parts, configurable either once under `global_profiler` (every role inherits it) or per role under `profiler` (which overrides the global value):

* **`relocate_results`** (bool, default `False`). When `True`, each profiled worker moves *its own* `worker_process_<pid>.*` reports out of `/tmp/ray/session_latest/logs/nsight/` into `save_path`, **on every profiled step**. Matching is by PID (nsys's `%p` token equals the worker process PID), so co-located ranks never touch each other's files and there is no race. Destination filenames are prefixed with the role and hostname (e.g. `actor_<host>_worker_process_<pid>.nsys-rep`) to stay unique across nodes. The move is best-effort: nsys only finalizes a report after the capture session shuts down, so files that are not present yet are simply skipped and nothing is ever deleted. The same flag also flattens rollout engine traces, which the engines write into `<save_path>/agent_loop_rollout_replica_<n>/`, into `save_path` — see [PyTorch profiling](torch_profiling.md).
* **`finish_hook_cmd`** (str, default `null`). A shell command executed on the selected ranks **once, after the last profiled step** (not once per step). Use it for post-processing, compression, or uploading traces to remote storage. Backend stop and `relocate_results` still run every profiled step, so by the last step `save_path` holds every profiled step's reports; because the command runs a single time, a command that uploads the whole directory sends each report exactly once. It runs with these environment variables:
  * `VERL_PROFILE_SAVE_PATH` — the configured `save_path` (holds all profiled steps' reports by the time the command runs).
  * `VERL_PROFILE_TOOL` — the profiler tool (e.g. `nsys`).
  * `VERL_PROFILE_RANK` — the global rank.
  * `VERL_PROFILE_PID` — the worker process PID (matches the `%p` in the report filename).
  * `VERL_PROFILE_ROLE` — the worker role, when known.
  * `VERL_PROFILE_RAY_NSIGHT_DIR` — Ray's fixed Nsight log dir (nsys only), handy for grabbing stragglers.

Running once is what keeps each report uploaded exactly once. If the command instead ran every profiled step, a directory upload would re-send earlier steps' reports each step (`save_path` accumulates), and an uploader that versions by upload time would then store one report as several files differing only by a trailing timestamp. A single upload at the end avoids that entirely.

**Choosing which ranks run the command.** `finish_hook_cmd` runs on `finish_hook_all_ranks`/`finish_hook_ranks`. When both are unset it falls back to the profiled ranks (`all_ranks`/`ranks`), so by default the command runs wherever profiling happened. `finish_hook_ranks` may also select ranks that were *not* profiled (for example to run a single aggregation step on rank 0). `relocate_results` always runs on the profiled ranks, since that is where the artifacts live. `save_path` is usually node-local, so the command runs on every selected rank/node: with multi-node profiling keep one selected rank per node (do not narrow it to rank 0 only, or the other nodes' artifacts are never picked up), but avoid selecting several ranks that share one node's directory, or each would upload that directory.

**Which role's config applies.** A worker reads exactly one role block, chosen by its own role: a collocated `actor_rollout_ref` worker reads `actor_rollout_ref.actor.profiler`, and `rollout.profiler` only applies to a standalone rollout worker. Setting the hook on `global_profiler` avoids having to guess.

Example: relocate every rank's reports into `save_path` every step and, once profiling is done, upload each node's directory to HDFS in one shot:

```yaml
    global_profiler:
        tool: nsys
        steps: [1, 2, 5]
        save_path: "outputs/profile"
        relocate_results: True
        finish_hook_cmd: 'hdfs dfs -put -f "$VERL_PROFILE_SAVE_PATH"/*.nsys-rep hdfs:///my/profiles/'
    actor_rollout_ref:
        actor:
            profiler:
                enable: True
                all_ranks: True
```

Because the command runs once (after step 5 here), globbing `save_path` uploads each report exactly
once — there is no per-step re-upload to deduplicate. When you reference a variable on the command
line, wrap it in single quotes so your shell does not expand it before the worker sees it. The value
itself must not contain double quotes — Hydra's override parser rejects them — so a command that
needs quoting (paths with spaces) belongs in a small script that the hook calls.

**Observing the hook.** After the last profiled step, the selected ranks print a `[Profiler][finish_hook]` line to the worker's stdout stating whether a command is configured and whether the current rank was selected, followed by the command line itself, its merged stdout/stderr streamed live, and its exit code. These are plain prints rather than logger calls so they show up in the Ray worker logs regardless of the logging level. The command failing (non-zero exit or launch error) never interrupts training.

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
