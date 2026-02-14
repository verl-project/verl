# Precision Debugger (msprobe) in verl

Last updated: 02/14/2026.

This guide explains how to collect precision data in verl using the
`msprobe` PrecisionDebugger.

## Prerequisites

* Install `msprobe` in the training environment.
* Prepare a `config.json` for msprobe (see examples below).

## Configuration

PrecisionDebugger is integrated through verl's unified profiler interface.
You configure it in two places:

* **Global profiling control** via `global_profiler` in the trainer config.
* **Role profiling control** via each role's `profiler` block.

### Global profiling control

In `global_profiler`, set the profiler tool to `precision_debugger` and
configure the msprobe-specific options under `global_tool_config`.

```yaml
global_profiler:
  tool: precision_debugger
  steps: [1, 2, 5]
  save_path: "outputs/profile" # optional, not used by msprobe
  global_tool_config:
    precision_debugger:
      _target_: verl.utils.profiler.config.PrecisionDebuggerToolConfig
      enable: True
      config_path: /path/to/config.json
      data_dir: outputs/precision_debug
      steps: [1, 2, 5]
      stages:
        - actor_update
        - actor_compute_log_prob
        - ref_compute_log_prob
        - compute_values
        - critic_update
        - compute_rm_score
      strict: False
```

Notes:

* `steps` in `global_profiler` controls the step window for start/stop.
* `precision_debugger.steps` provides an extra filter. If both are set,
  the intersection is applied.
* `data_dir` is the root directory for dumps. The actual path is
  `{data_dir}/step_{global_step}/{stage}`.

### Role profiling control

Enable profiling for the roles you want to collect:

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True
      all_ranks: False
      ranks: [0]
  ref:
    profiler:
      enable: True
      all_ranks: False
      ranks: [0]
critic:
  profiler:
    enable: True
    all_ranks: False
    ranks: [0]
```

## Supported stages

PrecisionDebugger collects data from the following stages:

* `actor_update`
* `actor_compute_log_prob`
* `ref_compute_log_prob`
* `compute_values`
* `critic_update`
* `compute_rm_score`

Rollout generation is intentionally skipped (`rollout_generate` is ignored).

## msprobe config.json examples

Example for `task: statistics`:

```json
{
  "task": "statistics",
  "dump_path": "/home/data_dump",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

Example for `task: tensor`:

```json
{
  "task": "tensor",
  "dump_path": "/home/data_dump",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,
  "tensor": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "bench_path": "/home/bench_data_dump",
    "summary_mode": "md5",
    "diff_nums": 5
  }
}
```

## Usage notes

* PrecisionDebugger uses `start(model) -> stop() -> step()` semantics.
  Verl maps this into `DistProfiler.annotate` wrappers for training stages.
* `global_steps` is read from batch `meta_info` or from worker attributes.
* If `strict` is `True`, missing msprobe or unknown stages raise errors.
