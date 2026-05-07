# Precision Debugger (msprobe) in verl

Last updated: 04/13/2026.

This guide explains how to collect precision data in verl using the
`msprobe` PrecisionDebugger.

## Prerequisites

* Install `msprobe` in the training environment:

```bash
pip install mindstudio-probe
```

* Prepare a `config.json` for msprobe (see examples below).
* Enable profiler for the roles you want to collect.

Reference:
* `https://gitcode.com/Ascend/msprobe.git`

## Configuration

PrecisionDebugger is integrated through verl's unified profiler interface.
Use a minimal two-part setup:

* `global_profiler` selects the tool and config file.
* role `profiler.enable=True` turns on profiling for that role.

### Global profiling control

In `global_profiler`, set the profiler tool to `precision_debugger` and
configure the msprobe-specific options under `global_tool_config`.

```yaml
global_profiler:
  tool: precision_debugger
  steps: [1, 2, 5]
  save_path: "outputs/profile"
  global_tool_config:
    precision_debugger:
      _target_: verl.utils.profiler.config.PrecisionDebuggerToolConfig
      config_path: /path/to/config.json
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

* `global_profiler.steps` is the only step filter for PrecisionDebugger.
* Dumps are written under `global_profiler.save_path`.
* Actual dump path is `{global_profiler.save_path}/step_{global_step}/{stage}`.
* Do not set `dump_path` in `config.json`; output path is controlled by verl.

### Role profiling control

Enable profiling for the roles you want to collect:

```yaml
actor_rollout_ref:
  actor:
    profiler:
      enable: True
  ref:
    profiler:
      enable: True
critic:
  profiler:
    enable: True
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

The current integration is designed for training-side stages. In a typical PPO
run, the most common useful combinations are:

* actor/ref only:
  `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update`
* actor/ref/critic:
  `actor_compute_log_prob`, `ref_compute_log_prob`, `compute_values`,
  `critic_update`, `actor_update`

## msprobe config.json common examples

### `statistics` mode

```json
{
  "task": "statistics",
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

### `tensor` mode

```json
{
  "task": "tensor",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,
  "tensor": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

## Minimal example

The following example enables PrecisionDebugger on steps `1` and `2`.
If you need rank filtering, configure it only in msprobe `config.json`.

```yaml
global_profiler:
  tool: precision_debugger
  steps: [1, 2]
  global_tool_config:
    precision_debugger:
      _target_: verl.utils.profiler.config.PrecisionDebuggerToolConfig
      config_path: /path/to/dump_config.json
      stages:
        - actor_compute_log_prob
        - ref_compute_log_prob
        - actor_update
      strict: False

actor_rollout_ref:
  actor:
    profiler:
      enable: True
  ref:
    profiler:
      enable: True
```

## Minimal CLI example

Use only the required flags:

```bash
python3 -m verl.trainer.main_ppo \
  global_profiler.tool=precision_debugger \
  global_profiler.steps='[1,2]' \
  global_profiler.save_path=outputs/profile \
  +global_profiler.global_tool_config.precision_debugger.config_path=/path/to/config.json \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.ref.profiler.enable=True
```

Optional stage filter:

```bash
+global_profiler.global_tool_config.precision_debugger.stages='[actor_compute_log_prob,ref_compute_log_prob,actor_update]'
```

## Output layout

Verl organizes PrecisionDebugger output by training global step and stage.
Inside each stage directory, msprobe creates its own `step*/rank*` layout.

Example:

```text
outputs/profile/
  step_1/
    actor_compute_log_prob/step0/rank0/dump.json
    actor_update/step0/rank0/dump.json
    ref_compute_log_prob/step0/rank0/dump.json
  step_2/
    actor_compute_log_prob/step0/rank0/dump.json
    actor_update/step0/rank0/dump.json
    ref_compute_log_prob/step0/rank0/dump.json
```

Observed output from a real run:

* Outer `step_<global_step>` directories are created by verl.
* Inner `step0/rank0/dump.json` directories are created by msprobe.
* With the current integration, each profiled stage is collected in an
  independent dump session, so stage-local output typically lands in `step0`.

## How results are written

The verl integration wraps each profiled stage with:

* `debugger.start(model=...)`
* execute the stage
* `debugger.stop()`
* `service.reset_status()` if the msprobe runtime exposes it

Verl does **not** manually call `debugger.step()` in the current integration.
Instead, each stage writes to its own dump directory and resets msprobe runtime
status after `stop()` to avoid stale `dump.json` cache growth across stages.

For L0 collection, PrecisionDebugger must bind to the actual model used in the
stage. The profiler resolves the model inside
`verl/utils/profiler/precision_debugger_profile.py` and supports both legacy
workers and the newer model-engine worker path.

## Overhead and disk usage

Below are measurements from a real PPO run on Ascend with:

* model: `Qwen2-0.5B`
* profiled steps: `[1, 2]`
* rank: `0`
* stages:
  * L1: `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update`
  * L0: `actor_compute_log_prob`, `ref_compute_log_prob`, `compute_values`,
    `critic_update`, `actor_update`

### Time overhead

| Run | Model | Profiled steps | Measured step time |
|---|---|---:|---:|
| Baseline | `Qwen2-0.5B` | None | about `16-18 s/step` in steady state |
| L0 | `Qwen2-0.5B` | `step 1` | `66.81 s` |
| L0 | `Qwen2-0.5B` | `step 2` | `48.78 s` |
| L0 | `Qwen2-0.5B` | non-profiled later steps | about `17 s/step` |
| L1 | `Qwen2-0.5B` | `step 1` | `177.35 s` |
| L1 | `Qwen2-0.5B` | `step 2` | `161.80 s` |
| L1 | `Qwen2-0.5B` | non-profiled later steps | about `17 s/step` |

In this experiment, profiled L0 steps were about `3x-4x` slower than the
baseline steady-state step time, and profiled L1 steps were about `9x-10x`
slower. Non-profiled later steps remained close to baseline in both cases.

In general, PrecisionDebugger should be treated as a heavy-weight precision
debugging tool rather than a lightweight profiler. In larger models or broader
stage coverage, it is common to observe `tens-X` performance inflation for
profiled steps.

### Disk usage

| Level | Model | Stages | Scope | Disk usage |
|---|---|---|---|---:|
| L1 | `Qwen2-0.5B` | `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update` | total for `step_1` and `step_2` | `21 MB` |
| L1 | `Qwen2-0.5B` | `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update` | per step | about `11 MB` |
| L1 | `Qwen2-0.5B` | `actor_update` | per step | about `5.1-5.2 MB` |
| L1 | `Qwen2-0.5B` | `actor_compute_log_prob` | per step | about `2.6 MB` |
| L1 | `Qwen2-0.5B` | `ref_compute_log_prob` | per step | about `2.6 MB` |
| L0 | `Qwen2-0.5B` | `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update` | total for `step_1` and `step_2` | `8.8 MB` |
| L0 | `Qwen2-0.5B` | `actor_compute_log_prob`, `ref_compute_log_prob`, `actor_update` | per step | about `4.4 MB` |
| L0 | `Qwen2-0.5B` | `actor_update` | per step | about `2.5 MB` |
| L0 | `Qwen2-0.5B` | `actor_compute_log_prob` | per step | about `1.1 MB` |
| L0 | `Qwen2-0.5B` | `ref_compute_log_prob` | per step | about `0.86-0.87 MB` |

In this experiment, total L1 disk usage was about `2.4x` the L0 disk usage for
the measured actor/ref stage set.

These numbers depend on:

* selected stages
* number of profiled steps
* dump level and task
* model shape and sequence length

## How to analyze results

At minimum, check:

* which `step_<global_step>` directory was generated
* which stage directories exist under that step
* whether `dump.json` exists under `step0/rank0`

For downstream analysis, use standard msprobe tools such as:

* `msprobe compare`
* `msprobe visualization`

Example compare usage:

```bash
msprobe compare \
  --target-path /path/to/target_dump/dump.json \
  --golden-path /path/to/golden_dump/dump.json
```

You can compare:

* the same stage across two runs
* different global steps of the same stage
* different ranks when multi-rank collection is enabled

For more advanced analysis workflows, refer to the official msprobe
documentation for compare and visualization commands.

## Usage notes

* Verl integrates PrecisionDebugger through `DistProfiler.annotate` wrappers on
  training stages.
* PrecisionDebugger is automatically discrete: each profiled stage is
  collected in an independent `start -> stop -> reset_status` session. It does
  not currently expose the unified profiler `discrete` configuration used by
  tools such as `nsys` or `npu`.
* `global_steps` is read from batch `meta_info` or from worker attributes.
* If `strict` is `True`, missing msprobe or unknown stages raise errors.
* If a stage prints `PrecisionDebugger model not resolved`, that stage ran
  normally but no dump was collected because verl could not bind msprobe to a
  valid model object.
* Because dump cost is high, prefer collecting a small number of representative
  steps first, then narrow the stage set if necessary.

## Quality checklist

Use this checklist to verify your setup is complete and reproducible:

* `global_profiler.tool=precision_debugger`
* `global_profiler.steps` includes the target step
* `+global_profiler.global_tool_config.precision_debugger.config_path=...` is set
* role `profiler.enable=True` is set for the stages you need
* `msprobe` is importable in the runtime environment
* output exists under `{global_profiler.save_path}/step_<global_step>/<stage>/...`

## Troubleshooting

### No dump directory is generated

Check:

* `global_profiler.tool=precision_debugger`
* `global_profiler.steps` contains the target step
* role profiler is enabled for the target role
* msprobe is installed in the training environment

### `PrecisionDebugger model not resolved`

This means the stage was reached, but verl could not find the actual model used
by that worker. The stage itself still runs, but dump is skipped. This usually
indicates:

* a new worker path was introduced and profiler model resolution needs to be
  updated
* the role or engine backend differs from the paths currently supported by the
  resolver

### `dump.json` keeps growing unexpectedly

If `stop()` is called without resetting msprobe runtime state, cached dump data
may continue to accumulate across stage invocations. The current verl
integration resets msprobe runtime status after `stop()` when the service API
supports it.
