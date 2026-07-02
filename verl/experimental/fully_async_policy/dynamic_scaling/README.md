# Dynamic Resource Scaling for Fully-Async Training

This module provides **hybrid inference resource dynamic scaling** for the fully-async training framework, enabling Trainer-node GPUs to participate in rollout generation during idle periods and thus improving overall GPU utilization.

---

## 1. Overview

### Problem

In the fully-async separated architecture, Trainer-node GPUs sit idle while waiting for rollout data, and Standalone Rollout nodes wait during training. This leads to suboptimal GPU utilization on both sides.

### Solution

A **Hybrid + Standalone dual-mode inference resource** design:

- **Standalone replicas**: Always-on inference replicas on dedicated rollout nodes.
- **Hybrid replicas**: Inference replicas that share Trainer-node GPUs. They are activated during training idle time; before each training step, weights are offloaded and GPU memory is returned to the training engine.

![Dynamic Resource Architecture](https://github.com/zpltys/Blob/blob/main/dynamic_resource.png?raw=true)

The figure above illustrates the **resource-time layout** of the dynamic scaling system across three consecutive training steps. The vertical axis shows GPU resources split into two pools:

- **Standalone Resource** (top, GPU 0–n): Dedicated rollout GPUs that **continuously perform rollout** across all steps — they are always active during the *Rollout* phase (blue) and idle during *Trainer* / *Weight Sync*.
- **Hybrid Resource** (bottom, GPU n+1–n+m): Trainer-node GPUs that **switch between rollout and training**. When in *Rollout* mode they join the Rollout LoadBalancer to generate samples (blue); when in *Trainer* mode they run PPO mini-batch updates (yellow); at each *Weight Sync* boundary the latest weights are broadcast to all replicas.

The horizontal axis shows three steps with distinct Hybrid behaviours:

- **Step 1 & 2**: At the start of each step, the MessageQueue does not yet have enough samples. Hybrid resources activate into *Rollout* mode to help generate samples alongside Standalone resources, then switch back to *Trainer* once sufficient samples are buffered.
- **Step 3**: Before this step begins, the MessageQueue **already contains enough samples** (buffered from previous steps). Hybrid resources skip Rollout entirely and go directly into *Trainer* mode — this is the key optimisation: when Standalone capacity is sufficient, Trainer GPUs stay focused on training without unnecessary context-switching overhead.

The transition threshold is controlled by `dynamic_scaling_deactivate_ratio`: the controller deactivates Hybrid replicas once `deactivate_ratio × required_samples × trigger_parameter_sync_step` samples have been collected. A central *Rollout LoadBalancer* dispatches generation requests from the *MessageQueue* across all active replicas.

Two request-handling mechanisms ensure smooth transitions between modes:

- **Hybrid → Trainer (deactivation)**: When Hybrid resources switch from Rollout back to Trainer, any **in-flight requests** running on them are aborted and automatically **redistributed by the LoadBalancer to Standalone resources**, which continue the rollout. This is transparent to upper layers thanks to the retry mechanism in `FullyAsyncLLMServerClient`.
- **Trainer → Hybrid (activation)**: After a training step ends, if Hybrid resources switch back into Rollout mode, the `dynamic_scaling_enable_rebalance` parameter controls whether to perform a **reshuffle**: when enabled, the controller clears the LoadBalancer's sticky-session cache and aborts in-flight requests across all active replicas, then resumes them so requests are redistributed via least-loaded routing — naturally balancing load toward the newly activated Hybrid replicas (which start with 0 in-flight requests).

- **Weight Sync (parameter synchronisation)**: During the *Weight Sync* phase, weights are first broadcast from Trainer to all **Standalone** rollout replicas (always required). The policy's `should_activate_after_step()` is then evaluated to decide whether Hybrid resources should enter Rollout mode for the next step. If activation is needed, weights are **additionally synced to Hybrid replicas**; otherwise this second sync is **skipped entirely**, saving significant communication overhead — this is exactly what happens in Step 3 of the figure above.

`DynamicResourceController` manages the lifecycle of hybrid replicas. A pluggable **Policy** decides when to activate and deactivate:

```
State machine:  STANDALONE_ONLY  <->  HYBRID_ACTIVE

Activate (after weight sync):
  1. add_replicas               — register hybrid replicas in the load balancer
  2. resume_generation_replicas — allow hybrid replicas to accept requests

Deactivate (order is critical):
  1. remove_replicas  — cut routing first; prevents retry loop re-routing to dying replicas
  2. abort_replicas   — abort in-flight requests; partial-rollout retries go to standalone
  3. sleep_replicas   — release KV cache + offload weights, return GPU to training engine
```

### Policy Call Order Per Training Step

```
1. should_deactivate()          — before training; decide whether to deactivate hybrid replicas
2. deactivate_wait_samples()    — if (1) is True; return the minimum buffered-sample threshold
3. should_activate_after_step() — after weight sync; decide whether to (re-)activate hybrid replicas
4. request_rebalance()          — after activation; redistribute requests across replicas (if enabled)
5. update_after_step()          — after weight sync; update policy internal state
```

---

## 2. Configuration Parameters

All dynamic scaling parameters live under the `async_training` section of your training config
(`fully_async_ppo_trainer.yaml` or `fully_async_ppo_megatron_trainer.yaml`):

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dynamic_resource_scaling` | bool | `False` | Master switch. When `True`, hybrid rollout replicas are initialised on Trainer-node GPUs at startup (sleeping, memory returned to the training engine). |
| `dynamic_scaling_policy` | str | `"default"` | Name of the scaling policy (`"default"`, `"static_fully_async"`, or custom registered name). |
| `dynamic_scaling_deactivate_ratio` | float | `0.3` | Sample-collection ratio threshold. The controller waits until `deactivate_ratio × required_samples × trigger_parameter_sync_step` samples are buffered before deactivating. Lower → earlier deactivation; `1.0` → wait for a full batch. |
| `dynamic_scaling_enable_rebalance` | bool | `False` | Whether to rebalance (abort + clear sticky cache + resume) in-flight requests across all active replicas after hybrid activation, via least-loaded routing. |

### Existing Async-Training Parameters (used by dynamic scaling)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `staleness_threshold` | float | `0.1` | Allowed sample staleness ratio; affects `buffer_samples = expected × staleness_threshold`. |
| `trigger_parameter_sync_step` | int | `4` | Number of collections per weight-sync step; used in the deactivate wait formula above. |
| `require_batches` | int | `1` | Number of ppo_mini_batches per collection; determines `required_samples`. |

### FSDP-Specific Parameter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `concurrent_samples_per_replica` | int | `16` | Maximum concurrent generation requests per replica. Used to compute `max_concurrent_samples = num_active_replicas × concurrent_samples_per_replica`. (FSDP config only.) |

### Rollout Resource Config (under `actor_rollout_ref.rollout`)

| Parameter | Description |
|-----------|-------------|
| `gpu_memory_utilization` | Memory utilization for **hybrid** replicas sharing GPU with training. Keep low (e.g. 0.3–0.5). |
| `standalone_gpu_memory_utilization` | Memory utilization for **standalone** replicas on dedicated rollout nodes (e.g. 0.6–0.8). Falls back to `gpu_memory_utilization` if `null`. |

### Standalone Node Config (under `rollout`)

| Parameter | Description |
|-----------|-------------|
| `rollout.nnodes` | Number of standalone rollout nodes. Set to `0` for pure colocated mode (hybrid only, no dedicated rollout nodes). |

---

## 3. Built-in Policies

### 3.1 `default` — DefaultDynamicScalingPolicy

**File:** `default_policy.py`

The recommended policy with **adaptive deactivate_ratio**:

| Method | Behaviour |
|--------|-----------|
| `should_deactivate()` | Returns `is_hybrid_active` (deactivate whenever active) |
| `deactivate_wait_samples()` | Returns `deactivate_ratio × required_samples × trigger_parameter_sync_step` |
| `should_activate_after_step()` | Activates when generated samples fall behind expectation |
| `update_after_step()` | Adapts `deactivate_ratio`: trainer wait > 10 s → `+0.05`; otherwise → `-0.2` (disabled when `only_hybrid=True`) |
| `request_rebalance()` | Clears sticky cache + aborts in-flight requests + resumes, so requests are redistributed via least-loaded routing |

**Use case:** Hybrid + Standalone mixed deployment targeting maximum GPU utilization.

### 3.2 `static_fully_async` — StaticFullyAsyncPolicy

**File:** `static_fully_async_policy.py`

Equivalent to the original fully-async strategy, designed for baseline comparisons or colocated fallback:

| Method | Behaviour |
|--------|-----------|
| `should_deactivate()` | Returns `is_hybrid_active` (deactivate whenever active) |
| `deactivate_wait_samples()` | **Always returns `0`** (deactivate immediately, no waiting) |
| `should_activate_after_step()` | **Always returns `False`** (never re-activate after weight sync) |
| `update_after_step()` | No-op |
| `request_rebalance()` | No-op (inherits base default) |

**Key properties:**

1. **Equivalent to standard Fully-Async**: Hybrid replicas are deactivated immediately at each training step start and never re-activated. Trainer GPUs are always 100% returned to training — same behaviour as running without `use_dynamic_resource_scaling`.
2. **Colocated fallback**: When `rollout.nnodes=0`, `only_hybrid=True` and the system runs in classic colocated mode (training + inference share the same GPUs, no separate rollout nodes).

---

## 4. Experimental Results

### Benchmark: Qwen3.5-35B-A3B on DAPO-Math-17K

| Item | Config |
|------|--------|
| Model | Qwen3.5-35B-A3B |
| Dataset | DAPO-Math-17k (train) / AIME-2024 (val) |
| Backend | Megatron (TP=4, PP=3, EP=8) |
| Hardware | H20 (8 GPUs/node) |
| Script | `verl/experimental/fully_async_policy/shell/run_qwen35_35b_a3b_math_dynamic_megatron.sh` |

**Baseline**: 16 GPU training + 16 GPU rollout (2 dedicated trainer nodes + 2 rollout nodes).  
**Dynamic scaling**: 24 GPU training (3 nodes) + 8 GPU rollout (1 node), with Hybrid replicas on Trainer GPUs.

#### Result: 12% Faster End-to-End Training Time

Dynamic resource scaling reduces total wall-clock training time by **~12%** compared to the 16+16 baseline, while producing an **identical reward curve** — confirming that the training quality is not compromised by the time-sliced GPU sharing.

**Reward curve** (dynamic scaling vs baseline):

![Reward Curve](https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_reward.png?raw=true)

Both curves overlap closely throughout training, demonstrating that dynamic scaling introduces no regression in model quality.

**Per-step runtime comparison**:

![Per-step Runtime](https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_time.png?raw=true)

The per-step runtime plot shows that dynamic scaling reduces step latency by leveraging Trainer GPUs during rollout idle windows, with the gap widening as training progresses and the policy adapts the `deactivate_ratio`.

---

## 5. Adding a Custom Policy

Four steps to support a new policy:

### Step 1: Subclass the base class

```python
from verl.experimental.fully_async_policy.dynamic_scaling import (
    DynamicScalingPolicyBase,
    DynamicScaleContext,
    register_policy,
)

@register_policy("my_policy")
class MyDynamicScalingPolicy(DynamicScalingPolicyBase):

    def __init__(self, deactivate_ratio: float = 0.5, only_hybrid: bool = False):
        self.deactivate_ratio = deactivate_ratio
        self.only_hybrid = only_hybrid

    def should_deactivate(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScaleContext,
    ) -> bool:
        """Return True to deactivate hybrid replicas this step."""
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScaleContext) -> int:
        """Return minimum buffered-sample count before deactivation proceeds."""
        return int(ctx.required_samples * ctx.trigger_parameter_sync_step * self.deactivate_ratio)

    def should_activate_after_step(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScaleContext,
    ) -> bool:
        """Return True to re-activate after weight sync."""
        return ctx.total_generated_samples < ctx.expected_samples + ctx.buffer_samples

    # Optional: override to update internal state after each step
    def update_after_step(self, global_steps: int, ctx: DynamicScaleContext) -> None:
        pass

    # Optional: override to customise request redistribution after activation
    def request_rebalance(self, global_steps: int, ctx: DynamicScaleContext) -> None:
        pass
```

### Step 2: Register the import

Place the file inside `dynamic_scaling/` and add an import in `__init__.py`:

```python
# dynamic_scaling/__init__.py
from .my_policy import MyDynamicScalingPolicy
```

Or import it manually in your entry script to trigger `@register_policy`.

### Step 3: Reference in config

```yaml
async_training:
  use_dynamic_resource_scaling: True
  dynamic_scaling_policy: "my_policy"
  dynamic_scaling_deactivate_ratio: 0.5
  dynamic_scaling_enable_rebalance: True
```

### Step 4: Use `DynamicScaleContext` fields

| Field | Type | Description |
|-------|------|-------------|
| `required_samples` | `int` | Min samples per collection (`ppo_mini_batch_size × require_batches`) |
| `trigger_parameter_sync_step` | `int` | Collections per weight-sync step |
| `total_generated_samples` | `int` | Cumulative rollout samples since training began |
| `expected_samples` | `int` | Theoretical samples needed up to current sync step |
| `buffer_samples` | `int` | Allowed buffer headroom (`expected × staleness_threshold`) |
| `step_wait_times` | `list[float]` | Per-collection wait times within latest step (seconds) |
| `only_hybrid` | `bool` | `True` when there are no standalone replicas |
| `last_activate_duration_s` | `float` | Duration of last activate cycle (weight sync + onload), seconds |
| `last_deactivate_duration_s` | `float` | Duration of last deactivate cycle (offload), seconds |

---

## 6. File Structure

```
dynamic_scaling/
├── __init__.py                        # Public exports + policy registry
├── base.py                            # DynamicScalingPolicyBase ABC, DynamicScaleContext, registry
├── default_policy.py                  # DefaultDynamicScalingPolicy (adaptive dynamic scaling)
├── static_fully_async_policy.py       # StaticFullyAsyncPolicy (original fully-async / colocated fallback)
├── dynamic_resource_controller.py     # DynamicResourceController (state machine + lifecycle)
└── README.md                          # This document
```
