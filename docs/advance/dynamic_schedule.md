# Dynamic Resource Scheduling for Fully-Async Training

**Author:** `https://github.com/meituan-search`

Last updated: 07/10/2026.

This module provides **hybrid inference resource dynamic scheduling** for the fully-async training framework, enabling Trainer-node GPUs to participate in rollout generation during idle periods and thus improving overall GPU utilization.

> **At a glance**
> - Two kinds of inference replicas: **Standalone** (dedicated rollout nodes, always on) + **Hybrid** (share Trainer-node GPUs; activated during training idle time, weights offloaded and memory returned before each training step).
> - A pluggable **Policy** decides when to activate/deactivate Hybrid replicas; `DynamicResourceController` owns the lifecycle (state machine `STANDALONE_ONLY <-> HYBRID_ACTIVE`).
> - When Standalone capacity is sufficient, Hybrid skips rollout entirely and stays focused on training, avoiding needless mode-switching; weight sync to Hybrid replicas can also be skipped, saving communication.
> - Measured (Qwen3.5-35B-A3B / DAPO-Math-17K): ~**15.3%** faster end-to-end training time, with a reward curve identical to baseline.

---

## 1. Overview

### Problem

In the fully-async separated architecture, Trainer-node GPUs sit idle while waiting for rollout data, and Standalone Rollout nodes wait during training. This leads to suboptimal GPU utilization on both sides.

### Solution

A **Hybrid + Standalone dual-mode inference resource** design:

- **Standalone replicas**: Always-on inference replicas on dedicated rollout nodes.
- **Hybrid replicas**: Inference replicas that share Trainer-node GPUs. They are activated during training idle time; before each training step, weights are offloaded and GPU memory is returned to the training engine.

![Dynamic Resource Architecture](https://github.com/zpltys/Blob/blob/main/dynamic_resource.png?raw=true)

The figure above illustrates the **resource-time layout** of the dynamic scheduling system across three consecutive training steps. The vertical axis shows GPU resources split into two pools:

- **Standalone Resource** (top, GPU 0–n): Dedicated rollout GPUs that **continuously perform rollout** across all steps — they are always active during the *Rollout* phase (blue) and idle during *Trainer* / *Weight Sync*.
- **Hybrid Resource** (bottom, GPU n+1–n+m): Trainer-node GPUs that **switch between rollout and training**. When in *Rollout* mode they join the Rollout LoadBalancer to generate samples (blue); when in *Trainer* mode they run PPO mini-batch updates (yellow); at each *Weight Sync* boundary the latest weights are broadcast to all replicas.

The horizontal axis shows three steps with distinct Hybrid behaviours:

- **Step 1 & 2**: At the start of each step, the MessageQueue does not yet have enough samples. Hybrid resources activate into *Rollout* mode to help generate samples alongside Standalone resources, then switch back to *Trainer* once sufficient samples are buffered.
- **Step 3**: Before this step begins, the MessageQueue **already contains enough samples** (buffered from previous steps). Hybrid resources skip Rollout entirely and go directly into *Trainer* mode — this is the key optimisation: when Standalone capacity is sufficient, Trainer GPUs stay focused on training without unnecessary context-switching overhead.

The transition threshold is controlled by `dynamic_schedule_deactivate_ratio`: the controller deactivates Hybrid replicas once `deactivate_ratio × required_samples × trigger_parameter_sync_step` samples have been collected. A central *Rollout LoadBalancer* dispatches generation requests from the *MessageQueue* across all active replicas.

Two request-handling mechanisms ensure smooth transitions between modes:

- **Hybrid → Trainer (deactivation)**: When Hybrid resources switch from Rollout back to Trainer, any **in-flight requests** running on them are aborted and automatically **redistributed by the LoadBalancer to Standalone resources**, which continue the rollout. This is transparent to upper layers thanks to the retry mechanism in `FullyAsyncLLMServerClient`.
- **Trainer → Hybrid (activation)**: After a training step ends, if Hybrid resources switch back into Rollout mode, the `dynamic_schedule_enable_rebalance` parameter controls whether to perform a **reshuffle**: when enabled, the controller clears the LoadBalancer's sticky-session cache and aborts in-flight requests across all active replicas, then resumes them so requests are redistributed via least-loaded routing — naturally balancing load toward the newly activated Hybrid replicas (which start with 0 in-flight requests).

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

All dynamic scheduling parameters live under the `async_training` section of your training config
(`fully_async_ppo_trainer.yaml` or `fully_async_ppo_megatron_trainer.yaml`):

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dynamic_resource_scheduling` | bool | `False` | Master switch. When `True`, hybrid rollout replicas are initialised on Trainer-node GPUs at startup (sleeping, memory returned to the training engine). |
| `dynamic_schedule_policy` | str | `"default"` | Name of the scheduling policy (`"default"`, `"static_fully_async"`, `"fixed_ratio"`, or a custom registered name). |
| `dynamic_schedule_deactivate_ratio` | float | `0.3` | Sample-collection ratio threshold. The controller waits until `deactivate_ratio × required_samples × trigger_parameter_sync_step` samples are buffered before deactivating. Lower → earlier deactivation; `1.0` → wait for a full batch. |
| `dynamic_schedule_enable_rebalance` | bool | `True` | Whether to rebalance (abort + clear sticky cache + resume) in-flight requests across all active replicas after hybrid activation, via least-loaded routing. |

### Existing Async-Training Parameters (used by dynamic scheduling)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `staleness_threshold` | float | `0.1` | Allowed sample staleness ratio; affects `buffer_samples = expected × staleness_threshold`. |
| `trigger_parameter_sync_step` | int | `4` | Number of collections per weight-sync step; used in the deactivate wait formula above. |
| `require_batches` | int | `1` | Number of ppo_mini_batches per collection; determines `required_samples`. |


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

### 3.1 `default` — DefaultDynamicSchedulePolicy

**File:** `verl/experimental/fully_async_policy/dynamic_schedule/default_policy.py`

The recommended policy with **adaptive deactivate_ratio** and a **cost/benefit activation gate**:

| Method | Behaviour |
|--------|-----------|
| `should_deactivate()` | Returns `is_hybrid_active` (deactivate whenever active) |
| `deactivate_wait_samples()` | Returns `deactivate_ratio × required_samples × trigger_parameter_sync_step` |
| `should_activate_after_step()` | Activates when generated samples fall behind expectation (a positive gap exists) **and** the estimated wall-clock saved exceeds the switch (activate+deactivate) cost |
| `update_after_step()` | Adapts `deactivate_ratio` from the `step_wait_samples` signal (see below); disabled when `only_hybrid=True` |
| `request_rebalance()` | Clears sticky cache + aborts in-flight requests + resumes, so requests are redistributed via least-loaded routing |

**`update_after_step()` adaptation logic** — keyed off `step_wait_samples` (the count of samples that actually had to be waited on for generation, excluding samples served instantly from queue backlog):

- This cycle had real waits (`total_wait_samples > 0`, rollout is the bottleneck): `ratio += clip(total_wait_samples / step_required_samples, 0.02, 0.1)` — the larger the shortfall fraction, the bigger the step (bounded), biasing toward deactivating **later**.
- No real waits this cycle (training is the bottleneck): `ratio -= 0.02`, biasing toward deactivating **earlier**.

**Cost/benefit gate** (`should_activate_after_step`): re-activating hybrid replicas costs one activate + one (future) deactivate cycle — `switch_cost`, the rolling average of the last 3 measured activate+deactivate durations (falls back to 10 s with no history). Activation only proceeds when the standalone-only generation shortfall, converted to wall-clock via a per-sample-time estimate (`total_wait / total_wait_samples`), exceeds `switch_cost`. Before any real generation-rate signal has been observed, the per-sample-time estimate defaults to a deliberately pessimistic 1000 s, biasing toward activation until real timing data narrows it.

**Use case:** Hybrid + Standalone mixed deployment targeting maximum GPU utilization.

### 3.2 `static_fully_async` — StaticFullyAsyncPolicy

**File:** `verl/experimental/fully_async_policy/dynamic_schedule/static_fully_async_policy.py`

Equivalent to the original fully-async strategy, designed for baseline comparisons or colocated fallback:

| Method | Behaviour |
|--------|-----------|
| `should_deactivate()` | Returns `is_hybrid_active` (deactivate whenever active) |
| `deactivate_wait_samples()` | **Always returns `0`** (deactivate immediately, no waiting) |
| `should_activate_after_step()` | **Always returns `False`** (never re-activate after weight sync) |
| `update_after_step()` | No-op |
| `request_rebalance()` | No-op (inherits base default) |

**Key properties:**

1. **Equivalent to standard Fully-Async**: Hybrid replicas are deactivated immediately at each training step start and never re-activated. Trainer GPUs are always 100% returned to training — same behaviour as running without `use_dynamic_resource_scheduling`.
2. **Colocated fallback**: When `rollout.nnodes=0`, `only_hybrid=True` and the system runs in classic colocated mode (training + inference share the same GPUs, no separate rollout nodes).

### 3.3 `fixed_ratio` — FixedRatioDynamicSchedulePolicy

**File:** `verl/experimental/fully_async_policy/dynamic_schedule/fixed_ratio_policy.py`

Identical to `default` except that `update_after_step()` is a no-op — `deactivate_ratio` stays at its initial value throughout training, with no adaptation. `should_activate_after_step()` is also simplified to a pure gap check (no cost/benefit gate). Useful for fixed-ratio ablation experiments.

> **Registration note:** `fixed_ratio` is **not** imported in `__init__.py`, so `@register_policy` does not fire on a default import. Add `from .fixed_ratio_policy import FixedRatioDynamicSchedulePolicy` to `verl/experimental/fully_async_policy/dynamic_schedule/__init__.py`, or import it manually in your entry script, before referencing it in config.

---

## 4. Monitoring Metrics

Dynamic resource scheduling emits a set of `dynamic_resource/*` metrics (in addition to the general
`fully_async/*` metrics documented in [`fully_async.md`](./fully_async.md))
to help quantify how effectively Hybrid and Standalone GPUs are utilized. This section gives the
formal definition of each metric.

### Notation

| Symbol | Meaning |
|--------|---------|
| $a$ | `hybrid_gpus` = `trainer.nnodes × trainer.n_gpus_per_node` (Trainer-node GPUs that can switch between rollout/train) |
| $b$ | `standalone_gpus` = `rollout.nnodes × rollout.n_gpus_per_node` (dedicated rollout-node GPUs) |
| $\text{compute}_i$ | Training-GPU compute time of micro fit-step $i$ (sum of `timing_raw` keys: `reward`, `old_log_prob`, `ref`, `values`, `adv`, `update_critic`, `update_actor`) |
| $\text{alloc}_i$ | Allocated wall-clock time of micro fit-step $i$ (from `_fit_generate()` through `_fit_dump_data()`, including `param_sync` and activation overhead) |
| $\text{cap}_k$ | Concurrency capacity of rollout interval $k$ = $\min(\text{servers}_k \times s,\ \text{maxReq})$ |
| $\text{active}_k$ | Active task count during rollout interval $k$ |
| sync cycle | The window between two consecutive parameter-sync events (i.e. between two `reset_staleness()` calls / every `trigger_parameter_sync_step` collections), possibly spanning multiple Trainer micro fit-steps (including partial-rollout resumption) |

### 4.1 `dynamic_resource/train_resource_utilization`

Fraction of the Trainer's "training turn" wall-clock time that is spent on actual training-GPU
compute, aggregated over an entire sync cycle.

For each micro fit-step $i = 1, \dots, n$ within the cycle:

- **Numerator** ($\text{compute}_i$): sum of the `timing_raw` training-compute keys
  (`reward`, `old_log_prob`, `ref`, `values`, `adv`, `update_critic`, `update_actor`).
  — `ref` is the timing key recorded via `marked_timer(str(Role.RefPolicy), …)` (the enum's
  string value is `"ref"`, not `"RefPolicy"`). Missing keys — e.g. `values`/`update_critic`
  when `use_critic=False` — default to 0.
- **Denominator** ($\text{alloc}_i$): wall-clock time from `_fit_generate()` through
  `_fit_update_weights()` and `_fit_dump_data()` (includes `timing_s/param_sync` and any hybrid
  activation time; these are intentionally **not** subtracted)

Both quantities are summed across all micro-steps in the cycle first, and the ratio is computed
once from the summed totals (not averaged per micro-step):

$$
U_{\text{train}} = \frac{\sum_{i=1}^{n} \text{compute}_i}{\sum_{i=1}^{n} \text{alloc}_i}
$$

The un-ratioed numerator and denominator are also emitted directly as the raw metrics
`dynamic_resource/train_compute_time_s` and `dynamic_resource/train_allocated_time_s`
(aggregated by **sum** across the cycle's micro-steps), so the ratio is computed once from the
summed totals by `MetricsAggregator` rather than averaged per micro-step.

### 4.2 `dynamic_resource/rollout_resource_utilization`

Fraction of rollout concurrency capacity actually used during the Rollouter's step (the window
between the previous and current `reset_staleness()` call).

The Rollouter records an event-driven history of `(len(active_tasks), max_concurrent_samples)`
tuples — appended every time a sample is submitted, completes, or is drained during a pause,
**and** whenever `max_concurrent_samples` itself changes (i.e. a replica is
activated/deactivated under dynamic resource scheduling). This gives timestamps
$t_0 < t_1 < \dots < t_m$ with two quantities held constant over each interval $[t_k, t_{k+1})$:
the active-task count $\text{active}_k$ and the concurrency capacity $\text{cap}_k$
($t_0$ is the previous step's end / this step's start; $t_m$ is "now", appended when
`reset_staleness()` is called, closing out the window including any trailing drain/idle time).

Recording the capacity alongside the active count per interval is essential because under dynamic
resource scheduling $\text{cap}_k$ is **not** constant over a step window — replicas can be
activated/deactivated mid-window. Using a single end-of-window capacity for the whole window
would misattribute utilization for intervals that had a different capacity, so each interval uses
its own $\text{cap}_k$.

The capacity for each interval is $\text{cap}_k = \min(\text{servers}_k \times s,\ \text{maxReq})$,
where $\text{servers}_k$ is `get_active_server_count()` at interval $k$, $s$ is
`concurrent_samples_per_replica`, and $\text{maxReq}$ is `max_required_samples`:

$$
\text{cap}_k = \min(\text{servers}_k \times s,\ \text{maxReq})
$$

Each interval's effective load is $\min(\text{cap}_k, \text{active}_k)$ (the capacity
$\text{cap}_k$ is already clamped, so the actual served concurrency is the smaller of capacity
and active tasks). Weighting each interval by its capacity-time (so a higher-capacity interval
counts proportionally more):

$$
U_{\text{rollout}} = \frac{\sum_k (t_{k+1} - t_k) \cdot \min(\text{cap}_k, \text{active}_k)}{\sum_k (t_{k+1} - t_k) \cdot \text{cap}_k}
$$

Returns `0.0` if there is no time span to integrate over (e.g. total capacity-time weight is 0;
intervals with $\text{cap}_k = 0$ contribute zero weight and are skipped).

### 4.3 `dynamic_resource/resource_utilization`

Cluster-wide GPU utilization for the sync cycle, combining the Hybrid GPUs' time-split between
rollout and train with the Standalone GPUs (which spend 100% of their time on rollout).

Let $x \in [0, 1]$ be the fraction of the cycle's wall-clock time that Hybrid GPUs spent doing
rollout (the rest, $1-x$, they spent training), estimated as the ratio of summed
"wait for enough samples" time (Hybrid-rollout wall-clock time, only recorded when dynamic
resource scheduling deactivates Hybrid replicas this cycle) over the summed step time:

$$
x = \mathrm{clip}\left(\frac{\text{wait}}{\text{step}},\ 0,\ 1\right)
$$

where $\text{wait}$ is the summed `timing_s/wait_for_enough_samples` and $\text{step}$ is the
summed `timing_s/step`. When dynamic resource scheduling never switches Hybrid GPUs into rollout
mode this cycle (e.g. disabled, or `wait_for_enough_samples` was skipped), $x = 0$ (100% of
Hybrid time is training).

Weighting each utilization by the GPU-seconds it was measured over — Hybrid GPUs spend
$(1-x) \cdot a$ GPU-time training at $U_{\text{train}}$ and $x \cdot a$ GPU-time doing
rollout at $U_{\text{rollout}}$; Standalone GPUs ($b$ of them) spend all their time doing
rollout at $U_{\text{rollout}}$:

$$
U_{\text{cluster}} = \frac{(1-x) \cdot a \cdot U_{\text{train}} + (x \cdot a + b) \cdot U_{\text{rollout}}}{a + b}
$$

### 4.4 `dynamic_resource/mq_size`

A snapshot (not an average) of the number of samples remaining in the MessageQueue for subsequent
steps, taken right after a Trainer `fit_step()` (including any partial-rollout resumption) finishes.
If we denote this snapshot as $Q$, then $Q$ equals the value returned by
`message_queue_client.get_queue_size_sync()`. Reported as the last value observed in the cycle
rather than an average across the cycle's
micro-steps.

### 4.5 Related rollouter-side metric: `fully_async/rollouter/step_generated_samples`

Number of samples fully generated by the Rollouter during the current param version (reset to 0 at
each `reset_staleness()` call). It is returned in the rollouter's `timing_raw` alongside
`dynamic_resource/rollout_resource_utilization` and is the underlying signal the Rollouter uses to
track per-step throughput; it is documented here because it is computed in the same code path as
the `dynamic_resource/*` metrics above.

### Caveats

- **`rollout_resource_utilization` per-interval capacity**: under dynamic resource scheduling the
  capacity $S_k$ can change mid-window as replicas are activated/deactivated; each interval uses
  its own $S_k$ rather than a single end-of-window value, so capacity changes are reflected
  faithfully. Intervals recorded before `max_concurrent_samples` is first known (a single seed
  point at init with placeholder capacity 0) are corrected once the real capacity is available.
- **`resource_utilization` assumes `should_deactivate() == is_hybrid_active`**: the $x$ estimate
  relies on `timing_s/wait_for_enough_samples` being a faithful proxy for "Hybrid GPUs are in
  rollout mode this cycle", which holds for the built-in `default`/`static_fully_async` policies
  but may not hold for custom policies with different activation semantics.
- **`train_resource_utilization` and `rollout_resource_utilization` are not perfectly comparable**:
  the former's denominator includes `param_sync`/activation overhead (not subtracted), while the
  latter's denominator is pure rollout window time — keep this in mind when interpreting the
  combined `resource_utilization`.

---

## 5. Experimental Results

### Benchmark: Qwen3.5-35B-A3B on DAPO-Math-17K

| Item | Config |
|------|--------|
| Model | Qwen3.5-35B-A3B |
| Dataset | DAPO-Math-17k (train) / AIME-2024 (val) |
| Backend | Megatron (TP=4, PP=2, EP=8) |
| Hardware | H20 (8 GPUs/node) |
| Script | `verl/experimental/fully_async_policy/shell/run_qwen35_35b_a3b_math_dynamic_megatron.sh` |
| Key hyperparams | `dynamic_schedule_policy="default"`, `dynamic_schedule_deactivate_ratio=0.6`, `dynamic_schedule_enable_rebalance=True`, `staleness_threshold=0.5`, `trigger_parameter_sync_step=4`; hybrid `gpu_memory_utilization=0.45`, standalone `standalone_gpu_memory_utilization=0.7` |

**Baseline 1**: 16 GPU training + 16 GPU rollout (2 dedicated trainer nodes + 2 rollout nodes).  
**Baseline 2**: 8 GPU training + 24 GPU rollout (1 dedicated trainer nodes + 3 rollout nodes).  
**Dynamic scheduling**: 16 GPU training (2 nodes) + 16 GPU rollout (2 node), with Hybrid replicas on Trainer GPUs.

#### Result: 15.3% Faster End-to-End Training Time

Dynamic resource scheduling reduces total wall-clock training time by **~15.3%** compared to the 8+24 baseline, and **~17.5%** compared to the 16+16 baseline, while producing an **identical reward curve** — confirming that the training quality is not compromised by the time-sliced GPU sharing.

**Reward curve** (dynamic scheduling vs baseline):

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_reward.png?raw=true" width="600" />
</div>

Both curves overlap closely throughout training, demonstrating that dynamic scheduling introduces no regression in model quality.

**Per-step runtime comparison**:

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_time.png?raw=true" width="600" />
</div>

The per-step runtime plot shows that dynamic scheduling reduces step latency by leveraging Trainer GPUs during rollout idle windows, with the gap widening as training progresses and the policy adapts the `deactivate_ratio`.

#### Resource Utilization Breakdown

The three static configurations differ in their **training/inference load balance**, which the metrics below make explicit. The key intuition: `dynamic_resource/mq_size` (the number of samples buffered in the MessageQueue after each `fit_step()`) directly reveals whether the bottleneck is on the production side (rollout) or the consumption side (trainer).

- **16-16-static** (16 GPU train + 16 GPU rollout): the trainer consumes samples faster than the 16-GPU rollout can produce them, so the **bottleneck is on the production side** — the queue is constantly drained to near zero and the trainer is left waiting for new samples.
- **24-8-static** (8 GPU train + 24 GPU rollout): the 24-GPU rollout produces samples faster than the 8-GPU trainer can consume them, so the **bottleneck is on the consumption (training) side** — samples pile up and the queue sits at its capacity limit.
- **dynamic**: by reallocating Trainer-node GPUs between rollout and training based on real-time queue depth, the controller keeps production and consumption matched — the queue stays at a moderate level, avoiding both the trainer starvation of 16-16 and the sample buildup of 24-8.

**MessageQueue size** (`dynamic_resource/mq_size`):

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_exp_mq_size.png?raw=true" width="600" />
</div>

The 16-16-static curve hugs zero (production-bound, trainer starves); the 24-8-static curve sits at the capacity ceiling (consumption-bound, samples back up); the dynamic curve stays in between, holding the generation/consumption balance.

**Cluster-wide GPU utilization** (`dynamic_resource/resource_utilization`):

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_resource_utilization.png?raw=true" width="600" />
</div>

16-16-static wastes training-side GPU time waiting for samples; 24-8-static wastes rollout-side GPU time over-producing unconsumed samples. Dynamic scheduling keeps both sides' GPU time on useful work, yielding the highest cluster-wide utilization.

**Rollout-side utilization** (`dynamic_resource/rollout_resource_utilization`):

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_rollout_resource_utilization.png?raw=true" width="600" />
</div>

16-16-static (production-bound) keeps its 16 rollout GPUs pinned near capacity; 24-8-static (production-rich) leaves much of its 24-GPU rollout capacity idle since the trainer cannot keep up. Dynamic scheduling holds rollout utilization at a high, stable level — neither bottlenecked like 16-16 nor idle like 24-8.

**Train-side utilization** (`dynamic_resource/train_resource_utilization`):

<div align="center">
<img src="https://github.com/zpltys/Blob/blob/main/dynamic_resource_train_resource_utilization.png?raw=true" width="600" />
</div>

24-8-static has samples in abundance, so the trainer never waits and achieves the highest train-side utilization. 16-16-static's trainer spends much of its turn waiting for samples (the wait is inside the metric's denominator, lowering the ratio). Dynamic scheduling approaches the 24-8 level when samples are sufficient, while avoiding the 16-16 wait waste.

**Summary of load-balance trade-offs**:

| Config | Bottleneck | MQ backlog | Rollout util | Train util |
|--------|-----------|------------|--------------|------------|
| 16-16-static | Production (rollout) | Near zero (trainer starves) | Pinned at capacity | Low (waits for samples) |
| 24-8-static | Consumption (trainer) | At capacity limit | Idle capacity | High (never waits) |
| dynamic | Matched dynamically | Moderate | High & stable | Near 24-8 |

Dynamic scheduling's value: by shifting GPUs between training and rollout to match the actual load, it simultaneously avoids the 16-16 trainer starvation and the 24-8 sample buildup, achieving higher end-to-end throughput without adding more total GPUs.

---

## 6. Adding a Custom Policy

Four steps to support a new policy:

### Step 1: Subclass the base class

```python
from verl.experimental.fully_async_policy.dynamic_schedule import (
    DynamicSchedulePolicyBase,
    DynamicScheduleContext,
    register_policy,
)

@register_policy("my_policy")
class MyDynamicSchedulePolicy(DynamicSchedulePolicyBase):

    def __init__(self, deactivate_ratio: float = 0.5, only_hybrid: bool = False):
        self.deactivate_ratio = deactivate_ratio
        self.only_hybrid = only_hybrid

    def should_deactivate(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScheduleContext,
    ) -> bool:
        """Return True to deactivate hybrid replicas this step."""
        return is_hybrid_active

    def deactivate_wait_samples(self, ctx: DynamicScheduleContext) -> int:
        """Return minimum buffered-sample count before deactivation proceeds."""
        return int(ctx.required_samples * ctx.trigger_parameter_sync_step * self.deactivate_ratio)

    def should_activate_after_step(
        self,
        global_steps: int,
        is_hybrid_active: bool,
        ctx: DynamicScheduleContext,
    ) -> bool:
        """Return True to re-activate after weight sync."""
        return ctx.total_generated_samples < ctx.expected_samples + ctx.buffer_samples

    # Optional: override to update internal state after each step
    def update_after_step(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        pass

    # Optional: override to customise request redistribution after activation
    def request_rebalance(self, global_steps: int, ctx: DynamicScheduleContext) -> None:
        pass
```

### Step 2: Register the import

Place the file inside `verl/experimental/fully_async_policy/dynamic_schedule/` and add an import in `__init__.py`:

```python
# verl/experimental/fully_async_policy/dynamic_schedule/__init__.py
from .my_policy import MyDynamicSchedulePolicy
```

Or import it manually in your entry script to trigger `@register_policy`.

### Step 3: Reference in config

```yaml
async_training:
  use_dynamic_resource_scheduling: True
  dynamic_schedule_policy: "my_policy"
  dynamic_schedule_deactivate_ratio: 0.5
  dynamic_schedule_enable_rebalance: True
```

### Step 4: Use `DynamicScheduleContext` fields

| Field | Type | Description |
|-------|------|-------------|
| `required_samples` | `int` | Min samples per collection (`ppo_mini_batch_size × require_batches`) |
| `trigger_parameter_sync_step` | `int` | Collections per weight-sync step |
| `step_required_samples` | `int` | Derived: `required_samples × trigger_parameter_sync_step` — total samples expected in one weight-sync cycle |
| `total_generated_samples` | `int` | Cumulative rollout samples since training began |
| `expected_samples` | `int` | Theoretical samples needed up to current sync step |
| `buffer_samples` | `int` | Allowed buffer headroom (`expected × staleness_threshold`) |
| `step_wait_times` | `list[float]` | Per-collection wait times within latest step (seconds) |
| `step_wait_samples` | `list[int]` | Per-collection count of samples that actually had to be waited on (`max(0, required_samples - queue_size_at_collection_start)`), parallel to `step_wait_times`; used together to compute a generation-rate signal not skewed by samples served instantly from queue backlog |
| `only_hybrid` | `bool` | `True` when there are no standalone replicas |
| `last_activate_duration_s` | `float` | Duration of last activate cycle (weight sync + onload), seconds |
| `last_deactivate_duration_s` | `float` | Duration of last deactivate cycle (offload), seconds |

---

## 7. File Structure

```
verl/experimental/fully_async_policy/dynamic_schedule/
├── __init__.py                        # Public exports + policy registry
├── base.py                            # DynamicSchedulePolicyBase ABC, DynamicScheduleContext, registry
├── default_policy.py                  # DefaultDynamicSchedulePolicy (adaptive dynamic scheduling)
├── static_fully_async_policy.py       # StaticFullyAsyncPolicy (original fully-async / colocated fallback)
├── fixed_ratio_policy.py              # FixedRatioDynamicSchedulePolicy (fixed ratio, no adaptation)
└── dynamic_resource_controller.py     # DynamicResourceController (state machine + lifecycle)
```
