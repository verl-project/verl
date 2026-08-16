# V1 Async Trainer

Last updated: 08/14/2026.

The V1 trainer provides two asynchronous PPO training modes under the standard `verl.trainer.main_ppo` entry point:

- `colocate_async` runs generation and training on the same GPU pool.
- `separate_async` runs generation continuously on standalone rollout GPUs and trains on a hybrid GPU pool. It can optionally lend idle hybrid trainer GPUs to generation.

Both modes use the V1 `TransferQueue`, asynchronous replay buffer, and partial rollout client. This guide explains their execution model, configuration, and tuning.

## Choose a Trainer Mode

Set the mode with `trainer.v1.trainer_mode`.


| Mode                         | Rollout resources                                                  | Typical use                                                                                   |
| ---------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `sync`                       | Hybrid rollout replicas colocated with the trainer                 | Baseline PPO and workloads where strict synchronization is preferred                          |
| `colocate_async`             | Hybrid rollout replicas colocated with the trainer                 | Use warmup batches + partial rollout to accelerate.                                           |
| `separate_async`             | Dedicated standalone rollout replicas plus hybrid trainer replicas | Separate resources without switch and offload costs. More compact and efficient rollout pool. |
| `separate_async` with switch | Same as `separate_async`                                           | Reduce trainer idle time when it's hard to set a perfect rollouter-trainer ratio.             |


`colocate_async` and `separate_async` both enable partial-rollout through `FullyAsyncLLMServerClient`. If generation is aborted during a mode transition, completed tokens are retained and the remaining generation is retried. A resumed trajectory can therefore span multiple model versions.

v1_trainer_timeline

## The V1 Training Loop

All trainer modes process the same number of PPO mini-batches per global step:

```text
mini-batches per PPO epoch
  = data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size
```

The modes differ in where this split happens. `sync` and `colocate_async` sample the full training batch in one controller round, then the actor worker divides it into PPO mini-batches. `separate_async` streams one PPO mini-batch through the controller at a time, overlapping rollout with mini-batch training.

The following invariant makes that value equal to the number of PPO mini-batches in `separate_async`:

```text
data.train_batch_size
  = trainer.v1.separate_async.parameter_sync_step
  × actor_rollout_ref.actor.ppo_mini_batch_size
```

For example, a train batch of 64 with a PPO mini-batch of 16 requires `parameter_sync_step=4`.

## Partial Rollout and Staleness

Partial rollout is part of the V1 async client behavior rather than a V1 configuration switch. When a request is aborted:

1. Tokens and log probabilities produced before the abort are retained.
2. The client retries the unfinished trajectory through the load balancer.
3. The resumed request run with a newer model version.
4. The KV cache is reconstructed for the retained prefix before decoding continues.

This avoids dropping long-running trajectories during rollout/trainer transitions, at the cost of extra prefill work and within-trajectory policy-version changes.

### Off-policy control

The V1 sampler controls staleness in model-version units:


| Parameter                                     | Default | Meaning                                                                                                             |
| --------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------- |
| `trainer.v1.sampler.max_off_policy_threshold` | `8`     | Maximum model versions from first generation to being trained before staleness handling is triggered                                         |
| `trainer.v1.sampler.max_off_policy_strategy`  | `drop`  | `drop` evicts stale prompt groups (for GRPO, one stale trajectory, the whole sample dropped); `wait` blocks for threshold-reaching in-flight groups instead of dropping them |


Monitor both forms of off-policy behavior:

- `training/off_policy/trajectory_spans/*`: number of model versions used within one trajectory. `1` means that the trajectory was generated entirely with one version.
- `training/off_policy/trajectory_staleness/*`: gap between the newest version used by a trajectory and the current training version.
- `training/off_policy/trajectory_staleness_worst/*`: gap between the oldest version used by a trajectory and the current training version.

## Separate Async Step Switching

Enable step switching with:

```bash
trainer.v1.separate_async.enable_switch=True
```

The switch addresses a specific idle window: after a PPO step finishes, the trainer may have to wait for standalone rollout to produce enough sampleable groups for the next step. During that window, the trainer's hybrid replicas can join the standalone load balancer and help generate samples.

![separate_async_switch_timeline](
https://github.com/Begunner/verl-link/blob/main/sepa_switch.svg?raw=true)

The upper timeline shows `separate_async` without switching; the lower timeline shows hybrid GPUs joining rollout during idle windows when switching is enabled.

### Lifecycle

```text
                              next step begins
                                      │
                                      ▼
                         Is enough data already sampleable?
                           │ yes                  │ no
                           ▼                      ▼
                    reclaim hybrid         keep hybrid in rollout
                           │                      │
                           │              submit this step's prompts
                           │                      │
                           │              wait until threshold
                           │                      │
                           └──────────────┬───────┘
                                          ▼
                             remove hybrid from balancer
                             abort unfinished requests
                             sleep hybrid replicas
                                          │
                                          ▼
                           parameter_sync_step PPO updates
                                          │
                                          ▼
                            sync standalone rollout weights
                                          │
                                          ▼
                      benefit > measured switching cost?
                           │ yes                  │ no
                           ▼                      ▼
                   lend hybrid again       remain in trainer mode
```

The reclaim order is important: hybrid replicas are removed from routing before their requests are aborted and their memory is returned to training. Aborted requests are then retried through the remaining standalone replicas.

The last training step never lends hybrid replicas back to generation.

### Reclaim threshold

The trainer converts `switch_threshold_ratio` into a number of sampleable groups:

```text
target = round(switch_threshold_ratio × train_batch_size)
threshold = clamp(target, one_mini_batch, train_batch_size)
```

The one-mini-batch floor ensures the trainer has useful work immediately after paying the reclaim cost. If the buffer is already at the threshold when a step begins, the trainer reclaims its hybrid replicas immediately. Otherwise, it keeps them in rollout mode until the threshold is reached.

### Adaptive threshold

With `adaptive_switch_threshold=True`, the threshold ratio reacts to observed sample wait:

- After `switch_threshold_release_steps` consecutive idle steps, increase the ratio by `switch_threshold_step_up`.
- After the same number of consecutive calm steps, decrease the ratio by `switch_threshold_step_down`.
- Clamp the ratio to `[1 / parameter_sync_step, 1]`.

The release interval applies in both directions and prevents one noisy step from changing the threshold.

### Benefit-versus-cost decision

At the end of a step, the trainer estimates whether lending hybrid replicas is worthwhile:

```text
remaining = threshold - sampleable_groups_for_next_step
scaling_factor = (hybrid_gpus + standalone_gpus) / standalone_gpus

benefit =
  remaining × observed_wait_per_sample × (1 - 1 / scaling_factor)

switch_cost =
  moving_average(switch_to_rollout) + moving_average(switch_to_trainer)
```

The trainer lends the hybrid replicas when `benefit > switch_cost`. During cold start, when either estimate is unavailable, it lends whenever the next-step buffer is below the threshold. The benefit model assumes near-linear generation scaling and can be optimistic when newly activated replicas have cold caches or the lending window is short.

### Switch configuration

All switch settings are inert unless `enable_switch=True`.


| Parameter                        | Default | Description                                                                          |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `enable_switch`                  | `false` | Allow hybrid trainer replicas to help rollout between steps                          |
| `switch_threshold_ratio`         | `0.3`   | Target sampleable fraction before hybrid replicas are reclaimed; must be in `(0, 1]` |
| `adaptive_switch_threshold`      | `true`  | Adapt the reclaim threshold from observed trainer idle time                          |
| `switch_threshold_step_up`       | `0.05`  | Ratio increase after sustained idle                                                  |
| `switch_threshold_step_down`     | `0.03`  | Ratio decrease after sustained calm                                                  |
| `switch_threshold_release_steps` | `2`     | Consecutive idle or calm steps required before adjustment                            |
| `switch_cost_window_size`        | `3`     | Number of recent transition costs used by the decision                               |


Step switching cannot be combined with rollout PD disaggregation. It also requires a replay buffer that implements `get_sampleable_count()` and `wait_for_sampleable()`, which the built-in `ReplayBufferAsync` provides.

## Configuration

### Colocated async

Add the following overrides to an existing V1 PPO launch:

```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=colocate_async \
trainer.v1.colocate_async.num_warmup_batches=1
```

The warmup batch starts generation before the first training step, reducing the initial empty-buffer wait.

### Separate async

The following example uses two trainer nodes and one standalone rollout node:

```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=separate_async \
trainer.nnodes=2 \
trainer.n_gpus_per_node=8 \
actor_rollout_ref.rollout.nnodes=1 \
actor_rollout_ref.rollout.n_gpus_per_node=8 \
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
data.train_batch_size=64 \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
trainer.v1.separate_async.parameter_sync_step=4 \
trainer.v1.separate_async.num_warmup_batches=1
```

`separate_async` requires a non-`naive` checkpoint-engine backend such as `nccl`, `nixl`, or `mooncake` for standalone rollout weight synchronization.

To enable step switching, add:

```bash
trainer.v1.separate_async.enable_switch=True \
trainer.v1.separate_async.adaptive_switch_threshold=True \
actor_rollout_ref.rollout.disaggregation.enabled=False
```

## Observability and Tuning

Start with the following timing metrics:

- `timing_s/gen`: trainer time spent waiting for the next trainable mini-batch.
- `timing_s/update_actor`: actor update time.
- `timing_s/update_weights`: standalone weight synchronization time in `separate_async`.
- `timing_s/switch_wait`: time during which lent hybrid replicas help fill the reclaim threshold.
- `timing_s/switch_to_rollout`: measured trainer-to-rollout transition time.
- `perf/throughput`: token throughput normalized by all trainer and standalone rollout GPUs in `separate_async`.

When switching is enabled, inspect:

- `separate_async/switch/threshold_ratio`
- `separate_async/switch/sample_wait_seconds`
- `separate_async/switch/idle`
- `separate_async/decision/sampleable_count`
- `separate_async/decision/remaining`
- `separate_async/decision/benefit_seconds`
- `separate_async/decision/effective_switch_cost_seconds`
- `separate_async/decision/switch_to_rollout`

Practical tuning order:

1. Balance trainer and standalone rollout resources before enabling switching.
2. Set `parameter_sync_step` from the required batch-size invariant.
3. Choose `max_off_policy_threshold` and `drop` or `wait` from the workload's policy-lag tolerance.
4. Enable switching when `timing_s/gen` shows sustained trainer idle time.
5. Keep adaptive thresholds enabled initially; use the decision metrics to determine whether the estimated benefit has a clear margin over switching cost.

The [RL-Insight guide](rl_insight.md) provides V1 rollout, TransferQueue, and resource-state dashboards. For large models where weight synchronization dominates, see [Delta Weight Sync](delta_weight_sync.md).

## Checkpoint and Validation Behavior

When the installed TransferQueue supports checkpointing, V1 async checkpoints persist its state alongside model and dataloader state. Finished trajectories are restored directly. Pending and running prompts are cleared and reissued after resume so prompts already fetched from the dataloader are not lost.

In `separate_async`, validation makes hybrid replicas available for rollout if they are currently in trainer mode. The next training step reclaims them before PPO updates when necessary.

## Benchmark

### Four-node mode comparison

> TODO: Add the four-node `sync` / `colocate_async` / `separate_async` comparison after the original run configuration and metrics are exported.

### Three-node step-switch comparison

The step switch was evaluated for the first 150 steps of two otherwise identical runs:

- 3 × 8 H100 GPUs: two trainer/hybrid nodes and one standalone rollout node.
- Qwen3.5-35B-A3B with Megatron training and vLLM rollout.
- `train_batch_size=64`, `ppo_mini_batch_size=16`, and `parameter_sync_step=4`.
- One run per setting; no seed sweep.

Enabling the switch reduced cumulative wall clock from **18.80 h to 16.43 h** (**12.6%**) and increased aggregate token throughput from **15,150 to 16,687 tokens/s** (**10.1%**). The runs differed by about 4% in mean response length, so aggregate tokens per wall-clock second is the more representative comparison.

Cumulative wall-clock comparison for separate_async with and without step switching

The result is specific to a rollout-constrained 2:1 hybrid-to-standalone resource ratio. Gains should shrink as standalone rollout capacity increases or as switch cost becomes a larger fraction of the available idle window.

## Relationship to Other Async Implementations


| Concept                 | V1 async trainer                                | Experimental `fully_async_policy`                       |
| ----------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| Entry point             | `verl.trainer.main_ppo`                         | `verl.experimental.fully_async_policy.fully_async_main` |
| Data exchange           | TransferQueue                                   | MessageQueue                                            |
| Async sampler           | `ReplayBufferAsync`                             | MessageQueue consumer and staleness controller          |
| Parameter sync interval | `trainer.v1.separate_async.parameter_sync_step` | `async_training.trigger_parameter_sync_step`            |
| Staleness control       | Model-version threshold with `drop` or `wait`   | Stale-sample production ratio                           |
| Partial rollout         | Built into the V1 async rollout client          | `async_training.partial_rollout`                        |
| Dynamic hybrid lending  | `trainer.v1.separate_async.enable_switch`       | `DynamicResourceController` policies                    |


The two hybrid-lending implementations solve a similar utilization problem but have different state machines and decision inputs. See [Dynamic Resource Scheduling for Fully-Async Training](dynamic_schedule.md) for the experimental implementation.