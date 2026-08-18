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


| Parameter                                     | Default | Meaning                                                                                                                                                                      |
| --------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trainer.v1.sampler.max_off_policy_threshold` | `8`     | Maximum model versions from first generation to being trained before staleness handling is triggered                                                                         |
| `trainer.v1.sampler.max_off_policy_strategy`  | `drop`  | `drop` evicts stale prompt groups (for GRPO, one stale trajectory, the whole sample dropped); `wait` blocks for threshold-reaching in-flight groups instead of dropping them |


Monitor both forms of off-policy behavior:

- `training/off_policy/trajectory_spans/*`: number of model versions used within one trajectory. `1` means that the trajectory was generated entirely with one version.
- `training/off_policy/trajectory_staleness/*`: gap between the newest version used by a trajectory and the current training version.
- `training/off_policy/trajectory_staleness_worst/*`: gap between the oldest version used by a trajectory and the current training version.

## Separate Async Step Switching (experimental)

Enable step switching with:

```bash
trainer.v1.separate_async.enable_switch=True
```

The switch addresses a specific idle window: after a PPO step finishes, the trainer may have to wait for standalone rollout to produce enough sampleable groups for the next step. During that window, the trainer's hybrid replicas can join the standalone load balancer and help generate samples.

separate_async_switch_timeline

The upper timeline shows `separate_async` without switching; the lower timeline shows hybrid GPUs joining rollout during idle windows when switching is enabled.

### Switch to trainer threshold

The trainer converts `switch_threshold_ratio` into a number of sampleable groups:

```text
target = round(switch_threshold_ratio × train_batch_size)
threshold = clamp(target, one_mini_batch, train_batch_size)
```

`switch_threshold_ratio` defines the target number of prompt groups ready for sampling before switching to trainer. At the end of a step, if the next step's buffer already meets the target or is expected to reach it soon without hybrid assistance (that is, the estimated benefit of lending does not exceed the measured switch cost), the hybrid replicas remain in trainer mode. If the buffer is below the target and switch is enabled, the hybrid replicas enter rollout mode and switch back to trainer mode once the target is reached. The one-mini-batch floor guarantees that at least one mini-batch is ready to train immediately.

### Adaptive threshold

With `adaptive_switch_threshold=True`, the threshold ratio reacts to observed sample wait:

- After `switch_threshold_release_steps` consecutive idle steps, increase the ratio by `switch_threshold_step_up`.
- After `switch_threshold_release_steps` consecutive non-idle steps, decrease the ratio by `switch_threshold_step_down`.

The release interval applies in both directions and prevents noisy steps from changing the threshold.

### Switch configuration

All switch-related settings are ignored unless `enable_switch=True`.


| Parameter                        | Default | Description                                                                          |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `enable_switch`                  | `false` | Allow hybrid trainer replicas to help rollout between steps                          |
| `switch_threshold_ratio`         | `0.3`   | Target sampleable fraction before hybrid replicas are reclaimed; must be in `(0, 1]` |
| `adaptive_switch_threshold`      | `true`  | Adapt the reclaim threshold from observed trainer idle time                          |
| `switch_threshold_step_up`       | `0.05`  | Ratio increase after sustained idle                                                  |
| `switch_threshold_step_down`     | `0.03`  | Ratio decrease after sustained calm                                                  |
| `switch_threshold_release_steps` | `2`     | Consecutive idle or calm steps required before adjustment                            |
| `switch_cost_window_size`        | `3`     | Number of recent transition costs used by the decision                               |


Temporarily, step switching cannot be combined with rollout PD disaggregation.

## Configuration

### Colocate async

Add the following overrides to an existing V1 PPO launch:

```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=colocate_async \
trainer.v1.colocate_async.num_warmup_batches=1
```

The warmup batch starts generation before the first training step, reducing the initial empty-buffer wait.

### Separate async

The following example uses two trainer nodes and two standalone rollout nodes:

```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=separate_async \
trainer.nnodes=2 \
trainer.n_gpus_per_node=8 \
actor_rollout_ref.rollout.nnodes=2 \
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
trainer.v1.separate_async.adaptive_switch_threshold=True
```

## Observability and Tuning

Start with the following timing metrics:

- `timing_s/gen`: trainer time spent waiting for the next trainable train-batch. (Also means idle time for separate_async's hybrid gpus)
- `timing_s/update_actor`: actor update time.
- `timing_s/update_weights`: standalone weight synchronization time in `separate_async`.
- `timing_s/switch_wait`: time during which lent hybrid replicas help fill the switch-to-trainer threshold. (It is not idle.)
- `timing_s/switch_to_rollout`: trainer-to-rollout transition time, including load-balancer registration, sticky-cache clearing, hybrid weight update, and generation resume.
- `timing_s/switch_to_trainer`: rollout-to-trainer transition time, including load-balancer removal, request abort, and hybrid replica sleep.

When switching is enabled, inspect:

- `separate_async/switch/threshold_ratio`
- `separate_async/switch/sample_wait_seconds`
- `separate_async/switch/idle`
- `separate_async/decision/sampleable_count`
- `separate_async/decision/remaining`
- `separate_async/decision/benefit_seconds`
- `separate_async/decision/effective_switch_cost_seconds`
- `separate_async/decision/should_switch_to_rollout`

Practical tuning order:

1. Balance trainer and standalone rollout resources before enabling switching.
2. Set `parameter_sync_step` from the required batch-size invariant.
3. Choose `max_off_policy_threshold` and `drop` or `wait` from the workload's policy-lag tolerance.
4. Enable switching when `timing_s/gen` shows sustained trainer idle time.

## Checkpoint and Validation Behavior

When the installed TransferQueue supports checkpointing, V1 async checkpoints persist its state alongside model and dataloader state. Finished samples are restored directly. Pending and running prompts are cleared and reissued after resume so prompts already fetched from the dataloader are not lost.

Validation shares the same AgentLoop and rollout server pool with unfinished training trajectories. Those partial trajectories continue running alongside validation requests, so `timing_s/testing` includes the contention and rollout capacity they consume rather than measuring validation generation in isolation.

In `separate_async`, validation makes hybrid replicas available for rollout if they are currently in trainer mode.

## Benchmark

### V1 Trainer all modes

The three modes were compared over their first 150 steps with the same four-node budget. `sync` and `colocate_async` used all four nodes as hybrid resources, while `separate_async` used two hybrid trainer nodes and two standalone rollout nodes.

- Qwen3.5-35B-A3B with Megatron training (TP2 PP2 CP2 EP4) and vLLM rollout (TP4).
- `train_batch_size=64`, `ppo_mini_batch_size=16`, and `parameter_sync_step=4`.
- DAPO-Math-17k, max_prompt_length=2048, max_response_length=32768


| Mode             | Resource split          | 150-step wall clock | Aggregate tokens/s | Mean response length |
| ---------------- | ----------------------- | ------------------- | ------------------ | -------------------- |
| `sync`           | 4 hybrid                | 22.79 h             | 12,053             | 12,720               |
| `colocate_async` | 4 hybrid                | 13.31 h             | 17,813             | 10,956               |
| `separate_async` | 2 hybrid + 2 standalone | 12.75 h             | 22,125             | 13,066               |


Compared with `sync`, `colocate_async` reduced wall clock by **41.6%** and increased aggregate token throughput by **47.8%**; `separate_async` reduced wall clock by **44.1%** and increased throughput by **83.6%**. The mean trainer wait for samples (`timing_s/gen`) fell from **380.8 s** in `sync` to **156.7 s** in `colocate_async` and **43.2 s** in `separate_async`.

`sync` and `colocate_async` recomputed old log probabilities in a similar **31.8 s** and **31.2 s** per step, while `separate_async` reused rollout log probabilities and spent only **0.3 s**. As a diagnostic scheduling comparison, subtracting `timing_s/old_log_prob` gives adjusted 150-step times of **21.47 h**, **12.01 h**, and **12.74 h**, respectively; these adjusted values are not measured end-to-end runtimes.

`separate_async` completed 150 steps 4.2% faster than `colocate_async` while processing 24.2% more tokens per second, but its mean response length was 19.3% higher. Report both time-to-step and aggregate token throughput rather than attributing the throughput difference entirely to scheduling. Each mode was measured with one run and no seed sweep.

### Separate_async switching

The step switch was evaluated for the first 150 steps of two otherwise identical runs:

- 3 × 8 H100 GPUs: two trainer/hybrid nodes and one standalone rollout node.
- Qwen3.5-35B-A3B with Megatron training (TP2 PP2 CP2 EP8) and vLLM rollout (TP4).
- `train_batch_size=64`, `ppo_mini_batch_size=16`, and `parameter_sync_step=4`.
- DAPO-Math-17k, max_prompt_length=2048, max_response_length=32768


| Mode        | Resource split          | 150-step wall clock | Aggregate tokens/s | Mean response length |
| ----------- | ----------------------- | ------------------- | ------------------ | -------------------- |
| `no-switch` | 2 hybrid + 1 standalone | 18.80 h             | 15,150             | 12,720               |
| `switch`    | 2 hybrid + 1 standalone | 16.43 h             | 16,687             | 10,956               |


Cumulative training-step-time comparison for `separate_async` with and without step switching

The no-switch baseline spent a mean **167.1 s** per step in `timing_s/gen`, or **37.0%** of its mean **451.2 s** step time. During this interval, the hybrid trainer GPUs were idle while waiting for the standalone rollout pool to fill the training buffer, showing that the tested 2:1 hybrid-to-standalone allocation was rollout-constrained rather than perfectly balanced.

A perfect static resource split is generally difficult to maintain because response lengths and rollout latency change throughout RL training. Step switching can therefore be enabled when no single allocation is expected to remain balanced: hybrid GPUs are lent only when the buffer is short and the estimated benefit exceeds the measured round-trip switch cost; otherwise, they remain in trainer mode. Gains should shrink as standalone rollout capacity increases or as switch cost becomes a larger fraction of the available idle window.

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
