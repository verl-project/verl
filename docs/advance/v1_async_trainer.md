# V1 Async Trainer

Last updated: 08/20/2026.

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

![v1_trainer_timeline](
https://github.com/Begunner/verl-link/blob/main/v1_trainer.svg?raw=true)

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

### Prompt refilling in replay buffer and checkpoint recovery

Both async modes use `ReplayBufferAsync`, which samples prompt groups from the TransferQueue. The refill unit is one prompt group and evicting the group produces exactly one replacement prompt.

During sampling, the replay buffer repeatedly synchronizes TransferQueue metadata and classifies terminal groups. A training group is evicted when any of the following applies:

- Its prompt age exceeds `max_off_policy_threshold` while `max_off_policy_strategy=drop`.
- It is removed by the configured DAPO group filter.
- Its rollout finishes with `failure`.

The reasons are unioned before eviction, so a group matching multiple conditions is removed and refilled only once. After evicting `k` groups, the buffer invokes `refill_fn(k)`, which fetches exactly `k` new prompts from the training dataloader, records them as `pending` in the TransferQueue, and dispatches them to the AgentLoop:

Refill can repeat if replacement groups also fail or are filtered. This keeps the requested training batch size stable without training on groups rejected by the active sampling policy. Refill applies to the training partition; validation does not perform staleness, filtering, or failure refill.

With `max_off_policy_strategy=wait`, stale groups are not evicted or refilled. Instead, sampling blocks when an in-flight prompt reaches the threshold, allowing it to finish and remain trainable. DAPO-filtered and failed groups are still evicted and replaced normally.

When async trainer saves a checkpoint, pending and running groups have already consumed dataloader entries, so they are reissued from their saved prompts during `load_checkpoint` recovery. Finished groups and their completed trajectories are restored as-is and remain available for sampling; they are not regenerated.

### Separate Async training Granularity and Decoupled PPO

`separate_async` uses a mini-batch training granularity, trying to overlap rollout and training in the same step. When a mini-batch is samplable, it is trained at once. This advances the timing of `update_weights` and reduces staleness.

When `separate_async` recomputes old log probabilities, actor weights may change between controller-level mini-batches. To keep the same `pi_old` for the entire `parameter_sync_step` cycle, the first mini-batch copies `pi_old` to CPU and computes with the weights already on GPU. Before each later mini-batch, the trainer:

1. Copies the current updated weights to CPU.
2. Restores `pi_old` to GPU and computes old log probabilities.
3. Restores the current weights to GPU and clears their temporary CPU copy.

For `N` mini-batches, this results in `N` `save_model_to_cpu` calls and `2 * (N - 1)` `restore_model_from_cpu` calls. The old-policy weights themselves are saved once and restored `N - 1` times; the remaining calls preserve the current weights around old-log-probability computation. For example, four mini-batches make four save calls and six restore calls, of which one save and three restores transfer `pi_old`.

These transfers and old log prob computings are skipped when `algorithm.rollout_correction.bypass_mode=True`.


## Separate Async Step Switching (experimental)

Enable step switching with:

```bash
trainer.v1.separate_async.hybrid_rollout.enable_switch=True
```

The switch addresses a specific idle window: after a PPO step finishes, the trainer may have to wait for standalone rollout to produce enough sampleable groups for the next step. During that window, the trainer's hybrid replicas can join the standalone load balancer and help generate samples.

![separate_async_switch_timeline](
https://github.com/Begunner/verl-link/blob/main/sepa_switch.svg?raw=true)

The upper timeline shows `separate_async` without switching; the lower timeline shows hybrid GPUs joining rollout during idle windows when switching is enabled.

### Switch to trainer threshold

The trainer converts `switch_threshold_ratio` into a number of sampleable groups:

```text
target = round(switch_threshold_ratio × train_batch_size)
threshold = clamp(target, one_mini_batch, train_batch_size)
```

`switch_threshold_ratio` defines the target number of prompt groups ready for sampling before switching to trainer. At the end of a step, if the next step's buffer already meets the target or is expected to reach it soon without hybrid assistance (that is, the estimated benefit of lending does not exceed the measured switch cost), the hybrid replicas remain in trainer mode. If the buffer is below the target and switch is enabled, the hybrid replicas enter rollout mode and switch back to trainer mode once the target is reached. The one-mini-batch floor guarantees that at least one mini-batch is ready to train immediately.

### Adaptive threshold

Since the sample-length distribution can evolve throughout RL training, the resource balance required by rollout and training can shift over time. The switch threshold therefore adapts to observed trainer idle time instead of assuming a fixed optimal resource split.

With `adaptive_switch_threshold=True`, the threshold ratio reacts to observed sample wait:

- After `switch_threshold_release_steps` consecutive idle steps, increase the ratio by `switch_threshold_step_up`.
- After `switch_threshold_release_steps` consecutive non-idle steps, decrease the ratio by `switch_threshold_step_down`.

The release interval applies in both directions and prevents noisy steps from changing the threshold.

### Switch configuration

All settings under `trainer.v1.separate_async.hybrid_rollout` are ignored unless `enable_switch=True`.


| Parameter                        | Default | Description                                                                          |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------ |
| `enable_switch`                  | `false` | Allow hybrid trainer replicas to help rollout between steps                          |
| `switch_threshold_ratio`         | `0.4`   | Target sampleable fraction before hybrid replicas are reclaimed; must be in `(0, 1]` |
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
trainer.v1.separate_async.hybrid_rollout.enable_switch=True \
trainer.v1.separate_async.hybrid_rollout.adaptive_switch_threshold=True
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

### V1 trainer all modes

The three modes were compared over their first 150 steps with the same four-node budget. `sync` and `colocate_async` used all four nodes as hybrid resources, while `separate_async` used two hybrid trainer nodes and two standalone rollout nodes.

- Qwen3.5-35B-A3B with Megatron training (TP2 PP2 CP2 EP4) and vLLM rollout (TP4).
- `train_batch_size=64`, `ppo_mini_batch_size=16`, and `parameter_sync_step=4`.
- DAPO-Math-17k, max_prompt_length=2048, max_response_length=32768
- Decoupled PPO enabled


| Mode             | Resource split          | 150-step training time | Aggregate tokens/s | Mean response length |
| ---------------- | ----------------------- | ---------------------- | ------------------ | -------------------- |
| `sync`           | 4 hybrid                | 22.79 h                | 12,053             | 12,720               |
| `colocate_async` | 4 hybrid                | 14.72 h (-35.4%)       | 18,852 (+56.4%)    | 12,854               |
| `separate_async` | 2 hybrid + 2 standalone | 14.10 h (-38.1%)       | 18,829 (+56.2%)    | 12,288               |


![v1_modes_quality_step](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_quality_step.png?raw=true)

![v1_modes_timing_components](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_timing_components.png?raw=true)

![v1_modes_offpolicy](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_offpolicy.png?raw=true)

![v1_modes_policy_alignment](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_policy_alignment.png?raw=true)


### Separate_async switching

The step switch was evaluated for the first 100 steps of two otherwise identical runs:

- 3 × 8 H100 GPUs: two trainer/hybrid nodes and one standalone rollout node.
- Qwen3.5-35B-A3B with Megatron training (TP2 PP2 CP2 EP8) and vLLM rollout (TP4).
- `train_batch_size=64`, `ppo_mini_batch_size=16`, and `parameter_sync_step=4`.
- DAPO-Math-17k, max_prompt_length=2048, max_response_length=32768
- Decoupled PPO disabled


| Mode        | Resource split          | 100-step training time | Aggregate tokens/s | Mean response length | Mean reward |
| ----------- | ----------------------- | ---------------------- | ------------------ | -------------------- | ----------- |
| `no-switch` | 2 hybrid + 1 standalone | 13.15 h                | 14,604             | 13,343               | 0.7755      |
| `switch`    | 2 hybrid + 1 standalone | 11.79 h (-10.3%)       | 16,445 (+12.6%)    | 13,478               | 0.7762      |

![switch_quality_step](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_quality_step.png?raw=true)

![switch_timing_components](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_timing_components.png?raw=true)

![switch_offpolicy](
https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_offpolicy.png?raw=true)
