# V1 Async Trainer

Last updated: 10/14/2026.

The V1 trainer provides two asynchronous PPO training modes under the standard `verl.trainer.main_ppo` entry point:

- `colocate_async` runs generation and training on the same GPU pool.
- `separate_async` runs generation continuously on standalone rollout GPUs and trains on a hybrid GPU pool. The trainer GPUs run no inference engine by default; setting `trainer.v1.separate_async.hybrid_rollout.enable_switch=True` additionally colocates hybrid replicas on the training GPUs and lends them to generation between steps.

Both modes use the V1 `TransferQueue`, the streaming dataloader actor, and the partial rollout client. This guide explains their execution model, configuration, and tuning.

## Choose a Trainer Mode

Set the mode with `trainer.v1.trainer_mode`.


| Mode                         | Rollout resources                                                  | Typical use                                                                                   |
| ---------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `sync`                       | Hybrid rollout replicas colocated with the trainer                 | Baseline PPO and workloads where strict synchronization is preferred                          |
| `colocate_async`             | Hybrid rollout replicas colocated with the trainer                 | Use watermark refill + partial rollout to accelerate.                                         |
| `separate_async`             | Standalone rollout replicas only                                   | Trainer GPUs never run an inference engine; separate resources without switch and offload costs. |
| `separate_async` with switch | Standalone rollout replicas plus hybrid trainer replicas           | Reduce trainer idle time when it's hard to set a perfect rollouter-trainer ratio.             |


`colocate_async` and `separate_async` both enable partial-rollout through `FullyAsyncLLMServerClient`. If generation is aborted during a mode transition, completed tokens are retained and the remaining generation is retried. A resumed trajectory can therefore span multiple model versions.

![v1_trainer_timeline](https://github.com/Begunner/verl-link/blob/main/v1_trainer.svg?raw=true)

## The V1 Training Loop

All trainer modes process the same number of PPO mini-batches per global step:

```text
mini-batches per PPO epoch
  = data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size
```

The modes differ in where this split happens. `sync` and `colocate_async` sample the full training batch in one controller round, then the actor worker divides it into PPO mini-batches. `separate_async` streams one PPO mini-batch through the controller at a time, overlapping rollout with mini-batch training.

The following invariant makes `parameter_sync_step` equal to the number of PPO mini-batches in `separate_async`:

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
3. The resumed request may run with a newer model version.
4. The KV cache is reconstructed for the retained prefix before decoding continues.

This avoids dropping long-running trajectories during rollout/trainer transitions, at the cost of extra prefill work and within-trajectory policy-version changes.

### Off-policy control

The V1 sampler controls staleness in model-version units:


| Parameter                                     | Default | Meaning                                                                                                                                                                                                                                                                                    |
| --------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `trainer.v1.sampler.max_off_policy_threshold` | `8`     | Maximum model versions from prompt dispatch to being trained before staleness handling is triggered. `null` disables off-policy version control entirely: no samples are dropped or awaited for staleness                                                                                  |
| `trainer.v1.sampler.max_off_policy_strategy`  | `drop`  | `drop` evicts stale finished prompt groups as a whole; `wait` blocks for threshold-reaching in-flight groups instead of dropping them and requires a threshold of at least `2`, because watermark refill always keeps prompts in flight. Ignored when `max_off_policy_threshold` is `null` |


Monitor both forms of off-policy behavior:

- `training/off_policy/trajectory_spans/*`: number of model versions used within one trajectory. `1` means that the trajectory was generated entirely with one version.
- `training/off_policy/trajectory_staleness/*`: gap between the newest version used by a trajectory and the current training version.
- `training/off_policy/trajectory_staleness_worst/*`: gap between the oldest version used by a trajectory and the current training version.



### Prompt refilling and checkpoint recovery

TransferQueue is the replay storage, and `StreamingDataloaderAsync` owns its control plane: it registers submitted prompts, polls prompt/session metadata, evicts terminal groups, reissues in-flight prompts on restore, and checkpoints TransferQueue state. The key and tag protocol shared with AgentLoop workers and the trainer lives in `verl/trainer/ppo/v1/tq_schema.py`. Each coordinator pass runs one non-blocking `poll_once(global_steps, partition_id, target_count)` and, once a request is ready, `materialize_sampleable(...)`. 

On each pass the coordinator synchronizes TransferQueue metadata and classifies terminal groups. A training group is evicted when any of the following applies:

- Its prompt age exceeds `max_off_policy_threshold` while `max_off_policy_strategy=drop`.
- It is removed by the configured DAPO group filter.
- Its rollout finishes with `failure`.

The reasons are unioned before eviction, so a group matching multiple conditions is removed only once. Terminal-group eviction does not create refill credit: its sessions already released their in-flight slots as they completed.

The coordinator continuously dispatches new prompts as session slots become free. Session-watermark refill applies to the training partition; validation does not dispatch replacement prompts.

With `max_off_policy_strategy=wait`, stale groups are not evicted. Instead, sampling blocks when an in-flight prompt reaches the threshold, allowing it to finish and remain trainable. DAPO-filtered and failed terminal groups are still evicted normally.

With `max_off_policy_threshold=null` (set via `trainer.v1.sampler.max_off_policy_threshold=null`), off-policy version control is disabled entirely: no samples are dropped or awaited for staleness.

When async trainer saves a checkpoint, pending and running groups have already consumed dataloader entries, so they are reissued from their saved prompts during `load_checkpoint` recovery. Finished groups and their completed trajectories are restored as-is and remain available for sampling; they are not regenerated.

### StreamingDataloader actor

![v1_streaming_dataloader_flow](https://github.com/Begunner/verl-link/blob/main/v1_streaming_dataloader_flow.svg?raw=true)

All V1 trainer modes run rollout data iteration and replay sampling in a CPU Ray actor:

```yaml
trainer:
  v1:
    colocate_async:
      watermark_batches: 2
    separate_async:
      watermark_batches: 2
      max_outstanding_batches: 4
```

The actor owns the train `StatefulDataLoader`, the TransferQueue replay control plane, and `AgentLoopManagerTQ`. Every mode runs a single-concurrency asyncio coordinator; only the refill policy differs. `sync` uses `StreamingDataloader`, which dispatches prompts only while a sample request is pending (one batch, plus DAPO/failure replacements) and leaves nothing in flight once the request is fulfilled. Both async modes use its `StreamingDataloaderAsync` subclass, which keeps the session watermark in flight between requests.

Custom samplers (`trainer.v1.sampler.custom_sampler`) subclass `StreamingDataloader` for `sync` or `StreamingDataloaderAsync` for async modes. They override sampling hooks such as `_terminal_eviction_reasons` and `_select_prompt_uids` (async subclasses may also override `poll_once` and `materialize_sampleable`), keeping the coordinator-owned watermark contract.

Trainer code uses these actor methods:

- `sample` submits a request and blocks on its future until the coordinator can return enough prompt groups.
- `save_checkpoint` saves the dataloader cursor and, in async modes, a TransferQueue snapshot while the coordinator is paused.
- `load_checkpoint` restores both before the coordinator resumes normal operation.
- Separate Async only: `count_sampleable` and `wait_sampleable` report or wait for sampleable train groups without consuming them. `separate_async` uses them to decide when lent Hybrid GPUs return to training. Each is served by a regular coordinator pass, so it also applies that pass's evictions; `count_sampleable` leaves accrued eviction metrics for the next `sample` or `wait_sampleable`.

Model and optimizer checkpointing remain in the trainer. The latest-checkpoint marker is updated only after both the model save and the actor's dataloader/TransferQueue save succeed.
Sync checkpoints contain only `data.pt`. Async checkpoints additionally contain the TransferQueue snapshot and reissue pending/running prompts. Async restore requires `data.pt` to keep the dataloader cursor consistent with restored TransferQueue state.
The actor is hard-affined to the TaskRunner node so local datasets, `data.pt`, and the latest-checkpoint marker share the same filesystem view.
`skip.rollout_tq.enable` is rejected until submit/sample cache semantics are implemented inside the actor; silently bypassing those hooks would execute real rollout during a skipped step.

The async coordinator runs a pass on each `sample`, `count_sampleable`, `wait_sampleable`, or `load_checkpoint` call, and every `poll_interval`. A pass syncs TransferQueue metadata, evicts terminal groups, serves the pending request once enough groups are sampleable (otherwise the request waits for a later pass), and then refills prompts. Refill starts once `load_checkpoint` has restored train state, and a validation request pauses it until the next train request.

Async rollout concurrency is measured in unfinished session slots:

```text
pending_slots = pending_prompt_count * rollout.n
running_slots = sum(rollout.n - completed_entries for running prompts)
watermark_sessions = watermark_batches * train_batch_size * default_rollout.n
free_slots = max(0, watermark_sessions - pending_slots - running_slots)
outstanding_groups = pending + running + finished/failed groups awaiting sampling
admission = max(0, max_outstanding_batches * train_batch_size - outstanding_groups)  # separate_async only
num_prompts = min(floor(free_slots / rollout.n), admission)
```

Each session creates one metadata-only `{uid}_{session_id}` entry with `status=running`, then updates it after all trajectory writes settle; `finished` and `failure` session statuses both release one slot. Finished and failed prompt groups occupy no rollout slots, but in `separate_async` they still count toward `max_outstanding_batches` until they are sampled or evicted. Async trainers require a fixed train `rollout.n` and fetch one prompt per dataloader call (other `data.gen_batch_size` values are overridden to `1`), so refill dispatches whole prompt groups.

`colocate_async` has no outstanding cap: generation runs only while the trainer blocks in `sample`, so rollout can never outpace training. Groups left unconsumed after a sample are the ones that finished within the final poll interval, fewer than one watermark, so unconsumed groups won't grow endlessly.

In `separate_async` the two limits govern different regimes. Prompt staleness (steps from dispatch to training) is roughly the number of unconsumed groups divided by `train_batch_size`:

- When rollout is slower than training, groups are sampled as soon as they finish, the watermark binds, and staleness stays near `watermark_batches`.
- When rollout outpaces training, finished groups wait for the trainer. Without a cap they would accumulate indefinitely, growing staleness and TransferQueue memory and, under `drop`, wasting generation on groups that are later evicted. The cap holds the unconsumed groups near `max_outstanding_batches` batches, so staleness settles near that value while the trainer still always finds a full mini-batch.

Keep `max_outstanding_batches` at least `watermark_batches` (enforced) and preferably one batch above it, so groups waiting for a full batch do not reduce rollout concurrency. Keep it at most `max_off_policy_threshold - 1` so steady-state staleness stays within the off-policy threshold (a warning is logged otherwise). Set it to `null` to disable the cap.

Each coordinator pass synchronizes TransferQueue metadata once and reuses that snapshot for request handling and refill.

`watermark_batches` replaces the old warmup-batch setting: it defines the steady-state in-flight session target rather than a one-time startup submission.

Trajectory tensors remain in TransferQueue and do not cross the actor boundary. Validation keeps its dataloader in the trainer but sends each validation batch through the actor's `sample` method, so the actor remains the sole owner of `AgentLoopManagerTQ`.

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

`enable_switch` also *creates* the hybrid replicas: without it, no inference engine exists on the training GPUs at all, so warmup and validation generation are served by the standalone pool alone.

The switch addresses a specific idle window: after a PPO step finishes, the trainer may have to wait for standalone rollout to produce enough sampleable groups for the next step. During that window, the trainer's hybrid replicas can join the standalone load balancer and help generate samples.

![separate_async_switch_timeline](https://github.com/Begunner/verl-link/blob/main/sepa_switch.svg?raw=true)

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

With `adaptive_switch_threshold=True`, the threshold ratio reacts to observed sample wait. A step is idle when its total sample wait exceeds `trainer.v1.sampler.poll_interval` (default 2 s), the streaming dataloader's TransferQueue poll period:

- After `switch_threshold_release_steps` consecutive idle steps, increase the ratio by `switch_threshold_step_up`, capped at `1.0`.
- After `switch_threshold_release_steps` consecutive non-idle steps, decrease the ratio by `switch_threshold_step_down`, floored at `1 / parameter_sync_step`.

The release interval applies in both directions and prevents noisy steps from changing the threshold.

### Switch configuration

All settings under `trainer.v1.separate_async.hybrid_rollout` are ignored unless `enable_switch=True`, but `switch_threshold_ratio` and `switch_cost_window_size` are always validated.


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

## Separate Async without Hybrid Replicas (default)

`separate_async` creates no rollout replicas on the training GPUs unless `trainer.v1.separate_async.hybrid_rollout.enable_switch=True`:

- No inference engine is initialized on the training GPUs; rollout (including warmup and validation) is served exclusively by the standalone rollout pool.
- The mode-switch hooks (`switch_to_rollout` / `switch_to_trainer`) and load-balancer updates are no-ops.

Without hybrid replicas, warmup takes longer (standalone pool only) and validation runs on the standalone pool while it keeps producing training data. In exchange, every training step avoids the colocated engine's costs: no engine init, no startup weight sync, and no `param_offload` / `optimizer_offload` or GPU headroom reserved for replica wake-ups.

## Configuration

### Colocate async

Add the following overrides to an existing V1 PPO launch:

```bash
trainer.use_v1=True \
trainer.v1.trainer_mode=colocate_async \
trainer.v1.colocate_async.watermark_batches=2
```

Watermark refill starts generation before the first training step, reducing the initial empty-buffer wait. Fractional values are supported: the target is rounded to a whole number of sessions, and refill submits whole prompt groups only.

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
trainer.v1.separate_async.watermark_batches=2
```

`watermark_batches` also accepts fractional values (see [colocate async](#colocate-async)).

Hybrid replicas use `actor_rollout_ref.rollout.gpu_memory_utilization`. To give dedicated standalone rollout GPUs a different memory budget, set the existing `standalone_gpu_memory_utilization` override:

```bash
actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
actor_rollout_ref.rollout.standalone_gpu_memory_utilization=0.85
```

Choose both fractions for the memory available on their respective GPU pools. By default, `standalone_gpu_memory_utilization` follows `gpu_memory_utilization`. Setting it to `null` also falls back to `gpu_memory_utilization`. The override applies whether or not hybrid replicas are enabled, and does not change the trainer's or hybrid replicas' configuration.

`separate_async` requires a non-`naive` checkpoint-engine backend such as `nccl`, `nixl`, or `mooncake` for standalone rollout weight synchronization.

To enable step switching, add:

```bash
trainer.v1.separate_async.hybrid_rollout.enable_switch=True \
trainer.v1.separate_async.hybrid_rollout.adaptive_switch_threshold=True
```

## Observability and Tuning

Start with the following timing metrics:

- `timing_s/gen`: trainer time spent sampling trainable mini-batches, summed over the step. In `separate_async`, this is idle time for the training GPUs; enabling switching lets hybrid replicas generate during part of that wait. In `colocate_async`, it also includes aborting and sleeping rollout replicas.
- `timing_s/update_actor`: actor update time.
- `timing_s/update_weights`: weight synchronization time; in `separate_async` it covers the standalone rollout replicas.
- `timing_s/switch_wait`: time during which lent hybrid replicas help fill the switch-to-trainer threshold. (It is not idle.)
- `timing_s/switch_to_rollout`: trainer-to-rollout transition time, including load-balancer registration, sticky-cache clearing, hybrid weight update, and generation resume.
- `timing_s/switch_to_trainer`: rollout-to-trainer transition time, including load-balancer removal, request abort, and hybrid replica sleep.

When switching is enabled, inspect:

- `separate_async/switch/threshold_ratio`
- `separate_async/switch/wait_samples`
- `separate_async/switch/idle`
- `separate_async/decision/sampleable_count`
- `separate_async/decision/remaining`
- `separate_async/decision/per_sample_time_seconds`
- `separate_async/decision/effective_switch_cost_seconds`
- `separate_async/decision/should_switch_to_rollout`

Practical tuning order:

1. Balance trainer and standalone rollout resources before enabling switching.
2. Set `parameter_sync_step` from the required batch-size invariant.
3. Choose `max_off_policy_threshold` and `drop` or `wait` from the workload's policy-lag tolerance.
4. Enable switching when `timing_s/gen` shows sustained trainer idle time.

## Checkpoint and Validation Behavior

When the installed TransferQueue supports checkpointing, V1 async checkpoints persist its state alongside model and dataloader state. Finished samples are restored directly. Pending and running prompts are cleared and reissued after resume so prompts already fetched from the dataloader are not lost.

Validation shares the same AgentLoop and rollout server pool with unfinished training trajectories. Existing partial trajectories continue running, but the coordinator pauses train polling and refill from the first validation request until the next train request. `timing_s/testing` therefore still includes contention from pre-existing train trajectories without dispatching additional train prompts.

In `separate_async` with `enable_switch=True`, validation makes hybrid replicas available for rollout if they are currently in trainer mode. Without it there are no hybrid replicas and validation is served by the standalone pool.

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


![v1_modes_quality_step](https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_quality_step.png?raw=true)

![v1_modes_timing_components](https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_timing_components.png?raw=true)

![v1_modes_offpolicy](https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_offpolicy.png?raw=true)

![v1_modes_policy_alignment](https://github.com/Begunner/verl-link/blob/main/v1_trainer/v1_modes_policy_alignment.png?raw=true)

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


![switch_quality_step](https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_quality_step.png?raw=true)

![switch_timing_components](https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_timing_components.png?raw=true)

![switch_offpolicy](https://github.com/Begunner/verl-link/blob/main/v1_trainer/switch_offpolicy.png?raw=true)