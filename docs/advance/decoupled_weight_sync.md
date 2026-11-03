# Decoupled Weight Sync for the V1 Separate-Async Trainer

**Authors:** Luosuu

**Last updated:** 2026-11-03

## Background

In the v1 `separate_async` trainer, weight sync is driven by the trainer. At the end of every step,
`CheckpointEngineManager.update_weights` aborts in-flight requests on every rollout replica, releases
their KV cache, builds a transfer topology, pushes the weights, and resumes generation. The trainer
waits for all of it, and every replica pauses at the same moment.

Decoupled weight sync turns this around. The trainer only **publishes** a weight version. Each
rollout replica **pulls** new versions with one-sided reads, and decides on its own when to load
them. The trainer never pauses, aborts or waits for rollout.

## Architecture

```
trainer rank 0 --copy--> WeightPublisher (versioned host slots, registered once)
                              | publish(v): manifest {version, layout, descriptors, endpoint}
                              v
                       WeightRegistry  <-- latest_ready / lease --  WeightUpdateAgent (per replica)
                                                                        | prefetch, then apply
                                                                        v
                     CheckpointEngineWorker (transport_pull) -- one-sided READ --> host mirror --> vLLM
```

All components live in `verl/checkpoint_engine/transport/`:

| Component | Role |
| --- | --- |
| `registry.py` | Directory of published versions (`READY` → `RETIRED` → reclaimed) and reader leases. Leases expire on the registry's clock, so a crashed reader cannot pin a version. `RayWeightRegistry` runs it as a named Ray actor scoped to the job. |
| `publisher.py` | Keeps the last `num_slots` versions in pinned host buffers that are registered once. A slot is overwritten only after every lease on its version ends. fp32 weights are cast to `rollout_dtype` (bf16 by default) on the way in. |
| `puller.py` | Leases a version, reads every bucket into a local registered mirror, and rejects the copy if the lease lapsed mid-read (torn read). |
| `transport.py`, `nixl_native.py` | Slot layout and transports. `nixl_native` uses NIXL's own bindings and runs on any torch. Other names go to `torch.distributed._transport` (torch >= 2.15). Trainer and rollout must use the same transport. |
| `engine.py` | The `transport_pull` checkpoint engine. `send_weights` publishes, `prefetch` pulls ahead of time, `receive_weights` hands the pulled version to the inference engine. It also works under the classic `CheckpointEngineManager` flow. |
| `update_agent.py` | `WeightUpdateAgent`: one per vLLM replica; decides when the replica updates. |

## When a replica updates

Each update has two phases, and only the second pauses generation:

1. **Prefetch.** Once the newest READY version is at least `min_version_lag` ahead of the loaded one,
   every checkpoint engine worker of the replica pulls it into its host mirror while vLLM keeps
   generating.
2. **Apply.** Abort in-flight requests, release the KV cache, load the mirror, restore the KV cache
   and resume. `FullyAsyncLLMServerClient` resubmits aborted requests with their partial output, so
   they are re-prefilled under the new weights. A failure during apply leaves the replica paused
   rather than serving mixed weights.

`min_version_lag` should stay below `trainer.v1.sampler.max_off_policy_threshold`. Otherwise rollout
produces samples the trainer drops.

## Staleness

With decoupled sync, the step at which a prompt was dispatched no longer tells which weights generated
it. The replay buffer therefore measures a group's lag from the oldest weight version any of its
trajectories started on (`min_global_steps`, already recorded per trajectory). Version `v` is published
at the end of step `v`, so a group generated entirely by the newest weights has lag 1, the same as a
freshly dispatched group under the default clock.

## Usage

```bash
bash tests/special_e2e/run_v1_separate_async.sh \
    actor_rollout_ref.rollout.checkpoint_engine.backend=transport_pull \
    trainer.v1.separate_async.decoupled_weight_sync.enable=True
```

| Option | Default | Meaning |
| --- | --- | --- |
| `trainer.v1.separate_async.decoupled_weight_sync.enable` | `false` | Turn on decoupled sync. Requires `transport_pull`, vLLM rollout, and `hybrid_rollout.enable_switch=false`. |
| `...decoupled_weight_sync.min_version_lag` | `1` | Versions a replica may fall behind before it updates. |
| `...decoupled_weight_sync.poll_interval` | `0.5` | Seconds between registry polls per replica. |
| `...decoupled_weight_sync.init_timeout` | `600` | Seconds the trainer waits at startup for every replica to load the initial weights. |
| `checkpoint_engine.engine_kwargs.transport_pull.device` | `cpu` | Slot and mirror memory. Each slot holds a full model copy, which rarely fits on a trainer GPU. |
| `checkpoint_engine.engine_kwargs.transport_pull.transport` | `nixl_native` | `nixl_native`, or a `torch.distributed._transport` backend such as `nixl`. |
| `checkpoint_engine.engine_kwargs.transport_pull.num_slots` | `2` | Versions the publisher keeps at once. |
| `checkpoint_engine.engine_kwargs.transport_pull.rollout_dtype` | `bfloat16` | dtype fp32 weights are cast to before publishing; `null` disables the cast. |

## Results

GSM8K async GRPO, Qwen2.5-0.5B-Instruct, 4 training + 4 rollout H100s on one node, 30 steps,
`parameter_sync_step=4`, 32 prompts × 4 responses per step, one seed:

| Metric | Baseline (`nccl`, trainer-driven) | Decoupled |
| --- | --- | --- |
| Training score, steps 26–30 | 0.428 | 0.420 |
| Validation accuracy at step 30 | 0.477 | 0.456 |
| Step time (mean, steps 2–30) | 5.80 s | 4.81 s |
| Trainer time in `update_weights` | 1.26 s | 0.15 s |
| Throughput | 1106 tok/s/GPU | 1319 tok/s/GPU |
| Replica pause / prefetch | — | ≤ 1.0 s / 0.33 s |
| `off_policy/trajectory_staleness/mean` | 1.0 | 1.05–1.19 |
| `off_policy/trajectory_spans/mean` | 1.0 | 1.8–1.9 |

Learning curves and validation accuracy match within single-seed noise. Most decoupled trajectories
span two versions: requests dispatched right after a publish start on the old weights and are
re-prefilled when the replica applies the new version.

Same setup with Qwen3-8B-Base (FSDP over the 4 training GPUs, 2 rollout replicas at TP=2), 20 steps:

| Metric | Baseline (`nccl`, trainer-driven) | Decoupled |
| --- | --- | --- |
| Validation accuracy at steps 0 / 10 / 20 | 0.638 / 0.900 / 0.908 | 0.642 / 0.892 / 0.899 |
| Step time (mean, steps 2–20) | 30.2 s | 29.5 s |
| `old_log_prob` | 25.6 s | 26.1 s |
| Trainer time in `update_weights` | 1.14 s | 0.20 s |
| Replica pause / prefetch | — | ≤ 1.5 s / 4.2 s |
| `off_policy/trajectory_spans/mean` | 1.0 | 2.0 |

Accuracy matches. Step time barely moves because this setup is bound by `old_log_prob`: with
`rollout_correction.bypass_mode=False` and `parameter_sync_step=4`, every later mini-batch saves the
8B model to CPU and restores the old weights to recompute log-probs. The trainer still saves the
~0.9 s per step that weight sync took, and rollout keeps generating while replicas prefetch.

## Limitations and follow-ups

- **Prompt dispatch and wasted re-prefill.** The trainer still pushes prompts through
  `generate_sequences`, and prompts enqueued right after a publish start on the old weights. Proposed
  fix: the trainer only writes `pending` prompts, tagged with a minimum weight version, to
  TransferQueue. A rollout-side `PromptDispatcher` claims them up to an in-flight cap once replicas
  have loaded that version ([Luosuu/verl#1](https://github.com/Luosuu/verl/issues/1)).
- **`keep` pause mode** to resume long in-flight requests on their old-weight KV cache instead of
  re-prefilling them.
- **Full copies.** Every rollout rank pulls the whole model from trainer rank 0. Sharded pulls (each
  rank reads only its TP/EP slice from each trainer rank) and P2P relays come next.
- **Checkpoint resume** needs a run epoch in the version once the registry outlives the job, so a
  resumed trainer can republish older weights.
