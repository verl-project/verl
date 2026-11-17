# Decoupled Async Trainer: Rollout Pulls Weights and Prompts

**Authors:** Luosuu

**Last updated:** 2026-11-03

## Background

In the v1 `separate_async` trainer, the trainer drives rollout. At the end of every step,
`CheckpointEngineManager.update_weights` aborts in-flight requests on every rollout replica, releases
their KV cache, builds a transfer topology, pushes the weights, and resumes generation. The trainer
waits for all of it, and every replica pauses at the same moment. Prompts are pushed to the agent loop
workers by the trainer as well.

The `decoupled_async` trainer mode turns this around. The trainer only **publishes** weight versions
and **enqueues** prompts in TransferQueue. Each rollout replica **pulls** new versions with one-sided
reads and decides on its own when to load them, and a rollout-side dispatcher starts enqueued prompts.
The trainer never pauses, aborts or waits for rollout.

## Architecture

```
trainer rank 0 --copy--> WeightPublisher (versioned host slots, registered once)
                              | publish(v): manifest {version, layout, descriptors, endpoint}
                              v
                       WeightRegistry  <-- latest_ready / lease --  WeightUpdateAgent (per replica)
                                                                        | prefetch, then apply
                                                                        v
                     CheckpointEngineWorker (transport_pull) -- one-sided READ --> host mirror --> vLLM

trainer --enqueue pending prompts (min_weight_version)--> TransferQueue
                                                              ^
                       PromptDispatcher -- claim when every replica has the version --> agent loop workers
```

| Component | Role |
| --- | --- |
| `checkpoint_engine/transport/registry.py` | Directory of published versions (`READY` → `RETIRED` → reclaimed) and reader leases. Leases expire on the registry's clock, so a crashed reader cannot pin a version. `RayWeightRegistry` runs it as a named Ray actor scoped to the job. |
| `checkpoint_engine/transport/publisher.py` | Keeps the last `num_slots` versions in pinned host buffers that are registered once. A slot is overwritten only after every lease on its version ends. fp32 weights are cast to `rollout_dtype` (bf16 by default) on the way in. |
| `checkpoint_engine/transport/puller.py` | Leases a version, reads every bucket into a local registered mirror, and rejects the copy if the lease lapsed mid-read (torn read). |
| `checkpoint_engine/transport/transport.py`, `nixl_native.py` | Slot layout and transports. `nixl_native` uses NIXL's own bindings and runs on any torch. Other names go to `torch.distributed._transport` (torch >= 2.15). Trainer and rollout must use the same transport. |
| `checkpoint_engine/transport/engine.py` | The `transport_pull` checkpoint engine. `send_weights` publishes, `prefetch` pulls ahead of time, `receive_weights` hands the pulled version to the inference engine. It also works under the classic `CheckpointEngineManager` flow. |
| `checkpoint_engine/transport/update_agent.py` | `WeightUpdateAgent`: one per vLLM replica; decides when the replica updates. |
| `trainer/ppo/v1/prompt_dispatcher.py` | `PromptDispatcher`: claims enqueued prompts and starts them on the agent loop workers. |
| `trainer/ppo/v1/trainer_decoupled_async.py` | `PPOTrainerDecoupledAsync`: publishes, enqueues, samples and trains. |

## When a replica updates

Each update has two phases, and only the second pauses generation:

1. **Prefetch.** Once the newest READY version is at least `min_version_lag` ahead of the loaded one,
   every checkpoint engine worker of the replica pulls it into its host mirror while vLLM keeps
   generating.
2. **Apply.** Abort in-flight requests, release the KV cache, load the mirror, restore the KV cache
   and resume. `FullyAsyncLLMServerClient` resubmits aborted requests with their partial output, so
   they are re-prefilled under the new weights. A failure during apply leaves the replica paused
   rather than serving mixed weights, and the trainer fails the job.

`min_version_lag` must stay below `trainer.v1.sampler.max_off_policy_threshold`. Otherwise rollout
produces samples the trainer drops.

## Prompt dispatch

The trainer enqueues prompts exactly where it used to dispatch them (warmup, once per step, and the
replay buffer's refills), but only writes them to TransferQueue as `pending`. Each prompt carries a
`min_weight_version`: the newest published version, minus `min_version_lag - 1`.

One `PromptDispatcher` per rollout service polls TransferQueue and claims pending prompts, oldest first
and up to `max_inflight_prompts`, once every replica has loaded their `min_weight_version`. Prompts
enqueued right after a publish therefore wait for the update instead of starting on the old weights and
being re-prefilled. There is a single dispatcher because TransferQueue tags are last-writer-wins, so
independent claimers could start the same prompt twice. On resume, the trainer only resets interrupted
prompts to `pending`; the dispatcher starts them again.

## Staleness

The step at which a prompt was enqueued no longer tells which weights generated it. The replay buffer
therefore measures a group's lag from the oldest weight version any of its trajectories started on
(`min_global_steps`, already recorded per trajectory). Version `v` is published at the end of step `v`,
so a group generated entirely by the newest weights has lag 1, the same as a freshly dispatched group
under the default clock.

## Usage

```bash
bash tests/special_e2e/run_v1_separate_async.sh \
    trainer.v1.trainer_mode=decoupled_async \
    actor_rollout_ref.rollout.checkpoint_engine.backend=transport_pull \
    algorithm.rollout_correction.bypass_mode=True
```

`bypass_mode=True` uses rollout's log-probs as the old policy. Without it, and with
`parameter_sync_step > 1`, the trainer saves and restores the full model on CPU for every later
mini-batch to recompute them, which dominates the step for large models.

| Option | Default | Meaning |
| --- | --- | --- |
| `trainer.v1.decoupled_async.num_warmup_batches` | `1` | Batches enqueued before the training loop starts. |
| `trainer.v1.decoupled_async.parameter_sync_step` | `4` | Local updates per published weight version. |
| `trainer.v1.decoupled_async.min_version_lag` | `1` | Versions a replica may fall behind before it updates. Keep `1`: prompts are admitted only on the newest weights, so a lazier replica makes them wait for its whole update. |
| `trainer.v1.decoupled_async.poll_interval` | `0.5` | Seconds between registry polls per replica. |
| `trainer.v1.decoupled_async.init_timeout` | `600` | Seconds the trainer waits at startup for every replica to load the initial weights. |
| `trainer.v1.decoupled_async.max_inflight_prompts` | `null` | Prompt groups generating at once across all replicas; `null` for no limit. |
| `trainer.v1.decoupled_async.dispatch_poll_interval` | `0.2` | Seconds between TransferQueue polls of the dispatcher. |
| `checkpoint_engine.engine_kwargs.transport_pull.device` | `cpu` | Slot and mirror memory. Each slot holds a full model copy, which rarely fits on a trainer GPU. |
| `checkpoint_engine.engine_kwargs.transport_pull.transport` | `nixl_native` | `nixl_native`, or a `torch.distributed._transport` backend such as `nixl`. |
| `checkpoint_engine.engine_kwargs.transport_pull.num_slots` | `2` | Versions the publisher keeps at once. |
| `checkpoint_engine.engine_kwargs.transport_pull.rollout_dtype` | `bfloat16` | dtype fp32 weights are cast to before publishing; `null` disables the cast. |

## Results

### What is decoupled

| | `separate_async` | `decoupled_async` |
| --- | --- | --- |
| Trainer calls that control rollout (abort, sleep, KV release/resume, weight push, load balancer add/remove, mode switch, `generate_sequences`) | 24 | 0 |
| Trainer mode size | 371 lines, plus hybrid switching state | 221 lines; shared standalone pieces in a 108-line mixin |
| What the trainer knows about rollout | Replica handles, load balancer, pause/resume protocol, KV memory lifecycle | Publish a version, enqueue prompts, sample from TransferQueue |

`decoupled_async` still starts the update agents and the dispatcher, because rollout runs in the same
Ray job, but after that it only checks their health.

### Correctness

All runs: GSM8K async GRPO, 4 training + 4 rollout H100s on one node, `parameter_sync_step=4`,
32 prompts × 4 responses per step, one seed.

| Run | Validation accuracy, baseline | Validation accuracy, `decoupled_async` | Weight versions per trajectory |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct, 30 steps, `bypass_mode=False` (steps 10 / 20 / 30) | 0.079 / 0.438 / 0.477 | 0.327 / 0.456 / 0.501 | 1.0 |
| Qwen3-8B-Base, 20 steps, `bypass_mode=True` (steps 0 / 10 / 20) | 0.638 / 0.870 / 0.896 | 0.644 / 0.860 / 0.887 | 1.0 |

Accuracy matches the `separate_async` + `nccl` baseline within single-seed noise, and every trajectory
is generated by a single weight version. Before the prompt dispatcher existed (weights pulled, prompts
still pushed by the trainer), 1.8–2.0 versions per trajectory were typical: prompts dispatched right
after a publish started on the old weights and were re-prefilled when the replica updated.

### Performance (not yet optimized)

| | Baseline | `decoupled_async` |
| --- | --- | --- |
| 0.5B: step time / trainer time in `update_weights` | 5.80 s / 1.26 s | 4.74 s / 0.14 s |
| 0.5B: replica prefetch / pause | — | 0.33 s / ≤ 0.93 s |
| 8B: step time / trainer time in `update_weights` | 4.78 s / 1.35 s | 6.08 s / 0.23 s |
| 8B: trainer waiting for samples (`timing_s/gen`) | 0.05 s | 2.47 s |
| 8B: replica prefetch / pause | — | 4.8 s / 1.5 s |

At 0.5B a replica update (~1.3 s) is much shorter than a step and decoupling saves the trainer's sync
time. At 8B it is slower: a replica update (~6.3 s) takes longer than a training step (~4.8 s), and
because the dispatcher admits prompts only once every replica has the newest version, new prompts wait
and the trainer runs short of samples. The baseline's NCCL broadcast moves the 16 GB in 1.35 s, while
each rollout rank here pulls the full model from one publisher through host memory. With
`min_version_lag=2` the replicas update only every other version, so the waits alternate between none
and a whole update (5.60 s per step); the dispatcher should keep admitting on the newest weights,
otherwise admitted prompts are aborted by the next update anyway. The fix is a faster transfer (sharded
pulls, GPU-direct publishing), not looser gating.

## Next: rollout outside the trainer's Ray job

`decoupled_async` removes the trainer's calls into rollout, but everything still runs in one Ray job: the
trainer launches the vLLM replicas, checkpoint engine workers, agent loop workers, update agents and
the dispatcher, and both sides find the weight registry and TransferQueue as named Ray actors. The next
step is a rollout service with its own lifecycle, possibly its own Ray cluster, that meets the trainer
only at the weight registry and TransferQueue.

What already works across that boundary:

- **Weight transfer.** One-sided reads need only the endpoint URL and serialized descriptors in the
  manifest; NIXL/UCX does not depend on Ray. Trainer and rollout nodes must be reachable over the
  network (RDMA or TCP).
- **TransferQueue data path.** In TransferQueue 0.1.11 a client (`TransferQueueClient` plus its storage
  manager) only *connects* ZMQ DEALER sockets to the controller and storage units, using addresses in
  the TransferQueue config (`controller.zmq_info`, `backend.SimpleStorage.zmq_info`). The
  `SimpleStorage` and `MooncakeStore` backends are reachable without Ray; the `RayStorage` backend is
  not.
- **Control.** The trainer never calls rollout; it only publishes weights and enqueues prompts.

What is missing:

| Gap | Today | Needed |
| --- | --- | --- |
| Weight registry | Named Ray actor (`RayWeightRegistry`) | An etcd or Redis implementation of `WeightRegistry`; leases already expire on the registry's clock |
| TransferQueue discovery | `tq.init()` finds the controller with `ray.get_actor("TransferQueueController")` and reads its config | Publish the TransferQueue config (controller and storage ZMQ addresses) through the registry store, and a public connect-from-config API in TransferQueue; today `interface._maybe_create_tq_client(conf)` does this but is private |
| TransferQueue outside Ray | `ray.get_runtime_context()` in the storage manager's thread pool sizing starts a local Ray instance in a process without Ray | Guard it with `ray.is_initialized()` (TransferQueue change) |
| Rollout launcher | The trainer creates replicas, agents, agent loop workers and the dispatcher | A rollout-service entry point that creates them, with reward workers if rewards are computed in the agent loop |
| Version identity | Versions are trainer steps; the registry dies with the job | A run epoch in each version, so a resumed trainer can republish older weights to a registry that outlived it |
| Lifecycle | TransferQueue's controller and storage live in the trainer's Ray job | The rollout service tolerates trainer restarts (reconnect), or TransferQueue runs as its own service |
| Security | ZMQ and NIXL endpoints are unauthenticated | Network isolation between the two clusters, or authenticated transports |

Inside the rollout service, Ray remains a convenient way to place each vLLM server and its checkpoint
engine worker on the same GPU and to call between agent, server and worker; it would simply be the
service's own cluster, sized independently of training.

### Reusing an existing vLLM deployment

Today each replica is launched by verl: a `vLLMHttpServer` Ray actor with verl's worker extension, and a
checkpoint engine worker per GPU that hands pulled weights to vLLM over CUDA IPC. vLLM 0.32 already has
what a standard `vllm serve` deployment would need instead:

- HTTP RL endpoints: `/pause`, `/resume`, `/abort_requests`, `/init_weight_transfer_engine`,
  `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/update_weight_version`.
- Pluggable weight transfer (`WeightTransferEngineFactory.register_engine`, selected at startup with
  `weight_transfer_config`), which verl already uses for delta sync.
- Per-token weight version spans in request outputs.

With a pull engine running inside the vLLM workers (registered as a plugin; also where sharded pulls
belong), an HTTP update agent, and agent loops calling vLLM over HTTP, rollout becomes a standard vLLM
deployment plus a pip-installed plugin, and `/pause`'s `keep` mode comes for free. The engine is chosen
at startup, so the deployment must be launched with the plugin; it cannot be attached to a vLLM that
is already running without it.

## Limitations and follow-ups

- **Weight transfer speed.** Full copies through host memory are slower than the NCCL path at 8B; see
  Performance.
- **`keep` pause mode** to resume long in-flight requests on their old-weight KV cache instead of
  re-prefilling them.
- **Version-aware routing.** The dispatcher waits until every replica has a prompt's version; routing
  to any replica that is already new enough would start prompts sooner.
- **Full copies.** Every rollout rank pulls the whole model from trainer rank 0. Sharded pulls (each
  rank reads only its TP/EP slice from each trainer rank) and P2P relays come next.
- **Checkpoint resume** needs a run epoch in the version once the registry outlives the job, so a
  resumed trainer can republish older weights.
- **Agent loop worker failures.** Prompts claimed by a worker that dies stay `running`; the dispatcher
  does not reclaim them yet.
