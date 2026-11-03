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
    actor_rollout_ref.rollout.checkpoint_engine.backend=transport_pull
```

| Option | Default | Meaning |
| --- | --- | --- |
| `trainer.v1.decoupled_async.num_warmup_batches` | `1` | Batches enqueued before the training loop starts. |
| `trainer.v1.decoupled_async.parameter_sync_step` | `4` | Local updates per published weight version. |
| `trainer.v1.decoupled_async.min_version_lag` | `1` | Versions a replica may fall behind before it updates. |
| `trainer.v1.decoupled_async.poll_interval` | `0.5` | Seconds between registry polls per replica. |
| `trainer.v1.decoupled_async.init_timeout` | `600` | Seconds the trainer waits at startup for every replica to load the initial weights. |
| `trainer.v1.decoupled_async.max_inflight_prompts` | `null` | Prompt groups generating at once across all replicas; `null` for no limit. |
| `trainer.v1.decoupled_async.dispatch_poll_interval` | `0.2` | Seconds between TransferQueue polls of the dispatcher. |
| `checkpoint_engine.engine_kwargs.transport_pull.device` | `cpu` | Slot and mirror memory. Each slot holds a full model copy, which rarely fits on a trainer GPU. |
| `checkpoint_engine.engine_kwargs.transport_pull.transport` | `nixl_native` | `nixl_native`, or a `torch.distributed._transport` backend such as `nixl`. |
| `checkpoint_engine.engine_kwargs.transport_pull.num_slots` | `2` | Versions the publisher keeps at once. |
| `checkpoint_engine.engine_kwargs.transport_pull.rollout_dtype` | `bfloat16` | dtype fp32 weights are cast to before publishing; `null` disables the cast. |

## Results

GSM8K async GRPO, Qwen2.5-0.5B-Instruct, 4 training + 4 rollout H100s on one node, 30 steps,
`parameter_sync_step=4`, 32 prompts × 4 responses per step, one seed. The middle column is an earlier
revision with the same weight path (publisher, puller, update agent) but prompts still pushed by the
trainer; the last column is `decoupled_async` with the prompt dispatcher.

| Metric | `separate_async` + `nccl` | Pulled weights, pushed prompts | `decoupled_async` |
| --- | --- | --- | --- |
| Training score, steps 26–30 | 0.428 | 0.420 | 0.483 |
| Validation accuracy at steps 10 / 20 / 30 | 0.079 / 0.438 / 0.477 | 0.190 / 0.459 / 0.456 | 0.327 / 0.456 / 0.501 |
| Step time (mean, steps 2–30) | 5.80 s | 4.81 s | 4.74 s |
| Trainer time in `update_weights` | 1.26 s | 0.15 s | 0.14 s |
| Throughput over all 8 GPUs | 1106 tok/s/GPU | 1319 tok/s/GPU | 1273 tok/s/GPU |
| Replica pause / prefetch | — | ≤ 1.0 s / 0.33 s | ≤ 0.93 s / 0.33 s |
| `off_policy/trajectory_staleness/mean` | 1.0 | 1.05–1.19 | 1.0 |
| `off_policy/trajectory_spans/mean` | 1.0 | 1.8–1.9 | 1.0 |

Learning curves and validation accuracy match within single-seed noise. With prompts pushed by the
trainer, most trajectories spanned two versions: prompts dispatched right after a publish started on
the old weights and were re-prefilled when the replica applied the new version. With the dispatcher
gating each prompt on its `min_weight_version`, every trajectory is generated by a single version and
staleness is back to 1.0. (`decoupled_async` logged 2563 tok/s/GPU because it divided by the 4 training
GPUs only; that is fixed, and the table shows the value over all 8 GPUs.)

Same setup with Qwen3-8B-Base (FSDP over the 4 training GPUs, 2 rollout replicas at TP=2), 20 steps:

| Metric | Baseline (`separate_async`, `nccl`) | Decoupled weight sync |
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

## Limitations and follow-ups

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
