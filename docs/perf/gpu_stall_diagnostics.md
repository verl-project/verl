# GPU stall diagnostics

This optional diagnostic is for investigating long periods of low GPU utilization
during PPO rollout. It is disabled by default and is deliberately independent
of `global_profiler.tool`, Nsight, PyTorch profiler, and RL Insight.

It does not change `enforce_eager`, CUDA Graph settings, fused-kernel settings,
SGLang server arguments, cache-release settings, or scheduling behavior. It is
an observability aid, not a claimed root-cause fix for [#3114](https://github.com/verl-project/verl/issues/3114).

## Enable it for a controlled run

    global_profiler:
      gpu_stall_diagnostics:
        enable: true
        sample_interval_s: 0.25
        zero_utilization_threshold: 1.0

The sample interval is validated in the range 0.2--0.5 seconds. No CUDA
context, Ray actor, sampler thread, NVML import, or diagnostics RPC occurs
while `enable: false`. Controllers execute only local disabled-state guards at
coarse phase boundaries; no rollout worker or per-token path is instrumented.

The built-in V1 `sync`, `colocate_async`, and `separate_async` trainers, as
well as the legacy `trainer.use_v1: false` controller, consume this setting.
The single controller creates one zero-GPU Ray actor pinned to each live Ray
node that advertises GPU resources. The actor imports `pynvml` lazily and
samples physical NVML devices from a daemon thread. Devices are identified by
PCI bus ID rather than by `CUDA_VISIBLE_DEVICES` ordinal, so the output remains
meaningful across multi-node jobs and differing local device masks.

The sampler is node-wide: it reports all physical GPUs visible to its node
actor. It does not infer Ray's per-process fractional allocation or attribute
a device to an individual co-tenant. Treat results from shared nodes as
node-level evidence, and correlate them with your resource-pool topology.

If NVML, its Python binding, or a node actor is unavailable, diagnostics logs a
warning and normal training continues. A transient per-device NVML sampling
failure is retried. After three consecutive failures, that physical device is
quarantined for the sampler lifetime and the actor logs the PCI bus ID and
failure reason. The feature never installs dependencies or falls back to
shelling out to a GPU utility.

The real-GPU qualification covered the sampler coordinator on one local Ray
node and GPU 0 with real NVML. V1 lifecycle wiring and hardening are covered
only by CPU tests. Multi-node node filtering, strict actor placement,
aggregation, RPC failure, and cleanup semantics are covered by CPU-only fake
Ray tests. No real multi-node GPU run or real V1 training run has been validated.

## Metrics

Metrics are emitted through the normal trainer tracking dictionary. The
logical node index is stable for one run because live Ray nodes are sorted by
NodeID before actors are created.

Typical keys are:

- `gpu_stall/training_step/node_0/utilization_mean`
- `gpu_stall/rollout_generate/node_1/gpu_0000_86_00_0/zero_utilization_fraction`
- `gpu_stall/rollout_wait/node_0/utilization_min`
- `gpu_stall/rollout_release_or_switch/node_0/sample_count`
- `gpu_stall/actor_update/node_0/sample_count`
- `gpu_stall/weight_sync_and_wake_or_resume/node_0/utilization_max`
- `gpu_stall/standalone_rollout_weight_sync/node_0/utilization_mean`

For every phase/node pair the diagnostic can emit `device_count`,
`sample_count`, `utilization_mean`, `utilization_min`, `utilization_max`, and
`zero_utilization_fraction`; per-device `sample_count`, mean, and zero-fraction
keys use the NVML PCI bus identifier. Empty windows are valid and emit a zero
sample count without a division-by-zero failure.

The legacy trainer records a full `training_step` window and nested,
non-contaminating windows for `rollout_generate`,
`rollout_sleep_or_release`, `actor_update`, and
`weight_sync_and_wake_or_resume`. In the current checkpoint-manager API,
weight synchronization and rollout wake/resume occur in one `update_weights()`
call, so the last label is intentionally combined rather than falsely
attributing samples to a separate wake stage.

V1 agent-loop generation is asynchronous: dispatch and server generation can
overlap controller work. V1 therefore records `rollout_wait` while the
controller waits for a generated trajectory from the replay buffer, and
`rollout_release_or_switch` while its mode releases replicas or switches the
hybrid engine. It records `actor_update` in the real actor update call. V1
`sync` and `colocate_async` use `weight_sync_and_wake_or_resume`; V1
`separate_async` instead uses `standalone_rollout_weight_sync`, because its
step-end operation synchronizes the always-running standalone rollout. V1 does
not currently emit a separate validation phase. These are diagnostic boundaries,
not conclusions about whether SGLang's cache lifecycle is responsible.

## Reading a result

A low legacy `rollout_generate` or V1 `rollout_wait` utilization window points
to a need for more request lifecycle evidence: queueing, Ray scheduling,
SGLang request admission, or server-side generation can all be inside that
caller-visible period. Low
utilization concentrated in `weight_sync_and_wake_or_resume` is a useful
signal to compare with SGLang release/resume and CUDA-Graph behavior, but does
not prove a verl-side defect.

Use this alongside existing profiler traces and RL Insight. The sampler adds
aggregate NVML evidence; it does not replace a timeline, request trace, or a
reproduction on current verl and SGLang versions.
