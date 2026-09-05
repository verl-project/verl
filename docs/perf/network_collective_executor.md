# Opt-in network collective executor

Last updated: 09/05/2026.

`verl.utils.communication_network.NetworkCollectiveExecutor` consumes a
validated `NetworkPolicyEligibility`, exact target telemetry, an immutable
operation sequence, and a separately approved execution digest. It creates
operation-specific NCCL communicators and issues actual asynchronous
broadcast, all-reduce and equal-split all-to-all operations through them.

This is an explicit SPMD transfer API, **not an automatic integration into
Ray's NCCL checkpoint engine**. Ray collective group names and PyTorch process
groups are different runtimes. Existing trainer/checkpoint paths are unchanged.
Use it only at a quiescent, post-autograd transfer boundary from one host thread,
with all world ranks participating in the same control sequence.

## Required operator evidence

- An exact measured multi-node topology and rank-complete rail/class policy.
- A full-world Gloo control group, a supported PyTorch/NCCL build and a current
  CUDA device selected by the caller on every rank.
- Operator-installed rail-specific NCCL network plugins. Stock `IB`/`Socket`
  cannot be used to claim per-communicator rail pinning. Different logical
  rails cannot alias the same plugin.
- A numeric service level (0–15, InfiniBand) or traffic class (0–255, RoCE)
  for each logical lane. This library does not configure switches, QoS or NICs.
- A telemetry observer returning the actual `NetworkLane` of each initialized
  communicator. Echoing requested settings is not measured evidence.
- A reviewed `network_plan_digest(...)` binding every operation, rank, lane,
  message size, dtype, policy fingerprint and source evidence SHA-256.

Conflicting NCCL environment/configuration-file overrides are rejected. All
local preflight errors are exchanged before launching the next data collective.
The runtime rejects plan disagreement, different host grouping, wrong order,
message-size/dtype drift, partial steps and reuse after failure. Close drains
owned work and destroys only owned communicators; it never touches world or
external groups. Transport failure poisons the instance and requires job-level
recovery; it is not retried as a new collective.

## Calling sequence

```python
# All ranks construct with externally reviewed kwargs and a real observer.
executor = NetworkCollectiveExecutor(**reviewed_runtime_kwargs)
try:
    executor.begin_step(weight_version)
    for operation, detached_contiguous_tensor in ordered_transfer_buckets:
        handle = executor.launch(operation, detached_contiguous_tensor)
        received = handle.wait()  # stream-safe result, not a host completion claim
    executor.finish_step()        # physical completion before publishing the version
finally:
    executor.close()
```

The API fences the previous operation before switching communicators, following
the NCCL cross-process-group wait contract. It therefore makes **no cross-lane
overlap or speedup claim**. Finish drains CUDA events before releasing retained
buffers. Missing physical rails/plugins or telemetry is a real deployment
blocker; logical two/four-rank mock tests are not a substitute.

API references: [NCCL communicator configuration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html),
[NCCL communicator QoS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html),
[PyTorch distributed process groups](https://docs.pytorch.org/docs/stable/distributed.html).
