# Native vLLM-Ascend full-weight updates

Last updated: 11/03/2026.

These optional adapters delegate model loading to vLLM-Ascend. Leaving the
options below unset preserves verl's existing weight path, including CUDA/ROCm
IPC, shared-memory fallback and delta updates. The selector does not import
`torch_npu` or `vllm_ascend`. Selecting an Ascend backend on another device is
an error; it does not silently fall back to a different transport.

## Requirements and scope

Use a compatible pair of vLLM and vLLM-Ascend revisions. Verify Ascend's
`.github/vllm-main-verified.commit` for the intended release rather than combining
arbitrary main branches. This adapter requires the public native weight-transfer
lifecycle and the packed consumer with a `device_index` argument; it is not a
compatibility shim for older releases. Preflight checks callable signatures on
the trainer and workers before starting the stream.

The supported payload is a checkpoint-format, unquantized full-weight stream.
LoRA, quantized/QAT models, a separately synchronized MTP drafter, shared-memory
IPC fallback and `wire_format="delta_flush"` are rejected on this opt-in path.
Set `additional_config.weight_nz_mode: 0` as required by Ascend's native update
API. No model-specific adaptation or worker monkey patch is installed here.

## NPU IPC

```yaml
actor_rollout_ref:
  rollout:
    engine_kwargs:
      vllm:
        weight_transfer_config:
          backend: npu_ipc
        additional_config:
          weight_nz_mode: 0
```

The trainer packs a bounded byte buffer and exports a fresh IPC reference for
each nonempty message. The worker calls `init_weight_transfer_engine`,
`start_weight_update`, `update_weights` for each bucket, then
`finish_weight_update` once. The official consumer owns independent copies;
the trainer reuses storage only after acknowledgement and synchronization.
An empty final message exports no unused IPC reference.

IPC requires colocated trainer/rollout workers on the same physical NPU.
`checkpoint_engine.update_weights_bucket_megabytes` controls buffer capacity.
A single tensor larger than the bucket is sent directly, is not sliced, and
must already reside on the sender buffer device. It stays alive until the ACK.
Consequently memory use is bounded by the bucket plus the largest exported
tensor and the official loader's own retained storage, not by the bucket alone.

## Failure handling

Preflight errors are collected across trainer ranks before receivers wait for
data. A stream failure marks the adapter invalid; restart the rollout workers
before retrying rather than generating with partially updated weights.
IPC sends an explicit abort between requests and bounds socket operations at
300 seconds. An outstanding native collective or crashed process still depends
on runtime timeouts and the job supervisor; this is not in-place recovery.

## Validation

CPU tests use real PyTorch tensors and ZeroMQ, with substituted device/Ray
collaborators. They cover stream ownership, mixed dtypes, empty/oversized tensors,
failure/abort ordering, preflight and unchanged default dispatch. The existing
CPU workflow discovers the `tests/workers/rollout/test_*_on_cpu.py` files.
CPU dispatch tests do not establish real CUDA transfer correctness.

The implementation was exercised in a downstream Step5 environment on Ascend:
one complete training step per path and a fixed-checkpoint comparison against
the previous loading path. Those results are prior integration evidence, not an
end-to-end pass on this rebased upstream commit. Re-run with the intended model
and paired dependencies before merging. Do not infer convergence from one step
or compare independent training runs as though their final weights were equal.

## HCCL

Select `engine_kwargs.vllm.weight_transfer_config.backend: hccl` instead of
`npu_ipc`. This adapter requires `HCCLTrainerWeightTransferEngine.trainer_init`,
`HCCLTrainerInitInfo`, a client and a replayable source, plus `send_weights` and
`shutdown`; it does not use the former static-send API.

The initial supported topology is one node, replicated FSDP full-weight
streams, at least two TP replicas, and inference DP=PP=1, with no prefill/decode
disaggregation. A replica's trainer leader sends to the next replica so the
source physical NPU is not also a receiver in its HCCL group. Other ranks still
consume the exporter to participate in FSDP gathers. This does not implement
the delta-sharded checkpoint engine or a general multi-node M-to-N transport.

The trainer materializes one bounded batch on its actor thread and invokes the
official engine on a device-bound transport thread. Start is forwarded once,
per-batch finish calls are deferred, and finalization occurs only after the full
stream succeeds. Producer tensors are copied before advancing a potentially
reusing exporter. A single oversized tensor can exceed the batch limit.
Cancellation queues cleanup after the outstanding transport call; it cannot
interrupt a stuck HCCL collective. Persistent groups are reused between rounds.

## Whole-stream reload

```yaml
actor_rollout_ref:
  rollout:
    custom:
      ascend_weight_update_mode: reload
    engine_kwargs:
      vllm:
        additional_config:
          weight_nz_mode: 0
```

Leave `weight_transfer_config.backend` unset, or set it to `npu_ipc`. Combining
reload with HCCL/NCCL/other backends is rejected. Reload is a loading API, not a
third wire transport: this adapter uses packed NPU IPC to feed one iterator to
`worker.reload_weights(weights_iterator=..., is_checkpoint_format=True)`.
The official consumer copies each bucket before the producer reuses it. The
last ACK is withheld until reload has consumed the entire iterator and finished
postprocessing. An early return or iterator failure is not successful loading.

In an additional downstream control, the same trained W1 and four image
questions were evaluated in the same TP4 engine, twice through each loading
path. Scheduling was paused using public APIs until all four prompts were
enqueued. Runtime parameters, all 2436 generated token IDs per repetition and
their log probabilities matched exactly (maximum absolute log-probability
difference 0); all four means were 0.675. An earlier uncontrolled-scheduling
repeatability failure remains unresolved. This scoped result is not a broad
accuracy or multi-step convergence claim.
