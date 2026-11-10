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
