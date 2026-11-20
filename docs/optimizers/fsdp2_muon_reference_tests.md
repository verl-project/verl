# What the Megatron Muon reference tests establish

Last updated: 11/09/2026

This is a source review of [verl PR #7120](https://github.com/verl-project/verl/pull/7120),
not a record of rerunning its tests. The final PR head is
`d1e0dfea94c75ca622b42d8d9f15d64f8a500530`; the PR is merged. All code links below
are pinned to that head, rather than an earlier review revision or current main.

In [issue #8549](https://github.com/verl-project/verl/issues/8549), a maintainer
[confirmed that native FSDP2 lacked Muon support](https://github.com/verl-project/verl/issues/8549#issuecomment-6647539187)
and [welcomed a contribution](https://github.com/verl-project/verl/issues/8549#issuecomment-6648644246).
Support in Megatron or VeOmni does not establish support in native FSDP2.

## Exact scope of the five reference test files

| Final-head test | Assertions in the source | Boundary of the evidence |
| --- | --- | --- |
| [test_optim_config_on_cpu.py](https://github.com/verl-project/verl/blob/d1e0dfea94c75ca622b42d8d9f15d64f8a500530/tests/workers/config/test_optim_config_on_cpu.py#L51-L88) | Mcore defaults: optimizer `adam`, layerwise `False`, momentum `0.95`, Nesterov `False`, split QKV `True`, scale mode `spectral`, coefficients `quintic`, NS steps `5`, TP mode `blockwise`, matmul precision `medium`, extra factor `1.0`, scalar optimizer `adam`. Overrides for optimizer, momentum, NS steps, TP mode, scalar optimizer and layerwise are retained. | Pure dataclass checks. The FSDP tests in this file cover scheduler defaults/types and cycle count, not Muon. The RMS-match switch has no assertion here. |
| [test_muon_optim_wiring_on_cpu.py](https://github.com/verl-project/verl/blob/d1e0dfea94c75ca622b42d8d9f15d64f8a500530/tests/utils/megatron/test_muon_optim_wiring_on_cpu.py#L112-L149) | Stub Megatron receives supported Muon overrides and default split QKV. When only momentum and NS steps are supported, those values are forwarded and the result has no `muon_tp_mode` attribute. Adam leaves Muon momentum at the stub default. No Muon-related fields raises `ValueError` mentioning `emerging_optimizers`. | Tests configuration plumbing against stubs, not optimizer construction or updates. It explicitly expects unsupported individual fields to be omitted. |
| [test_muon_optim_realcfg_on_cpu.py](https://github.com/verl-project/verl/blob/d1e0dfea94c75ca622b42d8d9f15d64f8a500530/tests/utils/megatron/test_muon_optim_realcfg_on_cpu.py#L35-L103) | Result is an actual Megatron `OptimizerConfig`, has optimizer `muon`, and retains each tested field that the installed build declares. Optimizer is outside `adam`/`sgd` and is in verl's Muon names. Adam remains `adam` and preserves Megatron's default Muon momentum. | Skips when Megatron cannot import or lacks `muon_momentum`. The optimizer routing check is a string predicate; it does not construct an optimizer or call `step`. |
| [test_muon_layerwise_bridge_ddp_on_cpu.py](https://github.com/verl-project/verl/blob/d1e0dfea94c75ca622b42d8d9f15d64f8a500530/tests/utils/megatron/test_muon_layerwise_bridge_ddp_on_cpu.py#L127-L196) | Layerwise mode defers provider DDP wrapping, passes the expected DDP override, calls the special wrapper exactly once, and retains module identity. Non-layerwise mode keeps standard DDP and makes no special-wrapper call. Layerwise plus Megatron FSDP raises `ValueError`. | Provider, wrapper and DDP configuration are stubs. No actual distributed buffers, communication, memory allocation or native FSDP2 behavior is tested. |
| [muon_optim_gpu_smoke.py](https://github.com/verl-project/verl/blob/d1e0dfea94c75ca622b42d8d9f15d64f8a500530/tests/utils/megatron/muon_optim_gpu_smoke.py#L37-L105) | Requires CUDA; builds a two-linear-layer BF16 network with real Megatron DDP, TP size 1 and distributed optimizer disabled. Checks `optimizer=muon`, a non-null optimizer, at least one inner class name containing `Muon`, and at least one changed parameter tensor after one backward/step. | Standalone `torchrun` program, not a collected CPU pytest. No exact `isinstance(TensorParallelMuon)` assertion, reference-update comparison, parameter-group coverage, finite-value assertion, multi-rank agreement, checkpoint test or resource measurement. |

## Assertions versus author reports

The PR description reports bitwise agreement, reduced memory and optimizer-update
time, and results from short RL runs and a Megatron-Bridge check. Those are author
experiment reports. The five files above do not assert those outcomes, and their
existence does not validate the same claims for native FSDP2. The PR also qualifies
reward differences as within run-to-run spread and does not establish an
end-to-end throughput improvement.

There is a concrete discrepancy between the prose and code: the PR/README says
unsupported Muon fields fail closed, but the final
[`_add_muon_args`](https://github.com/verl-project/verl/blob/d1e0dfea94c75ca622b42d8d9f15d64f8a500530/verl/utils/megatron/optimizer.py#L62-L88)
skips individual unsupported fields. It raises only when no field from its
Muon-related passthrough set is supported. The wiring test explicitly preserves
that behavior. The new FSDP2 implementation must define and test its own rejection
policy rather than inherit the broader prose claim.

The reference also implements an optional AdamW RMS-match factor and conflict/range
validation, but none of these five files asserts that logic. Its learning-rate
convention is not automatically the convention of the separate Dion dependency
used by the native FSDP2 implementation.

## CI caveat

The final head's [CPU unit-test workflow](https://github.com/verl-project/verl/actions/runs/30213862798)
reports failure. Its `headSha` matches the SHA above. During this review, fetching
the old job log returned HTTP 410; the available annotation only states exit code
1. This is insufficient to attribute the failure to a Muon test. The overall GPU
unit-test job reports success, but that does not show that the separate Muon smoke
was invoked. Merged status is not a substitute for per-test execution evidence.

## Additional native FSDP2 acceptance coverage

These are acceptance requirements for this implementation, not claims that the
Megatron tests or an unexecuted GPU script have passed them:

1. Configuration reaches the real native FSDP2 optimizer factory; AdamW remains
   the default without importing the optional Muon dependency. Unsupported
   strategies, layouts, dtypes and offload modes fail with an actionable error.
2. Parameter groups contain every trainable parameter exactly once, including
   shared/tied weights. Matrix selection uses global shape and module semantics;
   embeddings, output heads, norms, biases and unknown special parameters have an
   explicit AdamW fallback. Frozen parameters are excluded.
3. Real multi-GPU `fully_shard` updates agree with an unsharded reference over
   multiple steps, including momentum and rectangular matrices. Uneven/empty
   shards must either work correctly or be rejected before collectives. Check
   full-matrix orthogonalization, not independent orthogonalization of local shards.
4. FP32 parameter/state storage, supported forward dtypes, zero/missing gradients,
   weight decay, finite values, clipping and scheduler behavior are exercised.
   Dion's BF16 orthogonalization is part of the numerical contract even when
   stored parameters and forward computation are FP32.
5. Optimizer state and LR schedule survive save/load. A restarted process matches
   the next uninterrupted update; a real FSDP2 checkpoint round trip is separate
   from a CPU `state_dict` round trip.
6. A short real GRPO run exercises rollout, actor update, reward handling and
   weight synchronization through the existing training workflow. Resource or
   convergence claims require a separate controlled experiment on that backend.
