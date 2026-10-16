# Receiver dtype for rollout weight synchronization

Last updated: 10/04/2026.

VeOmni can cast temporary export shards to the dtype of the actual vLLM receiver before DTensor gathering and expert-parallel broadcast. For FP32 training parameters loaded into BF16 rollout parameters, this halves the tensor payload for those collectives and the weight transfer. The source training parameters, optimizer state, and checkpoint dtype are preserved. Buffers and actual FP32 receiver parameters keep their original dtype.

Enable with:

```text
actor_rollout_ref.rollout.checkpoint_engine.export_receiver_dtype=true
```

The default is `false`. The initial implementation supports the naive colocated backend, VeOmni, and unquantized BF16 `Qwen3MoeForCausalLM` receivers using tensor parallelism. Rollout data parallelism, rollout expert parallelism, LoRA, other model classes, and checkpoint converters that export weights are rejected. Trainer expert parallelism is supported.

Every trainer rank reads metadata from all actual receiver TP workers. Before any weight update starts, the manager requires all trainer ranks to complete preflight and all rollout replicas to agree about parameter names and dtypes. Preflight checks fused Q/K/V and expert mappings, export coverage, expert counts, and known buffers. Unknown names, unsupported dtypes, missing workers, or inconsistent metadata fail before synchronization. Metadata is checked on each update, including after a receiver restart.

The conversion moves BF16 rounding earlier in the export path. Q/K/V and expert conversion only rearrange or split values; no arithmetic is moved across the cast. Use full receiver comparisons and EP/DP export tests to verify byte equality with the existing receiver conversion path. A smaller payload does not establish training throughput or quality improvement; benchmark the complete synchronization path in the intended deployment before enabling it in production.

Validation commands, with a matching VeOmni/vLLM environment:

```bash
uv run python -m pytest tests/utils/test_rollout_weight_dtype_on_cpu.py tests/utils/test_receiver_dtype_integration_on_cpu.py -q
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 tests/special_distributed/test_rollout_weight_dtype.py
uv run python -m pytest tests/utils/test_bucketed_weight_transfer.py::TestBucketedWeightTransferIPC::test_mixed_dtypes -q
```
