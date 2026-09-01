# Real NVFP4 post-v0.26 hotfix validation

This bundle preserves the validated v8 training contract and layers two exact
vLLM upstream fixes onto its immutable image:

- `9c226684` / vLLM #50029: quantize MoE weights per expert, removing the
  whole-tensor FP32/BF16 temporary and the single oversized FP4 kernel launch.
- `3ac95255` / vLLM #50074: reuse the online NVFP4 MoE kernel across native
  BF16 weight reloads so captured execution never holds stale kernel state.

Run phases sequentially: `probe`, `build`, `preflight`, then `startup`.  The
`startup` phase uses all 8 nodes and three optimizer steps because the observed
hang only reproduced while 32 vLLM servers initialized concurrently.
