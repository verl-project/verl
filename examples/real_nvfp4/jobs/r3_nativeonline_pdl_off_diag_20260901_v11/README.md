# Real NVFP4 v11: TRTLLM MoE PDL-off diagnostic

This bundle changes one runtime variable relative to v10: vLLM's two
FlashInfer TRTLLM NVFP4 MoE call sites explicitly pass `enable_pdl=False`.

CUDA graphs remain enabled in `FULL_DECODE_ONLY` mode, `max_num_seqs` remains
128, and the vLLM 0.26 native online quantization/reload path is unchanged.
The first acceptance gate is an 8-node, 32-server startup run. This bundle is
diagnostic until that gate and a multi-step reload run both pass.

```bash
./submit.sh probe
./submit.sh build
./submit.sh preflight
./submit.sh startup
```
