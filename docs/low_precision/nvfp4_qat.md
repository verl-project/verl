# NVFP4 QAT

Last updated: 10/19/2026
verl supports NVFP4 as **W4A4 routed experts** on Blackwell:

- trainer: Megatron + ModelOpt QAT
- rollout: vLLM compressed-tensors
- kernel: FlashInfer CUTLASS
- dense linear layers: BF16 or MXFP8; NVFP4 linear is not supported

W4A16, FSDP NVFP4 and non-CUTLASS NVFP4 backends are not supported by this
path.

## Configuration

```yaml
actor_rollout_ref:
  actor:
    megatron:
      qat:
        enable: true
        mode: nvfp4
  rollout:
    quantization:
      linear: null
      moe: NVFP4
```

The weight uses E2M1 values in groups of 16, FP8 E4M3 local scales and one
tensor-level global divisor. Activations use the same NVFP4 hierarchy with
dynamic group scales. Before ModelOpt observes an activation, verl exports the
neutral tensor divisor `1.0`; subsequent syncs use the running QAT value.

ModelOpt fake quantization and compressed-tensors export share the same QDQ
implementation. The exporter also aligns gate/up global scales, as required by
vLLM's fused MoE loader.

## Validation

Use CUDA Graph mode; eager-mode results are not representative:

```bash
.venv-vllm/bin/python scripts/rollout_logprob_diff.py \
  --model-path /mnt/efs/model/Qwen/Qwen3.5-35B-A3B --ep 8 \
  --quant-moe NVFP4 --qat-mode nvfp4 \
  '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=false'
```

On Qwen3.5-35B-A3B with EP=8, CUDA Graph and no R3, the final validation
measured `k3_kl=0.25778` without QAT and `k3_kl=0.32112` with NVFP4 QAT
(Pearson correlation `0.991763` and `0.981248`, respectively). FlashInfer's
fused finalize is nondeterministic; repeated results ranged from `0.219–0.258`
and `0.263–0.380`. A synthetic single-expert CUTLASS comparison confirmed that
the exported weights and activation QDQ use the expected NVFP4 format, but the
fused kernel still differed from dequantize-then-BF16 matmul by mean absolute
`0.0084` per MoE layer (correlation `0.99855`). This validates the refit
lifecycle and kernel execution, but not numerical acceptance: the accumulated
model-level mismatch remains too large for training.
