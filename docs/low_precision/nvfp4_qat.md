# NVFP4 QAT

Last updated: 11/02/2026

verl supports NVFP4 QAT for **W4A4 routed experts** on Blackwell:

- trainer: Megatron + ModelOpt fake quantization
- rollout: vLLM, compressed-tensors NVFP4 experts with per-token activations on FlashInfer TRT-LLM
- dense linear layers: BF16 or MXFP8; NVFP4 linear is not supported

The rollout is the one [NVFP4 training](nvfp4.md) uses, with the same requirements. SGLang, W4A16, static
activation scales, FSDP and expert tensor parallelism are not supported.

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
      moe_activation: nvfp4_per_token
```

- Weights: E2M1 values in groups of 16 with E4M3 scales and one global scale per expert matrix; `gate_proj`
  and `up_proj` share one. Fake quantization and the trainer-side export run the same kernel, so the rollout
  gets the weights training used.
- Activations: each token gets a global scale from its own amax, as in the rollout's MoE kernel. Fake
  quantization matches FlashInfer's per-token quantization bitwise, including
  `FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH`; set it the same way for the trainer and the rollout. The rollout
  quantizes the `down_proj` input inside the fused kernel, so that input matches only approximately.
- Gradients pass straight through the fake quantization.

## Validation

```bash
.venv-vllm/bin/python scripts/rollout_logprob_diff.py \
  --model-path /mnt/efs/model/Qwen/Qwen3.5-35B-A3B --ep 8 \
  --quant-moe NVFP4 --quant-moe-activation nvfp4_per_token --qat-mode nvfp4
```

Measured results are in the [NVFP4 mismatch table](nvfp4.md#mismatch).
