# NVFP4 QAT (Quantization-Aware Training) in verl

Last updated: 07/06/2026

verl supports NVFP4 Quantization-Aware Training (QAT), which applies fake quantization during training so the model learns to tolerate NVFP4 quantization error. At rollout time, weights are packed into real NVFP4 format for vLLM inference. This closes the precision gap between training and inference, preventing KL divergence explosion.

| Training Backend | Training Precision | Rollout Precision | vLLM Quant Method |
|---|---|---|---|
| **FSDP** | BF16 + fake quantization | NVFP4 W4A16 | `compressed-tensors` |
| **FSDP (experimental)** | BF16 parameters + FP4 weight / FP8 activation fake quantization | W4A8 numerical simulation | `compressed-tensors` |
| **Megatron** | BF16 + fake quantization | NVFP4 W4A16 | `modelopt` |

> [!WARNING]
> W4A8 is an FSDP-only numerical simulation for dense models and the standard vLLM 0.15 NVFP4 MarlinExperts path. During rollout, weights use the existing NVFP4 W4A16 `compressed-tensors` kernels. Dense layer inputs are blockwise FP8 E4M3 quantized and dequantized; for fused MoE, both the gate/up input and the post-activation down-projection input receive the same Q/DQ. This does not execute a native W4A8 kernel and must not be used to claim W4A8 latency, throughput, or memory improvements. Within vLLM, non-Marlin NVFP4 backends and batched expert classes are rejected.

> [!TIP]
> For ready-to-run scripts, environment setup, and experimental results, see the [QAT recipe](https://github.com/verl-project/verl-recipe/tree/main/qat).

---

## Key Configuration

### FSDP Backend

Configured under `actor_rollout_ref.actor.fsdp_config.qat`:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      qat:
        enable: true
        mode: "w4a16"
        group_size: 16
        ignore_patterns:
          - "lm_head"
          - "embed_tokens"
          - "re:.*mlp.gate$"
        quantization_config_path: "recipe/qat/config/nvfp4_w4a16.json"
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `fsdp_config.qat.enable` | Enable QAT | `False` |
| `fsdp_config.qat.mode` | Quantization mode: `"w4a16"`, experimental `"w4a8"`, or `"w4a4"` | `"w4a16"` |
| `fsdp_config.qat.group_size` | Quantization group size | `16` |
| `fsdp_config.qat.ignore_patterns` | Layers to skip. Supports `re:` prefix for regex, otherwise substring match | `["lm_head", "embed_tokens", "re:.*mlp.gate$"]` |
| `fsdp_config.qat.quantization_config_path` | vLLM quantization config JSON path | Required |

For the experimental W4A8 simulation, set `mode: "w4a8"` but continue to use the W4A16 weight configuration:

```yaml
actor_rollout_ref:
  actor:
    fsdp_config:
      qat:
        enable: true
        mode: "w4a8"
        group_size: 16
        ignore_patterns:
          - "lm_head"
          - "embed_tokens"
          - "re:.*mlp.gate$"
        quantization_config_path: "recipe/qat/config/nvfp4_w4a16.json"
```

The W4A16 configuration is intentional: W4A8 simulation changes activation numerics only. FSDP still exports the same packed FP4 weights as W4A16, and verl automatically enables FP8 activation simulation in the vLLM subprocesses.

### Megatron Backend

Configured under `actor_rollout_ref.actor.megatron.qat`:

```yaml
actor_rollout_ref:
  actor:
    megatron:
      qat:
        enable: true
        mode: "w4a16"
        group_size: 16
        ignore_patterns:
          - "lm_head"
          - "*mlp.gate"
        quantization_config_path: "recipe/qat/config/nvfp4_w4a16_megatron.json"
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `megatron.qat.enable` | Enable QAT | `False` |
| `megatron.qat.mode` | Quantization mode | `"w4a16"` |
| `megatron.qat.group_size` | Quantization group size | `16` |
| `megatron.qat.ignore_patterns` | Layers to skip. Uses `fnmatch` glob syntax | `["lm_head", "*mlp.gate"]` |
| `megatron.qat.quantization_config_path` | vLLM quantization config JSON path | Required |

---

## Support Matrix

| Mode | Training Backend | Model Type | Rollout Path | Status |
|---|---|---|---|---|
| W4A16 | FSDP, Megatron | Dense, MoE | Native NVFP4 W4A16 | Supported |
| W4A8 simulation | FSDP | Dense, MoE (Marlin) | FP8 Q/DQ + W4A16 kernel | Experimental |
| W4A4 | FSDP | Dense, MoE | NVFP4 W4A4 | Experimental |

Full and FFN-only quantization strategies are available in the linked recipe. W4A16 has been verified on Qwen3-8B-Base and Qwen3-30B-A3B-Base; W4A8 numerical-simulation recipes cover both the dense Qwen3-8B-Base model and the MoE Qwen3-30B-A3B-Base model.

---

## Notes

- FSDP backend has scalability limitations for very large models. For large-scale training, use the Megatron backend.
- FSDP uses `re:` prefix regex for `ignore_patterns`, while Megatron uses `fnmatch` glob syntax. The two are not interchangeable.
- W4A8 uses dynamic per-token FP8 E4M3 activation blocks of shape `1 x 128`; no activation scale is stored in checkpoints or sent with the packed weights.
- W4A8 fused-MoE rollout currently requires vLLM 0.15's standard NVFP4 MarlinExperts path. Other vLLM NVFP4 backends and batched expert classes are rejected. Independent TensorRT-LLM and SGLang rollouts are outside this simulation's scope and have not been validated.
- Native W4A8 kernels and Megatron W4A8 support are outside the scope of the current simulation.
