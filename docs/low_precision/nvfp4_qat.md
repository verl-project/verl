# NVFP4 QAT (Quantization-Aware Training) in verl

Last updated: 04/02/2026

verl supports NVFP4 Quantization-Aware Training (QAT), which applies fake quantization during training so the model learns to tolerate NVFP4 quantization error. At rollout time, weights are packed into real NVFP4 format for vLLM inference. This closes the precision gap between training and inference, preventing KL divergence explosion.

| Training Backend | Training Precision | Rollout Precision | vLLM Quant Method |
|---|---|---|---|
| **FSDP** | BF16 + fake quantization | NVFP4 W4A16 | `compressed-tensors` |
| **Megatron** | BF16 + fake quantization | NVFP4 W4A16 | `modelopt` |

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
| `fsdp_config.qat.mode` | Quantization mode | `"w4a16"` |
| `fsdp_config.qat.group_size` | Quantization group size | `16` |
| `fsdp_config.qat.ignore_patterns` | Layers to skip. Supports `re:` prefix for regex, otherwise substring match | `["lm_head", "embed_tokens", "re:.*mlp.gate$"]` |
| `fsdp_config.qat.quantization_config_path` | vLLM quantization config JSON path | Required |

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

- NVFP4 W4A16 (weight-only FP4 quantization)
- Dense models and MoE models
- FSDP and Megatron training backends
- Full quantization and FFN-only quantization strategies
- Verified on Qwen3-8B-Base and Qwen3-30B-A3B-Base

---

## Notes

- FSDP backend has scalability limitations for very large models. For large-scale training, use the Megatron backend.
- FSDP uses `re:` prefix regex for `ignore_patterns`, while Megatron uses `fnmatch` glob syntax. The two are not interchangeable.
