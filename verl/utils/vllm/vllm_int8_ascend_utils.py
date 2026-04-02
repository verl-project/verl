import logging
import os
from typing import Generator, Optional

import torch

from verl.utils.kernel.int8_kernel import scaled_int8_per_channel

logger = logging.getLogger(__name__)


class _W8A8DynamicQuantDescription(dict):
    """A dict that returns 'W8A8_DYNAMIC' for quantizable layers and 'FLOAT' for others.

    vllm-ascend's AscendQuantConfig expects every layer weight to appear in
    the quant_description.  Instead of enumerating all possible layer names
    (which depend on model architecture), we use a smart dict that infers the
    quant type from the key name pattern when the key is not explicitly set.
    """

    _QUANT_PATTERNS = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2",
    ]
    _SKIP_PATTERNS = [
        "embed_tokens", "lm_head", "layernorm", "norm", "ln_",
        "embeddings", "mlp.gate.weight",
    ]

    def __missing__(self, key):
        key_lower = key.lower()
        for p in self._SKIP_PATTERNS:
            if p in key_lower:
                return "FLOAT"
        for p in self._QUANT_PATTERNS:
            if p in key_lower:
                return "W8A8_DYNAMIC"
        return "FLOAT"


def is_int8_ascend_model(vllm_config) -> bool:
    """Check if the model is configured for INT8 Ascend (W8A8_DYNAMIC) quantization."""
    if os.environ.get("VERL_VLLM_INT8_ASCEND_QUANT_ENABLED", "0") == "1":
        return True
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is None:
        return False
    quant_desc = getattr(quant_config, "quant_description", None)
    if quant_desc is None:
        return False
    if isinstance(quant_desc, _W8A8DynamicQuantDescription):
        return True
    if isinstance(quant_desc, dict) and any(
        v == "W8A8_DYNAMIC" for v in quant_desc.values()
    ):
        return True
    return False


def load_int8_ascend_weights(weights, model, dtype=torch.bfloat16):
    """Quantize bf16 weights to INT8 and load into the vllm model.

    Ascend W8A8_DYNAMIC stores weight as (input_size, output_size) — transposed.
    Scale/offset are 1D (output_size,).

    Handles vLLM's fused parameter mapping:
      q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj.
    """
    param_dict = dict(model.named_parameters())

    qkv_mod = None
    gate_up_mod = None
    for n, m in model.named_modules():
        if n.endswith(".self_attn") and qkv_mod is None:
            qkv_mod = getattr(m, "qkv_proj", None)
        if n.endswith(".mlp") and gate_up_mod is None:
            gate_up_mod = getattr(m, "gate_up_proj", None)
        if qkv_mod and gate_up_mod:
            break

    _FUSE_MAP = {}
    if qkv_mod:
        h = qkv_mod.num_heads * qkv_mod.head_size
        kv = qkv_mod.num_kv_heads * qkv_mod.head_size
        _FUSE_MAP["q_proj"] = ("qkv_proj", 0, h)
        _FUSE_MAP["k_proj"] = ("qkv_proj", h, kv)
        _FUSE_MAP["v_proj"] = ("qkv_proj", h + kv, kv)
    if gate_up_mod and hasattr(gate_up_mod, "output_partition_sizes"):
        sizes = gate_up_mod.output_partition_sizes
        _FUSE_MAP["gate_proj"] = ("gate_up_proj", 0, sizes[0])
        _FUSE_MAP["up_proj"] = ("gate_up_proj", sizes[0], sizes[1])

    loaded_params = set()

    for name, tensor in quant_weights_int8_ascend(weights, dtype=dtype):
        parts = name.rsplit(".", 1)
        if len(parts) != 2:
            continue
        prefix, attr = parts
        proj = prefix.rsplit(".", 1)[-1]

        if proj in _FUSE_MAP:
            fused_name, offset, size = _FUSE_MAP[proj]
            base = prefix.rsplit(".", 1)[0] if "." in prefix else ""
            full_name = f"{base}.{fused_name}.{attr}" if base else f"{fused_name}.{attr}"
            if full_name not in param_dict:
                continue
            param = param_dict[full_name]
            loaded = tensor

            if attr == "weight":
                if loaded.ndim == 2 and param.ndim == 2:
                    if loaded.shape[0] == size and loaded.shape[1] == param.shape[0]:
                        param.data.narrow(1, offset, size).copy_(loaded.t().contiguous())
                        loaded_params.add(full_name)
                    elif loaded.shape[1] == size and loaded.shape[0] == param.shape[0]:
                        param.data.narrow(1, offset, size).copy_(loaded)
                        loaded_params.add(full_name)
            else:
                if loaded.ndim == 2:
                    loaded = loaded.squeeze(-1)
                if param.ndim == 1 and loaded.ndim == 1:
                    if offset + loaded.shape[0] <= param.shape[0]:
                        param.data.narrow(0, offset, loaded.shape[0]).copy_(loaded)
                        loaded_params.add(full_name)
                    elif loaded.shape == param.shape:
                        param.data.copy_(loaded)
                        loaded_params.add(full_name)
        else:
            if name not in param_dict:
                continue
            param = param_dict[name]
            loaded = tensor
            if attr == "weight" and loaded.ndim == 2 and param.ndim == 2:
                if loaded.shape == (param.shape[1], param.shape[0]):
                    param.data.copy_(loaded.t().contiguous())
                    loaded_params.add(name)
                elif loaded.shape == param.shape and loaded.shape[0] != loaded.shape[1]:
                    param.data.copy_(loaded)
                    loaded_params.add(name)
                elif loaded.shape == param.shape:
                    param.data.copy_(loaded.t().contiguous())
                    loaded_params.add(name)
            else:
                if loaded.ndim == 2 and param.ndim == 1 and loaded.shape[0] == param.shape[0]:
                    loaded = loaded.squeeze(-1)
                if loaded.shape == param.shape:
                    param.data.copy_(loaded)
                    loaded_params.add(name)

    return loaded_params


INT8_ASCEND_QUANT_KWARGS = {
    "quant_method": "ascend",
    "default_quant_type": "W8A8_DYNAMIC",
}


def build_int8_ascend_quant_description(model_config) -> dict:
    """Build a quant_model_description dict for W8A8_DYNAMIC on the fly.

    vllm-ascend's AscendQuantConfig reads per-layer quant types from the
    quant_description dict.  For online INT8 we mark every Linear weight
    as W8A8_DYNAMIC and everything else (embed, norm, lm_head, gate) as FLOAT.

    We must explicitly enumerate all layer weight names because vLLM serializes
    the config dict to JSON when passing to subprocesses, which would lose any
    custom __missing__ behavior from dict subclasses.
    """
    desc = {}
    desc["quant_method"] = "ascend"

    num_layers = getattr(model_config, "num_hidden_layers", 0)

    desc["model.embed_tokens.weight"] = "FLOAT"
    desc["model.norm.weight"] = "FLOAT"
    desc["lm_head.weight"] = "FLOAT"

    quant_proj_names = [
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    float_names = [
        "input_layernorm", "post_attention_layernorm",
    ]

    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        for proj in quant_proj_names:
            desc[f"{prefix}.{proj}.weight"] = "W8A8_DYNAMIC"
        for norm in float_names:
            desc[f"{prefix}.{norm}.weight"] = "FLOAT"

    return desc


def should_quantize_int8(param_name: str) -> bool:
    """Determine whether to quantize a parameter to INT8.

    Rules (same as FP8QuantizerHelper.should_quantize_param):
    - Must end with .weight
    - Exclude embedding, norm, lm_head, MoE router
    - Include q/k/v/o_proj, gate/up/down_proj, fc1/fc2
    """
    if not param_name.endswith(".weight"):
        return False

    exclude_patterns = [
        "embed_tokens",
        "lm_head",
        "layernorm",
        "norm",
        "ln_",
        "embeddings",
        "mlp.gate.weight",
    ]
    param_lower = param_name.lower()
    for pattern in exclude_patterns:
        if pattern in param_lower:
            return False

    include_patterns = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "mlp",
    ]
    for pattern in include_patterns:
        if pattern in param_lower:
            return True

    return False


def quant_weights_int8_ascend(weights, dtype=torch.bfloat16):
    """Quantize weights to INT8 format for Ascend W8A8_DYNAMIC.

    For each weight that should be quantized:
      - Yields (name, int8_weight)
      - Yields (name.replace('.weight', '.weight_scale'), scale)
      - Yields (name.replace('.weight', '.weight_offset'), offset)

    Args:
        weights: Iterable of (name, tensor) pairs
        dtype: Data type for intermediate computation

    Yields:
        Tuples of (name, tensor)
    """
    for k, v in weights:
        if not should_quantize_int8(k):
            yield (k, v)
            continue

        try:
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                logger.debug(f"Quantizing to INT8 (Ascend): {k}")

            int8_weight, weight_scale, weight_offset = scaled_int8_per_channel(
                v.to(dtype),
            )

            yield (k, int8_weight)

            scale_name = k.replace(".weight", ".weight_scale")
            offset_name = k.replace(".weight", ".weight_offset")
            yield (scale_name, weight_scale)
            yield (offset_name, weight_offset)

            del int8_weight, weight_scale, weight_offset

        except Exception as e:
            logger.error(f"Failed to quantize {k} to INT8: {e}")
            yield (k, v)
