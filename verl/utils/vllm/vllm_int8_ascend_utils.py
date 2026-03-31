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
    if quant_desc is not None and isinstance(quant_desc, _W8A8DynamicQuantDescription):
        return True
    return False


def load_int8_ascend_weights(weights, model, dtype=torch.bfloat16):
    """Quantize bf16 weights to INT8 and load into the vllm model.

    For W8A8_DYNAMIC, the model already has int8 weight parameters plus
    weight_scale / weight_offset created by AscendLinearMethod.  We quantize
    the incoming bf16 weights to int8 and directly copy them into the
    existing parameters, bypassing vLLM weight_loader which would
    incorrectly try to shard already-sharded weights.

    Returns the set of loaded parameter names.
    """
    loaded_params = set()
    param_dict = dict(model.named_parameters())

    for name, tensor in quant_weights_int8_ascend(weights, dtype=dtype):
        if name not in param_dict:
            logger.debug("Skipping unknown parameter: %s", name)
            continue
        param = param_dict[name]
        loaded = tensor

        if loaded.shape != param.shape:
            if loaded.ndim == 2 and param.ndim == 1 and loaded.shape[0] == param.shape[0]:
                loaded = loaded.squeeze(-1)
            elif loaded.ndim == 2 and param.ndim == 2 and loaded.shape == (param.shape[1], param.shape[0]):
                loaded = loaded.t().contiguous()

        if param.shape != loaded.shape:
            logger.warning(
                "Shape mismatch for %s: param %s vs loaded %s, skipping",
                name, param.shape, loaded.shape,
            )
            continue
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
