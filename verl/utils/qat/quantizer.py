# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fast NVFP4 Quantizer for verl FSDP training.

Directly computes scales and quantizes weights using compressed_tensors APIs.
Includes scale computation utilities for weight quantization.
"""

import logging
import os
from typing import Optional

import torch
from compressed_tensors.compressors.quantized_compressors.fp4_quantized import NVFP4PackedCompressor
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import generate_gparam

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def compute_blockwise_scale(
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
) -> torch.Tensor:
    """Compute blockwise scale using pre-computed global_scale (for fusion).
    Returns FP8 E4M3 blockwise scale tensor.
    """
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    weight_reshaped = weight.view(out_features, num_groups, group_size)
    block_max = torch.amax(torch.abs(weight_reshaped), dim=-1).to(torch.float32)

    local_scale = block_max / FP4_E2M1_DATA.max
    blockwise_scale_f32 = torch.clamp(
        global_scale * local_scale,
        min=-FP8_E4M3_DATA.max,
        max=FP8_E4M3_DATA.max,
    )

    blockwise_scale = blockwise_scale_f32.to(torch.float8_e4m3fn)
    eps = torch.finfo(torch.float8_e4m3fn).eps
    blockwise_scale = torch.where(
        blockwise_scale == 0,
        torch.tensor(eps, dtype=blockwise_scale.dtype, device=weight.device),
        blockwise_scale,
    )

    return blockwise_scale


# Fusion patterns for transformer models
FUSE_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def fuse_global_scales(
    layer_global_scales: dict[str, torch.Tensor],
    strategy: str = "min",
) -> dict[str, torch.Tensor]:
    """Fuse global scales for QKV/GateUp groups (take min across group)."""
    if not layer_global_scales:
        return {}

    # Group by parent module
    parent_to_children: dict[str, dict[str, str]] = {}
    for name in layer_global_scales:
        parent, child = name.rsplit(".", 1) if "." in name else ("", name)
        parent_to_children.setdefault(parent, {})[child] = name

    fused_scales = {}
    processed = set()

    for parent, children in parent_to_children.items():
        for _, patterns in FUSE_PATTERNS.items():
            matched = [children[p] for p in patterns if p in children]
            if len(matched) == len(patterns):
                group_scales = [layer_global_scales[n] for n in matched]
                if strategy == "min":
                    fused_scale = torch.min(torch.cat(group_scales)).reshape([1])
                else:
                    raise ValueError(f"Unknown fuse strategy: {strategy}")
                for layer_name in matched:
                    fused_scales[layer_name] = fused_scale.clone()
                    processed.add(layer_name)

    for name, scale in layer_global_scales.items():
        if name not in processed:
            fused_scales[name] = scale

    return fused_scales


class QATQuantizer:
    """Quantizer for QAT-trained weights using compressed_tensors APIs."""

    def __init__(
        self,
        mode: str = "w4a16",
        group_size: int = 16,
        ignore_patterns: Optional[list] = None,
        device: Optional[torch.device] = None,
        param_dtype: Optional[torch.dtype] = None,
    ):
        self.mode = mode.lower()
        self._is_w4a4 = self.mode == "w4a4"  # W4A4 needs input_global_scale
        self.group_size = group_size
        self.ignore_patterns = ignore_patterns or ["lm_head", "embed_tokens", "re:.*mlp.gate$"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_dtype = param_dtype

        self._compressor = NVFP4PackedCompressor()
        self._quant_args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=group_size,
            scale_dtype=FP8_E4M3_DATA.dtype,
        )

    def _should_quantize(self, name: str, tensor: torch.Tensor) -> bool:
        """Check if parameter should be quantized."""
        import re

        if not name.endswith(".weight"):
            return False
        if tensor.dim() != 2:
            return False
        if tensor.shape[1] % self.group_size != 0:
            return False

        module_name = name.rsplit(".weight", 1)[0]

        for pattern in self.ignore_patterns:
            if pattern.startswith("re:"):
                # Regex pattern - use re.match like vLLM does
                regex = pattern[3:]
                if re.match(regex, module_name):
                    return False
            else:
                if pattern in module_name:
                    return False
        return True

    def _group_params_by_decoder_layer(
        self, params: dict[str, torch.Tensor]
    ) -> dict[Optional[int], dict[str, torch.Tensor]]:
        """Group parameters by decoder layer index. Returns {layer_idx: {name: tensor}}."""
        import re

        grouped: dict[Optional[int], dict[str, torch.Tensor]] = {}
        for name, tensor in params.items():
            match = re.search(r"layers\.(\d+)\.", name)
            layer_idx = int(match.group(1)) if match else None
            grouped.setdefault(layer_idx, {})[name] = tensor
        return grouped

    def quantize_with_fusion(
        self,
        params: dict[str, torch.Tensor],
        target_device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """Quantize all parameters with scale fusion for QKV/GateUp layers."""
        output_device = target_device or torch.device("cpu")
        quantization_device = self.device
        include_input_scale = self._is_w4a4

        output_params = {}
        input_global_scales = {}
        grouped_params = self._group_params_by_decoder_layer(params)
        layer_indices = sorted(grouped_params.keys(), key=lambda x: (x is not None, x or 0))

        # Pre-scan for input_global_scales (W4A4 mode)
        if include_input_scale:
            for name, tensor in params.items():
                if "input_global_scale" in name:
                    layer_name = name.replace(".input_global_scale", "")
                    if tensor.numel() == 1 and tensor.item() == -1.0:
                        logger.warning(f"W4A4: {layer_name} input_global_scale is uninitialized")
                    else:
                        input_global_scales[layer_name] = tensor
            logger.info(f"W4A4 mode: found {len(input_global_scales)} input_global_scales")

        total_quantized = 0
        total_passthrough = 0

        for layer_idx in layer_indices:
            layer_params = grouped_params[layer_idx]
            layer_weights = {}
            layer_passthrough = {}

            for name, tensor in layer_params.items():
                if "input_global_scale" in name or "input_amax" in name:
                    continue

                if self._should_quantize(name, tensor):
                    layer_name = name.rsplit(".weight", 1)[0]
                    layer_weights[layer_name] = (name, tensor)
                else:
                    layer_passthrough[name] = tensor

            if layer_idx is None and layer_weights:
                raise RuntimeError(
                    f"[QAT Quantizer] Unexpected quantizable weights outside decoder layers: "
                    f"{list(layer_weights.keys())}. These should be in ignore_patterns."
                )

            if not layer_weights:
                for name, tensor in layer_passthrough.items():
                    output_params[name] = tensor.to(output_device)
                    total_passthrough += 1
                continue

            weights_on_gpu = {}
            layer_global_scales = {}

            for layer_name, (_, tensor) in layer_weights.items():
                weight_gpu = tensor.to(device=quantization_device, dtype=self.param_dtype)
                weights_on_gpu[layer_name] = weight_gpu
                amax = torch.amax(torch.abs(weight_gpu)).to(torch.float32)
                layer_global_scales[layer_name] = generate_gparam(
                    -amax.unsqueeze(0),
                    amax.unsqueeze(0),
                    scale_data=FP8_E4M3_DATA,
                    quant_data=FP4_E2M1_DATA,
                    dtype=torch.float32,
                )

            fused_global_scales = fuse_global_scales(layer_global_scales, strategy="min")

            for layer_name, weight_gpu in weights_on_gpu.items():
                fused_global_scale = fused_global_scales[layer_name]
                weight_scale = compute_blockwise_scale(weight_gpu, fused_global_scale, self.group_size)
                weight_packed = self._compressor.compress_weight(
                    weight=weight_gpu,
                    scale=weight_scale.float(),
                    global_scale=fused_global_scale,
                    quantization_args=self._quant_args,
                )["weight_packed"]

                output_params[f"{layer_name}.weight_packed"] = weight_packed.to(output_device)
                output_params[f"{layer_name}.weight_scale"] = weight_scale.to(output_device)
                output_params[f"{layer_name}.weight_global_scale"] = fused_global_scale.to(output_device)

                if include_input_scale:
                    if layer_name in input_global_scales:
                        output_params[f"{layer_name}.input_global_scale"] = (
                            input_global_scales[layer_name].float().to(output_device)
                        )
                    else:
                        raise ValueError(
                            f"W4A4 mode requires input_global_scale for layer '{layer_name}', "
                            f"but it's not found or uninitialized (-1.0)."
                        )

                total_quantized += 1

            del weights_on_gpu, layer_global_scales, fused_global_scales

            for name, tensor in layer_passthrough.items():
                output_params[name] = tensor.to(output_device)
                total_passthrough += 1

        torch.cuda.empty_cache()
        logger.info(f"Quantized {total_quantized} layers, passed through {total_passthrough} params")
        return output_params


__all__ = [
    "QATQuantizer",
]
