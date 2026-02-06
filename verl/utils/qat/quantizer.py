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
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Constants
GROUP_SIZE = 16


# Lazy import to avoid circular dependency
_FP4_E2M1_DATA = None
_FP8_E4M3_DATA = None
_generate_gparam = None


def _ensure_imports():
    """Lazy import compressed_tensors dependencies."""
    global _FP4_E2M1_DATA, _FP8_E4M3_DATA, _generate_gparam
    if _FP4_E2M1_DATA is None:
        from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA, FP8_E4M3_DATA
        from compressed_tensors.quantization.utils.helpers import generate_gparam

        _FP4_E2M1_DATA = FP4_E2M1_DATA
        _FP8_E4M3_DATA = FP8_E4M3_DATA
        _generate_gparam = generate_gparam


def _resolve_param_dtype(
    weight: torch.Tensor,
    param_dtype: Optional[torch.dtype],
) -> torch.dtype:
    """Resolve param_dtype: use explicit value or infer from weight.dtype."""
    return param_dtype if param_dtype is not None else weight.dtype


@dataclass
class WeightScaleResult:
    """Result of weight scale computation."""

    blockwise_scale: torch.Tensor  # (out_features, num_groups), FP8 E4M3
    global_scale: torch.Tensor  # (1,), FP32
    block_max: Optional[torch.Tensor] = None


def compute_weight_scales(
    weight: torch.Tensor,
    group_size: int = 16,
    return_block_max: bool = False,
    param_dtype: Optional[torch.dtype] = None,
) -> WeightScaleResult:
    """Compute weight scales for NVFP4 quantization (TENSOR_GROUP strategy)."""
    _ensure_imports()
    param_dtype = _resolve_param_dtype(weight, param_dtype)

    device = weight.device
    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    weight_casted = weight.to(param_dtype) if weight.dtype != param_dtype else weight
    weight_reshaped = weight_casted.view(out_features, num_groups, group_size)
    block_max = torch.amax(torch.abs(weight_reshaped), dim=-1)

    # Compute global scale (amax in param_dtype, then f32 for division - matches vLLM)
    tensor_amax = torch.amax(torch.abs(weight_casted)).to(torch.float32)
    global_scale = _generate_gparam(
        -tensor_amax.unsqueeze(0),
        tensor_amax.unsqueeze(0),
        scale_data=_FP8_E4M3_DATA,
        quant_data=_FP4_E2M1_DATA,
        dtype=torch.float32,
    )

    blockwise_scale = _compute_blockwise_from_global(
        block_max=block_max,
        global_scale=global_scale,
        device=device,
    )

    return WeightScaleResult(
        blockwise_scale=blockwise_scale,
        global_scale=global_scale,
        block_max=block_max if return_block_max else None,
    )


def compute_blockwise_scale_only(
    weight: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int = 16,
    param_dtype: Optional[torch.dtype] = None,
) -> WeightScaleResult:
    """Compute blockwise scale using pre-computed global_scale (for fusion)."""
    _ensure_imports()
    param_dtype = _resolve_param_dtype(weight, param_dtype)

    device = weight.device
    out_features, in_features = weight.shape
    num_groups = in_features // group_size

    weight_casted = weight.to(param_dtype) if weight.dtype != param_dtype else weight
    weight_reshaped = weight_casted.view(out_features, num_groups, group_size)
    block_max = torch.amax(torch.abs(weight_reshaped), dim=-1)

    blockwise_scale = _compute_blockwise_from_global(
        block_max=block_max,
        global_scale=global_scale,
        device=device,
    )

    return WeightScaleResult(
        blockwise_scale=blockwise_scale,
        global_scale=global_scale,
    )


def compute_global_amax(
    weight: torch.Tensor,
    param_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute global amax of weight tensor (returns f32)."""
    param_dtype = _resolve_param_dtype(weight, param_dtype)
    weight_casted = weight.to(param_dtype) if weight.dtype != param_dtype else weight
    return torch.amax(torch.abs(weight_casted)).to(torch.float32)


def compute_global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute global_scale from amax value."""
    _ensure_imports()

    return _generate_gparam(
        -amax.unsqueeze(0) if amax.dim() == 0 else amax,
        amax.unsqueeze(0) if amax.dim() == 0 else amax,
        scale_data=_FP8_E4M3_DATA,
        quant_data=_FP4_E2M1_DATA,
        dtype=torch.float32,
    )


def _compute_blockwise_from_global(
    block_max: torch.Tensor,
    global_scale: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute blockwise scale from block_max and global_scale."""
    _ensure_imports()

    block_max_f32 = block_max.to(torch.float32)
    local_scale = block_max_f32 / _FP4_E2M1_DATA.max
    blockwise_scale_f32 = global_scale * local_scale

    blockwise_scale_f32 = torch.clamp(
        blockwise_scale_f32,
        min=-_FP8_E4M3_DATA.max,
        max=_FP8_E4M3_DATA.max,
    )

    blockwise_scale = blockwise_scale_f32.to(torch.float8_e4m3fn)
    eps = torch.finfo(torch.float8_e4m3fn).eps
    blockwise_scale = torch.where(
        blockwise_scale == 0,
        torch.tensor(eps, dtype=blockwise_scale.dtype, device=device),
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
    """Fuse global scales for QKV/GateUp groups."""
    if not layer_global_scales:
        return {}

    parent_to_children = _group_layers_by_parent(list(layer_global_scales.keys()))

    fused_scales = {}
    processed = set()

    for parent, children in parent_to_children.items():
        for _, patterns in FUSE_PATTERNS.items():
            matched = [children[p] for p in patterns if p in children]

            if len(matched) == len(patterns):
                group_scales = [layer_global_scales[n] for n in matched]

                if strategy == "min":
                    fused_scale = torch.min(torch.cat(group_scales)).reshape([1])
                elif strategy == "max":
                    fused_scale = torch.max(torch.cat(group_scales)).reshape([1])
                else:
                    raise ValueError(f"Unknown fuse strategy: {strategy}")

                for layer_name in matched:
                    fused_scales[layer_name] = fused_scale.clone()
                    processed.add(layer_name)

    for name, scale in layer_global_scales.items():
        if name not in processed:
            fused_scales[name] = scale

    return fused_scales


def _group_layers_by_parent(layer_names: list[str]) -> dict[str, dict[str, str]]:
    """Group layer names by parent module."""
    parent_to_children = {}
    for name in layer_names:
        if "." in name:
            parent, child = name.rsplit(".", 1)
        else:
            parent, child = "", name
        parent_to_children.setdefault(parent, {})[child] = name
    return parent_to_children


@dataclass
class ScaleInfo:
    """Quantization scale information."""

    weight_scale: torch.Tensor  # Blockwise scale, FP8 E4M3
    weight_global_scale: torch.Tensor  # Global scale, float32
    input_global_scale: Optional[torch.Tensor] = None  # For W4A4 mode


class QATQuantizer:
    """Quantizer for QAT-trained weights using compressed_tensors APIs."""

    def __init__(
        self,
        mode: str = "w4a16",
        ignore_patterns: Optional[list] = None,
        device: Optional[torch.device] = None,
        param_dtype: Optional[torch.dtype] = None,
    ):
        self.mode = mode.lower()
        self._is_w4a4 = self.mode == "w4a4"  # W4A4 needs input_global_scale
        self.ignore_patterns = ignore_patterns or ["lm_head", "embed_tokens", "re:.*mlp.gate$"]
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_dtype = param_dtype

        # Lazy import compressed_tensors
        self._init_compressed_tensors()

    def _init_compressed_tensors(self):
        """Initialize compressed-tensors dependencies."""
        from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
            NVFP4PackedCompressor,
        )
        from compressed_tensors.quantization.quant_args import (
            FP8_E4M3_DATA,
            QuantizationArgs,
            QuantizationStrategy,
            QuantizationType,
        )

        self._compressor = NVFP4PackedCompressor()

        self._quant_args = QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.TENSOR_GROUP,
            group_size=GROUP_SIZE,
            scale_dtype=FP8_E4M3_DATA.dtype,
        )

    def _should_quantize(self, name: str, tensor: torch.Tensor) -> bool:
        """Check if parameter should be quantized."""
        import re

        if not name.endswith(".weight"):
            return False
        if tensor.dim() != 2:
            return False
        if tensor.shape[1] % GROUP_SIZE != 0:
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

    def compute_scales(self, weight: torch.Tensor) -> ScaleInfo:
        """Compute quantization scales."""
        result = compute_weight_scales(weight, group_size=GROUP_SIZE)

        return ScaleInfo(
            weight_scale=result.blockwise_scale, weight_global_scale=result.global_scale, input_global_scale=None
        )

    def quantize_weight(self, weight: torch.Tensor, scale_info: ScaleInfo) -> torch.Tensor:
        """Quantize and pack a single weight tensor using compressed_tensors API."""
        compressed_dict = self._compressor.compress_weight(
            weight=weight,
            scale=scale_info.weight_scale.float(),  # FP8 -> float32
            global_scale=scale_info.weight_global_scale.float(),
            quantization_args=self._quant_args,
        )
        return compressed_dict["weight_packed"]

    def _to_regular_tensor(self, t):
        """Convert DTensor to regular tensor if needed."""
        if t is None:
            return None

        # Check for DTensor
        is_dtensor = False
        try:
            from torch.distributed.tensor import DTensor

            is_dtensor = isinstance(t, DTensor)
        except ImportError:
            try:
                from torch.distributed._tensor import DTensor

                is_dtensor = isinstance(t, DTensor)
            except ImportError:
                pass

        if is_dtensor:
            try:
                return t.full_tensor().clone().detach()
            except RuntimeError:
                if hasattr(t, "to_local"):
                    return t.to_local().clone().detach()
                if hasattr(t, "_local_tensor"):
                    return t._local_tensor.clone().detach()
                return torch.tensor(t.tolist(), dtype=t.dtype, device=t.device)

        if hasattr(t, "clone"):
            return t.clone().detach()
        return t

    def _extract_decoder_layer_idx(self, name: str) -> Optional[int]:
        """Extract decoder layer index from parameter name."""
        import re

        match = re.search(r"layers\.(\d+)\.", name)
        return int(match.group(1)) if match else None

    def _group_params_by_decoder_layer(
        self, params: dict[str, torch.Tensor]
    ) -> dict[Optional[int], dict[str, torch.Tensor]]:
        """Group parameters by decoder layer index."""
        grouped = {}
        for name, tensor in params.items():
            layer_idx = self._extract_decoder_layer_idx(name)
            if layer_idx not in grouped:
                grouped[layer_idx] = {}
            grouped[layer_idx][name] = tensor
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
                    tensor = self._to_regular_tensor(tensor)
                    layer_name = name.replace(".input_global_scale", "")
                    if tensor.numel() == 1 and tensor.item() == -1.0:
                        logger.warning(f"W4A4: {layer_name} input_global_scale is uninitialized")
                    else:
                        input_global_scales[layer_name] = tensor.clone()
            logger.info(f"W4A4 mode: found {len(input_global_scales)} input_global_scales")

        total_quantized = 0
        total_passthrough = 0

        for layer_idx in layer_indices:
            layer_params = grouped_params[layer_idx]
            layer_weights = {}
            layer_passthrough = {}

            for name, tensor in layer_params.items():
                tensor = self._to_regular_tensor(tensor)
                if tensor is None:
                    continue

                if "input_global_scale" in name or "input_amax" in name:
                    continue

                if self._should_quantize(name, tensor):
                    layer_name = name.rsplit(".weight", 1)[0]
                    layer_weights[layer_name] = (name, tensor)
                else:
                    layer_passthrough[name] = tensor

            if layer_idx is None:
                for name, tensor in layer_passthrough.items():
                    output_params[name] = tensor.to(output_device)
                    total_passthrough += 1

                for layer_name, (param_name, tensor) in layer_weights.items():
                    if self.param_dtype is not None:
                        weight_gpu = tensor.to(quantization_device, self.param_dtype)
                    else:
                        weight_gpu = tensor.to(quantization_device)
                    scale_info = self.compute_scales(weight_gpu)
                    weight_packed = self.quantize_weight(weight_gpu, scale_info)

                    output_params[f"{layer_name}.weight_packed"] = weight_packed.to(output_device)
                    output_params[f"{layer_name}.weight_scale"] = scale_info.weight_scale.to(torch.float8_e4m3fn).to(
                        output_device
                    )
                    output_params[f"{layer_name}.weight_global_scale"] = scale_info.weight_global_scale.float().to(
                        output_device
                    )

                    del weight_gpu
                    total_quantized += 1
                continue

            if not layer_weights:
                # No weights to quantize in this layer
                for name, tensor in layer_passthrough.items():
                    output_params[name] = tensor.to(output_device)
                    total_passthrough += 1
                continue

            weights_on_gpu = {}
            layer_amaxes = {}

            for layer_name, (param_name, tensor) in layer_weights.items():
                if self.param_dtype is not None:
                    weight_gpu = tensor.to(quantization_device, self.param_dtype)
                else:
                    weight_gpu = tensor.to(quantization_device)
                weights_on_gpu[layer_name] = weight_gpu
                layer_amaxes[layer_name] = compute_global_amax(weight_gpu, param_dtype=self.param_dtype)

            layer_global_scales = {name: compute_global_scale_from_amax(amax) for name, amax in layer_amaxes.items()}
            fused_global_scales = fuse_global_scales(layer_global_scales, strategy="min")

            for layer_name, weight_casted in weights_on_gpu.items():
                fused_global_scale = fused_global_scales[layer_name]

                # Compute blockwise scale with fused global scale
                result = compute_blockwise_scale_only(
                    weight_casted,
                    global_scale=fused_global_scale,
                    group_size=GROUP_SIZE,
                    param_dtype=self.param_dtype,
                )

                scale_info = ScaleInfo(
                    weight_scale=result.blockwise_scale, weight_global_scale=fused_global_scale, input_global_scale=None
                )

                weight_packed = self.quantize_weight(weight_casted, scale_info)

                output_params[f"{layer_name}.weight_packed"] = weight_packed.to(output_device)
                output_params[f"{layer_name}.weight_scale"] = scale_info.weight_scale.to(torch.float8_e4m3fn).to(
                    output_device
                )
                output_params[f"{layer_name}.weight_global_scale"] = scale_info.weight_global_scale.float().to(
                    output_device
                )

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

            del weights_on_gpu, layer_amaxes, layer_global_scales, fused_global_scales
            torch.cuda.empty_cache()

            for name, tensor in layer_passthrough.items():
                output_params[name] = tensor.to(output_device)
                total_passthrough += 1

        logger.info(f"Quantized {total_quantized} layers, passed through {total_passthrough} params")
        return output_params


__all__ = [
    "QATQuantizer",
    "ScaleInfo",
    # Scale computation utilities
    "WeightScaleResult",
    "compute_weight_scales",
    "compute_blockwise_scale_only",
    "compute_global_amax",
    "compute_global_scale_from_amax",
    "fuse_global_scales",
    "FUSE_PATTERNS",
]
