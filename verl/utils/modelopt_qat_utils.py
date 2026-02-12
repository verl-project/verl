# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""ModelOpt Quantization-Aware Training (QAT) utilities for Megatron models.

Includes:
- QAT application via ModelOpt (apply_qat)
- QAT weight post-processing for exporting quantized weights to vLLM rollout (QATWeightPostProcessor)
"""

import re
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.export.quant_utils import (
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    get_quantization_format,
    get_weight_block_size,
    to_quantized_weight,
)
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

from verl.utils.megatron_utils import unwrap_model

# ---------------------------------------------------------------------------
# NVFP4 quantization config
# ---------------------------------------------------------------------------

NVFP4_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {"enable": False},
        "nn.BatchNorm1d": {"*": {"enable": False}},
        "nn.BatchNorm2d": {"*": {"enable": False}},
        "nn.BatchNorm3d": {"*": {"enable": False}},
        "nn.LeakyReLU": {"*": {"enable": False}},
        "*lm_head*": {"enable": False},
        "*proj_out.*": {"enable": False},  # Whisper: lm_head has key name proj_out
        "*block_sparse_moe.gate*": {"enable": False},  # Skip MOE router
        "*router*": {"enable": False},  # Skip MOE router
        "*mlp.gate.*": {"enable": False},  # Skip MOE router
        "*mlp.shared_expert_gate.*": {"enable": False},  # Skip MOE router
        "*linear_attn.conv1d*": {"enable": False},
        "*mixer.conv1d*": {"enable": False},
        "*output_layer*": {"enable": False},
        "output.*": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

# ---------------------------------------------------------------------------
# QAT application
# ---------------------------------------------------------------------------


def apply_qat(model: nn.Module, quant_method: str):
    """Apply Quantization-Aware Training to the model.

    Args:
        model: The Megatron model to apply QAT to.
        quant_method: Quantization method (currently only ``"nvfp4"`` is supported).

    Returns:
        The quantized model.
    """
    if quant_method != "nvfp4":
        raise ValueError(f"Only 'nvfp4' is supported, got: {quant_method}")

    mtq.quantize(model, NVFP4_WEIGHT_ONLY_CFG)
    return model


# ---------------------------------------------------------------------------
# QAT weight post-processing (for exporting quantized weights to rollout)
# ---------------------------------------------------------------------------

@dataclass
class QuantizationMetadata:
    """Metadata for a quantized module."""

    qformat: str
    weight_quantizer: Any
    input_quantizer: Any
    module: torch.nn.Module
    vpp_idx: int
    block_size: int = 16  # Default NVFP4 block size


class QATWeightPostProcessor:
    """
    Post-processor for extracting quantization info from QAT trained modules
    and converting bf16 weights to quantized formats (e.g., NVFP4).

    Key Design:
        1. Collect quantization metadata (quantizers, amax, block_size) from QAT modules
        2. Process all_gathered bf16 weights to compute quantized weights and scaling factors
        3. The scaling factors are computed on the merged (all_gathered) weights to ensure
           correct block boundaries for per-block quantization (NVFP4)

    Note on TP (Tensor Parallelism):
        - For NVFP4, weight_scale_2 (global scale) should ideally be computed from the full
          (all_gathered) weight to ensure consistency across TP ranks.
        - If use_calibrated_scale_2=True (default), we use the QAT calibrated amax which may
          only reflect the local shard's statistics.
        - If use_calibrated_scale_2=False, we recompute weight_scale_2 from the merged weight.
    """

    def __init__(
        self,
        actor_module: list,
        quantization_method: str = "nvfp4",
        dtype: torch.dtype = torch.bfloat16,
        use_calibrated_scale_2: bool = True,
    ):
        """
        Initialize the QAT weight post-processor.

        Args:
            actor_module: List of QAT trained model chunks (vpp chunks)
            quantization_method: Quantization method (nvfp4, fp8, etc.)
            dtype: Original data type (bf16)
            use_calibrated_scale_2: If True, use QAT calibrated amax for weight_scale_2.
                If False, recompute weight_scale_2 from merged weights. Recommended to set
                False when using TP to ensure consistent global scale.
        """
        self.actor_module = actor_module
        self.quantization_method = quantization_method
        self.dtype = dtype
        self.use_calibrated_scale_2 = use_calibrated_scale_2
        self.quant_metadata: dict[str, QuantizationMetadata] = {}

        self._build_quantization_metadata()
        self._log_initialization_info()

    def _build_quantization_metadata(self):
        """
        Extract quantization metadata from all modules in actor_module.
        Stores: {param_name: QuantizationMetadata}

        Supports both dense and MoE (Mixture of Experts) models:
        - Dense: decoder.layers.X.mlp.linear_fc1, decoder.layers.X.mlp.linear_fc2
        - MoE SequentialMLP: decoder.layers.X.mlp.experts.local_experts.Y.linear_fc1/fc2
        - MoE shared_experts: decoder.layers.X.mlp.shared_experts.linear_fc1/fc2
        - MoE TEGroupedMLP: decoder.layers.X.mlp.experts.linear_fc1/fc2 (grouped)
        """
        for vpp_idx, module in enumerate(self.actor_module):
            model = unwrap_model(module)

            for name, submodule in model.named_modules():
                # Handle MoE SequentialMLP - need to iterate over local_experts
                if self._is_sequential_mlp(submodule):
                    self._build_moe_sequential_mlp_metadata(name, submodule, vpp_idx)
                    continue

                # Handle MoE TEGroupedMLP - grouped experts with linear_fc1/fc2
                if self._is_te_grouped_mlp(submodule):
                    self._build_moe_te_grouped_mlp_metadata(name, submodule, vpp_idx)
                    continue

                # Handle regular quantized modules (dense layers, shared_experts, etc.)
                qformat = get_quantization_format(submodule)
                if qformat == QUANTIZATION_NONE:
                    continue

                block_size = get_weight_block_size(submodule)
                if block_size == 0:
                    continue

                weight_quantizer = getattr(submodule, "weight_quantizer", None)
                input_quantizer = getattr(submodule, "input_quantizer", None)

                metadata = QuantizationMetadata(
                    qformat=qformat,
                    weight_quantizer=weight_quantizer,
                    input_quantizer=input_quantizer,
                    module=submodule,
                    vpp_idx=vpp_idx,
                    block_size=block_size,
                )

                for param_name, _ in submodule.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else param_name
                    self.quant_metadata[full_name] = metadata

    def _is_sequential_mlp(self, module: torch.nn.Module) -> bool:
        """Check if module is a MoE SequentialMLP."""
        module_type_name = type(module).__name__
        return "SequentialMLP" in module_type_name and hasattr(module, "local_experts")

    def _is_te_grouped_mlp(self, module: torch.nn.Module) -> bool:
        """Check if module is a MoE TEGroupedMLP (Transformer Engine Grouped MLP)."""
        module_type_name = type(module).__name__
        return "TEGroupedMLP" in module_type_name or "GroupedMLP" in module_type_name

    def _build_moe_sequential_mlp_metadata(
        self,
        base_name: str,
        sequential_mlp: torch.nn.Module,
        vpp_idx: int,
    ):
        """
        Build quantization metadata for MoE SequentialMLP.

        SequentialMLP structure:
        - local_experts: list of MLP experts, each with linear_fc1 and linear_fc2
        - Each expert's linear layers may have quantizers attached

        Args:
            base_name: Base module name (e.g., 'decoder.layers.0.mlp.experts')
            sequential_mlp: The SequentialMLP module
            vpp_idx: Virtual pipeline parallel index
        """
        if not hasattr(sequential_mlp, "local_experts"):
            return

        for expert_idx, expert in enumerate(sequential_mlp.local_experts):
            # Process linear_fc1 and linear_fc2 for each expert
            for linear_name in ["linear_fc1", "linear_fc2"]:
                linear_module = getattr(expert, linear_name, None)
                if linear_module is None:
                    continue

                qformat = get_quantization_format(linear_module)
                if qformat == QUANTIZATION_NONE:
                    continue

                block_size = get_weight_block_size(linear_module)
                if block_size == 0:
                    continue

                weight_quantizer = getattr(linear_module, "weight_quantizer", None)
                input_quantizer = getattr(linear_module, "input_quantizer", None)

                metadata = QuantizationMetadata(
                    qformat=qformat,
                    weight_quantizer=weight_quantizer,
                    input_quantizer=input_quantizer,
                    module=linear_module,
                    vpp_idx=vpp_idx,
                    block_size=block_size,
                )

                # Build full parameter name
                # Format: {base_name}.local_experts.{expert_idx}.{linear_name}.weight
                for param_name, _ in linear_module.named_parameters(recurse=False):
                    full_name = f"{base_name}.local_experts.{expert_idx}.{linear_name}.{param_name}"
                    self.quant_metadata[full_name] = metadata

    def _build_moe_te_grouped_mlp_metadata(
        self,
        base_name: str,
        te_grouped_mlp: torch.nn.Module,
        vpp_idx: int,
    ):
        """
        Build quantization metadata for MoE TEGroupedMLP.

        TEGroupedMLP structure (Transformer Engine):
        - linear_fc1: grouped linear layer for all experts
        - linear_fc2: grouped linear layer for all experts
        - Weights are stored as 3D tensors [num_experts, out_dim, in_dim]

        Args:
            base_name: Base module name (e.g., 'decoder.layers.0.mlp.experts')
            te_grouped_mlp: The TEGroupedMLP module
            vpp_idx: Virtual pipeline parallel index
        """
        for linear_name in ["linear_fc1", "linear_fc2"]:
            linear_module = getattr(te_grouped_mlp, linear_name, None)
            if linear_module is None:
                continue

            qformat = get_quantization_format(linear_module)
            if qformat == QUANTIZATION_NONE:
                continue

            block_size = get_weight_block_size(linear_module)
            if block_size == 0:
                continue

            weight_quantizer = getattr(linear_module, "weight_quantizer", None)
            input_quantizer = getattr(linear_module, "input_quantizer", None)

            metadata = QuantizationMetadata(
                qformat=qformat,
                weight_quantizer=weight_quantizer,
                input_quantizer=input_quantizer,
                module=linear_module,
                vpp_idx=vpp_idx,
                block_size=block_size,
            )

            # Build full parameter name
            # Format: {base_name}.{linear_name}.weight
            for param_name, _ in linear_module.named_parameters(recurse=False):
                full_name = f"{base_name}.{linear_name}.{param_name}"
                self.quant_metadata[full_name] = metadata

    def _log_initialization_info(self):
        """Log initialization information for debugging."""
        print(f"[QAT PostProcessor] Initialized with quantization method: {self.quantization_method}")
        print(f"[QAT PostProcessor] Found {len(self.quant_metadata)} quantized parameters")

        # Log sample parameters from layer 0 for debugging
        for name, metadata in self.quant_metadata.items():
            if "layers.0" in name and "weight" in name:
                print(
                    f"[QAT PostProcessor] Sample: {name}, qformat={metadata.qformat}, block_size={metadata.block_size}, module type: {type(metadata.module)}"
                )

    def _log_initialization_info(self):
        """Log initialization information for debugging."""
        print(f"[QAT PostProcessor] Initialized with quantization method: {self.quantization_method}")
        print(f"[QAT PostProcessor] Found {len(self.quant_metadata)} quantized parameters")

        # Log sample parameters from layer 0 for debugging (including MoE experts)
        moe_expert_count = 0
        for name, metadata in self.quant_metadata.items():
            if "layers.0" in name and "weight" in name:
                if "local_experts" in name:
                    moe_expert_count += 1
                    if moe_expert_count <= 2:  # Only log first 2 experts
                        print(
                            f"[QAT PostProcessor] MoE Expert Sample: {name}, qformat={metadata.qformat}, block_size={metadata.block_size}"
                        )
                elif "shared_experts" in name:
                    print(
                        f"[QAT PostProcessor] Shared Expert Sample: {name}, qformat={metadata.qformat}, block_size={metadata.block_size}"
                    )
                else:
                    print(
                        f"[QAT PostProcessor] Dense Sample: {name}, qformat={metadata.qformat}, block_size={metadata.block_size}, module type: {type(metadata.module)}"
                    )

        if moe_expert_count > 0:
            print(f"[QAT PostProcessor] Total MoE expert layers in layer 0: {moe_expert_count}")

    def _find_matching_metadata(self, param_name: str) -> QuantizationMetadata | None:
        """
        Find matching quantization metadata for a parameter name.
        Handles potential name variations between training and export.
        """
        # Direct match
        if param_name in self.quant_metadata:
            return self.quant_metadata[param_name]

        # Try removing common prefixes/suffixes
        variations = [
            param_name,
            param_name.replace("module.", ""),
            param_name.replace("model.", ""),
        ]

        for var in variations:
            if var in self.quant_metadata:
                return self.quant_metadata[var]

        return None

    def _quantize_weight(
        self,
        name: str,
        weight: torch.Tensor,
        metadata: QuantizationMetadata,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Quantize a single weight parameter.

        Args:
            name: Parameter name
            weight: The all_gathered bf16 weight tensor
            metadata: Quantization metadata

        Yields:
            (param_name, param_tensor) for quantized weight and scaling factors
        """
        qformat = metadata.qformat

        if qformat == QUANTIZATION_NVFP4:
            # print("[lark]: quantize_weight name:", name, "weight:", weight.shape, "metadata:", metadata)
            yield from self._quantize_nvfp4(name, weight, metadata)
        else:
            # Unknown format, pass through with warning
            print(f"[QAT PostProcessor] Warning: Unknown qformat {qformat} for {name}, passing through")
            yield (name, weight)

    def _quantize_nvfp4(
        self,
        name: str,
        weight: torch.Tensor,
        metadata: QuantizationMetadata,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        NVFP4 quantization implementation.

        NVFP4 uses two-level scaling:
        - weight_scale_2 (global): per-tensor scale = amax / (6.0 * 448.0)
        - weight_scale (per-block): per-block scale in FP8 format

        The weight is packed into uint8 format (2 x FP4 values per byte).

        Yields:
            (name, quantized_weight): Packed uint8 weight
            (name + "_scale", weight_scale): Per-block FP8 scaling factors
            (name + "_scale_2", weight_scale_2): Global scaling factor
            (name + "_input_scale", input_scale): Input activation scale (if available)
        """
        weight_quantizer = metadata.weight_quantizer
        input_quantizer = metadata.input_quantizer
        block_size = metadata.block_size
        qformat = metadata.qformat

        # # Ensure weight is in float for quantization computation
        # weight_float = weight.float()

        # Step 1: Compute weight_scale_2 (global scale)
        # For TP sharding, we should recompute weight_scale_2 from merged weight
        # to ensure consistent global scale across all TP ranks.
        if self.use_calibrated_scale_2 and weight_quantizer is not None and hasattr(weight_quantizer, "_amax"):
            # Use QAT calibrated amax (may only reflect local shard statistics)
            # weight_scale_2 = amax / (6.0 * 448.0)
            weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(weight_quantizer)
        else:
            # Compute from all_gathered weight directly (recommended for TP)
            # weight_scale_2 = max(abs(weight)) / (6.0 * 448.0)
            weight_scale_2 = NVFP4QTensor.get_weights_scaling_factor_2(weight)

        # Step 2: Compute weight_scale (per-block scale)
        # This MUST be computed on the all_gathered (merged) weight to ensure
        # correct block boundaries
        # weight_scale shape: [out_dim, in_dim / block_size], dtype: float8_e4m3fn
        weight_scale = NVFP4QTensor.get_weights_scaling_factor(
            weight,
            block_size,
            weights_scaling_factor_2=weight_scale_2.to(weight.device),
        )[0]

        # Step 3: Quantize weight to NVFP4 packed format
        quantized_weight = to_quantized_weight(
            weight,
            weight_scale,
            qformat,
            weight_scale_2,
            block_size,
        )

        # Yield quantized weight
        yield (name, quantized_weight)

        # Yield scaling factors
        # Note: Use consistent naming convention with ModelOpt export
        scale_name = name.replace(".weight", ".weight_scale")
        if scale_name == name:
            scale_name = name + "_scale"
        yield (scale_name, weight_scale)

        scale_2_name = name.replace(".weight", ".weight_scale_2")
        if scale_2_name == name:
            scale_2_name = name + "_scale_2"
        yield (scale_2_name, weight_scale_2)

        # Step 4: Export input_scale (activation quantization) if available
        if input_quantizer is not None:
            input_scale = self._get_input_scale(input_quantizer)
            if input_scale is not None:
                input_scale_name = name.replace(".weight", ".input_scale")
                if input_scale_name == name:
                    input_scale_name = name + "_input_scale"
                yield (input_scale_name, input_scale)

    def _get_input_scale(self, input_quantizer) -> torch.Tensor | None:
        """
        Get input activation scaling factor from quantizer.

        Args:
            input_quantizer: The input quantizer from the module

        Returns:
            Input scaling factor tensor or None
        """
        if input_quantizer is None:
            return None

        if not hasattr(input_quantizer, "_amax"):
            return None

        amax = input_quantizer._amax
        if amax is None:
            return None

        # For NVFP4, use the NVFP4QTensor method
        if hasattr(NVFP4QTensor, "get_activation_scaling_factor"):
            return NVFP4QTensor.get_activation_scaling_factor(input_quantizer)

        return amax.float() / (6.0 * 448.0)

    def process_weights_iterator(
        self,
        per_tensor_param: Iterator[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Process an iterator of weights and yield quantized results.

        This method wraps per_tensor_generator output and applies quantization
        to each weight, yielding the quantized weights and scaling factors.

        Args:
            per_tensor_param: Iterator of (name, bf16_weight) from per_tensor_generator

        Yields:
            (name, tensor): Quantized weight and associated scaling factors
        """
        for name, param in per_tensor_param:
            # quantize_single_tensor returns a list of (name, tensor) tuples
            # For NVFP4: [(name, quant_weight), (name_scale, scale), (name_scale_2, scale_2), ...]
            # For non-quantized: [(name, original_weight)]
            quantized_results = self.quantize_single_tensor(name, param)
            for q_name, q_tensor in quantized_results:
                yield (q_name, q_tensor)

    def quantize_single_tensor(
        self,
        name: str,
        weight: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Quantize a single tensor and return all related tensors as a list.

        This method is designed to be called AFTER weight_converter.convert_param,
        so the name should already be in HF format (e.g., 'model.layers.0.self_attn.q_proj.weight').

        Args:
            name: Parameter name in HF format
            weight: Single tensor to quantize

        Returns:
            List of (param_name, param_tensor) tuples:
            - (name, quantized_weight)
            - (name.replace('.weight', '.weight_scale'), weight_scale)  # for NVFP4
            - (name.replace('.weight', '.weight_scale_2'), weight_scale_2)  # for NVFP4
        """
        # Find matching metadata using the original mcore name pattern
        # Since name is now in HF format, we need to check if this layer type should be quantized
        metadata = self._find_matching_metadata_by_hf_name(name)

        if metadata is None:
            # Not quantized, return original tensor
            return [(name, weight)]

        # Quantize this tensor
        return list(self._quantize_weight(name, weight, metadata))

    def _find_matching_metadata_by_hf_name(self, hf_name: str) -> QuantizationMetadata | None:
        """
        Find matching quantization metadata for an HF-format parameter name.

        This maps HF names back to the original mcore names to find metadata.
        E.g., 'model.layers.0.self_attn.q_proj.weight' -> check if qkv layer is quantized

        The mapping logic:
        - HF q_proj/k_proj/v_proj.weight -> mcore linear_qkv.weight
        - HF o_proj.weight -> mcore linear_proj.weight
        - HF gate_proj/up_proj.weight -> mcore linear_fc1.weight
        - HF down_proj.weight -> mcore linear_fc2.weight
        """
        import re

        # Only process weight parameters
        if not hf_name.endswith(".weight") or hf_name.endswith("._amax") or "norm" in hf_name:
            return None

        # Extract layer number from HF name
        layer_match = re.search(r"layers?\.(\d+)\.", hf_name)
        if not layer_match:
            # Not a layer parameter (e.g., embed_tokens, lm_head, norm)
            # Check for direct matches
            return self._find_non_layer_metadata(hf_name)

        layer_num = layer_match.group(1)

        # Determine the mcore module name based on HF name pattern
        mcore_patterns = []

        if "self_attn" in hf_name:
            if any(proj in hf_name for proj in ["q_proj", "k_proj", "v_proj"]):
                mcore_patterns.append(f"decoder.layers.{layer_num}.self_attention.linear_qkv.weight")
            elif "o_proj" in hf_name:
                mcore_patterns.append(f"decoder.layers.{layer_num}.self_attention.linear_proj.weight")
        elif "mlp" in hf_name:
            if any(proj in hf_name for proj in ["gate_proj", "up_proj"]):
                mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.linear_fc1.weight")
            elif "down_proj" in hf_name:
                mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.linear_fc2.weight")

        # Try to find matching metadata
        for pattern in mcore_patterns:
            if pattern in self.quant_metadata:
                return self.quant_metadata[pattern]

        # If no exact match, try to find any metadata from the same layer
        # This handles cases where the exact name might be slightly different
        for mcore_name, metadata in self.quant_metadata.items():
            if f"layers.{layer_num}." in mcore_name:
                # Found a quantized module in the same layer
                # For QAT, if any module in the layer is quantized, all Linear layers should be
                if ".weight" in mcore_name:
                    return metadata

        return None

    def _find_non_layer_metadata(self, hf_name: str) -> QuantizationMetadata | None:
        """Find metadata for non-layer parameters (embed_tokens, lm_head, etc.)."""
        # Map HF names to mcore names for non-layer parameters
        name_mapping = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "lm_head.weight": "output_layer.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
        }

        mcore_name = name_mapping.get(hf_name)
        if mcore_name and mcore_name in self.quant_metadata:
            return self.quant_metadata[mcore_name]

        return None

    def _find_matching_metadata_by_hf_name(self, hf_name: str) -> QuantizationMetadata | None:
        """
        Find matching quantization metadata for an HF-format parameter name.

        This maps HF names back to the original mcore names to find metadata.
        E.g., 'model.layers.0.self_attn.q_proj.weight' -> check if qkv layer is quantized

        The mapping logic:
        - HF q_proj/k_proj/v_proj.weight -> mcore linear_qkv.weight
        - HF o_proj.weight -> mcore linear_proj.weight
        - HF gate_proj/up_proj.weight -> mcore linear_fc1.weight
        - HF down_proj.weight -> mcore linear_fc2.weight
        - HF experts.X.gate_proj/up_proj.weight -> mcore experts.local_experts.X.linear_fc1.weight
        - HF experts.X.down_proj.weight -> mcore experts.local_experts.X.linear_fc2.weight
        - HF shared_expert.gate_proj/up_proj.weight -> mcore shared_experts.linear_fc1.weight
        - HF shared_expert.down_proj.weight -> mcore shared_experts.linear_fc2.weight
        """
        import re

        # Only process weight parameters
        if not hf_name.endswith(".weight") or hf_name.endswith("._amax") or "norm" in hf_name:
            return None

        # Extract layer number from HF name
        layer_match = re.search(r"layers?\.(\d+)\.", hf_name)
        if not layer_match:
            # Not a layer parameter (e.g., embed_tokens, lm_head, norm)
            # Check for direct matches
            return self._find_non_layer_metadata(hf_name)

        layer_num = layer_match.group(1)

        # Determine the mcore module name based on HF name pattern
        mcore_patterns = []

        if "self_attn" in hf_name:
            if any(proj in hf_name for proj in ["q_proj", "k_proj", "v_proj"]):
                mcore_patterns.append(f"decoder.layers.{layer_num}.self_attention.linear_qkv.weight")
            elif "o_proj" in hf_name:
                mcore_patterns.append(f"decoder.layers.{layer_num}.self_attention.linear_proj.weight")
        elif "mlp" in hf_name:
            # Check for MoE expert patterns first
            expert_match = re.search(r"experts\.(\d+)\.", hf_name)
            if expert_match:
                expert_id = expert_match.group(1)
                if any(proj in hf_name for proj in ["gate_proj", "up_proj"]):
                    # MoE expert gate_proj/up_proj -> local_experts.X.linear_fc1
                    mcore_patterns.append(
                        f"decoder.layers.{layer_num}.mlp.experts.local_experts.{expert_id}.linear_fc1.weight"
                    )
                elif "down_proj" in hf_name:
                    # MoE expert down_proj -> local_experts.X.linear_fc2
                    mcore_patterns.append(
                        f"decoder.layers.{layer_num}.mlp.experts.local_experts.{expert_id}.linear_fc2.weight"
                    )
            elif "shared_expert" in hf_name:
                # Shared expert patterns (Qwen2Moe, DeepSeekV3, etc.)
                if any(proj in hf_name for proj in ["gate_proj", "up_proj"]):
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.shared_experts.linear_fc1.weight")
                elif "down_proj" in hf_name:
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.shared_experts.linear_fc2.weight")
            elif "gate.weight" in hf_name:
                # MoE router gate
                mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.router.weight")
            elif any(proj in hf_name for proj in ["gate_proj", "up_proj"]):
                # Dense MLP gate_proj/up_proj
                mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.linear_fc1.weight")
            elif "down_proj" in hf_name:
                # Dense MLP down_proj
                mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.linear_fc2.weight")

        # Try to find matching metadata
        for pattern in mcore_patterns:
            if pattern in self.quant_metadata:
                return self.quant_metadata[pattern]

        # If no exact match, try to find any metadata from the same layer
        # This handles cases where the exact name might be slightly different
        for mcore_name, metadata in self.quant_metadata.items():
            if f"layers.{layer_num}." in mcore_name:
                # For MoE, check if we're looking for expert weights
                if "experts" in hf_name:
                    if "experts" in mcore_name and ".weight" in mcore_name:
                        return metadata
                # For dense, check if any module in the layer is quantized
                elif ".weight" in mcore_name:
                    return metadata

        return None