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


import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional

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


@dataclass
class QuantizationMetadata:
    """Metadata for a quantized module."""

    qformat: str
    weight_quantizer: Any
    input_quantizer: Any
    module: torch.nn.Module
    vpp_idx: int
    block_size: int = 16  # Default NVFP4 block size
    # Fields for EP synchronization - store amax values for non-local experts
    weight_amax: Optional[torch.Tensor] = None
    input_amax: Optional[torch.Tensor] = None
    is_local: bool = True  # Whether this expert is local to current EP rank
    global_expert_idx: Optional[int] = None  # Global expert index for MoE experts
    local_expert_idx: Optional[int] = None  # Local expert index on this EP rank


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
    Note on EP (Expert Parallelism):
        - When EP is enabled, each rank only holds a subset of experts (local_experts)
        - We synchronize metadata across all EP ranks to ensure complete metadata for all experts
        - Local expert indices are converted to global expert indices for proper mapping
    """

    def __init__(
        self,
        actor_module: list,
        quantization_method: str = "nvfp4",
        dtype: torch.dtype = torch.bfloat16,
        use_calibrated_scale_2: bool = False,
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
        self.ep_size, self.ep_rank, self.ep_group = self._get_ep_info()
        self.pp_size, self.pp_rank, self.pp_group = self._get_pp_info()
        self.num_local_experts = 0  # Will be determined during metadata building

        self._build_quantization_metadata()

        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # print(f"[QAT PostProcessor][Rank {global_rank}] After _build_quantization_metadata: "
        #       f"metadata_count={len(self.quant_metadata)}, ep_size={self.ep_size}, pp_size={self.pp_size}")

        # Synchronize metadata across EP ranks if EP is enabled
        if self.ep_size > 1:
            print(f"[QAT PostProcessor][Rank {global_rank}] Starting EP metadata sync...")
            self._sync_quantization_metadata_across_ep()
            print(f"[QAT PostProcessor][Rank {global_rank}] After EP sync: metadata_count={len(self.quant_metadata)}")

        # Synchronize metadata across PP ranks if PP is enabled
        # This ensures all PP ranks have complete metadata for all layers
        if self.pp_size > 1:
            print(f"[QAT PostProcessor][Rank {global_rank}] Starting PP metadata sync...")
            self._sync_quantization_metadata_across_pp()
            print(f"[QAT PostProcessor][Rank {global_rank}] After PP sync: metadata_count={len(self.quant_metadata)}")
        else:
            print(f"[QAT PostProcessor][Rank {global_rank}] PP sync skipped: pp_size={self.pp_size}")

        self._log_initialization_info()

    def _get_ep_info(self) -> tuple[int, int, Any]:
        """
        Get Expert Parallel information from Megatron parallel state.

        Returns:
            (ep_size, ep_rank, ep_group): EP world size, rank, and process group
        """
        try:
            from megatron.core import parallel_state as mpu

            ep_size = mpu.get_expert_model_parallel_world_size()
            if ep_size > 1:
                ep_rank = mpu.get_expert_model_parallel_rank()
                ep_group = mpu.get_expert_model_parallel_group()
                return ep_size, ep_rank, ep_group
        except Exception:
            # EP not enabled or mpu not available
            pass
        return 1, 0, None

    def _get_pp_info(self) -> tuple[int, int, Any]:
        """
        Get Pipeline Parallel information from Megatron parallel state.

        Returns:
            (pp_size, pp_rank, pp_group): PP world size, rank, and process group
        """
        try:
            from megatron.core import parallel_state as mpu

            pp_size = mpu.get_pipeline_model_parallel_world_size()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_group = mpu.get_pipeline_model_parallel_group()

            if torch.distributed.get_rank() == 0:
                print(f"[QAT PostProcessor] PP info: pp_size={pp_size}, pp_rank={pp_rank}, pp_group={pp_group}")

            if pp_size > 1:
                return pp_size, pp_rank, pp_group
            else:
                return pp_size, pp_rank, None
        except Exception as e:
            if torch.distributed.get_rank() == 0:
                print(f"[QAT PostProcessor] Warning: Failed to get PP info: {e}")
            pass
        return 1, 0, None

    def _extract_layer_index(self, name: str) -> Optional[int]:
        """
        Extract layer index from parameter name.

        For mcore format: decoder.layers.{layer_idx}.xxx

        Returns:
            Layer index or None if not a layer parameter
        """
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            return int(match.group(1))
        return None

    def _get_num_layers_per_pp_stage(self) -> int:
        """
        Get the number of layers per PP stage from local metadata.

        This is calculated as max(local_layer_indices) + 1
        """
        max_layer_idx = -1
        for name in self.quant_metadata.keys():
            layer_idx = self._extract_layer_index(name)
            if layer_idx is not None and layer_idx > max_layer_idx:
                max_layer_idx = layer_idx
        return max_layer_idx + 1 if max_layer_idx >= 0 else 0

    def _convert_local_to_global_layer_name(self, name: str, source_pp_rank: int, num_layers_per_stage: int) -> str:
        """
        Convert parameter name from local layer index to global layer index.

        Args:
            name: Parameter name with local layer index (e.g., decoder.layers.0.xxx)
            source_pp_rank: The PP rank this name came from
            num_layers_per_stage: Number of layers per PP stage

        Returns:
            Parameter name with global layer index
        """
        local_layer_idx = self._extract_layer_index(name)
        if local_layer_idx is None:
            return name

        global_layer_idx = source_pp_rank * num_layers_per_stage + local_layer_idx
        return re.sub(r"layers\.(\d+)\.", f"layers.{global_layer_idx}.", name, count=1)

    def _extract_local_expert_index(self, name: str) -> Optional[int]:
        """
        Extract local expert index from parameter name.

        For SequentialMLP structure, the pattern is:
        decoder.layers.{layer}.mlp.experts.local_experts.{local_idx}.linear_fc1/fc2.weight

        Args:
            name: Parameter name in mcore format

        Returns:
            Local expert index or None if not an expert parameter
        """
        match = re.search(r"local_experts\.(\d+)\.", name)
        if match:
            return int(match.group(1))
        return None

    def _local_to_global_expert_index(self, local_idx: int) -> int:
        """
        Convert local expert index to global expert index.

        Global index = ep_rank * num_local_experts + local_idx

        Args:
            local_idx: Local expert index on this EP rank

        Returns:
            Global expert index
        """
        return self.ep_rank * self.num_local_experts + local_idx

    def _convert_name_to_global_index(self, name: str, local_idx: int, global_idx: int) -> str:
        """
        Convert parameter name from local to global expert index.

        Args:
            name: Original parameter name with local index
            local_idx: Local expert index
            global_idx: Global expert index

        Returns:
            Parameter name with global expert index
        """
        return name.replace(f"local_experts.{local_idx}.", f"local_experts.{global_idx}.")

    def _build_quantization_metadata(self):
        """
        Extract quantization metadata from all modules in actor_module.
        Stores: {param_name: QuantizationMetadata}

        For EP training with SequentialMLP:
        - Detects local expert indices and computes global indices
        - Stores metadata with global expert indices as keys
        """
        # First pass: collect all local expert indices to determine num_local_experts
        local_expert_indices = set()

        for vpp_idx, module in enumerate(self.actor_module):
            model = unwrap_model(module)
            for name, submodule in model.named_modules():
                local_idx = self._extract_local_expert_index(name)
                if local_idx is not None:
                    local_expert_indices.add(local_idx)

        if local_expert_indices:
            self.num_local_experts = max(local_expert_indices) + 1
            if torch.distributed.get_rank() == 0:
                print(f"[QAT PostProcessor] Detected {self.num_local_experts} local experts per EP rank")

        # Second pass: build metadata with global indices
        for vpp_idx, module in enumerate(self.actor_module):
            model = unwrap_model(module)

            for name, submodule in model.named_modules():
                # Check if this module is quantized
                qformat = get_quantization_format(submodule)
                if qformat == QUANTIZATION_NONE:
                    continue

                block_size = get_weight_block_size(submodule)
                if block_size == 0:
                    continue

                weight_quantizer = getattr(submodule, "weight_quantizer", None)
                input_quantizer = getattr(submodule, "input_quantizer", None)

                # Extract amax values for synchronization
                weight_amax = None
                input_amax = None
                if weight_quantizer is not None and hasattr(weight_quantizer, "_amax"):
                    weight_amax = weight_quantizer._amax.clone().cpu() if weight_quantizer._amax is not None else None
                if input_quantizer is not None and hasattr(input_quantizer, "_amax"):
                    input_amax = input_quantizer._amax.clone().cpu() if input_quantizer._amax is not None else None

                # Determine global expert index for MoE experts
                local_expert_idx = self._extract_local_expert_index(name)
                global_expert_idx = None
                if local_expert_idx is not None and self.ep_size > 1:
                    global_expert_idx = self._local_to_global_expert_index(local_expert_idx)

                metadata = QuantizationMetadata(
                    qformat=qformat,
                    weight_quantizer=weight_quantizer,
                    input_quantizer=input_quantizer,
                    module=submodule,
                    vpp_idx=vpp_idx,
                    block_size=block_size,
                    weight_amax=weight_amax,
                    input_amax=input_amax,
                    is_local=True,
                    global_expert_idx=global_expert_idx,
                    local_expert_idx=local_expert_idx,
                )

                for param_name, _ in submodule.named_parameters(recurse=False):
                    full_name = f"{name}.{param_name}" if name else param_name

                    # For EP training, store with global expert index as key
                    if local_expert_idx is not None and self.ep_size > 1:
                        global_name = self._convert_name_to_global_index(full_name, local_expert_idx, global_expert_idx)
                        self.quant_metadata[global_name] = metadata
                    else:
                        self.quant_metadata[full_name] = metadata

    def _log_initialization_info(self):
        """Log initialization information for debugging."""
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        print(
            f"[QAT PostProcessor][Rank {global_rank}] Initialized with quantization method: {self.quantization_method}"
        )
        print(f"[QAT PostProcessor][Rank {global_rank}] Found {len(self.quant_metadata)} quantized parameters")
        if self.ep_size > 1:
            print(
                f"[QAT PostProcessor][Rank {global_rank}] EP enabled: ep_size={self.ep_size}, ep_rank={self.ep_rank}, "
                f"num_local_experts={self.num_local_experts}"
            )
        if self.pp_size > 1:
            local_count = sum(1 for m in self.quant_metadata.values() if m.is_local)
            remote_count = sum(1 for m in self.quant_metadata.values() if not m.is_local)
            print(
                f"[QAT PostProcessor][Rank {global_rank}] PP enabled: pp_size={self.pp_size}, pp_rank={self.pp_rank}, "
                f"local_params={local_count}, remote_params={remote_count}"
            )

        # Log all metadata entries for debugging
        for name, metadata in self.quant_metadata.items():
            extra_info = ""
            if metadata.global_expert_idx is not None:
                extra_info = f", global_expert_idx={metadata.global_expert_idx}"
            if not metadata.is_local:
                extra_info += ", is_local=False"
            print(
                f"[QAT PostProcessor][Rank {global_rank}] Metadata: {name}, qformat={metadata.qformat}, "
                f"block_size={metadata.block_size}{extra_info}"
            )

    def _sync_quantization_metadata_across_ep(self):
        """
        Synchronize quantization metadata across all EP (Expert Parallel) ranks.

        When EP is enabled, each rank only holds metadata for its local experts.
        This method gathers metadata from all EP ranks and merges them so that
        every rank has complete metadata for all experts.

        For SequentialMLP structure:
        - Local expert indices are converted to global indices
        - Metadata is gathered and merged using global indices as keys
        - Non-local experts have is_local=False and module/quantizers set to None
        """
        if self.ep_size <= 1 or self.ep_group is None:
            return

        # Prepare serializable metadata info for all_gather
        # We can't send module/quantizer objects, so we extract necessary info
        local_metadata_info = {}
        for name, metadata in self.quant_metadata.items():
            # Only sync MoE expert metadata (containing "local_experts")
            if "local_experts" not in name:
                continue

            local_metadata_info[name] = {
                "qformat": metadata.qformat,
                "block_size": metadata.block_size,
                "vpp_idx": metadata.vpp_idx,
                "weight_amax": metadata.weight_amax,
                "input_amax": metadata.input_amax,
                "global_expert_idx": metadata.global_expert_idx,
                "local_expert_idx": metadata.local_expert_idx,
            }

        # Also send num_local_experts for validation
        sync_data = {
            "metadata": local_metadata_info,
            "num_local_experts": self.num_local_experts,
            "ep_rank": self.ep_rank,
        }

        # Gather metadata from all EP ranks
        all_sync_data = [None] * self.ep_size
        torch.distributed.all_gather_object(all_sync_data, sync_data, group=self.ep_group)

        # Validate that all ranks have the same num_local_experts
        for rank_idx, data in enumerate(all_sync_data):
            if data is not None and data["num_local_experts"] != self.num_local_experts:
                print(
                    f"[QAT PostProcessor] Warning: EP rank {rank_idx} has "
                    f"{data['num_local_experts']} local experts, expected {self.num_local_experts}"
                )

        # Merge metadata from all ranks
        for rank_idx, data in enumerate(all_sync_data):
            if rank_idx == self.ep_rank:
                # Skip local metadata (already have it)
                continue

            if data is None:
                continue

            rank_metadata = data["metadata"]
            for name, info in rank_metadata.items():
                if name in self.quant_metadata:
                    # Already have this metadata (shouldn't happen with proper global indices)
                    continue

                # Create metadata entry for non-local experts
                # Note: module and quantizers are not available for non-local experts
                metadata = QuantizationMetadata(
                    qformat=info["qformat"],
                    weight_quantizer=None,  # Not available for non-local
                    input_quantizer=None,  # Not available for non-local
                    module=None,  # Not available for non-local
                    vpp_idx=info["vpp_idx"],
                    block_size=info["block_size"],
                    weight_amax=info["weight_amax"],
                    input_amax=info["input_amax"],
                    is_local=False,  # Mark as non-local
                    global_expert_idx=info["global_expert_idx"],
                    local_expert_idx=info["local_expert_idx"],
                )
                self.quant_metadata[name] = metadata

        # Count local vs non-local experts
        num_local = sum(1 for m in self.quant_metadata.values() if m.is_local and m.global_expert_idx is not None)
        num_remote = sum(1 for m in self.quant_metadata.values() if not m.is_local and m.global_expert_idx is not None)

        if torch.distributed.get_rank() == 0:
            print(
                f"[QAT PostProcessor] EP metadata sync complete. "
                f"EP size: {self.ep_size}, Local expert params: {num_local}, "
                f"Remote expert params: {num_remote}, Total metadata entries: {len(self.quant_metadata)}"
            )

    def _sync_quantization_metadata_across_pp(self):
        """
        Synchronize quantization metadata across all PP (Pipeline Parallel) ranks.

        When PP is enabled, each rank only holds layers for its pipeline stage.
        This method gathers metadata from all PP ranks and merges them so that
        every rank has complete metadata for all layers.

        IMPORTANT: In Megatron's PP mode, each PP rank uses LOCAL layer indices
        (starting from 0), not global layer indices. For example:
        - PP rank 0 has decoder.layers.0 (globally layer 0)
        - PP rank 1 has decoder.layers.0 (globally layer 1)

        This method converts local layer indices to global layer indices during sync.

        For MoE SequentialMLP structure with PP:
        - Different PP ranks hold different decoder layers
        - Each PP rank builds metadata only for its local layers
        - We gather and merge metadata from all PP ranks
        - Layer indices are converted from local to global during merge
        - Non-local layers have is_local=False and module/quantizers set to None
        """
        global_rank = torch.distributed.get_rank()

        print(
            f"[QAT PostProcessor][Rank {global_rank}] PP sync starting: "
            f"pp_size={self.pp_size}, pp_rank={self.pp_rank}, pp_group={self.pp_group}, "
            f"local_metadata_count={len(self.quant_metadata)}"
        )

        if self.pp_size <= 1:
            print(f"[QAT PostProcessor][Rank {global_rank}] PP sync skipped: pp_size <= 1")
            return

        if self.pp_group is None:
            print(f"[QAT PostProcessor][Rank {global_rank}] PP sync skipped: pp_group is None")
            return

        # Verify PP group size matches expected pp_size
        actual_pp_group_size = torch.distributed.get_world_size(group=self.pp_group)
        print(
            f"[QAT PostProcessor][Rank {global_rank}] PP group size verification: "
            f"expected={self.pp_size}, actual={actual_pp_group_size}"
        )

        # Calculate number of layers per PP stage (needed for global layer index conversion)
        num_layers_per_stage = self._get_num_layers_per_pp_stage()
        print(f"[QAT PostProcessor][Rank {global_rank}] Detected {num_layers_per_stage} layers per PP stage")

        # First, convert our local metadata to use global layer indices
        # This is needed so we can properly merge with other PP ranks
        local_metadata_with_global_indices = {}
        for name, metadata in self.quant_metadata.items():
            global_name = self._convert_local_to_global_layer_name(name, self.pp_rank, num_layers_per_stage)
            local_metadata_with_global_indices[global_name] = metadata

        # Update our metadata dict to use global layer indices
        self.quant_metadata = local_metadata_with_global_indices

        # Prepare serializable metadata info for all_gather
        # We can't send module/quantizer objects, so we extract necessary info
        local_metadata_info = {}
        for name, metadata in self.quant_metadata.items():
            local_metadata_info[name] = {
                "qformat": metadata.qformat,
                "block_size": metadata.block_size,
                "vpp_idx": metadata.vpp_idx,
                "weight_amax": metadata.weight_amax,
                "input_amax": metadata.input_amax,
                "global_expert_idx": metadata.global_expert_idx,
                "local_expert_idx": metadata.local_expert_idx,
                "is_local": metadata.is_local,
            }

        # Include PP rank info and num_layers_per_stage for global index conversion
        sync_data = {
            "metadata": local_metadata_info,
            "pp_rank": self.pp_rank,
            "num_local_experts": self.num_local_experts,
            "num_layers_per_stage": num_layers_per_stage,
            "global_rank": global_rank,
        }

        print(
            f"[QAT PostProcessor][Rank {global_rank}] Preparing to sync {len(local_metadata_info)} metadata entries, "
            f"sample keys (global indices): {list(local_metadata_info.keys())[:3]}"
        )

        # Gather metadata from all PP ranks
        all_sync_data = [None] * actual_pp_group_size
        torch.distributed.all_gather_object(all_sync_data, sync_data, group=self.pp_group)

        # Debug: print what we received
        print(f"[QAT PostProcessor][Rank {global_rank}] Received data from {len(all_sync_data)} PP ranks")
        for i, data in enumerate(all_sync_data):
            if data is not None:
                sample_keys = list(data.get("metadata", {}).keys())[:2]
                print(
                    f"[QAT PostProcessor][Rank {global_rank}] PP rank {i}: "
                    f"received from global_rank={data.get('global_rank', 'unknown')}, "
                    f"pp_rank={data.get('pp_rank', 'unknown')}, "
                    f"metadata_count={len(data.get('metadata', {}))}, "
                    f"sample_keys={sample_keys}"
                )

        # Merge metadata from all PP ranks
        local_metadata_before = len(self.quant_metadata)
        for rank_idx, data in enumerate(all_sync_data):
            if data is None:
                print(f"[QAT PostProcessor][Rank {global_rank}] Skipping rank_idx={rank_idx}: data is None")
                continue

            source_pp_rank = data.get("pp_rank")

            # Skip our own data - compare by pp_rank from the data, not by index
            if source_pp_rank == self.pp_rank:
                print(
                    f"[QAT PostProcessor][Rank {global_rank}] Skipping rank_idx={rank_idx}: same pp_rank={self.pp_rank}"
                )
                continue

            rank_metadata = data["metadata"]
            added_count = 0
            skipped_existing = 0

            for name, info in rank_metadata.items():
                # The name already has global layer indices (converted by the sender)
                if name in self.quant_metadata:
                    # Already have this metadata (shouldn't happen with correct global indices)
                    existing = self.quant_metadata[name]
                    if existing.is_local:
                        skipped_existing += 1
                        continue
                    # If both are non-local, just keep existing
                    skipped_existing += 1
                    continue

                # Create metadata entry for layers from other PP ranks
                # Note: module and quantizers are not available for non-local layers
                metadata = QuantizationMetadata(
                    qformat=info["qformat"],
                    weight_quantizer=None,  # Not available for non-local PP rank
                    input_quantizer=None,  # Not available for non-local PP rank
                    module=None,  # Not available for non-local PP rank
                    vpp_idx=info["vpp_idx"],
                    block_size=info["block_size"],
                    weight_amax=info["weight_amax"],
                    input_amax=info["input_amax"],
                    is_local=False,  # Mark as non-local (from other PP rank)
                    global_expert_idx=info["global_expert_idx"],
                    local_expert_idx=info["local_expert_idx"],
                )
                self.quant_metadata[name] = metadata
                added_count += 1

            print(
                f"[QAT PostProcessor][Rank {global_rank}] From pp_rank={source_pp_rank}: "
                f"added {added_count} metadata entries, skipped {skipped_existing} existing"
            )

        # Log statistics
        metadata_added = len(self.quant_metadata) - local_metadata_before
        local_count = sum(1 for m in self.quant_metadata.values() if m.is_local)
        remote_count = sum(1 for m in self.quant_metadata.values() if not m.is_local)

        print(
            f"[QAT PostProcessor][Rank {global_rank}] PP metadata sync complete. "
            f"PP size: {self.pp_size}, PP rank: {self.pp_rank}, "
            f"Local params: {local_count}, Remote params: {remote_count}, "
            f"Metadata added from other PP ranks: {metadata_added}, "
            f"Total metadata entries: {len(self.quant_metadata)}"
        )

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
        elif metadata.weight_amax is not None:
            # Non-local expert (EP): Use synchronized amax from metadata
            weight_amax = metadata.weight_amax.to(weight.device)
            weight_scale_2 = weight_amax.float() / (6.0 * 448.0)
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
        - MoE experts: model.layers.X.mlp.experts.Y.gate_proj/up_proj/down_proj.weight
        - MoE router (gate): model.layers.X.mlp.gate.weight -> NOT quantized (returns None)
        """

        # Only process weight parameters
        if not hf_name.endswith(".weight") or hf_name.endswith("._amax") or "norm" in hf_name:
            return None

        # Check for MoE router (gate) - should NOT be quantized
        # HF formats: model.layers.X.mlp.gate.weight (Qwen)
        #             model.layers.X.block_sparse_moe.gate.weight (Mixtral)
        if self._is_moe_router(hf_name):
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
            # Check for MoE experts first
            # HF format: model.layers.X.mlp.experts.Y.gate_proj/up_proj/down_proj.weight
            # HF Mixtral format: model.layers.X.block_sparse_moe.experts.Y.w1/w2/w3.weight
            expert_match = re.search(r"\.experts\.(\d+)\.", hf_name)
            if expert_match:
                expert_id = expert_match.group(1)  # This is the global expert ID in HF format
                # MoE expert layers - use global expert ID for SequentialMLP
                if any(proj in hf_name for proj in ["gate_proj", "up_proj", "w1", "w3"]):
                    # Try TEGroupedMLP pattern first (all experts share same linear layer)
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.experts.linear_fc1.weight")
                    # Try SequentialMLP pattern with global expert index
                    mcore_patterns.append(
                        f"decoder.layers.{layer_num}.mlp.experts.local_experts.{expert_id}.linear_fc1.weight"
                    )
                elif any(proj in hf_name for proj in ["down_proj", "w2"]):
                    # Try TEGroupedMLP pattern first
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.experts.linear_fc2.weight")
                    # Try SequentialMLP pattern with global expert index
                    mcore_patterns.append(
                        f"decoder.layers.{layer_num}.mlp.experts.local_experts.{expert_id}.linear_fc2.weight"
                    )
            # Check for shared_expert (Qwen2 MoE)
            elif "shared_expert" in hf_name:
                if any(proj in hf_name for proj in ["gate_proj", "up_proj"]):
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.shared_experts.linear_fc1.weight")
                elif "down_proj" in hf_name:
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.shared_experts.linear_fc2.weight")
            else:
                # Dense MLP
                if any(proj in hf_name for proj in ["gate_proj", "up_proj"]):
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.linear_fc1.weight")
                elif "down_proj" in hf_name:
                    mcore_patterns.append(f"decoder.layers.{layer_num}.mlp.linear_fc2.weight")

        # Try to find matching metadata
        for pattern in mcore_patterns:
            if pattern in self.quant_metadata:
                return self.quant_metadata[pattern]

        # # If no exact match, try to find any metadata from the same layer
        # # This handles cases where the exact name might be slightly different
        # for mcore_name, metadata in self.quant_metadata.items():
        #     if f"layers.{layer_num}." in mcore_name:
        #         # Found a quantized module in the same layer
        #         # Skip router metadata - router should not be used for other layers
        #         if ".router." in mcore_name:
        #             continue
        #         # For QAT, if any module in the layer is quantized, all Linear layers should be
        #         if ".weight" in mcore_name:
        #             return metadata

        return None

    def _is_moe_router(self, hf_name: str) -> bool:
        """
        Check if the HF parameter name corresponds to a MoE router (gate).

        MoE router should NOT be quantized to maintain routing precision.

        Router naming patterns:
        - Qwen/Qwen2/Qwen3 MoE: model.layers.X.mlp.gate.weight
        - Mixtral: model.layers.X.block_sparse_moe.gate.weight
        - Shared expert gate (Qwen2 MoE): model.layers.X.mlp.shared_expert_gate.weight

        Note: gate_proj is NOT the router, it's part of the MLP expert.
        """

        # Pattern 1: Qwen/Qwen3 MoE router - model.layers.X.mlp.gate.weight
        # Must be exactly ".mlp.gate.weight" not ".mlp.gate_proj.weight"
        if re.search(r"\.mlp\.gate\.weight$", hf_name):
            return True

        # Pattern 2: Mixtral router - model.layers.X.block_sparse_moe.gate.weight
        if re.search(r"\.block_sparse_moe\.gate\.weight$", hf_name):
            return True

        # Pattern 3: Qwen2 MoE shared expert gate - model.layers.X.mlp.shared_expert_gate.weight
        if re.search(r"\.mlp\.shared_expert_gate\.weight$", hf_name):
            return True

        return False

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