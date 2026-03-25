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

"""QAT weight exporter for Megatron-to-vLLM NVFP4 quantized weight sync."""

import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch
from modelopt.torch.export.quant_utils import (
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    get_quantization_format,
    get_weight_block_size,
    to_quantized_weight,
)
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

from verl.utils.megatron_utils import unwrap_model

# NVFP4 two-level scaling denominator: FP4_MAX (6.0) * FP8_MAX (448.0).
_NVFP4_AMAX_DENOMINATOR = 6.0 * 448.0


@dataclass
class _QuantMeta:
    """Quantization metadata for a single parameter."""

    qformat: str
    block_size: int
    weight_amax: Optional[torch.Tensor]
    input_amax: Optional[torch.Tensor] = None
    input_quantizer: Any = None


class QATWeightExporter:
    """Export QAT-trained bf16 weights as quantized weights (e.g. NVFP4)."""

    def __init__(
        self,
        actor_module: list,
        bridge: Any,
        qat_mode: str = "w4a16",
    ):
        self.qat_mode = qat_mode
        self._actor_module = actor_module

        self._registry = self._get_mapping_registry(bridge)

        from megatron.core import parallel_state as mpu

        self._pp_size = mpu.get_pipeline_model_parallel_world_size()
        self._pp_rank = mpu.get_pipeline_model_parallel_rank()
        self._pp_group = mpu.get_pipeline_model_parallel_group() if self._pp_size > 1 else None

        self._ep_size = mpu.get_expert_model_parallel_world_size()
        self._ep_rank = mpu.get_expert_model_parallel_rank() if self._ep_size > 1 else 0
        self._ep_group = mpu.get_expert_model_parallel_group() if self._ep_size > 1 else None

        self._config = self._get_model_config(actor_module)
        self._num_local_experts = self._count_local_experts(actor_module)

        self._metadata: dict[str, _QuantMeta] = {}
        self._collect_metadata(actor_module)

        if self._pp_size > 1 and self._pp_group is not None:
            self._sync_metadata(self._pp_group)
        if self._ep_size > 1 and self._ep_group is not None:
            self._sync_metadata(self._ep_group)

    def process_weights_iterator(
        self,
        per_tensor_param: Iterator[tuple[str, torch.Tensor]],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Wrap a weight iterator to apply quantization.

        For each ``(hf_name, bf16_weight)`` from the iterator, yields the
        quantized weight plus its scaling factors when the parameter is
        quantized, or the original tensor unchanged otherwise.
        """
        for hf_name, weight in per_tensor_param:
            meta = self._resolve_quant_metadata(hf_name)
            if meta is None:
                yield (hf_name, weight)
            else:
                assert meta.qformat == QUANTIZATION_NVFP4, f"Unsupported qformat: {meta.qformat}"
                yield from self._quantize_nvfp4(hf_name, weight, meta)

    @staticmethod
    def _get_mapping_registry(bridge):
        return bridge._model_bridge.mapping_registry()

    @staticmethod
    def _get_model_config(actor_module):
        model = unwrap_model(actor_module[0])
        return getattr(model, "config", None)

    @staticmethod
    def _count_local_experts(actor_module) -> int:
        indices: set[int] = set()
        for module in actor_module:
            model = unwrap_model(module)
            for name, _ in model.named_modules():
                m = re.search(r"local_experts\.(\d+)", name)
                if m:
                    indices.add(int(m.group(1)))
        return max(indices) + 1 if indices else 0

    def _collect_metadata(self, actor_module: list) -> None:
        for vpp_idx, module in enumerate(actor_module):
            model = unwrap_model(module)
            for name, submodule in model.named_modules():
                qformat = get_quantization_format(submodule)
                if qformat == QUANTIZATION_NONE:
                    continue
                block_size = get_weight_block_size(submodule)
                if block_size == 0:
                    continue

                w_q = getattr(submodule, "weight_quantizer", None)
                i_q = getattr(submodule, "input_quantizer", None)
                w_amax = w_q._amax.clone().cpu() if w_q and getattr(w_q, "_amax", None) is not None else None
                i_amax = i_q._amax.clone().cpu() if i_q and getattr(i_q, "_amax", None) is not None else None

                meta = _QuantMeta(
                    qformat=qformat,
                    block_size=block_size,
                    weight_amax=w_amax,
                    input_amax=i_amax,
                    input_quantizer=i_q,
                )

                for pname, _ in submodule.named_parameters(recurse=False):
                    full_name = f"{name}.{pname}" if name else pname
                    global_name = self._local_to_global_param_name(full_name, vpp_idx)
                    self._metadata[global_name] = meta

    def _local_to_global_param_name(self, name: str, vpp_idx: int) -> str:
        if self._pp_size > 1 and "layers." in name and self._config is not None:
            from megatron.bridge.models.conversion.model_bridge import (
                _megatron_local_name_to_global,
            )

            name = _megatron_local_name_to_global(self._actor_module, self._config, name, vpp_idx)

        # SequentialMLP ``local_experts.{idx}`` needs manual global conversion;
        # TEGroupedMLP is already handled by ``_megatron_local_name_to_global``.
        if self._ep_size > 1 and self._num_local_experts > 0:
            m = re.search(r"local_experts\.(\d+)\.", name)
            if m:
                local_idx = int(m.group(1))
                global_idx = self._ep_rank * self._num_local_experts + local_idx
                name = name.replace(
                    f"local_experts.{local_idx}.",
                    f"local_experts.{global_idx}.",
                    1,
                )

        return name

    def _sync_metadata(self, group) -> None:
        world_size = torch.distributed.get_world_size(group=group)

        local_info = {
            name: {
                "qformat": m.qformat,
                "block_size": m.block_size,
                "weight_amax": m.weight_amax,
                "input_amax": m.input_amax,
            }
            for name, m in self._metadata.items()
        }

        gathered: list[dict | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_info, group=group)

        for rank_info in gathered:
            if rank_info is None:
                continue
            for name, info in rank_info.items():
                if name in self._metadata:
                    continue
                self._metadata[name] = _QuantMeta(
                    qformat=info["qformat"],
                    block_size=info["block_size"],
                    weight_amax=info["weight_amax"],
                    input_amax=info["input_amax"],
                    input_quantizer=None,
                )

    def _resolve_quant_metadata(self, hf_name: str) -> Optional[_QuantMeta]:
        if not hf_name.endswith(".weight") or "norm" in hf_name:
            return None

        for resolved in _iter_hf_to_megatron_matches(self._registry, hf_name):
            meta = self._metadata.get(resolved.megatron_param)
            if meta is not None:
                return meta

        return None

    def _quantize_nvfp4(
        self,
        name: str,
        weight: torch.Tensor,
        meta: _QuantMeta,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """NVFP4 two-level quantization.

        Produces up to four tensors:
          ``(name, packed_uint8_weight)``
          ``(weight_scale, per_block_fp8_scale)``
          ``(weight_scale_2, global_scale_from_amax)``
          ``(input_scale, activation_scale)`` -- only when available
        """
        w_amax = meta.weight_amax.to(weight.device)
        w_scale_2 = w_amax.float() / _NVFP4_AMAX_DENOMINATOR

        w_scale = NVFP4QTensor.get_weights_scaling_factor(
            weight,
            meta.block_size,
            weights_scaling_factor_2=w_scale_2.to(weight.device),
        )[0]

        quantized = to_quantized_weight(weight, w_scale, meta.qformat, w_scale_2, meta.block_size)

        yield (name, quantized)
        yield (_derive_scale_name(name, "weight_scale"), w_scale)
        yield (_derive_scale_name(name, "weight_scale_2"), w_scale_2)

        input_scale = _compute_input_scale(meta)
        if input_scale is not None:
            yield (_derive_scale_name(name, "input_scale"), input_scale)


def _iter_hf_to_megatron_matches(registry, hf_name: str):
    """Yield all resolved mappings whose HF pattern matches *hf_name*."""
    for pattern_info, mapping in registry._reverse_patterns:
        if isinstance(mapping.hf_param, str):
            pattern = pattern_info
            if pattern is None:
                if mapping.hf_param == hf_name:
                    yield mapping
            else:
                match = pattern.match(hf_name)
                if match:
                    yield mapping.resolve(match.groups())
        else:
            patterns_dict = pattern_info
            for key, pattern in patterns_dict.items():
                if pattern is None:
                    if mapping.hf_param[key] == hf_name:
                        yield mapping.resolve(())
                else:
                    match = pattern.match(hf_name)
                    if match:
                        yield mapping.resolve(match.groups())


def _derive_scale_name(weight_name: str, suffix: str) -> str:
    result = weight_name.replace(".weight", f".{suffix}")
    return result if result != weight_name else f"{weight_name}_{suffix}"


def _compute_input_scale(meta: _QuantMeta) -> Optional[torch.Tensor]:
    if meta.input_quantizer is not None:
        if hasattr(NVFP4QTensor, "get_activation_scaling_factor"):
            return NVFP4QTensor.get_activation_scaling_factor(meta.input_quantizer)
        if hasattr(meta.input_quantizer, "_amax") and meta.input_quantizer._amax is not None:
            return meta.input_quantizer._amax.float() / _NVFP4_AMAX_DENOMINATOR

    if meta.input_amax is not None:
        return meta.input_amax.float() / _NVFP4_AMAX_DENOMINATOR

    return None
