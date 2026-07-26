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

"""QAT weight exporter for Megatron-to-vLLM FP4 quantized weight sync."""

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Iterator, Optional

import torch
from modelopt.torch.export.quant_utils import (
    QUANTIZATION_MXFP4,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    get_quantization_format,
    get_weight_block_size,
    to_quantized_weight,
)
from modelopt.torch.quantization.qtensor.mxfp4_tensor import MXFP4QTensor
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
        qat_config: Any = "w4a16",
    ):
        if isinstance(qat_config, str):
            self.qat_mode = qat_config
            self._block_size = 32 if qat_config == "mxfp4" else 16
            self._ignore_patterns = []
            self._use_modelopt_fake_quant = True
        else:
            self.qat_mode = getattr(qat_config, "mode", "w4a16")
            self._block_size = getattr(qat_config, "group_size", 32 if self.qat_mode == "mxfp4" else 16)
            self._ignore_patterns = list(getattr(qat_config, "ignore_patterns", []))
            self._use_modelopt_fake_quant = getattr(qat_config, "apply_modelopt_fake_quant", True)
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
            if "_quantizer." in hf_name:
                continue
            meta = self._resolve_quant_metadata(hf_name)
            if meta is None:
                yield (hf_name, weight)
            elif meta.qformat == QUANTIZATION_NVFP4:
                yield from self._quantize_nvfp4(hf_name, weight, meta)
            elif meta.qformat == QUANTIZATION_MXFP4:
                yield from self._quantize_mxfp4(hf_name, weight, meta)
            else:
                raise ValueError(f"Unsupported qformat: {meta.qformat}")

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
        if self._config is None:
            return name

        from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global

        return _megatron_local_name_to_global(self._actor_module, self._config, name, vpp_idx)

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

        if not self._use_modelopt_fake_quant and not self._is_ignored(hf_name):
            qformat = QUANTIZATION_MXFP4 if self.qat_mode == "mxfp4" else QUANTIZATION_NVFP4
            return _QuantMeta(qformat=qformat, block_size=self._block_size, weight_amax=None)

        return None

    def _is_ignored(self, hf_name: str) -> bool:
        module_name = hf_name.removesuffix(".weight")
        for pattern in self._ignore_patterns:
            if pattern.startswith("re:"):
                if re.search(pattern[3:], module_name):
                    return True
            elif pattern in module_name or fnmatch(module_name, pattern):
                return True
        return False

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
        w_amax = weight.detach().abs().amax() if meta.weight_amax is None else meta.weight_amax.to(weight.device)
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

    def _quantize_mxfp4(
        self,
        name: str,
        weight: torch.Tensor,
        meta: _QuantMeta,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """OCP MXFP4 quantization with one E8M0 scale per 32 values.

        Packing is always along the input (last) dimension. This preserves the
        projection layout consumed by vLLM's fused MoE loader:
        gate/up projections concatenate into ``w13`` while down projections
        populate ``w2``.
        """
        if meta.block_size != 32:
            raise ValueError(f"MXFP4 requires block size 32, got {meta.block_size}")
        if weight.shape[-1] % meta.block_size != 0:
            raise ValueError(
                f"MXFP4 input dimension must be divisible by block size 32, got shape {tuple(weight.shape)}"
            )

        _, weight_scale = MXFP4QTensor.quantize(weight, meta.block_size)
        scale_shape = (*weight.shape[:-1], weight.shape[-1] // meta.block_size)
        weight_scale = weight_scale.reshape(scale_shape)
        quantized = to_quantized_weight(
            weight,
            weight_scale,
            meta.qformat,
            block_size=meta.block_size,
        )

        yield (name, quantized)
        yield (_derive_scale_name(name, "weight_scale"), weight_scale)


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
