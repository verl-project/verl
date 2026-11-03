# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Export vs engine on vLLM: the parameter that receives each synced HF weight.

The check itself is ``verl.utils.quant_sync_format_check.SyncFormatCheck``. The colocated worker extension
(``verl.workers.rollout.vllm_rollout.utils``) calls it around its quantized weight loading: ``begin_sync``
before the first bucket, ``check_bucket`` before each bucket is loaded and ``end_sync`` after the kernel
layouts are re-derived.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from verl.utils.quant_sync_format_check import SyncFormatCheck


def _vllm_receiving_module(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    """The vLLM module that receives HF weight ``name``, or ``None`` when the name does not resolve.

    Applies the model's ``hf_to_vllm_mapper``, then its ``packed_modules_mapping`` for the last
    module name (``q_proj`` -> ``qkv_proj``), and stops at a fused-MoE block, whose remaining
    per-expert parts (``experts.3.gate_proj``) its own weight loader resolves.
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is not None:
        mapped = mapper.apply_list([name])
        if not mapped:
            return None
        name = mapped[0]
    parts = name.split(".")
    path = parts[:-1] if parts[-1] in ("weight", "weight_packed") else parts
    packed = getattr(model, "packed_modules_mapping", None) or {}
    owner = {sub: fused for fused, subs in packed.items() for sub in subs}
    module = model
    for i, part in enumerate(path):
        if hasattr(module, "routed_experts") or hasattr(module, "w13_weight"):
            break
        child = getattr(module, part, None)
        if child is None and i == len(path) - 1 and part in owner:
            child = getattr(module, owner[part], None)
        if not isinstance(child, torch.nn.Module):
            return None
        module = child
    for holder in ("routed_experts", "base_layer"):  # the MoE weight holder; the layer under a LoRA wrapper
        inner = getattr(module, holder, None)
        if isinstance(inner, torch.nn.Module):
            module = inner
    return module


def resolve_vllm_param(model: torch.nn.Module, hf_name: str) -> Optional[torch.Tensor]:
    """The vLLM parameter that receives HF weight ``hf_name``, or ``None`` when the name does not resolve.

    A linear layer receives ``.weight`` in its ``weight`` and compressed-tensors' packed FP4 ``.weight_packed``
    in its ``weight_packed``; a layer the engine keeps in BF16 has only ``weight``, which then receives the
    packed weight. A fused-MoE block receives every expert's gate and up projections in ``w13_weight`` and
    its down projection in ``w2_weight``. Layers another pipeline stage owns are unresolved on this rank.
    """
    module = _vllm_receiving_module(model, hf_name)
    if module is None or type(module).__name__ == "PPMissingLayer":
        return None
    if hasattr(module, "w13_weight"):
        parts = hf_name.split(".")
        expert_parts = parts[parts.index("experts") + 1 :] if "experts" in parts else parts
        return module.w2_weight if any(p in ("down_proj", "w2") for p in expert_parts) else module.w13_weight
    candidates = ("weight_packed", "weight") if hf_name.endswith(".weight_packed") else ("weight",)
    for attr in candidates:
        param = getattr(module, attr, None)
        if isinstance(param, torch.Tensor):
            return param
    return None


def vllm_sync_format_check(model: torch.nn.Module, quant_config: Any = None) -> Optional[SyncFormatCheck]:
    """The export-vs-engine check of a vLLM model, or ``None`` when the engine quantizes what it receives.

    verl syncs weights the trainer-side exporter has already quantized into the checkpoint format of
    ``rollout.quantization`` or of the model's own ``quantization_config``. An engine started with online
    quantization (an unquantized checkpoint, ``is_checkpoint_fp8_serialized=False``) receives BF16 by design.
    """
    if getattr(quant_config, "is_checkpoint_fp8_serialized", True) is False:
        return None
    return SyncFormatCheck.of(model, "vllm", resolve_vllm_param)
