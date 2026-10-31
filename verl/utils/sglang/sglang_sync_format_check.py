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
"""Export vs engine on SGLang: the parameter that receives each synced HF weight.

The check itself is ``verl.utils.quant_sync_format_check.SyncFormatCheck``. The MXFP8/MXFP4/NVFP4 refit loader
(``verl.utils.sglang.sglang_quant_utils.load_and_reprocess``) calls it: ``begin_sync`` on the first request of a
sync, ``check_bucket`` before each bucket is loaded and ``end_sync`` after the kernel layouts are re-derived.
Formats SGLang loads without that loader (FP8 blockwise) are not checked.
"""

from __future__ import annotations

from typing import Optional

import torch

from verl.utils.quant_sync_format_check import SyncFormatCheck

# HF projections SGLang fuses into one module (the model's stacked_params_mapping, restated).
_STACKED = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}
_W2_LEAVES = ("down_proj", "w2")
_PARAM_LEAVES = ("weight", "weight_packed")


def resolve_sglang_param(model: torch.nn.Module, hf_name: str) -> Optional[torch.Tensor]:
    """The SGLang parameter that receives HF weight ``hf_name``, or ``None`` when the name does not resolve.

    Walks the module tree by name, trying the fused module name for projections SGLang stacks
    (``q_proj`` -> ``qkv_proj``, ``gate_proj`` -> ``gate_up_proj``) and stopping at a fused-MoE block
    (a module holding ``w13_weight``), where the remaining per-expert parts select ``w13`` or ``w2``.
    A packed FP4 ``.weight_packed`` lands in the module's ``weight_packed``, or in ``weight`` on a layer
    the engine keeps in BF16.
    """
    parts = hf_name.split(".")
    leaf = parts[-1] if parts[-1] in _PARAM_LEAVES else None
    module_path = parts[:-1] if leaf else parts
    cur: object = model
    for i, part in enumerate(module_path):
        if hasattr(cur, "w13_weight"):
            return cur.w2_weight if any(r in _W2_LEAVES for r in module_path[i:]) else cur.w13_weight
        nxt = getattr(cur, part, None)
        if nxt is None and part in _STACKED:
            nxt = getattr(cur, _STACKED[part], None)
        if nxt is None:
            return None
        cur = nxt
    if hasattr(cur, "w13_weight"):
        return cur.w2_weight if parts[-1] in _W2_LEAVES else cur.w13_weight
    if isinstance(cur, torch.nn.Module):
        for attr in ("weight_packed", "weight") if leaf == "weight_packed" else ("weight",):
            param = getattr(cur, attr, None)
            if isinstance(param, torch.Tensor):
                return param
        return None
    return cur if isinstance(cur, torch.Tensor) else None


def sglang_sync_format_check(model: torch.nn.Module) -> SyncFormatCheck:
    """The export-vs-engine check of an SGLang model."""
    return SyncFormatCheck.of(model, "sglang", resolve_sglang_param)
