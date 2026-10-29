# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapt VeOmni 0.1.11 uncached Qwen masks to Transformers 5.12.1.

Transformers removed the deprecated ``cache_position`` argument. Uncached
packed training uses input length and ``position_ids`` instead. Install only
in VeOmni's generated CUDA model modules, without modifying Transformers.
"""

import importlib
import importlib.metadata
import inspect
from functools import wraps

import torch


def _uncached_causal_mask_adapter(original):
    @wraps(original)
    def adapted(*args, cache_position=None, **kwargs):
        if args:
            raise ValueError("VeOmni compatibility expects keyword-only mask arguments")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("VeOmni mask compatibility supports uncached training only")
        if cache_position is not None:
            embeds = kwargs["inputs_embeds"]
            if cache_position.ndim != 1 or cache_position.numel() != embeds.shape[1]:
                raise ValueError("Uncached cache_position must span the current packed sequence")
            expected = torch.arange(embeds.shape[1], device=cache_position.device)
            # CUDA assertion avoids a host synchronization on every training forward.
            torch._assert_async(
                (cache_position == expected).all(),
                "Uncached cache_position must be contiguous and start at zero",
            )
        return original(**kwargs)

    adapted._verl_uncached_mask_compat = True
    return adapted


def install_qwen_uncached_mask_compat(model_type):
    """Adapt the released generated mask call sites, only for the tested pair.

    Qwen3 MoE 0.1.11 passes cache_position to both causal and sliding-window
    helpers. Qwen3.5 MoE uses the causal helper only. The adapters retain
    position_ids/padding and reject cached inference; Transformers is untouched.
    """
    modules = {
        "qwen3_moe": ("qwen3_moe", ("create_causal_mask", "create_sliding_window_causal_mask")),
        "qwen3_5_moe": ("qwen3_5_moe", ("create_causal_mask",)),
        "qwen3_5_moe_text": ("qwen3_5_moe", ("create_causal_mask",)),
    }
    if model_type not in modules:
        return False
    if importlib.metadata.version("veomni") != "0.1.11":
        return False
    if importlib.metadata.version("transformers") != "5.12.1":
        return False
    family, functions = modules[model_type]
    module = importlib.import_module(f"veomni.models.transformers.{family}.generated.patched_modeling_{family}_gpu")
    active = False
    for name in functions:
        original = getattr(module, name)
        if getattr(original, "_verl_uncached_mask_compat", False):
            active = True
        elif "cache_position" not in inspect.signature(original).parameters:
            setattr(module, name, _uncached_causal_mask_adapter(original))
            active = True
    return active
