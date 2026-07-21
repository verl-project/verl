# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""PP-local HF weight export for disaggregated P2P sync.

``bridge.export_hf_weights`` broadcasts every parameter across PP ranks and
yields the full model on each rank. For P2P, each PP stage should export and
send only the layers it owns (after TP/EP gather inside bridge mappings).

This module reuses megatron-bridge conversion tasks/mappings but skips PP
broadcast so non-owning PP ranks never materialize foreign layers.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

import torch

from verl.utils.megatron_utils import unwrap_model

logger = logging.getLogger(__name__)


def filter_local_conversion_tasks(conversion_tasks: Iterable[Any]) -> list[Any]:
    """Keep bridge tasks whose parameter tensor lives on this rank."""
    local_tasks = []
    for task in conversion_tasks:
        if task is None:
            continue
        if task.megatron_module is None or task.param_weight is None:
            continue
        local_tasks.append(task)
    return local_tasks


@contextmanager
def disable_megatron_bridge_pp_broadcast():
    """No-op PP broadcast helpers so export stays on the owning PP stage only."""
    try:
        from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
    except ImportError:
        yield
        return

    original_tensor_broadcast = MegatronParamMapping.broadcast_from_pp_rank
    original_obj_broadcast = MegatronParamMapping.broadcast_obj_from_pp_rank

    def _tensor_without_pp_broadcast(self, tensor, cache_key=None):
        return tensor

    def _obj_without_pp_broadcast(self, obj, cache_key=None):
        return obj

    MegatronParamMapping.broadcast_from_pp_rank = _tensor_without_pp_broadcast
    MegatronParamMapping.broadcast_obj_from_pp_rank = _obj_without_pp_broadcast
    try:
        yield
    finally:
        MegatronParamMapping.broadcast_from_pp_rank = original_tensor_broadcast
        MegatronParamMapping.broadcast_obj_from_pp_rank = original_obj_broadcast


def _yield_hf_pairs(weights: Iterable[Any]) -> Generator[tuple[str, torch.Tensor], None, None]:
    for item in weights:
        if isinstance(item, tuple):
            if len(item) >= 2:
                yield item[0], item[1]
            continue
        hf_name = getattr(item, "param_name", None) or getattr(item, "hf_param_name", None)
        tensor = getattr(item, "weight", None)
        if hf_name is not None and tensor is not None:
            yield hf_name, tensor


@torch.no_grad()
def export_pp_local_hf_weights(
    bridge: Any,
    megatron_model: torch.nn.Module | list[torch.nn.Module],
    *,
    merge_adapter_weights: bool = True,
    show_progress: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Export HF weights owned by this PP stage (TP/EP gather via bridge, no PP broadcast).

    All ranks in the PP group must still call ``build_conversion_tasks`` (it
    all-gathers global param names). Only local tasks are converted and yielded.
    """
    model_list = unwrap_model(megatron_model)
    if not isinstance(model_list, list):
        model_list = [model_list]

    model_bridge = bridge._model_bridge
    hf_pretrained = bridge.hf_pretrained

    conversion_tasks = bridge.get_conversion_tasks(model_list)
    local_tasks = filter_local_conversion_tasks(conversion_tasks)
    if not local_tasks:
        logger.warning("PP-local export produced zero local conversion tasks on this rank.")
        return

    with disable_megatron_bridge_pp_broadcast():
        streamed = model_bridge.stream_weights_megatron_to_hf(
            model_list,
            hf_pretrained,
            cpu=False,
            show_progress=show_progress,
            conversion_tasks=local_tasks,
            merge_adapter_weights=merge_adapter_weights,
        )
        yield from _yield_hf_pairs(streamed)
