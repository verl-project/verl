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

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from verl.utils.megatron_pp_local_export import (
    disable_megatron_bridge_pp_broadcast,
    export_pp_local_hf_weights,
    filter_local_conversion_tasks,
)


def test_filter_local_conversion_tasks_keeps_only_local_params():
    local = SimpleNamespace(megatron_module=object(), param_weight=torch.ones(2))
    remote = SimpleNamespace(megatron_module=None, param_weight=None)
    placeholder = None

    assert filter_local_conversion_tasks([local, remote, placeholder]) == [local]


def test_disable_megatron_bridge_pp_broadcast_skips_collectives(monkeypatch: pytest.MonkeyPatch):
    param_mapping_module = types.ModuleType("megatron.bridge.models.conversion.param_mapping")

    class DummyMapping:
        def broadcast_from_pp_rank(self, tensor, cache_key=None):
            raise AssertionError("PP tensor broadcast should be disabled")

        def broadcast_obj_from_pp_rank(self, obj, cache_key=None):
            raise AssertionError("PP object broadcast should be disabled")

    param_mapping_module.MegatronParamMapping = DummyMapping
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.bridge", types.ModuleType("megatron.bridge"))
    monkeypatch.setitem(sys.modules, "megatron.bridge.models", types.ModuleType("megatron.bridge.models"))
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.models.conversion",
        types.ModuleType("megatron.bridge.models.conversion"),
    )
    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.models.conversion.param_mapping",
        param_mapping_module,
    )

    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    mapping = MegatronParamMapping()
    with disable_megatron_bridge_pp_broadcast():
        assert torch.equal(mapping.broadcast_from_pp_rank(torch.tensor([1.0])), torch.tensor([1.0]))
        assert mapping.broadcast_obj_from_pp_rank({"x": 1}) == {"x": 1}


def test_export_pp_local_hf_weights_uses_local_tasks_only(monkeypatch: pytest.MonkeyPatch):
    local_task = SimpleNamespace(megatron_module=object(), param_weight=torch.ones(2))
    remote_task = SimpleNamespace(megatron_module=None, param_weight=None)

    model_bridge = MagicMock()
    model_bridge.stream_weights_megatron_to_hf.return_value = [("layer.weight", torch.ones(3))]

    bridge = MagicMock()
    bridge.hf_pretrained = object()
    bridge._model_bridge = model_bridge
    bridge.get_conversion_tasks.return_value = [local_task, remote_task]

    monkeypatch.setattr(
        "verl.utils.megatron_pp_local_export.disable_megatron_bridge_pp_broadcast",
        lambda: _nullcontext(),
    )

    weights = list(export_pp_local_hf_weights(bridge, [MagicMock()]))
    assert len(weights) == 1
    assert weights[0][0] == "layer.weight"
    assert torch.equal(weights[0][1], torch.ones(3))

    model_bridge.stream_weights_megatron_to_hf.assert_called_once()
    _, kwargs = model_bridge.stream_weights_megatron_to_hf.call_args
    assert kwargs["conversion_tasks"] == [local_task]


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
