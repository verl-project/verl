# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import torch
import torch.nn as nn

from verl.utils import fsdp_utils


def test_offload_fsdp2_model_to_cpu_uses_non_blocking_copy():
    model = Mock()

    fsdp_utils.offload_fsdp2_model_to_cpu(model, empty_cache=False)

    model.to.assert_called_once_with("cpu", non_blocking=True)


def test_load_fsdp2_model_to_gpu_uses_non_blocking_copy(monkeypatch):
    model = Mock()
    device = object()
    monkeypatch.setattr(fsdp_utils, "get_device_id", lambda: device)

    fsdp_utils.load_fsdp2_model_to_gpu(model)

    model.to.assert_called_once_with(device, non_blocking=True)


class _NoPlacementModel(nn.Module):
    _no_placement_params = ["large.weight"]

    def __init__(self):
        super().__init__()
        self.large = nn.Embedding(4, 2)
        self.small = nn.Linear(2, 2)
        self.large.weight.requires_grad_(False)


def test_no_placement_param_is_detached_during_model_move(monkeypatch):
    model = _NoPlacementModel()
    registrations = fsdp_utils.get_no_placement_param_registrations(model)
    fsdp_utils.set_no_placement_param_registrations(model, registrations)
    large_weight = model.large.weight
    small_weight = model.small.weight

    def fake_to(device, non_blocking):
        assert device is target_device
        assert non_blocking
        assert model.large.weight is None
        model.small.weight = nn.Parameter(torch.empty_like(small_weight, device="meta"))
        return model

    target_device = object()
    monkeypatch.setattr(model, "to", fake_to)
    monkeypatch.setattr(fsdp_utils, "get_device_id", lambda: target_device)

    fsdp_utils.load_fsdp2_model_to_gpu(model)

    assert model.large.weight is large_weight
    assert model.large.weight.device.type == "cpu"
    assert model.small.weight.device.type == "meta"


def test_no_placement_param_must_be_frozen():
    model = _NoPlacementModel()
    model.large.weight.requires_grad_(True)

    try:
        fsdp_utils.get_no_placement_param_registrations(model)
    except ValueError as error:
        assert "cannot train" in str(error)
    else:
        raise AssertionError("expected trainable no-placement parameter to fail")


def test_no_placement_param_is_reused_from_process_cache(monkeypatch):
    first = _NoPlacementModel()
    second = _NoPlacementModel()
    first_registrations = fsdp_utils.get_no_placement_param_registrations(first)
    second_registrations = fsdp_utils.get_no_placement_param_registrations(second)
    monkeypatch.setattr(fsdp_utils, "_NO_PLACEMENT_CACHE", {})
    monkeypatch.setattr(fsdp_utils.dist, "is_initialized", lambda: False)

    first_registrations = fsdp_utils.materialize_no_placement_params(first_registrations, cache_scope="same-checkpoint")
    second_registrations = fsdp_utils.materialize_no_placement_params(
        second_registrations, cache_scope="same-checkpoint"
    )

    assert first_registrations[0][2] is second_registrations[0][2]
    assert first.large.weight is second.large.weight
