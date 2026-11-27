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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch.distributed


@pytest.fixture
def distributed_module(monkeypatch):
    device = ModuleType("verl.utils.device")
    device.get_device_name = lambda: "cpu"
    device.get_nccl_backend = lambda: "gloo"
    device.get_resource_name = lambda: "CPU"
    device.get_torch_device = lambda: SimpleNamespace(set_device=lambda _: None)
    device.is_npu_available = False
    net_utils = ModuleType("verl.utils.net_utils")
    net_utils.is_ipv6 = lambda _: False
    monkeypatch.setitem(sys.modules, device.__name__, device)
    monkeypatch.setitem(sys.modules, net_utils.__name__, net_utils)

    source = Path(__file__).resolve().parents[2] / "verl" / "utils" / "distributed.py"
    spec = importlib.util.spec_from_file_location("_distributed_on_cpu", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("ray_initialized", [False, True])
def test_cpu_default_backend_is_gloo(distributed_module, monkeypatch, ray_initialized):
    calls = []

    def fake_init_process_group(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(torch.distributed, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    if ray_initialized:
        distributed_module.initialize_global_process_group_ray()
    else:
        distributed_module.initialize_global_process_group()

    assert calls[0]["backend"] == "gloo"


@pytest.mark.parametrize(
    ("device_name", "communication_backend", "expected"),
    [
        ("cuda", "nccl", "cpu:gloo,cuda:nccl"),
        ("npu", "hccl", "cpu:gloo,npu:hccl"),
    ],
)
def test_accelerator_default_backend_keeps_cpu_gloo(
    distributed_module, monkeypatch, device_name, communication_backend, expected
):
    distributed_module.get_device_name = lambda: device_name
    distributed_module.get_nccl_backend = lambda: communication_backend
    calls = []

    monkeypatch.setattr(torch.distributed, "init_process_group", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    distributed_module.initialize_global_process_group()

    assert calls[0]["backend"] == expected
