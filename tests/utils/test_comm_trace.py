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
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[2] / "verl" / "utils" / "comm_trace.py"
_SPEC = importlib.util.spec_from_file_location("comm_trace_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
comm_trace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comm_trace)


def _reload_comm_trace(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ulysses_module(monkeypatch):
    verl_package = types.ModuleType("verl")
    verl_package.__path__ = []
    utils_package = types.ModuleType("verl.utils")
    utils_package.__path__ = []
    monkeypatch.setitem(sys.modules, "verl", verl_package)
    monkeypatch.setitem(sys.modules, "verl.utils", utils_package)
    monkeypatch.setitem(sys.modules, "verl.utils.comm_trace", comm_trace)

    module_path = _MODULE_PATH.with_name("ulysses.py")
    spec = importlib.util.spec_from_file_location("ulysses_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_device_api(monkeypatch, *, vendor, device_module):
    verl_package = types.ModuleType("verl")
    verl_package.__path__ = []
    utils_package = types.ModuleType("verl.utils")
    utils_package.__path__ = []
    device_api = types.ModuleType("verl.utils.device")
    device_api.get_vendor = lambda: vendor
    device_api.get_torch_device = lambda: device_module
    monkeypatch.setitem(sys.modules, "verl", verl_package)
    monkeypatch.setitem(sys.modules, "verl.utils", utils_package)
    monkeypatch.setitem(sys.modules, "verl.utils.device", device_api)


class _FakeGroup:
    group_name = "sp-group-0"


class _UnnamedGroup:
    pass


def test_trace_is_disabled_by_default_and_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("VERL_COMM_TRACE", raising=False)
    default_module = _reload_comm_trace("comm_trace_default_under_test")
    assert default_module._COMM_TRACE_ENABLED is False

    monkeypatch.setenv("VERL_COMM_TRACE", "1")
    enabled_module = _reload_comm_trace("comm_trace_enabled_under_test")
    assert enabled_module._COMM_TRACE_ENABLED is True


def test_format_communication_range_has_stable_field_order_and_escaping():
    label = comm_trace.format_communication_range(
        "ulysses_a2a",
        requested_offset_us=500,
        direction="scatter=2|gather=1",
        step=7,
        message_bytes=4096,
    )
    assert label == (
        "verl.comm/ulysses_a2a|step=7|direction=scatter%3D2%7Cgather%3D1|message_bytes=4096|requested_offset_us=500"
    )


def test_communication_trace_context_is_nested_and_restored(monkeypatch):
    labels = []

    @contextmanager
    def fake_range(message):
        labels.append(message)
        yield

    monkeypatch.setattr(comm_trace, "_COMM_TRACE_ENABLED", True)
    monkeypatch.setattr(comm_trace, "_get_nvtx_range", lambda: fake_range)

    tensor = torch.empty(16, dtype=torch.float32)
    with comm_trace.communication_trace_context(step=3, microbatch=2, layer=11, logical_sequence_id="3/2/11"):
        with comm_trace.communication_nvtx_range(
            "ulysses_a2a", tensor=tensor, group=_FakeGroup(), direction="scatter_dim=2,gather_dim=1"
        ):
            pass

    assert labels == [
        "verl.comm/ulysses_a2a|step=3|microbatch=2|layer=11|direction=scatter_dim%3D2,gather_dim%3D1|"
        "message_bytes=64|process_group_id=sp-group-0|logical_sequence_id=3/2/11"
    ]
    assert comm_trace._COMM_CONTEXT.get() is None


def test_disabled_trace_is_a_noop(monkeypatch):
    monkeypatch.setattr(comm_trace, "_COMM_TRACE_ENABLED", False)
    monkeypatch.setattr(comm_trace, "_get_nvtx_range", lambda: None)
    with comm_trace.communication_trace_context(step=9):
        assert comm_trace._COMM_CONTEXT.get() is None
        with comm_trace.communication_nvtx_range("ulysses_a2a"):
            pass


def test_unnamed_process_group_uses_membership_not_only_size(monkeypatch):
    group = _UnnamedGroup()
    monkeypatch.setattr(comm_trace.dist, "_get_process_group_name", None, raising=False)
    monkeypatch.setattr(comm_trace.dist, "get_process_group_ranks", lambda candidate: [0, 2])
    assert comm_trace._process_group_id(group) == "ranks-0,2"


def test_nvtx_factory_uses_platform_device_abstraction(monkeypatch):
    @contextmanager
    def fake_range(_message):
        yield

    device_module = types.SimpleNamespace(is_available=lambda: True, nvtx=types.SimpleNamespace(range=fake_range))
    _install_fake_device_api(monkeypatch, vendor="nvidia", device_module=device_module)
    monkeypatch.setattr(comm_trace, "_COMM_TRACE_ENABLED", True)
    assert comm_trace._get_nvtx_range() is fake_range


def test_non_nvidia_trace_is_a_noop(monkeypatch):
    device_module = types.SimpleNamespace(is_available=lambda: pytest.fail("device probe should not run"))
    _install_fake_device_api(monkeypatch, vendor="amd", device_module=device_module)
    monkeypatch.setattr(comm_trace, "_COMM_TRACE_ENABLED", True)
    assert comm_trace._get_nvtx_range() is None
    with comm_trace.communication_nvtx_range("ulysses_a2a"):
        pass


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="unsupported communication trace fields"):
        with comm_trace.communication_trace_context(typo_field=1):
            pass


def test_explicit_process_group_id_supports_non_torch_collectives(monkeypatch):
    labels = []

    @contextmanager
    def fake_range(message):
        labels.append(message)
        yield

    monkeypatch.setattr(comm_trace, "_COMM_TRACE_ENABLED", True)
    monkeypatch.setattr(comm_trace, "_get_nvtx_range", lambda: fake_range)
    with comm_trace.communication_nvtx_range(
        "weight_sync",
        step=19,
        direction="send",
        message_bytes=1024,
        process_group_id="rollout-update",
        logical_sequence_id="bucket-2",
    ):
        pass

    assert labels[0].startswith(
        "verl.comm/weight_sync|step=19|direction=send|message_bytes=1024|process_group_id=rollout-update|"
        "logical_sequence_id=bucket-2"
    )


def test_ulysses_collectives_have_semantic_ranges(monkeypatch):
    ulysses = _load_ulysses_module(monkeypatch)
    ranges = []

    @contextmanager
    def fake_range(operation, **metadata):
        ranges.append((operation, metadata))
        yield

    group = object()
    monkeypatch.setattr(ulysses, "communication_nvtx_range", fake_range)
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda group=None: 2)

    def fake_all_to_all(output_list, input_list, **kwargs):
        for output, input_ in zip(output_list, input_list, strict=True):
            output.copy_(input_)

    def fake_all_gather_into_tensor(output, input_, **kwargs):
        output[: input_.shape[0]].copy_(input_)
        output[input_.shape[0] :].copy_(input_)

    monkeypatch.setattr(ulysses.dist, "all_to_all", fake_all_to_all)
    monkeypatch.setattr(ulysses.dist, "all_gather_into_tensor", fake_all_gather_into_tensor)

    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert torch.equal(ulysses.all_to_all_tensor(tensor, 0, 0, group), tensor)
    assert torch.equal(ulysses.all_gather_tensor(tensor, group), torch.cat((tensor, tensor)))

    assert [operation for operation, _ in ranges] == ["ulysses_a2a", "ulysses_all_gather"]
    assert ranges[0][1]["tensor"] is tensor
    assert ranges[0][1]["group"] is group
    assert ranges[0][1]["direction"] == "scatter_dim=0,gather_dim=0"
    assert set(ranges[1][1]) == {"tensor", "group"}
    assert ranges[1][1]["tensor"] is tensor
    assert ranges[1][1]["group"] is group
