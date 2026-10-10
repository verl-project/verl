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
"""Check cleanup when training or NPU profiling raises an exception."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from verl.utils.profiler.config import NPUToolConfig, ProfilerConfig


@pytest.fixture
def mstx_module(monkeypatch):
    npu = ModuleType("torch_npu.npu")
    npu.mstx = SimpleNamespace()
    torch_npu = ModuleType("torch_npu")
    torch_npu.npu = npu
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setitem(sys.modules, "torch_npu.npu", npu)
    path = Path(__file__).resolve().parents[2] / "verl/utils/profiler/mstx_profile.py"
    spec = importlib.util.spec_from_file_location("verl.utils.profiler._mstx_cleanup_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "mark_start_range", Mock(return_value=42))
    monkeypatch.setattr(module, "mark_end_range", Mock())
    monkeypatch.setattr(module, "get_npu_profiler", Mock(return_value=Mock()))
    return module


def _profiler(module, discrete):
    return module.NPUProfiler(0, ProfilerConfig(tool="npu", enable=True), NPUToolConfig(discrete=discrete))


@pytest.mark.parametrize("discrete", [False, True])
def test_stage_error_keeps_exception_and_closes_scope(mstx_module, discrete):
    failure = RuntimeError("training stage failed")
    body = Mock(side_effect=failure)
    profiled = _profiler(mstx_module, discrete).annotate(role="update_actor")(body)
    with pytest.raises(RuntimeError) as raised:
        profiled()
    assert raised.value is failure
    body.assert_called_once()
    mstx_module.mark_end_range.assert_called_once_with(42)
    if discrete:
        backend = mstx_module.get_npu_profiler.return_value
        backend.step.assert_called_once()
        backend.stop.assert_called_once()
    else:
        mstx_module.get_npu_profiler.assert_not_called()


def test_timer_error_closes_range(mstx_module):
    failure = RuntimeError("timed operation failed")
    with pytest.raises(RuntimeError) as raised, mstx_module.marked_timer("forward", {}):
        raise failure
    assert raised.value is failure
    mstx_module.mark_end_range.assert_called_once_with(42)


@pytest.mark.parametrize("discrete", [False, True])
def test_step_error_still_stops_backend(mstx_module, discrete):
    profiler = _profiler(mstx_module, discrete)
    backend = mstx_module.get_npu_profiler.return_value
    failure = RuntimeError("profiler step failed")
    backend.step.side_effect = failure
    if discrete:
        call = profiler.annotate()(lambda: "result")
    else:
        profiler.start()
        call = profiler.stop
    with pytest.raises(RuntimeError) as raised:
        call()
    assert raised.value is failure
    backend.stop.assert_called_once()
    assert mstx_module.NPUProfiler._define_count == 0


@pytest.mark.parametrize("marker", ["mark_start_range", "mark_end_range"])
def test_marker_error_stops_discrete_profiler(mstx_module, marker):
    failure = RuntimeError("marker failed")
    getattr(mstx_module, marker).side_effect = failure
    profiled = _profiler(mstx_module, True).annotate()(lambda: "result")
    with pytest.raises(RuntimeError) as raised:
        profiled()
    assert raised.value is failure
    mstx_module.get_npu_profiler.return_value.stop.assert_called_once()
