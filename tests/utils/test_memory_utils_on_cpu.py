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

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from verl.utils import memory_utils


class _FakeDevice:
    def __init__(self):
        self._allocated = iter((20 * 1024**2, 12 * 1024**2))
        self._reserved = iter((24 * 1024**2, 24 * 1024**2))

    def is_available(self):
        return True

    def memory_allocated(self):
        return next(self._allocated)

    def memory_reserved(self):
        return next(self._reserved)


@pytest.mark.parametrize(
    ("gc_setting", "expected_result", "expected_calls"),
    [
        (True, 7, [call()]),
        (False, 0, []),
        (0, 7, [call(0)]),
        (1, 7, [call(1)]),
    ],
)
def test_collect_garbage_honors_setting(monkeypatch, gc_setting, expected_result, expected_calls):
    collect = Mock(return_value=7)
    monkeypatch.setattr(memory_utils.gc, "collect", collect)

    assert memory_utils.collect_garbage(gc_setting) == expected_result
    assert collect.call_args_list == expected_calls


def test_collect_garbage_reports_diagnostics(monkeypatch, capsys):
    gc_stats = iter(
        (
            [
                {"collections": 1, "collected": 2, "uncollectable": 0},
                {"collections": 3, "collected": 4, "uncollectable": 0},
            ],
            [
                {"collections": 1, "collected": 2, "uncollectable": 0},
                {"collections": 4, "collected": 11, "uncollectable": 1},
            ],
        )
    )
    rss = iter((10 * 1024**2, 8 * 1024**2))
    wall_times = iter((10.0, 10.25))
    cpu_times = iter((5.0, 5.2))
    device = _FakeDevice()
    process = Mock()
    print_spy = Mock(wraps=print)
    process.memory_info.side_effect = [SimpleNamespace(rss=next(rss)), SimpleNamespace(rss=next(rss))]
    monkeypatch.setattr(memory_utils.gc, "get_stats", lambda: next(gc_stats))
    monkeypatch.setattr(memory_utils.gc, "collect", lambda: 7)
    monkeypatch.setattr(memory_utils.psutil, "Process", lambda: process)
    monkeypatch.setattr(memory_utils, "get_torch_device", lambda: device)
    monkeypatch.setattr(memory_utils.torch.distributed, "is_available", lambda: False)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(memory_utils.time, "perf_counter", lambda: next(wall_times))
    monkeypatch.setattr(memory_utils.time, "thread_time", lambda: next(cpu_times))
    monkeypatch.setattr("builtins.print", print_spy)

    assert memory_utils.collect_garbage(True, diagnostics_point="test_point") == 7

    line = capsys.readouterr().out.strip()
    assert line.startswith("[gc_diagnostics] point=test_point rank=3 generation=full")
    assert "wall_ms=250.000 thread_cpu_ms=200.000 collected=7 uncollectable=1" in line
    assert "rss_delta_mib=-2.000" in line
    assert "cuda_allocated_delta_mib=-8.000" in line
    assert "cuda_reserved_delta_mib=0.000" in line
    assert "generation_1_collections=1" in line
    assert "generation_1_collected=7" in line
    assert "generation_1_uncollectable=1" in line
    print_spy.assert_called_once()
    assert print_spy.call_args.kwargs == {"flush": True}


def test_collect_garbage_diagnostics_forwards_generation(monkeypatch, capsys):
    collect = Mock(return_value=0)
    process = Mock()
    process.memory_info.return_value = SimpleNamespace(rss=0)
    monkeypatch.setattr(memory_utils.gc, "collect", collect)
    monkeypatch.setattr(memory_utils.gc, "get_stats", lambda: [])
    monkeypatch.setattr(memory_utils.psutil, "Process", lambda: process)
    monkeypatch.setattr(memory_utils, "get_torch_device", lambda: Mock(is_available=lambda: False))
    monkeypatch.setattr(memory_utils.torch.distributed, "is_available", lambda: False)

    memory_utils.collect_garbage(1, diagnostics_point="test_point")

    collect.assert_called_once_with(1)
    assert "generation=1" in capsys.readouterr().out


def test_collect_garbage_diagnostics_can_be_disabled(monkeypatch, capsys):
    collect = Mock()
    monkeypatch.setattr(memory_utils.gc, "collect", collect)

    assert memory_utils.collect_garbage(False, diagnostics_point="test_point") == 0

    collect.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", [-1, 1.5, "0", None])
def test_validate_gc_setting_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="custom_gc must be a boolean or non-negative integer"):
        memory_utils.validate_gc_setting(value, name="custom_gc")


@pytest.mark.parametrize(
    "gc_setting",
    [True, False, 1],
)
def test_aggressive_empty_cache_forwards_gc_configuration(monkeypatch, gc_setting):
    device = Mock()
    device.is_available.return_value = True
    # No memory is freed, so aggressive_empty_cache stops after one attempt.
    device.memory_reserved.return_value = 2 * 1024**3
    device.memory_allocated.return_value = 1024**3
    collect = Mock()
    monkeypatch.setattr(memory_utils, "get_torch_device", lambda: device)
    monkeypatch.setattr(memory_utils, "collect_garbage", collect)

    memory_utils.aggressive_empty_cache(gc_setting=gc_setting, gc_diagnostics_point="test_point")

    collect.assert_called_once_with(gc_setting, diagnostics_point="test_point")
    device.empty_cache.assert_called_once_with()
    device.synchronize.assert_called_once_with()
