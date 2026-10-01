# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Unit tests for the platform-plugin profiler hooks: lazy marker resolution
(verl.utils.profiler._resolve_markers) and DistProfiler's plugin tool dispatch
(verl.utils.profiler.profile.DistProfiler).
"""

from unittest import mock

import pytest

import verl.plugin.platform.platform_manager as pm
import verl.utils.profiler as profiler_pkg
from verl.plugin.platform import set_platform
from verl.utils.profiler import mark_annotate, mark_end_range, mark_start_range, marked_timer
from verl.utils.profiler.config import ProfilerConfig
from verl.utils.profiler.profile import DistProfiler


@pytest.fixture
def reset_profiler_and_platform():
    pm._current_platform = None
    profiler_pkg._mark_start_range = None
    profiler_pkg._mark_end_range = None
    profiler_pkg._mark_annotate = None
    profiler_pkg._marked_timer = None
    yield
    pm._current_platform = None
    profiler_pkg._mark_start_range = None
    profiler_pkg._mark_end_range = None
    profiler_pkg._mark_annotate = None
    profiler_pkg._marked_timer = None


def test_plugin_markers_resolved_lazily_and_forwarded(reset_profiler_and_platform):
    """profiler_markers() should be picked up on first use when nvtx/npu are unavailable,
    and the four returned callables must be the ones actually invoked."""
    plugin_start = mock.Mock(return_value="range-id")
    plugin_end = mock.Mock()
    plugin_annotate = mock.Mock(side_effect=lambda *a, **k: lambda f: f)
    plugin_timer = mock.Mock()

    platform = mock.Mock()
    platform.profiler_markers.return_value = (plugin_start, plugin_end, plugin_annotate, plugin_timer)
    set_platform(platform)

    with (
        mock.patch("verl.utils.profiler.is_nvtx_available", return_value=False),
        mock.patch("verl.utils.profiler.is_npu_available", False),
    ):
        # Not resolved until first use.
        assert profiler_pkg._mark_start_range is None

        result = mark_start_range("hello")
        mark_end_range("range-id")
        mark_annotate()(lambda: None)
        marked_timer()

    assert result == "range-id"
    plugin_start.assert_called_once_with("hello")
    plugin_end.assert_called_once_with("range-id")
    plugin_annotate.assert_called_once()
    plugin_timer.assert_called_once()
    # Resolved once and cached for subsequent calls.
    assert profiler_pkg._mark_start_range is plugin_start


def test_dist_profiler_dispatches_to_plugin_tool(reset_profiler_and_platform):
    """A tool name unknown to verl-core but recognized by the platform's dist_profiler_cls()
    must be instantiated as the DistProfiler's backend, receiving rank/config/tool_config."""

    class _FakePluginProfiler:
        def __init__(self, rank, config, tool_config, **kwargs):
            self.rank = rank
            self.config = config
            self.tool_config = tool_config
            self.started = False

        def start(self, **kwargs):
            self.started = True

        def stop(self):
            self.started = False

    platform = mock.Mock()
    platform.dist_profiler_cls.side_effect = lambda tool: _FakePluginProfiler if tool == "my_plugin_tool" else None
    set_platform(platform)

    tool_config = object()
    config = ProfilerConfig(tool="my_plugin_tool", enable=True, all_ranks=True, ranks=[], tool_config=tool_config)
    dp = DistProfiler(rank=0, config=config, tool_config=tool_config)

    assert isinstance(dp._impl, _FakePluginProfiler)
    assert dp._impl.rank == 0
    assert dp._impl.tool_config is tool_config

    dp.start()
    assert dp._impl.started is True
    dp.stop()
    assert dp._impl.started is False


def test_dist_profiler_falls_back_to_noop_when_plugin_does_not_recognize_tool(reset_profiler_and_platform):
    platform = mock.Mock()
    platform.dist_profiler_cls.return_value = None
    set_platform(platform)

    config = ProfilerConfig(tool="unknown_tool", enable=True, all_ranks=True, ranks=[])
    dp = DistProfiler(rank=0, config=config)

    # Falls back to the no-op implementation; start()/stop() must not raise.
    dp.start()
    dp.stop()
