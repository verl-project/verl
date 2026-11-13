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
"""Unit tests for lazy profiler-marker resolution (verl.utils.profiler._resolve_markers)."""

from unittest import mock

import pytest

import verl.plugin.platform.platform_manager as pm
import verl.utils.profiler as profiler_pkg
from verl.plugin.platform import set_platform
from verl.utils.profiler import mark_annotate, mark_end_range, mark_start_range, marked_timer


@pytest.fixture
def reset_profiler_and_platform():
    def _clear():
        pm._current_platform = None
        profiler_pkg._mark_start_range = None
        profiler_pkg._mark_end_range = None
        profiler_pkg._mark_annotate = None
        profiler_pkg._marked_timer = None

    _clear()
    yield
    _clear()


def _platform_with_markers():
    """A platform supplying all four markers, plus the mocks it returns."""
    markers = (
        mock.Mock(return_value="range-id"),
        mock.Mock(),
        mock.Mock(side_effect=lambda *a, **k: lambda f: f),
        mock.Mock(),
    )
    platform = mock.Mock()
    platform.profiler_markers.return_value = markers
    return platform, markers


def test_platform_markers_win_over_an_importable_nvtx(reset_profiler_and_platform):
    """nvtx ships in the verl-core extra, so is_nvtx_available() can be True on a device that
    cannot use it. A platform that supplies its own markers must not be shadowed by that."""
    platform, (plugin_start, plugin_end, plugin_annotate, plugin_timer) = _platform_with_markers()
    set_platform(platform)

    with mock.patch("verl.utils.profiler.is_nvtx_available", return_value=True):
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
    # Resolved once, then cached for subsequent calls.
    assert profiler_pkg._mark_start_range is plugin_start
    platform.profiler_markers.assert_called_once()


def test_nvtx_still_used_when_platform_supplies_no_markers(reset_profiler_and_platform):
    """CUDA/NPU behaviour is unchanged: PlatformBase.profiler_markers() returns None there."""
    pytest.importorskip("nvtx", reason="nvtx_profile imports the nvtx package at module level")

    platform = mock.Mock()
    platform.profiler_markers.return_value = None
    set_platform(platform)

    with mock.patch("verl.utils.profiler.is_nvtx_available", return_value=True):
        profiler_pkg._resolve_markers()

    from verl.utils.profiler import nvtx_profile

    assert profiler_pkg._mark_start_range is nvtx_profile.mark_start_range
    assert profiler_pkg._marked_timer is nvtx_profile.marked_timer


def test_npu_falls_through_to_mstx(reset_profiler_and_platform):
    """NPU without nvtx keeps using mstx markers."""
    pytest.importorskip("torch_npu", reason="mstx_profile imports torch_npu at module level")

    platform = mock.Mock()
    platform.profiler_markers.return_value = None
    set_platform(platform)

    with (
        mock.patch("verl.utils.profiler.is_nvtx_available", return_value=False),
        mock.patch("verl.utils.profiler.is_npu_available", True),
    ):
        profiler_pkg._resolve_markers()

    from verl.utils.profiler import mstx_profile

    assert profiler_pkg._mark_start_range is mstx_profile.mark_start_range


def test_generic_fallback_when_nothing_applies(reset_profiler_and_platform):
    """No platform markers, no nvtx, no npu -> the pure-Python no-op markers."""
    platform = mock.Mock()
    platform.profiler_markers.return_value = None
    set_platform(platform)

    with (
        mock.patch("verl.utils.profiler.is_nvtx_available", return_value=False),
        mock.patch("verl.utils.profiler.is_npu_available", False),
    ):
        profiler_pkg._resolve_markers()

    from verl.utils.profiler import performance, profile

    assert profiler_pkg._mark_start_range is profile.mark_start_range
    assert profiler_pkg._marked_timer is performance.marked_timer
