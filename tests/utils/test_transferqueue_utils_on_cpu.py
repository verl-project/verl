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

import gc
import os

import pytest

from verl.utils import transferqueue_utils as tqu


async def _noop():
    return None


def _open_fd_count() -> int:
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("requires Linux /proc file descriptor accounting")
    return len(os.listdir("/proc/self/fd"))


def test_temp_event_loop_releases_file_descriptors():
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        before = _open_fd_count()
        for _ in range(32):
            tqu._run_async_in_temp_loop(_noop)
        leaked = _open_fd_count() - before
    finally:
        tqu._shutdown_async_bridge_runtime()
        if gc_was_enabled:
            gc.enable()
        gc.collect()

    assert leaked == 0


def test_async_bridge_loop_reused_between_calls():
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        first_thread = tqu._ASYNC_BRIDGE_THREAD
        first_loop = tqu._ASYNC_BRIDGE_LOOP

        assert first_thread is not None
        assert first_loop is not None
        assert first_thread.is_alive()
        assert not first_loop.is_closed()

        tqu._run_async_in_temp_loop(_noop)
        assert tqu._ASYNC_BRIDGE_THREAD is first_thread
        assert tqu._ASYNC_BRIDGE_LOOP is first_loop
    finally:
        tqu._shutdown_async_bridge_runtime()
