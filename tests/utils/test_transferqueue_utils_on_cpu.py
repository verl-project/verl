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

from verl.utils.transferqueue_utils import _run_async_in_temp_loop


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
    try:
        before = _open_fd_count()
        for _ in range(32):
            _run_async_in_temp_loop(_noop)
        leaked = _open_fd_count() - before
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()

    assert leaked == 0
