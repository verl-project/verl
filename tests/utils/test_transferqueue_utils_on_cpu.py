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


class _FakeMeta:
    """Duck-typed stand-in for transfer_queue.BatchMeta (only what _async_meta_to_realdata reads)."""

    def __init__(self, size, field_names):
        self.size = size
        self.field_names = field_names
        self.extra_info = {"temperature": 1.0}


class _FakeClient:
    def __init__(self, tensordict):
        self._tensordict = tensordict

    async def async_get_data(self, meta):
        return self._tensordict


class _FakeTQ:
    def __init__(self, tensordict):
        self._client = _FakeClient(tensordict)

    def get_client(self):
        return self._client


def test_meta_to_realdata_raises_when_tq_returns_no_samples(monkeypatch):
    # TransferQueue answers an empty TensorDict (with only a warning) when the controller never
    # registered the samples' fields, e.g. after a missed data-status update ACK.
    import torch
    from tensordict import TensorDict

    monkeypatch.setattr(tqu, "tq", _FakeTQ(TensorDict({}, batch_size=0)))
    with pytest.raises(RuntimeError, match="TQ_DATA_UPDATE_RESPONSE_TIMEOUT"):
        tqu._meta_to_realdata(_FakeMeta(size=4, field_names=[]))

    # A short read is caught too, not just an empty one.
    monkeypatch.setattr(tqu, "tq", _FakeTQ(TensorDict({"loss_mask": torch.ones(3, 2)}, batch_size=3)))
    with pytest.raises(RuntimeError, match="returned 3 samples for a BatchMeta of size 4"):
        tqu._meta_to_realdata(_FakeMeta(size=4, field_names=["loss_mask"]))
    tqu._shutdown_async_bridge_runtime()


def test_meta_to_realdata_passes_complete_batch(monkeypatch):
    import torch
    from tensordict import TensorDict

    monkeypatch.setattr(tqu, "tq", _FakeTQ(TensorDict({"loss_mask": torch.ones(4, 2)}, batch_size=4)))
    try:
        data = tqu._meta_to_realdata(_FakeMeta(size=4, field_names=["loss_mask"]))
    finally:
        tqu._shutdown_async_bridge_runtime()
    assert data.batch_size[0] == 4
    assert "loss_mask" in data.keys()
