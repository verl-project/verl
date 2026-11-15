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
import asyncio

import pytest

from verl.experimental.das.contracts import TrajectoryPush
from verl.experimental.das.push_client import DASPushClient
from verl.workers.config.das import DASConfig


class FakeRemoteMethod:
    def __init__(self, calls, *, fail=False):
        self._calls = calls
        self._fail = fail

    def remote(self, worker_id, batch):
        if self._fail:
            raise RuntimeError("service down")
        self._calls.append((worker_id, list(batch)))
        return object()  # stands in for the discarded ObjectRef


class FakeService:
    def __init__(self, *, fail=False):
        self.calls = []
        self.add_trajectories = FakeRemoteMethod(self.calls, fail=fail)


def make_push(i=0):
    return TrajectoryPush(problem_id=f"p{i}", token_ids=(1, 2, 3))


def cfg(**overrides):
    return DASConfig(**overrides)


@pytest.mark.asyncio
async def test_flush_on_batch_size():
    # Async context: a timer task is scheduled, so pushes batch up to size.
    service = FakeService()
    client = DASPushClient(service, "w1", cfg(push_batch_size=3, push_flush_interval_s=60))
    for i in range(3):
        client.enqueue(make_push(i))
    assert len(service.calls) == 1
    worker_id, batch = service.calls[0]
    assert worker_id == "w1"
    assert len(batch) == 3
    assert client.pushed == 3


@pytest.mark.asyncio
async def test_flush_on_interval():
    service = FakeService()
    client = DASPushClient(service, "w1", cfg(push_batch_size=100, push_flush_interval_s=0.01))
    client.enqueue(make_push())
    assert service.calls == []
    await asyncio.sleep(0.05)
    assert len(service.calls) == 1


@pytest.mark.asyncio
async def test_overflow_drops_with_counter():
    service = FakeService()
    # batch_size > capacity so nothing auto-flushes before the timer fires.
    client = DASPushClient(service, "w1", cfg(push_batch_size=100, push_queue_capacity=2, push_flush_interval_s=60))
    for i in range(5):
        client.enqueue(make_push(i))
    assert client.dropped == 3
    assert len(client._buffer) == 2


def test_flush_errors_never_raise():
    service = FakeService(fail=True)
    client = DASPushClient(service, "w1", cfg(push_batch_size=1, push_queue_capacity=1))
    client.enqueue(make_push())
    assert client.flush_errors == 1


def test_flush_without_event_loop_is_synchronous():
    service = FakeService()
    client = DASPushClient(service, "w1", cfg(push_batch_size=100, push_flush_interval_s=60))
    # No running loop: _ensure_flusher must flush inline rather than schedule.
    client.enqueue(make_push())
    assert len(service.calls) == 1
