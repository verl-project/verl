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
"""RemoteCall deadline, wait/gather, and await behavior."""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.runtime.workers import EchoWorker
from verl.runtime import ClassWithInitArgs, RemoteCall, RPCTimeoutError


def test_result_uses_submit_deadline_and_is_non_terminal_on_timeout(runtime, cpu_pool):
    wg = runtime.create_worker_group(EchoWorker, on=cpu_pool)
    call = wg.sleep_then(1.0, 7, timeout=0.05)
    assert isinstance(call, RemoteCall)
    with pytest.raises(RPCTimeoutError):
        call.result()
    # Timeout does not terminalize the call; a later zero-time observation may
    # still see the completed backend result.
    stop = time.monotonic() + 15.0
    while True:
        try:
            assert call.result() == [7, 8]
            break
        except RPCTimeoutError:
            if time.monotonic() >= stop:
                pytest.fail("backend work did not become observable after timeout")
            time.sleep(0.05)

    calls = [wg.rank(0).sleep_then(0.5, value, timeout=0.01) for value in (1, 2)]
    with pytest.raises(RPCTimeoutError, match="0 of 2"):
        RemoteCall.wait(calls, count=2).result()


def test_gather_preserves_input_order(runtime, cpu_pool):
    one_rank = cpu_pool.slice(0)
    slow_wg = runtime.create_worker_group(
        ClassWithInitArgs(EchoWorker, **{"base": 0}),
        on=one_rank,
    )
    fast_wg = runtime.create_worker_group(
        ClassWithInitArgs(EchoWorker, **{"base": 0}),
        on=one_rank,
    )
    slow = slow_wg.sleep_then(2.0, 1, timeout=10.0)
    fast = fast_wg.ping_async(2, timeout=10.0)

    done, remaining = RemoteCall.wait([slow, fast], count=1).result()
    assert [call.result() for call in done] == [[2]]
    assert len(remaining) == 1

    # A larger count waits for more of them, still in input order.
    done_both, remaining_none = RemoteCall.wait([slow, fast], count=2).result()
    assert len(done_both) == 2
    assert remaining_none == []

    values = RemoteCall.gather([slow, fast]).result()
    assert values == [[1], [2]]


def test_async_wait_and_gather_use_backend_awaitables(runtime, cpu_pool):
    wg = runtime.create_worker_group(EchoWorker, on=cpu_pool.slice(0))
    calls = [wg.ping_async(1), wg.ping_async(2)]

    async def consume():
        empty = RemoteCall.gather([])
        assert empty.done()
        assert empty.result() == await empty == []
        waiting = RemoteCall.wait(calls, count=1)
        with pytest.raises(RuntimeError, match="cannot run in an active event loop"):
            waiting.result()
        done, remaining = await waiting
        assert waiting.result() == (done, remaining)
        assert len(done) == 1
        assert len(done) + len(remaining) == 2
        values = await RemoteCall.gather(calls)
        repeated = await RemoteCall.gather([calls[0], calls[0]])
        return values, repeated

    assert asyncio.run(consume()) == ([[1], [2]], [[1], [1]])


@pytest.mark.parametrize("collect", [RemoteCall.gather, RemoteCall.wait], ids=["gather", "wait"])
def test_cancelled_aggregate_observation_can_be_retried(runtime, cpu_pool, collect):
    wg = runtime.create_worker_group(EchoWorker, on=cpu_pool.slice(0))
    call = wg.sleep_then(0.2, 9)
    aggregate = collect([call])

    async def consume():
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(aggregate, timeout=0.01)
        assert not aggregate.done()
        return await aggregate

    expected = [[9]] if collect is RemoteCall.gather else ([call], [])
    assert asyncio.run(consume()) == expected


def test_ray_sync_gather_remains_usable_from_an_active_loop(ray_only_runtime):
    wg = ray_only_runtime.create_worker_group(
        EchoWorker,
        on=ray_only_runtime.create_resource_pool(nnodes=1, processes_per_node=2, device_type="cpu"),
    )
    calls = wg.execute_all_async("raw_add", 4)

    async def consume():
        return RemoteCall.gather(calls).result()

    assert asyncio.run(consume()) == [4, 5]
