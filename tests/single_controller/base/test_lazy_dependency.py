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

"""RemoteCall arguments implicitly order dependent WorkerGroup submissions."""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.runtime.workers import EchoWorker
from verl.runtime import RemoteCall, RPCTimeoutError


@pytest.mark.parametrize(
    ("upstream_timeout", "expected"),
    [(30.0, "would-block"), (0.01, "expired")],
    ids=["live-dependency", "expired-dependency"],
)
def test_result_in_event_loop_reports_would_block_or_the_expired_deadline(
    runtime, cpu_pool, upstream_timeout, expected
):
    """Synchronous ``.result()`` inside a loop refuses to block, but an expired
    dependency deadline wins over that refusal."""
    producer = runtime.create_worker_group(EchoWorker, on=cpu_pool)
    consumer = runtime.create_worker_group(EchoWorker, on=cpu_pool)
    upstream = producer.sleep_then(0.5, 7, timeout=upstream_timeout)
    downstream = consumer.record_later(upstream, timeout=30.0)

    async def observe():
        if expected == "expired":
            await asyncio.sleep(0.02)
            with pytest.raises(RPCTimeoutError):
                downstream.result()
            return None
        with pytest.raises(RuntimeError, match="active event loop"):
            downstream.result()
        # A refused observation is non-terminal: awaiting still succeeds.
        return await downstream

    assert asyncio.run(observe()) == (None if expected == "expired" else [[7, 8], [7, 8]])


def test_repeated_dependency_call_is_submitted_once(runtime, cpu_pool):
    producer = runtime.create_worker_group(EchoWorker, on=cpu_pool)
    consumer = runtime.create_worker_group(EchoWorker, on=cpu_pool)
    upstream = producer.sleep_then(0.05, 7, timeout=30.0)
    downstream = consumer.record_later(upstream, timeout=30.0)

    async def observe_twice():
        return await RemoteCall.gather([downstream, downstream])

    result = [[7, 8], [7, 8]]
    assert asyncio.run(observe_twice()) == [result, result]
    assert consumer.get_seen() == [[[7, 8]], [[7, 8]]]


def test_dependency_failure_settles_nonblocking_call_without_submitting_target(runtime, cpu_pool):
    producer = runtime.create_worker_group(EchoWorker, on=cpu_pool)
    consumer = runtime.create_worker_group(EchoWorker, on=cpu_pool)

    failed = producer.raise_async_error("dep-fail")
    downstream = consumer.record_later(failed, timeout=30.0)

    assert isinstance(downstream, RemoteCall)
    with pytest.raises(RuntimeError, match="dep-fail"):
        downstream.result()
    assert consumer.get_seen() == [[], []]


def test_concrete_dependency_failure_and_sibling_timeout_are_equals(runtime, cpu_pool):
    from verl.single_controller.base.errors import ExceptionGroup, RPCTimeoutError

    producer = runtime.create_worker_group(EchoWorker, on=cpu_pool.slice(0))
    consumer = runtime.create_worker_group(EchoWorker, on=cpu_pool.slice(0))
    pending = producer.sleep_then(0.5, 1, timeout=0.01)
    failed = producer.raise_async_error("dep-concrete-fail")

    with pytest.raises(ExceptionGroup) as caught:
        consumer.record([pending, failed], timeout=30.0)
    types = {type(error) for error in caught.value.exceptions}
    messages = {str(error) for error in caught.value.exceptions}
    assert RuntimeError in types
    assert RPCTimeoutError in types
    assert any("dep-concrete-fail" in message for message in messages)


def test_downstream_timeout_bounds_aggregate_dependency_observation(runtime, cpu_pool):
    producer = runtime.create_worker_group(EchoWorker, on=cpu_pool.slice(0))
    consumer = runtime.create_worker_group(EchoWorker, on=cpu_pool.slice(0))
    aggregate = RemoteCall.gather([producer.sleep_then(1.0, 5)])

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        consumer.record(aggregate, timeout=0.05)
    assert time.monotonic() - started < 0.5

    assert aggregate.result() == [[5]]
