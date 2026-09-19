# Copyright 2026 Individual Contributor: Lirui Luo
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

"""Run checked-in queue/wait methods with no Ray actor or model dependencies."""

import ast
import asyncio
import logging
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "verl/experimental/fully_async_policy"


def _methods(filename, class_name, names, namespace):
    tree = ast.parse((SOURCE / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [
        node for node in cls.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    program = ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[]))
    exec(compile(program, str(SOURCE / filename), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _Clock:
    def __init__(self, on_poll=None):
        self.now = 0.0
        self.polls = 0
        self.on_poll = on_poll

    def time(self):
        return self.now

    async def sleep(self, interval):
        self.now += interval
        self.polls += 1
        assert self.polls <= 10, "wait never returned after the producer finished"
        if self.on_poll:
            await self.on_poll(self.polls)
        await asyncio.sleep(0)


def _queue():
    methods = _methods(
        "message_queue.py",
        "MessageQueue",
        ("__init__", "put_sample", "get_sample", "get_queue_size"),
        dict(asyncio=asyncio, deque=deque, logger=logging.getLogger(__name__)),
    )
    return type("LocalMessageQueue", (), methods)(config=None, max_queue_size=128)


def _rollouter(queue, clock):
    methods = _methods(
        "fully_async_rollouter.py",
        "FullyAsyncRollouter",
        ("wait_for_enough_samples",),
        dict(asyncio=SimpleNamespace(sleep=clock.sleep), time=clock),
    )
    result = SimpleNamespace(message_queue_client=queue, running=True)
    result.wait_for_enough_samples = MethodType(methods["wait_for_enough_samples"], result)
    return result


@pytest.mark.parametrize("tail_size", [0, 3, 8])
def test_finished_producer_releases_wait_without_consuming_tail(tail_size):
    async def run():
        queue = _queue()
        clock = _Clock()
        rollouter = _rollouter(queue, clock)
        for value in range(tail_size):
            await queue.put_sample(value)
        await queue.put_sample(None)
        rollouter.running = False

        assert await rollouter.wait_for_enough_samples(8) == tail_size + 1
        assert clock.polls == 0
        assert await queue.get_queue_size() == tail_size + 1
        received = [(await queue.get_sample())[0] for _ in range(tail_size + 1)]
        assert received == [*range(tail_size), None]

    asyncio.run(run())


def test_producer_completion_during_wait_is_observed():
    async def run():
        queue = _queue()

        async def finish(poll):
            if poll == 1:
                await queue.put_sample("last sample")
                await queue.put_sample(None)
                rollouter.running = False

        clock = _Clock(finish)
        rollouter = _rollouter(queue, clock)
        assert await rollouter.wait_for_enough_samples(8) == 2
        assert clock.polls == 1
        assert (await queue.get_sample())[0] == "last sample"
        assert (await queue.get_sample())[0] is None

    asyncio.run(run())


def test_live_producer_still_waits_for_threshold():
    async def run():
        queue = _queue()

        async def produce(poll):
            await queue.put_sample(poll)

        clock = _Clock(produce)
        rollouter = _rollouter(queue, clock)
        assert await rollouter.wait_for_enough_samples(3) == 3
        assert clock.polls == 3
        assert rollouter.running

    asyncio.run(run())


def test_live_producer_timeout_is_preserved():
    async def run():
        rollouter = _rollouter(_queue(), _Clock())
        with pytest.raises(TimeoutError, match="Timed out waiting for 8 samples"):
            await rollouter.wait_for_enough_samples(8, poll_interval=1.0, timeout=2.0)

    asyncio.run(run())


def test_queue_error_is_not_hidden_by_completion():
    async def run():
        queue = SimpleNamespace(get_queue_size=AsyncMock(side_effect=RuntimeError("queue unavailable")))
        rollouter = _rollouter(queue, _Clock())
        rollouter.running = False
        with pytest.raises(RuntimeError, match="queue unavailable"):
            await rollouter.wait_for_enough_samples(8)

    asyncio.run(run())


def test_wait_can_be_cancelled():
    async def run():
        polled = asyncio.Event()

        async def block(_poll):
            polled.set()
            await asyncio.Event().wait()

        rollouter = _rollouter(_queue(), _Clock(block))
        task = asyncio.create_task(rollouter.wait_for_enough_samples(8))
        try:
            await asyncio.wait_for(polled.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_producer_marks_completion_after_publishing_eos():
    async def run():
        queue = _queue()
        publishing = asyncio.Event()
        release = asyncio.Event()

        async def put_eos(sample):
            assert sample is None
            publishing.set()
            await release.wait()
            return await queue.put_sample(sample)

        methods = _methods(
            "fully_async_rollouter.py",
            "FullyAsyncRollouter",
            ("_streaming_generation_main",),
            dict(asyncio=asyncio, safe_create_task=asyncio.create_task),
        )
        rollouter = SimpleNamespace(
            async_rollout_manager=object(),
            max_concurrent_samples=1,
            _feed_samples=AsyncMock(),
            _processor_worker=AsyncMock(),
            pending_queue=SimpleNamespace(join=AsyncMock()),
            message_queue_client=SimpleNamespace(put_sample=put_eos),
            lock=asyncio.Lock(),
            running=True,
        )
        task = asyncio.create_task(methods["_streaming_generation_main"](rollouter))
        try:
            await asyncio.wait_for(publishing.wait(), timeout=1)
            assert rollouter.running
            assert await queue.get_queue_size() == 0
            release.set()
            await asyncio.wait_for(task, timeout=1)
            assert not rollouter.running
            assert (await queue.get_sample())[0] is None
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
