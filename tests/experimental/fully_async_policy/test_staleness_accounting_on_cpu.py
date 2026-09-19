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

"""Exercise checked-in controller/queue methods with deterministic asyncio tasks.

AST loading avoids importing the GPU stack. Queue, dispatch, reset and task-helper
methods are real; sample generation is an async publisher and metrics are stubbed.
"""

import ast
import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "verl/experimental/fully_async_policy"


def _load(filename, class_name, names):
    tree = ast.parse((SOURCE / filename).read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    functions = [
        node for node in body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = dict(asyncio=asyncio, deque=deque, logger=logging.getLogger(__name__), time=time)
    program = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(program, str(SOURCE / filename), "exec"), namespace)
    return namespace


def _queue(capacity=128):
    names = ("__init__", "put_sample", "get_sample", "get_queue_size", "get_statistics", "clear_queue")
    methods = _load("message_queue.py", "MessageQueue", names)
    return type("LocalMessageQueue", (), {name: methods[name] for name in names})(None, capacity)


def _rollouter(queue, dispatched=0):
    names = ("_init_async_objects", "reset_staleness", "_should_pause_generation", "_processor_worker")
    methods = _load("fully_async_rollouter.py", "FullyAsyncRollouter", names)
    task_helpers = _load("detach_utils.py", None, ("safe_create_task", "task_exception_handler"))
    methods["safe_create_task"] = task_helpers["safe_create_task"]
    rollouter = SimpleNamespace(
        message_queue_client=queue,
        pending_queue=asyncio.Queue(),
        active_tasks=set(),
        paused=False,
        staleness_samples=dispatched,
        _total_dispatched_samples=dispatched,
        max_required_samples=70,
        max_queue_size=128,
        max_concurrent_samples=128,
        step_start_time=time.time(),
        idle_start_time=0,
        _step_generated_samples=0,
        _completed_steps=1,
        _STEP_HISTORY_SIZE=10,
        _step_samples_history=deque(maxlen=10),
        _compute_rollout_resource_utilization=lambda: 0.0,
        _record_active_count=lambda: None,
    )
    for name in names:
        setattr(rollouter, name, MethodType(methods[name], rollouter))
    rollouter._init_async_objects()

    async def generate(sample):
        await queue.put_sample(sample.sample_id)

    rollouter._process_single_sample_streaming = generate
    rollouter.create_task = task_helpers["safe_create_task"]
    return rollouter


async def _until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    pytest.fail("controlled async schedule did not reach the expected state")


async def _cleanup(*tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _completed_samples(rollouter, count):
    # Seed an already-dispatched state; use the real helper to retain done tasks.
    for index in range(count):
        rollouter.create_task(
            rollouter.message_queue_client.put_sample(index), str(index), task_set=rollouter.active_tasks
        )
    await asyncio.gather(*rollouter.active_tasks)
    assert len(rollouter.active_tasks) == count
    assert all(task.done() for task in rollouter.active_tasks)


@pytest.mark.parametrize("reap_first", [False, True])
def test_reset_counts_six_remaining_samples_independent_of_task_reaping(reap_first):
    async def run():
        queue = _queue()
        rollouter = _rollouter(queue, dispatched=70)
        await _completed_samples(rollouter, 70)
        for _ in range(64):
            await queue.get_sample()
        if reap_first:
            rollouter.active_tasks.clear()

        await rollouter.reset_staleness()
        assert rollouter.staleness_samples == 6
        assert not await rollouter._should_pause_generation()

    asyncio.run(run())


@pytest.mark.parametrize("snapshot_before_publish", [False, True])
def test_queue_snapshot_is_independent_of_delayed_put_ack(snapshot_before_publish):
    async def run():
        queue = _queue()
        requested, snapshot_taken, take_snapshot, reply = (asyncio.Event() for _ in range(4))
        publish, published, acknowledge = (asyncio.Event() for _ in range(3))

        async def snapshot(size_only=False):
            requested.set()
            await take_snapshot.wait()
            stats = await queue.get_statistics()
            snapshot_taken.set()
            await reply.wait()
            return stats["queue_size"] if size_only else stats

        async def get_size():
            return await snapshot(size_only=True)

        rollouter = _rollouter(SimpleNamespace(get_statistics=snapshot, get_queue_size=get_size), dispatched=1)

        async def generate():
            await publish.wait()
            await queue.put_sample("sample")
            published.set()
            await acknowledge.wait()  # Queue publication precedes the RPC reply.

        task = rollouter.create_task(generate(), "sample", task_set=rollouter.active_tasks)
        reset = asyncio.create_task(rollouter.reset_staleness())
        try:
            await asyncio.wait_for(requested.wait(), timeout=2)
            assert rollouter.lock.locked()
            if snapshot_before_publish:
                take_snapshot.set()
                await asyncio.wait_for(snapshot_taken.wait(), timeout=2)
            publish.set()
            await asyncio.wait_for(published.wait(), timeout=2)
            assert (await queue.get_sample())[0] == "sample"
            assert not task.done()
            take_snapshot.set()
            reply.set()
            await asyncio.wait_for(reset, timeout=2)
            # A sample consumed after the snapshot is conservatively still counted.
            assert rollouter.staleness_samples == int(snapshot_before_publish)
        finally:
            await _cleanup(reset, task)

    asyncio.run(run())


def test_reset_counts_unpublished_tasks_and_queue_backlog_once():
    async def run():
        queue = _queue()
        rollouter = _rollouter(queue, dispatched=5)
        for index in range(3):
            await queue.put_sample(index)
        for index in range(2):
            rollouter.create_task(asyncio.Event().wait(), str(index), task_set=rollouter.active_tasks)
        try:
            await rollouter.reset_staleness()
            assert rollouter.staleness_samples == 5
        finally:
            await _cleanup(*rollouter.active_tasks)

    asyncio.run(run())


def test_queue_eviction_and_clear_release_reservations():
    async def run():
        queue = _queue(capacity=2)
        rollouter = _rollouter(queue, dispatched=3)
        await _completed_samples(rollouter, 3)
        stats = await queue.get_statistics()
        assert stats["total_produced"] == 3
        assert stats["dropped_samples"] == 1
        await rollouter.reset_staleness()
        assert rollouter.staleness_samples == 2
        await queue.clear_queue()
        await rollouter.reset_staleness()
        assert rollouter.staleness_samples == 0

    asyncio.run(run())


def test_reset_during_capacity_wait_does_not_erase_next_dispatch():
    async def run():
        queue = _queue()
        rollouter = _rollouter(queue, dispatched=1)
        rollouter.max_concurrent_samples = 1
        finish_first, next_started = asyncio.Event(), asyncio.Event()

        async def first_sample():
            await finish_first.wait()
            await queue.put_sample("first")

        async def next_sample(_sample):
            next_started.set()
            await asyncio.Event().wait()

        rollouter._process_single_sample_streaming = next_sample
        first = rollouter.create_task(first_sample(), "first", task_set=rollouter.active_tasks)
        await rollouter.pending_queue.put(SimpleNamespace(sample_id="next"))
        processor = asyncio.create_task(rollouter._processor_worker())
        reset = None
        try:
            await _until(lambda: rollouter.pending_queue.empty())
            reset = asyncio.create_task(rollouter.reset_staleness())
            await asyncio.sleep(0)  # Let reset run or wait for the state lock before capacity is released.
            finish_first.set()
            await asyncio.wait_for(reset, timeout=2)
            await asyncio.wait_for(next_started.wait(), timeout=2)
            assert rollouter.staleness_samples == 2
            assert rollouter._total_dispatched_samples == 2
        finally:
            await _cleanup(processor, first, *([reset] if reset else []), *rollouter.active_tasks)

    asyncio.run(run())


def test_reset_allows_next_consumer_batch_to_fill():
    async def run():
        queue = _queue()
        rollouter = _rollouter(queue, dispatched=70)
        await _completed_samples(rollouter, 70)
        for _ in range(64):
            await queue.get_sample()
        await rollouter.reset_staleness()
        rollouter.active_tasks.clear()  # Reaping after reset cannot repair an overcount.
        for index in range(58):
            await rollouter.pending_queue.put(SimpleNamespace(sample_id=f"next-{index}"))
        await rollouter.pending_queue.put(None)
        processor = asyncio.create_task(rollouter._processor_worker())
        try:
            await _until(lambda: processor.done() or rollouter.paused)
            assert not rollouter.paused, "phantom reservations block the next batch"
            await asyncio.wait_for(processor, timeout=2)
            assert await queue.get_queue_size() == 64
            assert rollouter.staleness_samples == 64
            assert rollouter._total_dispatched_samples == 128
            for _ in range(64):
                await queue.get_sample()
            await rollouter.reset_staleness()
            assert rollouter.staleness_samples == 0
        finally:
            await _cleanup(processor, *rollouter.active_tasks)

    asyncio.run(run())


def test_pending_eof_is_not_a_sample_dispatch():
    async def run():
        rollouter = _rollouter(_queue())
        await rollouter.pending_queue.put(None)
        await rollouter._processor_worker()
        assert rollouter.staleness_samples == 0
        assert rollouter._total_dispatched_samples == 0

    asyncio.run(run())


@pytest.mark.parametrize("consume_eos", [False, True])
def test_normal_eos_does_not_leave_negative_reservations(consume_eos):
    async def run():
        queue = _queue()
        rollouter = _rollouter(queue, dispatched=1)
        await queue.put_sample("sample")
        await queue.get_sample()
        await queue.put_sample(None)
        if consume_eos:
            await queue.get_sample()
        await rollouter.reset_staleness()
        assert rollouter.staleness_samples == 0

    asyncio.run(run())


def test_reset_propagates_queue_failure():
    async def run():
        async def unavailable():
            raise RuntimeError("queue unavailable")

        rollouter = _rollouter(SimpleNamespace(get_statistics=unavailable, get_queue_size=unavailable))
        with pytest.raises(RuntimeError, match="queue unavailable"):
            await rollouter.reset_staleness()
        assert not rollouter.lock.locked()

    asyncio.run(run())


def test_cancelled_reset_releases_state_lock():
    async def run():
        requested = asyncio.Event()

        async def blocked():
            requested.set()
            await asyncio.Event().wait()

        rollouter = _rollouter(SimpleNamespace(get_statistics=blocked, get_queue_size=blocked))
        reset = asyncio.create_task(rollouter.reset_staleness())
        try:
            await asyncio.wait_for(requested.wait(), timeout=2)
            reset.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reset
            assert not rollouter.lock.locked()
        finally:
            await _cleanup(reset)

    asyncio.run(run())


def test_lifetime_dispatch_counter_is_initialized_to_zero():
    tree = ast.parse((SOURCE / "fully_async_rollouter.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullyAsyncRollouter")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    assignments = [
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_total_dispatched_samples" for target in node.targets
        )
    ]
    assert len(assignments) == 1
    instance = SimpleNamespace()
    program = ast.fix_missing_locations(ast.Module(body=assignments, type_ignores=[]))
    exec(compile(program, str(SOURCE / "fully_async_rollouter.py"), "exec"), {"self": instance})
    assert instance._total_dispatched_samples == 0
