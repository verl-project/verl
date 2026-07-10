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

from verl.experimental.fully_async_policy import fully_async_rollouter


class _RolloutSample:
    sample_id = "test_sample"


@pytest.mark.asyncio
async def test_processor_releases_lock_while_waiting_for_capacity(monkeypatch):
    rollouter_class = fully_async_rollouter.FullyAsyncRollouter.__ray_metadata__.modified_class
    rollouter = object.__new__(rollouter_class)
    rollouter.paused = False
    rollouter.pending_queue = asyncio.Queue()
    rollouter.staleness_samples = 0
    rollouter.max_concurrent_samples = 1
    rollouter.lock = asyncio.Lock()
    rollouter._resume_event = asyncio.Event()
    rollouter._resume_event.set()

    async def should_not_pause():
        return False

    sample_processed = asyncio.Event()

    async def process_sample(_sample):
        sample_processed.set()

    rollouter._should_pause_generation = should_not_pause
    rollouter._process_single_sample_streaming = process_sample

    blocker_release = asyncio.Event()
    blocker_task = asyncio.create_task(blocker_release.wait())
    rollouter.active_tasks = {blocker_task}
    await rollouter.pending_queue.put(_RolloutSample())
    await rollouter.pending_queue.put(None)

    wait_started = asyncio.Event()
    original_wait = asyncio.wait

    async def observed_wait(*args, **kwargs):
        wait_started.set()
        return await original_wait(*args, **kwargs)

    monkeypatch.setattr(fully_async_rollouter.asyncio, "wait", observed_wait)
    processor_task = asyncio.create_task(rollouter._processor_worker())

    try:
        await asyncio.wait_for(wait_started.wait(), timeout=1)
        await asyncio.wait_for(rollouter.lock.acquire(), timeout=0.5)
        rollouter.lock.release()
    finally:
        blocker_release.set()
        await asyncio.gather(blocker_task, return_exceptions=True)

        if not processor_task.done():
            try:
                await asyncio.wait_for(processor_task, timeout=1)
            except TimeoutError:
                processor_task.cancel()
                await asyncio.gather(processor_task, return_exceptions=True)

    assert sample_processed.is_set()
