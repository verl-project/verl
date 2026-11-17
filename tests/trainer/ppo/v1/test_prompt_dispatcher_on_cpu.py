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
"""PromptDispatcher against a real (CPU-only) TransferQueue, with fake agent loop workers and update agents."""

import asyncio
import uuid

import pytest
import torch
import transfer_queue as tq

from verl.trainer.ppo.v1 import prompt_dispatcher
from verl.trainer.ppo.v1.prompt_dispatcher import PromptDispatcher
from verl.utils import tensordict_utils as tu


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id(monkeypatch):
    # The dispatcher calls tq.init() as a Ray actor would; the module fixture already did.
    monkeypatch.setattr(prompt_dispatcher.tq, "init", lambda *args, **kwargs: None)
    return f"test-{uuid.uuid4().hex}"


class _Remote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _Worker:
    """Records the uids and global_steps of every batch it is asked to generate."""

    def __init__(self):
        self.batches = []
        self.generate_sequences = _Remote(
            lambda batch: self.batches.append((list(batch["uid"]), tu.get_non_tensor_data(batch, "global_steps", None)))
        )

    @property
    def uids(self):
        return [uid for uids, _ in self.batches for uid in uids]


class _Agent:
    def __init__(self, version=0, failed=False):
        self.version, self.failed = version, failed
        self.get_state = _Remote(lambda: {"version": self.version, "failed": self.failed})


def _enqueue(partition_id, global_steps, min_weight_version, n=1):
    uids = [uuid.uuid4().hex for _ in range(n)]
    batch = tu.get_tensordict({"uid": uids, "index": torch.arange(n)})
    tag = {
        "is_prompt": True,
        "status": "pending",
        "global_steps": global_steps,
        "min_weight_version": min_weight_version,
    }
    tq.kv_batch_put(keys=uids, partition_id=partition_id, tags=[dict(tag) for _ in uids], fields=batch)
    return uids


def _set_status(partition_id, uids, status):
    tq.kv_batch_put(keys=uids, partition_id=partition_id, tags=[{"status": status} for _ in uids])


def _dispatcher(partition_id, workers, agents, **kwargs):
    return PromptDispatcher(workers, agents, partition_id=partition_id, poll_interval=0.01, **kwargs)


def test_waits_until_every_replica_has_the_required_version(tq_init, partition_id):
    async def main():
        workers, agents = [_Worker()], [_Agent(version=1), _Agent(version=0)]
        dispatcher = _dispatcher(partition_id, workers, agents)
        old = _enqueue(partition_id, global_steps=1, min_weight_version=0)
        new = _enqueue(partition_id, global_steps=2, min_weight_version=1)

        assert await dispatcher.dispatch_once() == 1
        assert workers[0].uids == old
        assert dispatcher.get_state()["waiting_for_version"] == 1

        # Polling again must not start the claimed prompt twice, even though it is still pending.
        assert await dispatcher.dispatch_once() == 0

        agents[1].version = 1
        assert await dispatcher.dispatch_once() == 1
        assert workers[0].uids == old + new

    asyncio.run(main())


def test_caps_inflight_and_claims_oldest_first(tq_init, partition_id):
    async def main():
        workers = [_Worker()]
        dispatcher = _dispatcher(partition_id, workers, [_Agent(version=5)], max_inflight_prompts=2)
        late = _enqueue(partition_id, global_steps=3, min_weight_version=0, n=2)
        early = _enqueue(partition_id, global_steps=1, min_weight_version=0, n=2)

        assert await dispatcher.dispatch_once() == 2
        assert sorted(workers[0].uids) == sorted(early)
        assert workers[0].batches[0][1] == 1, "the batch carries the steps the prompts were enqueued at"

        # Running prompts still count toward the cap; finished ones free a slot.
        _set_status(partition_id, early, "running")
        assert await dispatcher.dispatch_once() == 0
        _set_status(partition_id, early[:1], "finished")
        assert await dispatcher.dispatch_once() == 1
        assert workers[0].uids[-1] in late
        assert dispatcher.get_state()["inflight"] == 2

    asyncio.run(main())


def test_spreads_batches_over_workers(tq_init, partition_id):
    async def main():
        workers = [_Worker(), _Worker(), _Worker()]
        dispatcher = _dispatcher(partition_id, workers, [_Agent()])
        for step in range(4):
            _enqueue(partition_id, global_steps=step, min_weight_version=0)
            await dispatcher.dispatch_once()
        # One prompt per poll, rotating so the first worker does not get them all.
        assert [len(worker.uids) for worker in workers] == [2, 1, 1]

    asyncio.run(main())


def test_failed_update_agent_stops_dispatch(tq_init, partition_id):
    async def main():
        dispatcher = _dispatcher(partition_id, [_Worker()], [_Agent(), _Agent(failed=True)])
        _enqueue(partition_id, global_steps=1, min_weight_version=0)
        task = asyncio.create_task(dispatcher.run())
        with pytest.raises(RuntimeError, match="weight update agent failed"):
            await asyncio.wait_for(task, timeout=5)
        assert dispatcher.get_state()["failed"]

    asyncio.run(main())
