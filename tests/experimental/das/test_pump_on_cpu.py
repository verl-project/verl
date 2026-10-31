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
"""``DeltaPump`` against a real in-process store and a recording collective_rpc."""

import pytest

from tests.experimental.das.fakes import FakeActor, PySuffixTree
from verl.experimental.das import service as service_module
from verl.experimental.das.contracts import TrajectoryPush, deserialize_delta_batch
from verl.experimental.das.drafter_state import DASDrafterState
from verl.experimental.das.pump import DeltaPump
from verl.experimental.das.service import SuffixDataService
from verl.workers.config.das import DASConfig


class Receiver:
    """Stands in for every TP rank: applies payloads to one drafter state, records calls."""

    def __init__(self):
        self.state = DASDrafterState(DASConfig(), max_tree_depth=24, tree_factory=PySuffixTree)
        self.calls: list[tuple[str, int]] = []

    async def collective_rpc(self, method, args=(), **kwargs):
        assert method == "das_apply_deltas"
        (payload,) = args
        self.state.apply_delta_batch(payload)
        self.calls.append((method, deserialize_delta_batch(payload).to_version))
        return [self.state.state_version()]


def make(store=None, **pump_kwargs):
    store = store or SuffixDataService(DASConfig(window_iterations=4, fresh_iterations=1))
    receiver = Receiver()
    pump = DeltaPump(
        "svc", "replica-1", receiver.collective_rpc, get_actor=lambda name: FakeActor(store), **pump_kwargs
    )
    return pump, store, receiver


@pytest.mark.asyncio
async def test_nothing_to_apply_before_the_boundary():
    pump, store, receiver = make()
    store.begin_iteration(1)
    store.add_trajectories("w", [TrajectoryPush("p1", (1, 2, 3, 4))])
    assert await pump.run_once() == 0  # withheld until the iteration advances
    assert receiver.calls == []
    assert pump.version == -1


@pytest.mark.asyncio
async def test_applies_in_order_and_tracks_version():
    pump, store, receiver = make()
    store.begin_iteration(1)
    store.add_trajectories("w", [TrajectoryPush("p1", (1, 2, 3, 4))])
    store.begin_iteration(2)
    applied = await pump.run_once()
    assert applied >= 1
    assert pump.version == store.get_metrics()["das_servable_version"]
    assert [v for _, v in receiver.calls] == sorted(v for _, v in receiver.calls)
    assert receiver.state.speculate("p1", [1, 2, 3], budget=8, max_spec_factor=2.0).token_ids == [4]
    assert await pump.run_once() == 0  # caught up


@pytest.mark.asyncio
async def test_buffer_cap_bounds_prefetch_per_poll():
    store = SuffixDataService(DASConfig(window_iterations=4, fresh_iterations=0, max_delta_batch_tokens=2))
    pump, store, receiver = make(store, buffer_cap_bytes=1)
    store.begin_iteration(1)
    store.add_trajectories(
        "w", [TrajectoryPush("p1", (1, 2)), TrajectoryPush("p1", (3, 4)), TrajectoryPush("p1", (5, 6))]
    )
    store.begin_iteration(2)
    # One payload per poll under a 1-byte cap; the next poll continues from the buffered version.
    assert await pump.run_once() == 1
    assert await pump.run_once() == 1
    assert await pump.run_once() == 1
    assert await pump.run_once() == 0
    assert pump.applied_batches == 3


@pytest.mark.asyncio
async def test_store_unavailable_then_recovers():
    store = SuffixDataService(DASConfig(window_iterations=4, fresh_iterations=1))
    receiver = Receiver()
    available = {"ok": False}

    def get_actor(name):
        if not available["ok"]:
            raise ValueError("actor not found")
        return FakeActor(store)

    pump = DeltaPump("svc", "r", receiver.collective_rpc, get_actor=get_actor)
    store.begin_iteration(1)
    store.add_trajectories("w", [TrajectoryPush("p1", (1, 2, 3))])
    store.begin_iteration(2)
    assert await pump.run_once() == 0
    available["ok"] = True
    assert await pump.run_once() >= 1
    assert receiver.state.tree_stats()["problem_trees"] == 1


@pytest.mark.asyncio
async def test_snapshot_resync_after_log_truncation(monkeypatch):
    monkeypatch.setattr(service_module, "_LOG_CAPACITY", 2)
    store = SuffixDataService(DASConfig(window_iterations=8, fresh_iterations=0))
    pump, store, receiver = make(store)
    store.begin_iteration(1)
    for i in range(5):
        store.add_trajectories("w", [TrajectoryPush("p1", (i, i + 1, i + 2))])
    store.begin_iteration(2)
    assert await pump.run_once() == 1  # a single snapshot batch
    assert receiver.state.snapshots_applied == 1
    assert receiver.state.tree_stats()["problem_trees"] == 1
    assert pump.version == store.get_metrics()["das_servable_version"]


@pytest.mark.asyncio
async def test_start_and_stop_task():
    pump, store, receiver = make(poll_interval_s=0.01)
    task = pump.start()
    assert pump.start() is task  # idempotent
    store.begin_iteration(1)
    store.add_trajectories("w", [TrajectoryPush("p1", (1, 2, 3))])
    store.begin_iteration(2)
    import asyncio

    for _ in range(100):
        await asyncio.sleep(0.01)
        if pump.applied_batches:
            break
    assert pump.applied_batches >= 1
    pump.stop()
    with pytest.raises(asyncio.CancelledError):
        await task
