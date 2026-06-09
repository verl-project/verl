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
"""Unit tests for :class:`verl.trainer.ppo.v2.replay_buffer.ReplayBuffer`.

The tests run against a real (CPU-only) TransferQueue instance. Each test uses a
unique ``partition_id`` so that data written by one test never leaks into the
metadata polled by another (``async_kv_list`` returns *all* partitions, but
``ReplayBuffer`` tracks keys per partition).
"""

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq

from verl.trainer.ppo.v2.replay_buffer import ReplayBuffer

# Small poll interval so the background task reflects TransferQueue changes quickly.
POLL_INTERVAL = 0.05


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _trajectory_key(uid: str, session_id: int = 0, index: int = 0) -> str:
    return f"{uid}_{session_id}_{index}"


@dataclass
class PromptSpec:
    """One GRPO group to produce: ``sessions`` trajectories followed by a terminal
    prompt status."""

    uid: str
    status: str
    sessions: int = 1
    trajectory_keys: list[str] = field(default_factory=list)


class RolloutProducer(threading.Thread):
    """Simulates the rollout side feeding TransferQueue from a *separate thread*.

    It mirrors the real producer ordering in ``main_ppo_sync`` (``_run_prompt``):
    every trajectory of a GRPO group is written *first*, and only then is the
    prompt marked terminal (``finished``/``failure``). Writing the prompt status
    last guarantees that whenever the consumer observes a terminal prompt, all of
    its trajectories are already present -- avoiding a producer/consumer race.

    Uses the synchronous ``tq.kv_put`` API which talks to the client directly and
    is therefore safe to call from a non-asyncio thread.
    """

    def __init__(self, partition_id: str, specs: list[PromptSpec], delay: float = 0.0):
        super().__init__(daemon=True)
        self.partition_id = partition_id
        self.specs = specs
        self.delay = delay
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            for spec in self.specs:
                for session_id in range(spec.sessions):
                    key = _trajectory_key(spec.uid, session_id)
                    tq.kv_put(
                        key=key,
                        partition_id=self.partition_id,
                        fields={"input_ids": torch.tensor([1, 2, 3])},
                        tag={"is_prompt": False, "seq_len": 3},
                    )
                    spec.trajectory_keys.append(key)
                tq.kv_put(
                    key=spec.uid,
                    partition_id=self.partition_id,
                    tag={"is_prompt": True, "status": spec.status},
                )
                if self.delay:
                    time.sleep(self.delay)
        except Exception as e:  # surfaced to the test via join_and_check()
            self.error = e

    def join_and_check(self, timeout: float = 10.0) -> None:
        self.join(timeout)
        assert not self.is_alive(), "RolloutProducer thread did not finish in time"
        if self.error is not None:
            raise self.error


def _produce(partition_id: str, specs: list[PromptSpec], delay: float = 0.0) -> RolloutProducer:
    producer = RolloutProducer(partition_id, specs, delay=delay)
    producer.start()
    return producer


async def _clear_partition(partition_id: str):
    """Best-effort cleanup of every key written into a partition."""
    data = await tq.async_kv_list(partition_id=partition_id)
    keys = list(data.get(partition_id, {}).keys())
    if keys:
        await tq.async_kv_clear(partition_id=partition_id, keys=keys)


async def _make_buffer(poll_interval: float = POLL_INTERVAL) -> ReplayBuffer:
    return ReplayBuffer(poll_interval=poll_interval)


def _uids_of(keys: list[str]) -> set[str]:
    return {key.split("_")[0] for key in keys}


async def _wait_for_poll_to_drop(rb: ReplayBuffer, partition_id: str, gone_uids: set[str], timeout: float = 10.0):
    """Wait until the background poll refreshes terminal tracking so that already
    sampled prompts are no longer selectable.

    ``sample`` only clears the sampled *prompt* keys in TransferQueue; the in-memory
    ``finished_keys`` / ``failure_keys`` are refreshed by the next background poll.
    The real trainer always has a training gap between two ``sample`` calls, during
    which the poll runs; this helper reproduces that gap deterministically so a
    follow-up ``sample`` cannot re-select consumed prompts.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        live = rb.finished_keys[partition_id] | rb.failure_keys[partition_id]
        if not (gone_uids & live):
            return
        await asyncio.sleep(POLL_INTERVAL)
    raise TimeoutError("background poll did not drop sampled prompts in time")


# --------------------------------------------------------------------------- #
# _update_date: pure classification logic (no TransferQueue interaction).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_update_date_classifies_keys(tq_init):
    """_update_date splits prompts by status and collects trajectory tags."""
    rb = await _make_buffer(poll_interval=1e6)  # effectively disable background polling
    await rb.close()  # stop background task; we drive _update_date manually below

    data = {
        "train": {
            "p0": {"is_prompt": True, "status": "pending"},
            "p1": {"is_prompt": True, "status": "running"},
            "p2": {"is_prompt": True, "status": "finished"},
            "p3": {"is_prompt": True, "status": "failure"},
            "p2_0_0": {"is_prompt": False, "seq_len": 10},
            "p2_0_1": {"is_prompt": False, "seq_len": 11},
            # missing "is_prompt" key defaults to a trajectory.
            "p3_0_0": {"seq_len": 12},
        }
    }
    rb._update_date(data)

    assert rb.pending_keys["train"] == {"p0"}
    assert rb.running_keys["train"] == {"p1"}
    assert rb.finished_keys["train"] == {"p2"}
    assert rb.failure_keys["train"] == {"p3"}
    assert set(rb.partitions["train"].keys()) == {"p2_0_0", "p2_0_1", "p3_0_0"}
    # The full tag dict is retained for trajectories (including the is_prompt flag).
    assert rb.partitions["train"]["p2_0_0"] == {"is_prompt": False, "seq_len": 10}
    assert rb.partitions["train"]["p3_0_0"] == {"seq_len": 12}


@pytest.mark.asyncio
async def test_update_date_none_clears_state(tq_init):
    """Polling None (empty metadata) resets all tracking structures."""
    rb = await _make_buffer(poll_interval=1e6)
    await rb.close()

    rb._update_date({"train": {"p0": {"is_prompt": True, "status": "finished"}}})
    assert rb.finished_keys["train"] == {"p0"}

    rb._update_date(None)
    assert rb.finished_keys == {}
    assert rb.partitions == {}


@pytest.mark.asyncio
async def test_update_date_unknown_status_raises(tq_init):
    """An unrecognized prompt status is a hard error."""
    rb = await _make_buffer(poll_interval=1e6)
    await rb.close()

    with pytest.raises(ValueError, match="Unknown status"):
        rb._update_date({"train": {"p0": {"is_prompt": True, "status": "bogus"}}})


# --------------------------------------------------------------------------- #
# sample: end-to-end against a real TransferQueue.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sample_returns_finished_and_failure_trajectories(tq_init, partition_id):
    """sample picks trajectories belonging to finished/failure prompts and clears
    the sampled prompt keys from TransferQueue."""
    finished = PromptSpec(uid=_uid(), status="finished", sessions=2)
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    # Running prompt's trajectory must NOT be sampled.
    running = PromptSpec(uid=_uid(), status="running", sessions=1)

    # Producer thread writes everything; join so all data is in TransferQueue.
    producer = _produce(partition_id, [finished, failure, running])
    producer.join_and_check()

    expected_keys = set(finished.trajectory_keys) | set(failure.trajectory_keys)

    rb = await _make_buffer()
    try:
        batch = await asyncio.wait_for(rb.sample(partition_id, batch_size=2), timeout=10)

        assert batch.partition_id == partition_id
        assert set(batch.keys) == expected_keys
        assert len(batch.tags) == len(batch.keys)

        # The two sampled prompt keys are consumed from TransferQueue.
        remaining = (await tq.async_kv_list(partition_id=partition_id)).get(partition_id, {})
        assert finished.uid not in remaining
        assert failure.uid not in remaining
        assert running.uid in remaining
    finally:
        await rb.close()
        await _clear_partition(partition_id)


@pytest.mark.asyncio
async def test_sample_blocks_until_enough_then_unblocks(tq_init, partition_id):
    """sample waits while fewer than batch_size prompts are ready, and returns once
    enough finished prompts appear.

    The producer thread writes each group's trajectories before its terminal status,
    so a terminal prompt is never observed without its trajectories already present.
    """
    rb = await _make_buffer()
    try:
        # First group fully produced -> exactly one prompt ready.
        _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

        # batch_size=2 with a single ready prompt -> must stay blocked (times out).
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(rb.sample(partition_id, batch_size=2), timeout=0.5)

        # A second group arrives from the producer thread; sample can now complete.
        producer = _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)])
        batch = await asyncio.wait_for(rb.sample(partition_id, batch_size=2), timeout=10)
        producer.join_and_check()

        assert len(batch.keys) == 2
    finally:
        await rb.close()
        await _clear_partition(partition_id)


@pytest.mark.asyncio
async def test_sample_concurrent_with_streaming_producer(tq_init, partition_id):
    """sample(batch_size=N) returns as soon as a slow streaming producer has emitted
    N terminal groups, even though the consumer started waiting first."""
    rb = await _make_buffer()
    batch_size = 3
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=2) for _ in range(batch_size)]
    # Stream groups one-by-one with a delay; consumer is already waiting in sample().
    producer = _produce(partition_id, specs, delay=0.1)
    try:
        batch = await asyncio.wait_for(rb.sample(partition_id, batch_size=batch_size), timeout=15)
        producer.join_and_check()

        expected_keys = {k for spec in specs for k in spec.trajectory_keys}
        assert set(batch.keys) == expected_keys
        assert len(batch.keys) == batch_size * 2
    finally:
        await rb.close()
        await _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# Scenario: synchronous PPO/GRPO step (main_ppo_sync).
# A step submits exactly batch_size prompts, each a GRPO group of n sessions;
# once they all finish, a single sample must return every trajectory as whole
# groups (the sampling unit is the prompt, not the trajectory).
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sync_grpo_step_returns_complete_groups(tq_init, partition_id):
    n_prompts = 3
    n_sessions = 4  # GRPO rollout.n
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()

    rb = await _make_buffer()
    try:
        batch = await asyncio.wait_for(rb.sample(partition_id, batch_size=n_prompts), timeout=10)

        # Every prompt's full GRPO group is present, nothing more, nothing less.
        assert len(batch.keys) == n_prompts * n_sessions
        assert set(batch.keys) == {k for spec in specs for k in spec.trajectory_keys}
        # Each sampled prompt contributes exactly n_sessions trajectories.
        per_uid: dict[str, int] = {}
        for key in batch.keys:
            per_uid[key.split("_")[0]] = per_uid.get(key.split("_")[0], 0) + 1
        assert set(per_uid.values()) == {n_sessions}
    finally:
        await rb.close()
        await _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# Scenario: fully-async rollouter over-production.
# The rollouter keeps producing ahead of the trainer, so more finished prompts
# accumulate than a single batch. sample(batch_size) must take exactly
# batch_size *complete groups*, leave the surplus for later, and sequential
# samples must drain the surplus without ever re-selecting a consumed prompt.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_async_overproduction_drains_in_batches_without_duplicates(tq_init, partition_id):
    n_prompts = 5
    n_sessions = 2
    batch_size = 2
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()
    all_uids = {spec.uid for spec in specs}

    rb = await _make_buffer()
    try:
        collected_keys: list[str] = []
        consumed_uids: set[str] = set()

        # Drain 2 + 2 + 1 prompts across three samples.
        for bs in (batch_size, batch_size, n_prompts - 2 * batch_size):
            batch = await asyncio.wait_for(rb.sample(partition_id, batch_size=bs), timeout=10)

            sampled_uids = _uids_of(batch.keys)
            # Each sample takes exactly `bs` complete groups.
            assert len(sampled_uids) == bs
            assert len(batch.keys) == bs * n_sessions
            # No prompt is ever handed out twice.
            assert not (sampled_uids & consumed_uids)

            collected_keys.extend(batch.keys)
            consumed_uids |= sampled_uids
            await _wait_for_poll_to_drop(rb, partition_id, sampled_uids)

        # The whole surplus was drained exactly once.
        assert consumed_uids == all_uids
        assert len(collected_keys) == n_prompts * n_sessions
        assert len(set(collected_keys)) == len(collected_keys)
    finally:
        await rb.close()
        await _clear_partition(partition_id)


@pytest.mark.asyncio
async def test_async_overproduction_leaves_surplus_available(tq_init, partition_id):
    """A single sample consumes only batch_size prompts; the surplus stays in
    TransferQueue (and remains sampleable)."""
    n_prompts = 4
    batch_size = 1
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()

    rb = await _make_buffer()
    try:
        batch = await asyncio.wait_for(rb.sample(partition_id, batch_size=batch_size), timeout=10)
        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == batch_size

        # Surplus prompts are NOT cleared from TransferQueue.
        remaining = (await tq.async_kv_list(partition_id=partition_id)).get(partition_id, {})
        remaining_finished = {
            key for key, tag in remaining.items() if tag.get("is_prompt") and tag.get("status") == "finished"
        }
        assert remaining_finished == ({spec.uid for spec in specs} - sampled_uids)
        assert len(remaining_finished) == n_prompts - batch_size
    finally:
        await rb.close()
        await _clear_partition(partition_id)


@pytest.mark.asyncio
async def test_sample_zero_batch_size_raises_on_empty_clear(tq_init, partition_id):
    """batch_size=0 selects no prompts; clearing an empty key list is rejected by
    TransferQueue, so sample surfaces a ValueError (degenerate, documented case)."""
    rb = await _make_buffer()
    try:
        with pytest.raises(ValueError, match="empty list"):
            await asyncio.wait_for(rb.sample(partition_id, batch_size=0), timeout=10)
    finally:
        await rb.close()
        await _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# Background task / lifecycle error handling.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_close_propagates_background_error(tq_init, partition_id):
    """A bad prompt status surfaced by the background poll is re-raised on close."""
    _produce(partition_id, [PromptSpec(uid=_uid(), status="bogus", sessions=0)]).join_and_check()
    try:
        rb = await _make_buffer()
        # Wait for the background task to poll and fail.
        for _ in range(200):
            if rb.background_error is not None:
                break
            await asyncio.sleep(0.05)

        with pytest.raises(ValueError, match="Unknown status"):
            await rb.close()
    finally:
        await _clear_partition(partition_id)


@pytest.mark.asyncio
async def test_close_is_clean_without_errors(tq_init):
    """Closing a healthy buffer stops the background task and does not raise."""
    rb = await _make_buffer()
    await asyncio.sleep(POLL_INTERVAL * 2)  # let it poll at least once
    await rb.close()
    assert rb.stop_event.is_set()
    assert rb.background_task.done()
