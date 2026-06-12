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

The tests run against a real (CPU-only) TransferQueue instance. ``ReplayBuffer``
is fully synchronous: :meth:`ReplayBuffer.sample` blocks the calling thread,
re-polling TransferQueue every ``poll_interval`` seconds until ``batch_size``
terminal (``finished``/``failure``) prompts are available.

To exercise the blocking consumer without deadlocking the test, the *producer*
side -- the rollout that feeds TransferQueue -- runs in a dedicated thread (see
:class:`RolloutProducer`). The producer mirrors the real ordering: it writes
every trajectory of a GRPO group first, and only then marks the prompt terminal,
so the consumer never observes a terminal prompt before its trajectories exist.

Each test uses a unique ``partition_id`` so that data written by one test never
leaks into another (``_sync_metadata_from_transfer_queue`` lists *all*
partitions, but ``ReplayBuffer`` tracks keys per partition).
"""

import threading
import time
import uuid
from dataclasses import dataclass, field

import pytest
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer

# Small poll interval so the blocking consumer reacts to producer writes quickly.
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
    prompt status (``finished``/``failure``/``running``/``pending``)."""

    uid: str
    status: str
    sessions: int = 1
    trajectory_keys: list[str] = field(default_factory=list)


class RolloutProducer(threading.Thread):
    """Simulates the rollout side feeding TransferQueue from a *separate thread*.

    For every spec it writes all trajectory values first and only then writes the
    prompt status. Writing the prompt status last guarantees that whenever the
    consumer observes a terminal prompt, all of its trajectories are already
    present -- avoiding a producer/consumer race.

    Uses the synchronous ``tq.kv_put`` API which is safe to call from a plain
    (non-asyncio) thread.
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


class SampleConsumer(threading.Thread):
    """Runs the blocking ``ReplayBuffer.sample`` in a background thread so the test
    can assert that it stays blocked until the producer supplies enough data."""

    def __init__(self, rb: ReplayBuffer, partition_id: str, batch_size: int):
        super().__init__(daemon=True)
        self.rb = rb
        self.partition_id = partition_id
        self.batch_size = batch_size
        self.result: KVBatchMeta | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result = self.rb.sample(self.partition_id, self.batch_size)
        except Exception as e:
            self.error = e

    def result_or_raise(self, timeout: float = 10.0) -> KVBatchMeta:
        self.join(timeout)
        assert not self.is_alive(), "SampleConsumer thread did not finish in time"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _produce(partition_id: str, specs: list[PromptSpec], delay: float = 0.0) -> RolloutProducer:
    producer = RolloutProducer(partition_id, specs, delay=delay)
    producer.start()
    return producer


def _clear_partition(partition_id: str) -> None:
    """Best-effort cleanup of every key written into a partition."""
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def _uids_of(keys: list[str]) -> set[str]:
    return {key.split("_")[0] for key in keys}


# --------------------------------------------------------------------------- #
# _sync_metadata_from_transfer_queue: classification of polled metadata.
# --------------------------------------------------------------------------- #


def test_sync_metadata_classifies_keys(tq_init, partition_id):
    """The poll splits prompts by status and collects trajectory tags."""
    pending = PromptSpec(uid=_uid(), status="pending", sessions=0)
    running = PromptSpec(uid=_uid(), status="running", sessions=1)
    finished = PromptSpec(uid=_uid(), status="finished", sessions=2)
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    _produce(partition_id, [pending, running, finished, failure]).join_and_check()

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    try:
        rb._sync_metadata_from_transfer_queue()

        assert rb.pending_keys[partition_id] == {pending.uid}
        assert rb.running_keys[partition_id] == {running.uid}
        assert rb.finished_keys[partition_id] == {finished.uid}
        assert rb.failure_keys[partition_id] == {failure.uid}

        # All trajectory keys (and only those) land in the partition value map.
        expected_traj = set(running.trajectory_keys) | set(finished.trajectory_keys) | set(failure.trajectory_keys)
        assert set(rb.partitions[partition_id].keys()) == expected_traj
        for key in expected_traj:
            assert rb.partitions[partition_id][key] == {"is_prompt": False, "seq_len": 3}
    finally:
        _clear_partition(partition_id)


def test_sync_metadata_unknown_status_raises(tq_init, partition_id):
    """An unrecognized prompt status is a hard error during the poll."""
    _produce(partition_id, [PromptSpec(uid=_uid(), status="bogus", sessions=0)]).join_and_check()

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    try:
        with pytest.raises(ValueError, match="Unknown status"):
            rb._sync_metadata_from_transfer_queue()
    finally:
        # The bogus prompt must be removed: every poll lists *all* partitions, so
        # leaving it behind would break unrelated tests.
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# sample: end-to-end against a real TransferQueue.
# --------------------------------------------------------------------------- #


def test_sample_returns_finished_and_failure_trajectories(tq_init, partition_id):
    """sample picks trajectories belonging to finished/failure prompts and clears
    the sampled prompt keys from TransferQueue."""
    finished = PromptSpec(uid=_uid(), status="finished", sessions=2)
    failure = PromptSpec(uid=_uid(), status="failure", sessions=1)
    # Running prompt's trajectory must NOT be sampled.
    running = PromptSpec(uid=_uid(), status="running", sessions=1)

    _produce(partition_id, [finished, failure, running]).join_and_check()
    expected_keys = set(finished.trajectory_keys) | set(failure.trajectory_keys)

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    try:
        batch = rb.sample(partition_id, batch_size=2)

        assert batch.partition_id == partition_id
        assert set(batch.keys) == expected_keys
        assert len(batch.tags) == len(batch.keys)

        # The two sampled prompt keys are consumed from TransferQueue; the running
        # prompt and all trajectory values remain.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert finished.uid not in remaining
        assert failure.uid not in remaining
        assert running.uid in remaining
    finally:
        _clear_partition(partition_id)


def test_sample_blocks_until_enough_then_unblocks(tq_init, partition_id):
    """sample stays blocked while fewer than batch_size prompts are ready and
    returns once the producer thread supplies the missing group."""
    # One group ready up front -> not enough for batch_size=2.
    _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    consumer = SampleConsumer(rb, partition_id, batch_size=2)
    try:
        consumer.start()

        # Give the consumer time to poll a few times; it must still be blocked.
        time.sleep(POLL_INTERVAL * 5)
        assert consumer.is_alive(), "sample returned before batch_size prompts were ready"

        # The producer thread supplies a second group; sample can now complete.
        _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

        batch = consumer.result_or_raise()
        assert len(batch.keys) == 2
    finally:
        consumer.join(timeout=10.0)
        _clear_partition(partition_id)


def test_sample_concurrent_with_streaming_producer(tq_init, partition_id):
    """sample(batch_size=N) returns as soon as a slow streaming producer has emitted
    N terminal groups, even though the consumer started waiting first."""
    batch_size = 3
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=2) for _ in range(batch_size)]

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    # Stream groups one-by-one with a delay; consumer blocks in sample() meanwhile.
    producer = _produce(partition_id, specs, delay=0.1)
    try:
        batch = rb.sample(partition_id, batch_size=batch_size)
        producer.join_and_check()

        expected_keys = {k for spec in specs for k in spec.trajectory_keys}
        assert set(batch.keys) == expected_keys
        assert len(batch.keys) == batch_size * 2
    finally:
        producer.join(timeout=10.0)
        _clear_partition(partition_id)


def test_sync_grpo_step_returns_complete_groups(tq_init, partition_id):
    """A synchronous PPO/GRPO step submits batch_size prompts, each a GRPO group of
    n sessions; one sample must return every trajectory as whole groups."""
    n_prompts = 3
    n_sessions = 4  # GRPO rollout.n
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    try:
        batch = rb.sample(partition_id, batch_size=n_prompts)

        # Every prompt's full GRPO group is present, nothing more, nothing less.
        assert len(batch.keys) == n_prompts * n_sessions
        assert set(batch.keys) == {k for spec in specs for k in spec.trajectory_keys}
        # Each sampled prompt contributes exactly n_sessions trajectories.
        per_uid: dict[str, int] = {}
        for key in batch.keys:
            per_uid[key.split("_")[0]] = per_uid.get(key.split("_")[0], 0) + 1
        assert set(per_uid.values()) == {n_sessions}
    finally:
        _clear_partition(partition_id)


def test_async_overproduction_drains_in_batches_without_duplicates(tq_init, partition_id):
    """An async rollouter over-produces; sequential samples drain the surplus
    batch_size complete groups at a time without ever re-selecting a prompt.

    ``sample`` only re-polls TransferQueue while it is under-filled, and it clears
    just the sampled *prompt* keys (trajectory values stay). The real trainer
    re-polls metadata in the gap between two ``sample`` calls; we reproduce that
    gap deterministically by re-syncing so a follow-up ``sample`` cannot re-select
    consumed prompts.
    """
    n_prompts = 5
    n_sessions = 2
    batch_size = 2
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=n_sessions) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()
    all_uids = {spec.uid for spec in specs}

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    try:
        collected_keys: list[str] = []
        consumed_uids: set[str] = set()

        # Drain 2 + 2 + 1 prompts across three samples.
        for bs in (batch_size, batch_size, n_prompts - 2 * batch_size):
            batch = rb.sample(partition_id, batch_size=bs)

            sampled_uids = _uids_of(batch.keys)
            assert len(sampled_uids) == bs
            assert len(batch.keys) == bs * n_sessions
            assert not (sampled_uids & consumed_uids), "a prompt was handed out twice"

            collected_keys.extend(batch.keys)
            consumed_uids |= sampled_uids
            # Mimic the trainer's metadata refresh between two sample() calls.
            rb._sync_metadata_from_transfer_queue()

        # The whole surplus was drained exactly once.
        assert consumed_uids == all_uids
        assert len(collected_keys) == n_prompts * n_sessions
        assert len(set(collected_keys)) == len(collected_keys)
    finally:
        _clear_partition(partition_id)


def test_async_overproduction_leaves_surplus_available(tq_init, partition_id):
    """A single sample consumes only batch_size prompts; the surplus stays in
    TransferQueue (and remains sampleable)."""
    n_prompts = 4
    batch_size = 1
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1) for _ in range(n_prompts)]
    _produce(partition_id, specs).join_and_check()

    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    try:
        batch = rb.sample(partition_id, batch_size=batch_size)
        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == batch_size

        # Surplus prompts are NOT cleared from TransferQueue.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        remaining_finished = {
            key for key, tag in remaining.items() if tag.get("is_prompt") and tag.get("status") == "finished"
        }
        assert remaining_finished == ({spec.uid for spec in specs} - sampled_uids)
        assert len(remaining_finished) == n_prompts - batch_size
    finally:
        _clear_partition(partition_id)


def test_sample_zero_batch_size_raises_on_empty_clear(tq_init, partition_id):
    """batch_size=0 selects no prompts; clearing an empty key list is rejected by
    TransferQueue, so sample surfaces a ValueError (degenerate, documented case)."""
    rb = ReplayBuffer(poll_interval=POLL_INTERVAL)
    try:
        with pytest.raises(ValueError, match="empty list"):
            rb.sample(partition_id, batch_size=0)
    finally:
        _clear_partition(partition_id)
