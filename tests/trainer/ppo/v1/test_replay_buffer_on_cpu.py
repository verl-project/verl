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
"""Unit tests for :class:`verl.trainer.ppo.v1.replay_buffer.ReplayBuffer`.

The tests run against a real (CPU-only) TransferQueue instance. ``ReplayBuffer``
is fully synchronous: :meth:`ReplayBuffer.sample` blocks the calling thread,
re-polling TransferQueue every ``poll_interval`` seconds until enough terminal
(``finished``/``failure``) prompts are available, then returns a
``(KVBatchMeta, metrics)`` tuple.

To exercise the blocking consumer without deadlocking the test, the *producer*
side -- the rollout that feeds TransferQueue -- runs in a dedicated thread (see
:class:`RolloutProducer`). The producer mirrors the real ordering: it writes
every trajectory of a GRPO group first, and only then marks the prompt terminal,
so the consumer never observes a terminal prompt before its trajectories exist.

Each test uses a unique ``partition_id`` so that data written by one test never
leaks into another (``_sync_metadata_from_transfer_queue`` lists *all*
partitions, but ``ReplayBuffer`` tracks keys per partition).

### Off-policy control

``global_steps`` is the model weight version (one weight sync per global step, with
``parameter_sync_step`` local updates performed inside a step), so ``ReplayBuffer``
measures a trajectory's staleness (number of model versions it spans) directly as
``(global_steps - prompt_global_steps + 1)`` and supports two strategies once a
trajectory crosses ``max_off_policy_threshold``:

- ``drop``: train eagerly, dropping any sampled trajectory strictly above the
  threshold (``staleness > threshold``) and reporting drop metrics.
- ``wait``: never drop -- block sampling while any in-flight (pending/running)
  prompt has reached the threshold (``staleness >= threshold``).
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


def _make_rb(
    *,
    max_off_policy_threshold: int = 8,
    max_off_policy_strategy: str = "drop",
    poll_interval: float = POLL_INTERVAL,
    refill_fn=None,
) -> ReplayBuffer:
    """Construct a ReplayBuffer with test-friendly defaults.

    Defaults (drop strategy, threshold 8) make the off-policy filter a no-op for
    the generic tests that ``sample`` at ``global_steps=0`` over freshly produced
    trajectories. ``refill_fn`` mirrors the trainer-injected drop-and-refill hook.
    """
    return ReplayBuffer(
        trainer_mode="sync",
        trainer_config={},
        max_off_policy_threshold=max_off_policy_threshold,
        max_off_policy_strategy=max_off_policy_strategy,
        sampler_kwargs={},
        poll_interval=poll_interval,
        refill_fn=refill_fn,
    )


class FakeRefiller:
    """Stand-in for the trainer's refill hook: on ``refill(k)``, produces ``k`` fresh finished
    prompts into TransferQueue so the blocking ``sample`` can progress. Records calls for asserts.
    """

    def __init__(self, partition_id: str, global_steps: int, sessions: int = 1):
        self.partition_id = partition_id
        self.global_steps = global_steps
        self.sessions = sessions
        self.calls: list[int] = []
        self.produced_uids: list[str] = []

    def __call__(self, num_prompts: int) -> int:
        self.calls.append(num_prompts)
        specs = [
            PromptSpec(uid=_uid(), status="finished", sessions=self.sessions, global_steps=self.global_steps)
            for _ in range(num_prompts)
        ]
        # Synchronous produce (already-terminal) so the sample loop sees them on its next poll.
        _produce(self.partition_id, specs).join_and_check()
        self.produced_uids.extend(spec.uid for spec in specs)
        return num_prompts


def _sample(rb: ReplayBuffer, partition_id: str, batch_size: int, global_steps: int = 0) -> KVBatchMeta:
    """Call ``sample`` and return just the batch (dropping the metrics dict)."""
    batch, _metrics = rb.sample(global_steps=global_steps, partition_id=partition_id, batch_size=batch_size)
    return batch


def _uid() -> str:
    # uid must not contain "_" because ReplayBuffer derives it via key.split("_")[0].
    return uuid.uuid4().hex


def _trajectory_key(uid: str, session_id: int = 0, index: int = 0) -> str:
    return f"{uid}_{session_id}_{index}"


def _set_prompt_status(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Transition an existing prompt to a new status (e.g. running -> finished).

    Mirrors the rollout side flipping a GRPO group's status once it terminates.
    The prompt key is rewritten in place; its trajectory values are untouched.
    """
    tq.kv_clear(keys=[uid], partition_id=partition_id)
    tq.kv_put(
        key=uid,
        partition_id=partition_id,
        tag={"is_prompt": True, "status": status, "global_steps": global_steps},
    )


@dataclass
class PromptSpec:
    """One GRPO group to produce: ``sessions`` trajectories followed by a terminal
    prompt status (``finished``/``failure``/``running``/``pending``)."""

    uid: str
    status: str
    sessions: int = 1
    global_steps: int = 0
    trajectory_keys: list[str] = field(default_factory=list)


class RolloutProducer(threading.Thread):
    """Simulates the rollout side feeding TransferQueue from a *separate thread*.

    For every spec it writes all trajectory values first and only then writes the
    prompt status. Writing the prompt status last guarantees that whenever the
    consumer observes a terminal prompt, all of its trajectories are already
    present -- avoiding a producer/consumer race.

    Trajectory tags carry ``global_steps`` (the dataloader dispatch step) exactly
    like the real producer (see ``agent_loop_tq.py``); ``ReplayBuffer`` reads it
    to decide which trajectories to drop.

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
                        tag={"is_prompt": False, "seq_len": 3, "global_steps": spec.global_steps},
                    )
                    spec.trajectory_keys.append(key)
                tq.kv_put(
                    key=spec.uid,
                    partition_id=self.partition_id,
                    tag={"is_prompt": True, "status": spec.status, "global_steps": spec.global_steps},
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

    def __init__(self, rb: ReplayBuffer, partition_id: str, batch_size: int, global_steps: int = 0):
        super().__init__(daemon=True)
        self.rb = rb
        self.partition_id = partition_id
        self.batch_size = batch_size
        self.global_steps = global_steps
        self.result: KVBatchMeta | None = None
        self.metrics: dict | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            self.result, self.metrics = self.rb.sample(
                global_steps=self.global_steps,
                partition_id=self.partition_id,
                batch_size=self.batch_size,
            )
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
# __init__: configuration validation.
# --------------------------------------------------------------------------- #


def test_init_rejects_non_positive_threshold():
    """max_off_policy_threshold must be a positive integer."""
    with pytest.raises(AssertionError, match="max off policy threshold"):
        _make_rb(max_off_policy_threshold=0)


def test_init_rejects_unknown_strategy():
    """max_off_policy_strategy must be one of {drop, wait}."""
    with pytest.raises(AssertionError, match="max off policy strategy"):
        _make_rb(max_off_policy_strategy="bogus")


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

    rb = _make_rb()
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
            assert rb.partitions[partition_id][key] == {"is_prompt": False, "seq_len": 3, "global_steps": 0}
    finally:
        _clear_partition(partition_id)


def test_sync_metadata_unknown_status_raises(tq_init, partition_id):
    """An unrecognized prompt status is a hard error during the poll."""
    _produce(partition_id, [PromptSpec(uid=_uid(), status="bogus", sessions=0)]).join_and_check()

    rb = _make_rb()
    try:
        with pytest.raises(ValueError, match="Unknown status"):
            rb._sync_metadata_from_transfer_queue()
    finally:
        # The bogus prompt must be removed: every poll lists *all* partitions, so
        # leaving it behind would break unrelated tests.
        _clear_partition(partition_id)


def test_sync_metadata_records_prompt_global_steps(tq_init, partition_id):
    """The poll records each prompt's ``global_steps`` tag for staleness ordering."""
    a = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=7)
    b = PromptSpec(uid=_uid(), status="failure", sessions=1, global_steps=3)
    _produce(partition_id, [a, b]).join_and_check()

    rb = _make_rb()
    try:
        rb._sync_metadata_from_transfer_queue()

        assert rb.prompt_global_steps[partition_id] == {a.uid: 7, b.uid: 3}
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# _has_enough_samples: gating logic for the two strategies (pure, no TransferQueue).
# --------------------------------------------------------------------------- #


def test_has_enough_samples_drop_ignores_inflight():
    """drop ignores in-flight prompts when deciding whether enough samples are available."""
    rb = _make_rb(max_off_policy_strategy="drop", max_off_policy_threshold=2)
    pid = "p"
    rb.finished_keys[pid] |= {"a", "b"}
    # A very stale in-flight prompt must not influence the drop gate.
    rb.running_keys[pid] |= {"c"}
    rb.prompt_global_steps[pid]["c"] = 0

    assert rb._has_enough_samples(1000, pid, batch_size=2) is True
    assert rb._has_enough_samples(1000, pid, batch_size=3) is False


def test_has_enough_samples_drop_counts_only_fresh_terminal():
    """Stale terminal prompts do not satisfy the drop strategy's sample-count gate."""
    rb = _make_rb(max_off_policy_strategy="drop", max_off_policy_threshold=1)
    pid = "p"
    rb.finished_keys[pid] |= {"stale", "fresh"}
    rb.prompt_global_steps[pid] = {"stale": 0, "fresh": 5}

    assert rb._has_enough_samples(global_steps=5, partition_id=pid, batch_size=1) is True
    assert rb._has_enough_samples(global_steps=5, partition_id=pid, batch_size=2) is False


def test_has_enough_samples_wait_blocks_on_stale_inflight():
    """wait blocks while any in-flight prompt has reached the staleness threshold."""
    rb = _make_rb(max_off_policy_strategy="wait", max_off_policy_threshold=2)
    pid = "p"
    rb.finished_keys[pid] |= {"a", "b"}
    rb.running_keys[pid] |= {"c"}
    rb.prompt_global_steps[pid]["c"] = 0

    # staleness = (g - 0 + 1); >= 2 (threshold) exactly at g == 1.
    assert rb._has_enough_samples(global_steps=1, partition_id=pid, batch_size=2) is False
    # g == 0 -> staleness 1 < 2 -> in-flight is fresh, terminal count suffices.
    assert rb._has_enough_samples(global_steps=0, partition_id=pid, batch_size=2) is True
    # wait still needs the terminal count even when nothing is stale.
    assert rb._has_enough_samples(global_steps=0, partition_id=pid, batch_size=5) is False


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

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=2)

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


def test_sample_prioritizes_smallest_global_steps(tq_init, partition_id):
    """When more prompts are ready than ``batch_size``, sample must hand out the
    oldest ones first (smallest ``global_steps``), leaving the newer surplus."""
    # Produced out of step order on purpose; selection must follow global_steps.
    oldest = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=1)
    middle = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=2)
    newest = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5)
    _produce(partition_id, [newest, oldest, middle]).join_and_check()

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=2)

        # The two smallest-step prompts are picked; the newest is left behind.
        assert _uids_of(batch.keys) == {oldest.uid, middle.uid}

        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert oldest.uid not in remaining
        assert middle.uid not in remaining
        assert newest.uid in remaining
    finally:
        _clear_partition(partition_id)


def test_sample_orders_by_global_steps_across_finished_and_failure(tq_init, partition_id):
    """Ordering by ``global_steps`` spans both finished and failure prompts, not
    just within a single status bucket."""
    failure_old = PromptSpec(uid=_uid(), status="failure", sessions=1, global_steps=0)
    finished_mid = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=4)
    finished_new = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=9)
    _produce(partition_id, [finished_new, failure_old, finished_mid]).join_and_check()

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=2)
        assert _uids_of(batch.keys) == {failure_old.uid, finished_mid.uid}
    finally:
        _clear_partition(partition_id)


def test_sample_blocks_until_enough_then_unblocks(tq_init, partition_id):
    """sample stays blocked while fewer than batch_size prompts are ready and
    returns once the producer thread supplies the missing group."""
    # One group ready up front -> not enough for batch_size=2.
    _produce(partition_id, [PromptSpec(uid=_uid(), status="finished", sessions=1)]).join_and_check()

    rb = _make_rb()
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

    rb = _make_rb()
    # Stream groups one-by-one with a delay; consumer blocks in sample() meanwhile.
    producer = _produce(partition_id, specs, delay=0.1)
    try:
        batch = _sample(rb, partition_id, batch_size=batch_size)
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

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=n_prompts)

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

    rb = _make_rb()
    try:
        collected_keys: list[str] = []
        consumed_uids: set[str] = set()

        # Drain 2 + 2 + 1 prompts across three samples.
        for bs in (batch_size, batch_size, n_prompts - 2 * batch_size):
            batch = _sample(rb, partition_id, batch_size=bs)

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

    rb = _make_rb()
    try:
        batch = _sample(rb, partition_id, batch_size=batch_size)
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
    rb = _make_rb()
    try:
        with pytest.raises(ValueError, match="empty list"):
            _sample(rb, partition_id, batch_size=0)
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# sample: off-policy "drop" strategy.
# --------------------------------------------------------------------------- #


def test_drop_refills_stale_groups_and_reports_metrics(tq_init, partition_id):
    """drop discards finished groups whose staleness strictly exceeds the threshold *inside the
    polling loop*, calls ``refill_fn`` for an equal count, and returns a full, fresh batch."""
    # staleness = (global_steps - prompt_global_steps + 1); drop when staleness > threshold(=2).
    # At global_steps=5:
    #   stale    gs=0 -> 6 > 2 -> dropped (and refilled)
    #   boundary gs=4 -> 2 not > 2 -> kept (boundary is inclusive on the keep side)
    #   fresh    gs=5 -> 1 -> kept
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    boundary = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=4)
    fresh = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5)
    _produce(partition_id, [stale, boundary, fresh]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=5)
    rb = _make_rb(max_off_policy_strategy="drop", max_off_policy_threshold=2, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=3)

        # Full batch (batch_size=3), no stale group, dropped slot backfilled by the refiller.
        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == 3
        assert stale.uid not in sampled_uids
        assert {boundary.uid, fresh.uid} <= sampled_uids
        assert len(sampled_uids & set(refiller.produced_uids)) == 1

        # The dropped group's prompt and trajectory keys are gone from TransferQueue.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert stale.uid not in remaining
        assert stale.trajectory_keys[0] not in remaining

        assert refiller.calls == [1]

        # partition_id is a random test id (not "train") -> "validation" prefix.
        assert metrics["validation/off_policy/dropped_samples"] == 1
        assert metrics["validation/off_policy/dropped_samples_staleness/mean"] == 6
        assert metrics["validation/off_policy/dropped_samples_staleness/max"] == 6
        assert metrics["validation/off_policy/dropped_samples_staleness/min"] == 6
    finally:
        _clear_partition(partition_id)


def test_drop_keeps_all_within_threshold_without_metrics(tq_init, partition_id):
    """When nothing exceeds the threshold, drop keeps the full batch and emits no
    drop metrics."""
    specs = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5) for _ in range(2)]
    _produce(partition_id, specs).join_and_check()

    rb = _make_rb(max_off_policy_strategy="drop", max_off_policy_threshold=2)
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {spec.uid for spec in specs}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_drop_uses_version_based_staleness(tq_init, partition_id):
    """staleness is measured directly in model-version units: (global_steps -
    prompt_global_steps + 1), since global_steps is the weight version."""
    # threshold=8, at global_steps=10, drop when staleness > 8:
    #   stale gs=0 -> 10 - 0 + 1 = 11 > 8 -> dropped (and refilled)
    #   fresh gs=8 -> 10 - 8 + 1 = 3       -> kept
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    fresh = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=8)
    _produce(partition_id, [stale, fresh]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=10)
    rb = _make_rb(max_off_policy_strategy="drop", max_off_policy_threshold=8, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=10, partition_id=partition_id, batch_size=2)

        # Full batch, stale group dropped and backfilled.
        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == 2
        assert stale.uid not in sampled_uids
        assert fresh.uid in sampled_uids
        assert refiller.calls == [1]
        assert metrics["validation/off_policy/dropped_samples"] == 1
        assert metrics["validation/off_policy/dropped_samples_staleness/mean"] == 11
    finally:
        _clear_partition(partition_id)


def test_drop_refills_over_multiple_iterations(tq_init, partition_id):
    """When staleness reaches the batch over several poll iterations, each dropped group triggers
    a refill and the accumulated drop metrics reflect every dropped sample."""
    # threshold=1, at global_steps=3, drop when staleness > 1 (i.e. prompt_global_steps < 3).
    # Three stale groups at gs=0 will be dropped; refiller backfills fresh (gs=3) ones.
    stale = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0) for _ in range(3)]
    _produce(partition_id, stale).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=3)
    rb = _make_rb(max_off_policy_strategy="drop", max_off_policy_threshold=1, refill_fn=refiller)
    try:
        batch, metrics = rb.sample(global_steps=3, partition_id=partition_id, batch_size=3)

        # All 3 original stale groups dropped and replaced by refills.
        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == 3
        assert not (sampled_uids & {s.uid for s in stale})
        assert sampled_uids <= set(refiller.produced_uids)
        assert sum(refiller.calls) == 3
        # Staleness of every dropped group was (3 - 0 + 1) = 4.
        prefix = "validation"
        assert metrics[f"{prefix}/off_policy/dropped_samples"] == 3
        assert metrics[f"{prefix}/off_policy/dropped_samples_staleness/mean"] == 4
        assert metrics[f"{prefix}/off_policy/dropped_samples_staleness/max"] == 4
        assert metrics[f"{prefix}/off_policy/dropped_samples_staleness/min"] == 4
    finally:
        _clear_partition(partition_id)


def test_drop_uses_one_snapshot_per_poll_iteration(tq_init, partition_id):
    """A stale running prompt that finishes during refill is dropped on the next snapshot.

    This reproduces the production race where a second metadata sync after refill exposed newly
    finished stale prompts to selection without running the drop pass again.
    """
    stale_finished = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    stale_running = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, [stale_finished, stale_running]).join_and_check()

    refiller = FakeRefiller(partition_id, global_steps=5)

    def finish_during_first_refill(num_prompts: int) -> int:
        if not refiller.calls:
            _set_prompt_status(partition_id, stale_running.uid, "finished", global_steps=0)
        return refiller(num_prompts)

    rb = _make_rb(
        max_off_policy_strategy="drop",
        max_off_policy_threshold=1,
        refill_fn=finish_during_first_refill,
    )
    try:
        batch, metrics = rb.sample(global_steps=5, partition_id=partition_id, batch_size=1)

        sampled_uids = _uids_of(batch.keys)
        assert len(sampled_uids) == 1
        assert not (sampled_uids & {stale_finished.uid, stale_running.uid})
        assert sampled_uids <= set(refiller.produced_uids)
        assert refiller.calls == [1, 1]
        assert metrics["validation/off_policy/dropped_samples"] == 2
    finally:
        _clear_partition(partition_id)


def test_drop_without_refill_fn_drops_without_replacing(tq_init, partition_id):
    """With no refill_fn, stale groups are dropped but not replaced, so the batch completes only
    from the remaining fresh groups (the test-only path where the trainer hook is absent)."""
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    fresh = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=5) for _ in range(2)]
    _produce(partition_id, [stale] + fresh).join_and_check()

    rb = _make_rb(max_off_policy_strategy="drop", max_off_policy_threshold=2, refill_fn=None)
    consumer = SampleConsumer(rb, partition_id, batch_size=2, global_steps=5)
    try:
        consumer.start()
        # Only the 2 fresh groups satisfy batch_size=2; the stale one is dropped, not counted.
        batch = consumer.result_or_raise()
        assert _uids_of(batch.keys) == {f.uid for f in fresh}
        # Stale group removed from TransferQueue.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert stale.uid not in remaining
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# sample: off-policy "wait" (dropless) strategy.
# --------------------------------------------------------------------------- #


def test_wait_blocks_until_stale_inflight_finishes(tq_init, partition_id):
    """wait holds back a full batch while a stale in-flight prompt exists, then
    proceeds (without dropping it) once it terminates."""
    threshold, g = 2, 5
    fresh = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=g) for _ in range(2)]
    # In-flight prompt: staleness (5 - 0 + 1) = 6 >= 2 -> blocks sampling.
    stale = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=0)
    _produce(partition_id, fresh + [stale]).join_and_check()

    rb = _make_rb(max_off_policy_strategy="wait", max_off_policy_threshold=threshold)
    consumer = SampleConsumer(rb, partition_id, batch_size=2, global_steps=g)
    try:
        consumer.start()

        time.sleep(POLL_INTERVAL * 5)
        assert consumer.is_alive(), "wait must block while a stale in-flight prompt exists"

        # The stale group terminates -> no in-flight prompt at threshold -> unblock.
        # (clear+put is not atomic, so the consumer may proceed on the two fresh
        # groups the instant the running prompt disappears; either way it returns a
        # full batch_size batch -- droplessness is asserted separately below.)
        _set_prompt_status(partition_id, stale.uid, "finished", global_steps=0)

        batch = consumer.result_or_raise()
        assert len(_uids_of(batch.keys)) == 2
    finally:
        consumer.join(timeout=10.0)
        _clear_partition(partition_id)


def test_wait_keeps_stale_terminal_trajectories(tq_init, partition_id):
    """wait is dropless: a finished-but-very-stale group that ``drop`` would
    discard is still returned, with no drop metrics."""
    # staleness (100 - 0 + 1) = 101, far above threshold=2; "drop" would
    # remove it, "wait" keeps it. No in-flight prompts, so sampling never blocks.
    stale = PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=0)
    _produce(partition_id, [stale]).join_and_check()

    rb = _make_rb(max_off_policy_strategy="wait", max_off_policy_threshold=2)
    try:
        batch, metrics = rb.sample(global_steps=100, partition_id=partition_id, batch_size=1)

        assert set(batch.keys) == set(stale.trajectory_keys)
        assert metrics == {}
    finally:
        _clear_partition(partition_id)


def test_wait_does_not_block_when_inflight_is_fresh(tq_init, partition_id):
    """wait proceeds immediately when every in-flight prompt is below threshold,
    and never drops (no drop metrics)."""
    threshold, g = 4, 3
    finished = [PromptSpec(uid=_uid(), status="finished", sessions=1, global_steps=g) for _ in range(2)]
    # In-flight prompt staleness (3 - 3 + 1) = 1 < 4 -> does not block.
    inflight = PromptSpec(uid=_uid(), status="running", sessions=1, global_steps=g)
    _produce(partition_id, finished + [inflight]).join_and_check()

    rb = _make_rb(max_off_policy_strategy="wait", max_off_policy_threshold=threshold)
    try:
        batch, metrics = rb.sample(global_steps=g, partition_id=partition_id, batch_size=2)

        assert _uids_of(batch.keys) == {spec.uid for spec in finished}
        assert metrics == {}
    finally:
        _clear_partition(partition_id)
