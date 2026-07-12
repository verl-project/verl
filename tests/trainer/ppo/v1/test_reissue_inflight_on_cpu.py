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
"""Unit tests for :meth:`PPOTrainer._reissue_inflight_prompts` (checkpoint recovery).

These run against a real (CPU-only) TransferQueue, mirroring
``test_replay_buffer_on_cpu.py``. They exercise checkpoint resume for async
trainers: after ``tq.load_checkpoint`` restores the queue, prompts that were
still generating (``pending``/``running``) must be re-submitted from the prompt
data persisted at submit time, while finished trajectories are left untouched.

Because half-generated tokens are not durable, a re-issued prompt restarts
generation from scratch; its original ``global_steps`` (the staleness anchor) is
preserved by grouping re-issues per step.

The method is bound to a lightweight stub ``self`` (it only touches
``self.agent_loop_manager``), avoiding the need to build a full trainer.

A second group of tests covers the ``tq.save_checkpoint``/``tq.load_checkpoint``
round-trip that backs checkpoint consistency (matching ``_save_checkpoint``/
``_load_checkpoint``). These call the real checkpoint APIs, so they skip on
builds that lack them (the same ``_tq_supports_checkpoint`` gate the trainer uses
to decide whether to save/load); one test additionally asserts the trainer
short-circuits re-issue when the gate reports the feature unsupported.
"""

import uuid

import pytest
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu

# TODO: remove this later
# Capture the real version gate before the autouse fixture patches it. The save/load round-trip
# tests call tq.save_checkpoint/load_checkpoint for real, so they must skip on builds that lack the
# APIs (e.g. TransferQueue 0.1.8) rather than force the gate open.
_REAL_TQ_SUPPORTS_CHECKPOINT = trainer_base._tq_supports_checkpoint
requires_tq_checkpoint = pytest.mark.skipif(
    not _REAL_TQ_SUPPORTS_CHECKPOINT(),
    reason="installed TransferQueue does not support save_checkpoint/load_checkpoint",
)


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


@pytest.fixture(autouse=True)
def _force_tq_checkpoint_supported(monkeypatch):
    """Force the version gate open so the re-issue logic itself is exercised, not the gate.

    The locally installed TransferQueue may predate checkpoint support (making
    ``_tq_supports_checkpoint`` return False, which short-circuits re-issue). This only patches the
    gate used by the trainer methods; the save/load round-trip tests instead gate on the real
    availability via ``requires_tq_checkpoint`` (they call the actual tq APIs).
    """
    monkeypatch.setattr(trainer_base, "_tq_supports_checkpoint", lambda: True)


@pytest.fixture
def partition_id():
    """A unique partition per test to isolate TransferQueue state across tests."""
    return f"test-{uuid.uuid4().hex}"


def _uid() -> str:
    # uid must not contain "_" because trajectory keys are "{uid}_{session}_{index}".
    return uuid.uuid4().hex


class FakeAgentLoopManager:
    """Records every batch handed to ``generate_sequences`` so re-issues can be asserted."""

    def __init__(self):
        self.batches: list[TensorDict] = []

    def generate_sequences(self, batch: TensorDict) -> None:
        self.batches.append(batch)


def _make_trainer_stub(trainer_mode: str = "separate_async"):
    """Bind the real ``_reissue_inflight_prompts`` to a stub exposing only what it uses
    (``agent_loop_manager`` and ``trainer_mode``)."""
    stub = type("Stub", (), {})()
    stub.trainer_mode = trainer_mode
    stub.agent_loop_manager = FakeAgentLoopManager()
    stub._reissue_inflight_prompts = PPOTrainer._reissue_inflight_prompts.__get__(stub)
    return stub


def _submit_prompt(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Mirror ``_submit_one_gen_batch`` for async modes: store the per-row prompt data as fields
    plus the status tag under the ``{uid}`` key (a single-row batch).

    ``global_steps`` is a scalar broadcast across the batch, which TransferQueue cannot store as a
    field; the trainer excludes it (``_storable_prompt_fields``) and re-derives it from the tag, so
    the producer here does the same.
    """
    batch = tu.get_tensordict(
        {
            "uid": [uid],
            "raw_prompt": [f"prompt-for-{uid}"],
            "index": torch.tensor([0]),
        }
    )
    tag = {"is_prompt": True, "status": status, "global_steps": global_steps}
    tq.kv_batch_put(keys=[uid], partition_id=partition_id, tags=[tag], fields=batch)


def _add_trajectory(partition_id: str, uid: str, session_id: int, global_steps: int) -> str:
    """Attach one finished trajectory to a prompt (composite key), as the rollout would."""
    key = f"{uid}_{session_id}_0"
    tq.kv_put(
        key=key,
        partition_id=partition_id,
        fields={"input_ids": torch.tensor([1, 2, 3])},
        tag={"is_prompt": False, "seq_len": 3, "global_steps": global_steps},
    )
    return key


def _prompt_status(partition_id: str, uid: str) -> str | None:
    tag = tq.kv_list(partition_id=partition_id).get(partition_id, {}).get(uid)
    return None if tag is None else tag.get("status")


def _clear_partition(partition_id: str) -> None:
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


# --------------------------------------------------------------------------- #
# _reissue_inflight_prompts
# --------------------------------------------------------------------------- #


def test_reissue_resubmits_only_inflight_prompts(tq_init, partition_id):
    """Only pending/running prompts are re-issued; finished/failure are left untouched."""
    pending = _uid()
    running = _uid()
    finished = _uid()
    _submit_prompt(partition_id, pending, "pending", global_steps=2)
    _submit_prompt(partition_id, running, "running", global_steps=2)
    _submit_prompt(partition_id, finished, "finished", global_steps=2)
    _add_trajectory(partition_id, finished, session_id=0, global_steps=2)

    stub = _make_trainer_stub()
    try:
        reissued = stub._reissue_inflight_prompts(partition_id)

        assert reissued == 2
        # Exactly the two in-flight prompts were re-submitted for generation.
        submitted_uids = {uid for batch in stub.agent_loop_manager.batches for uid in batch["uid"]}
        assert submitted_uids == {pending, running}

        # Their status is reset to pending; finished is unchanged and still present.
        assert _prompt_status(partition_id, pending) == "pending"
        assert _prompt_status(partition_id, running) == "pending"
        assert _prompt_status(partition_id, finished) == "finished"
    finally:
        _clear_partition(partition_id)


def test_reissue_preserves_prompt_data_and_global_steps(tq_init, partition_id):
    """A re-issued batch carries back the original prompt data and its submission step."""
    uid = _uid()
    _submit_prompt(partition_id, uid, "running", global_steps=7)

    stub = _make_trainer_stub()
    try:
        stub._reissue_inflight_prompts(partition_id)

        assert len(stub.agent_loop_manager.batches) == 1
        batch = stub.agent_loop_manager.batches[0]
        assert list(batch["uid"]) == [uid]
        assert list(batch["raw_prompt"]) == [f"prompt-for-{uid}"]
        # global_steps is re-stamped to the original submission step (scalar broadcast).
        assert int(batch["global_steps"]) == 7
        # The re-put prompt tag also carries the original step for staleness ordering.
        tag = tq.kv_list(partition_id=partition_id).get(partition_id, {})[uid]
        assert tag["global_steps"] == 7
    finally:
        _clear_partition(partition_id)


def test_reissue_groups_by_global_steps(tq_init, partition_id):
    """Prompts submitted at different steps are re-issued as separate per-step batches."""
    step2 = [_uid(), _uid()]
    step5 = [_uid()]
    for uid in step2:
        _submit_prompt(partition_id, uid, "pending", global_steps=2)
    for uid in step5:
        _submit_prompt(partition_id, uid, "running", global_steps=5)

    stub = _make_trainer_stub()
    try:
        reissued = stub._reissue_inflight_prompts(partition_id)
        assert reissued == 3

        # One batch per distinct global_steps; each batch's step matches its prompts.
        by_step = {int(batch["global_steps"]): set(batch["uid"]) for batch in stub.agent_loop_manager.batches}
        assert by_step == {2: set(step2), 5: set(step5)}
    finally:
        _clear_partition(partition_id)


def test_reissue_noop_without_inflight(tq_init, partition_id):
    """With only terminal prompts, nothing is re-issued."""
    finished = _uid()
    failure = _uid()
    _submit_prompt(partition_id, finished, "finished", global_steps=1)
    _submit_prompt(partition_id, failure, "failure", global_steps=1)

    stub = _make_trainer_stub()
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 0
        assert stub.agent_loop_manager.batches == []
    finally:
        _clear_partition(partition_id)


def test_reissue_noop_on_empty_partition(tq_init, partition_id):
    """An empty (never-written) partition re-issues nothing and does not raise."""
    stub = _make_trainer_stub()
    assert stub._reissue_inflight_prompts(partition_id) == 0
    assert stub.agent_loop_manager.batches == []


def test_reissue_noop_for_sync_mode(tq_init, partition_id):
    """Sync mode never persists prompts, so re-issue is a guarded no-op even with in-flight prompts."""
    _submit_prompt(partition_id, _uid(), "pending", global_steps=1)

    stub = _make_trainer_stub(trainer_mode="sync")
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 0
        assert stub.agent_loop_manager.batches == []
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# tq.save_checkpoint / tq.load_checkpoint round-trip
#
# These call the real TransferQueue checkpoint APIs (matching _save_checkpoint/_load_checkpoint),
# so they skip on builds that lack them. The defensive gate (_tq_supports_checkpoint) is what the
# trainer uses to decide whether to save/load at all; here we skip on the same condition and
# additionally assert the trainer short-circuits when the gate reports unsupported.
# --------------------------------------------------------------------------- #


@requires_tq_checkpoint
def test_save_load_roundtrip_restores_prompts(tq_init, partition_id, tmp_path):
    """save_checkpoint then load_checkpoint (into a cleared queue) restores prompt status and data."""
    pending = _uid()
    running = _uid()
    finished = _uid()
    _submit_prompt(partition_id, pending, "pending", global_steps=3)
    _submit_prompt(partition_id, running, "running", global_steps=3)
    _submit_prompt(partition_id, finished, "finished", global_steps=3)
    _add_trajectory(partition_id, finished, session_id=0, global_steps=3)

    ckpt_dir = str(tmp_path / "transfer_queue")
    tq.save_checkpoint(ckpt_dir, metadata={"global_steps": 3})

    # Simulate a fresh process: wipe the live partition so the assertions below only pass if
    # load_checkpoint actually repopulates it.
    _clear_partition(partition_id)
    assert tq.kv_list(partition_id=partition_id).get(partition_id, {}) == {}

    tq.load_checkpoint(ckpt_dir)

    try:
        # Statuses are restored verbatim (load does not mutate pending/running -> finished).
        assert _prompt_status(partition_id, pending) == "pending"
        assert _prompt_status(partition_id, running) == "running"
        assert _prompt_status(partition_id, finished) == "finished"

        # Per-row prompt fields survive the round-trip so re-issue can regenerate from them.
        batch = tq.kv_batch_get(keys=[pending], partition_id=partition_id)
        assert list(batch["uid"]) == [pending]
        assert list(batch["raw_prompt"]) == [f"prompt-for-{pending}"]

        # The finished trajectory is restored too (composite key preserved).
        traj = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert f"{finished}_0_0" in traj
    finally:
        _clear_partition(partition_id)


@requires_tq_checkpoint
def test_save_load_then_reissue_only_inflight(tq_init, partition_id, tmp_path):
    """End-to-end recovery: after a save/load round-trip, only in-flight prompts are re-issued."""
    pending = _uid()
    finished = _uid()
    _submit_prompt(partition_id, pending, "pending", global_steps=4)
    _submit_prompt(partition_id, finished, "finished", global_steps=4)
    _add_trajectory(partition_id, finished, session_id=0, global_steps=4)

    ckpt_dir = str(tmp_path / "transfer_queue")
    tq.save_checkpoint(ckpt_dir, metadata={"global_steps": 4})
    _clear_partition(partition_id)
    tq.load_checkpoint(ckpt_dir)

    stub = _make_trainer_stub()
    try:
        reissued = stub._reissue_inflight_prompts(partition_id)
        assert reissued == 1
        submitted_uids = {uid for batch in stub.agent_loop_manager.batches for uid in batch["uid"]}
        assert submitted_uids == {pending}
        # Its restored global_steps anchor is preserved through save/load/re-issue.
        assert int(stub.agent_loop_manager.batches[0]["global_steps"]) == 4
    finally:
        _clear_partition(partition_id)


def test_reissue_short_circuits_when_checkpoint_unsupported(tq_init, partition_id, monkeypatch):
    """The defensive gate: when TransferQueue lacks checkpoint support, re-issue is a no-op.

    This overrides the autouse fixture (which forces the gate open) to assert the real guard, so
    an old TransferQueue never tries to read back / re-submit prompts that were never persisted.
    """
    monkeypatch.setattr(trainer_base, "_tq_supports_checkpoint", lambda: False)
    _submit_prompt(partition_id, _uid(), "running", global_steps=1)

    stub = _make_trainer_stub()
    try:
        assert stub._reissue_inflight_prompts(partition_id) == 0
        assert stub.agent_loop_manager.batches == []
    finally:
        _clear_partition(partition_id)
