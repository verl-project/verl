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
"""

import uuid

import pytest
import torch
import transfer_queue as tq
from tensordict import TensorDict

from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils import tensordict_utils as tu


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
    # uid must not contain "_" because trajectory keys are "{uid}_{session}_{index}".
    return uuid.uuid4().hex


class FakeAgentLoopManager:
    """Records every batch handed to ``generate_sequences`` so re-issues can be asserted."""

    def __init__(self):
        self.batches: list[TensorDict] = []

    def generate_sequences(self, batch: TensorDict) -> None:
        self.batches.append(batch)


def _make_trainer_stub():
    """Bind the real ``_reissue_inflight_prompts`` to a stub exposing only what it uses."""
    stub = type("Stub", (), {})()
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
