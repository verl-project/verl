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
"""Unit tests for StreamingDataloader inflight-prompt checkpoint recovery.

These run against a real (CPU-only) TransferQueue, mirroring
``test_streaming_dataloader_sampling_on_cpu.py``. They exercise checkpoint resume for async
trainers: after ``tq.load_checkpoint`` restores the queue, prompts that were
still generating (``pending``/``running``) must be re-submitted from the prompt
data persisted at submit time, while finished trajectories are left untouched.

Because half-generated tokens are not durable, a re-issued prompt restarts
generation from scratch. Existing session trajectories under an in-flight prompt
are removed, and the prompt is re-stamped with the resumed training step.

The method is bound to a lightweight stub ``self`` exposing a fake
``agent_loop_manager`` and ``global_steps``, avoiding the need to build a full
trainer.

A second group of tests covers the ``tq.save_checkpoint``/``tq.load_checkpoint``
round-trip that backs checkpoint consistency (matching ``save_checkpoint``/
``load_checkpoint``), calling the real checkpoint APIs.
"""

import uuid

import pytest
import torch
import transfer_queue as tq
from tensordict import TensorDict

import verl.trainer.ppo.v1.streaming_dataloader as streaming_module
from verl.trainer.ppo.v1.streaming_dataloader import StreamingDataloaderAsync
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


def _make_dataloader_stub(global_steps: int = 8):
    """Build an actor implementation stub with the real reissue method."""
    stub = object.__new__(StreamingDataloaderAsync)
    stub.global_steps = global_steps
    stub.default_rollout_n = 1
    stub._default_rollout_n = lambda _partition_id: 1
    stub.agent_loop_manager = FakeAgentLoopManager()
    return stub


def _submit_prompt(partition_id: str, uid: str, status: str, global_steps: int) -> None:
    """Mirror async prompt submission: store the per-row prompt data as fields
    plus the status tag under the ``{uid}`` key (a single-row batch).

    ``global_steps`` is a scalar broadcast across the batch, which TransferQueue cannot store as a
    field; the trainer excludes it and re-derives it from the tag on restore.
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
        tag={"is_prompt": False, "status": "success", "seq_len": 3, "global_steps": global_steps},
    )
    return key


def _prompt_status(partition_id: str, uid: str) -> str | None:
    tag = tq.kv_list(partition_id=partition_id).get(partition_id, {}).get(uid)
    return None if tag is None else tag.get("status")


def _clear_partition(partition_id: str) -> None:
    keys = list(tq.kv_list(partition_id=partition_id).get(partition_id, {}).keys())
    if keys:
        tq.kv_clear(keys=keys, partition_id=partition_id)


def test_async_submission_persists_reissuable_prompt_fields(monkeypatch):
    stub = _make_dataloader_stub(global_steps=4)
    stub.global_steps = 4
    batch = tu.get_tensordict(
        {
            "uid": [_uid(), _uid()],
            "raw_prompt": ["prompt-a", "prompt-b"],
            "index": torch.tensor([0, 1]),
        }
    )
    tu.assign_non_tensor_data(batch, "global_steps", stub.global_steps)
    puts = []
    monkeypatch.setattr(streaming_module.tq, "kv_batch_put", lambda **kwargs: puts.append(kwargs))

    assert stub._submit_batch_to_rollout(batch, "train") == 2
    assert len(puts) == 1
    assert set(puts[0]["fields"].keys()) == {"uid", "raw_prompt", "index"}
    assert stub.agent_loop_manager.batches == [batch]


def test_async_submission_rejects_dynamic_rollout_n():
    stub = _make_dataloader_stub(global_steps=4)
    batch = tu.get_tensordict(
        {
            "uid": [_uid(), _uid()],
            "raw_prompt": ["prompt-a", "prompt-b"],
            "index": torch.tensor([0, 1]),
            "__rollout_n__": torch.tensor([1, 2]),
        }
    )

    with pytest.raises(ValueError, match="fixed train rollout_n"):
        stub._submit_batch_to_rollout(batch, "train")


def test_next_train_batch_coalesces_gen_batches():
    stub = _make_dataloader_stub(global_steps=5)
    stub.gen_batch_size = 2
    stub.global_steps = 5
    chunks = iter(
        [
            tu.get_tensordict({"uid": ["a", "b"], "raw_prompt": ["a", "b"], "index": torch.tensor([0, 1])}),
            tu.get_tensordict({"uid": ["c", "d"], "raw_prompt": ["c", "d"], "index": torch.tensor([2, 3])}),
        ]
    )
    stub._fetch_one_gen_batch = lambda: next(chunks)

    batch = stub._next_train_batch(4)

    assert list(batch["uid"]) == ["a", "b", "c", "d"]
    assert list(batch["index"]) == [0, 1, 2, 3]
    assert int(batch["global_steps"]) == 5


# --------------------------------------------------------------------------- #
# _reissue_inflight_prompts
# --------------------------------------------------------------------------- #


def test_reissue_resubmits_only_inflight_prompts(tq_init, partition_id):
    """In-flight groups restart cleanly; finished groups and their trajectories remain untouched."""
    pending = _uid()
    running = _uid()
    finished = _uid()
    _submit_prompt(partition_id, pending, "pending", global_steps=2)
    _submit_prompt(partition_id, running, "running", global_steps=2)
    _submit_prompt(partition_id, finished, "finished", global_steps=2)
    pending_trajectory = _add_trajectory(partition_id, pending, session_id=0, global_steps=2)
    running_trajectory = _add_trajectory(partition_id, running, session_id=0, global_steps=2)
    finished_trajectory = _add_trajectory(partition_id, finished, session_id=0, global_steps=2)
    running_failure = f"{running}_1"
    tq.kv_put(
        key=running_failure,
        partition_id=partition_id,
        tag={"is_session": True, "status": "failure", "session_id": 1},
    )

    stub = _make_dataloader_stub(global_steps=3)
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

        # A restarted group keeps only its prompt fields. Terminal groups are not modified.
        remaining = tq.kv_list(partition_id=partition_id).get(partition_id, {})
        assert pending_trajectory not in remaining
        assert running_trajectory not in remaining
        assert running_failure not in remaining
        assert finished_trajectory in remaining
    finally:
        _clear_partition(partition_id)


def test_reissue_preserves_prompt_data_and_resets_global_steps(tq_init, partition_id):
    """A restarted group keeps its prompt fields but uses the resumed dispatch step."""
    uid = _uid()
    _submit_prompt(partition_id, uid, "running", global_steps=7)

    stub = _make_dataloader_stub(global_steps=8)
    try:
        stub._reissue_inflight_prompts(partition_id)

        assert len(stub.agent_loop_manager.batches) == 1
        batch = stub.agent_loop_manager.batches[0]
        assert list(batch["uid"]) == [uid]
        assert list(batch["raw_prompt"]) == [f"prompt-for-{uid}"]
        assert int(batch["global_steps"]) == 8
        tag = tq.kv_list(partition_id=partition_id).get(partition_id, {})[uid]
        assert tag["global_steps"] == 8
    finally:
        _clear_partition(partition_id)


# --------------------------------------------------------------------------- #
# tq.save_checkpoint / tq.load_checkpoint round-trip
#
# These call the real TransferQueue checkpoint APIs (matching save_checkpoint/load_checkpoint).
# --------------------------------------------------------------------------- #


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

    stub = _make_dataloader_stub(global_steps=5)
    try:
        assert _prompt_status(partition_id, pending) == "pending"
        assert _prompt_status(partition_id, finished) == "finished"
        restored = tq.kv_batch_get(keys=[pending], partition_id=partition_id)
        assert list(restored["raw_prompt"]) == [f"prompt-for-{pending}"]
        assert f"{finished}_0_0" in tq.kv_list(partition_id=partition_id)[partition_id]

        reissued = stub._reissue_inflight_prompts(partition_id)
        assert reissued == 1
        submitted_uids = {uid for batch in stub.agent_loop_manager.batches for uid in batch["uid"]}
        assert submitted_uids == {pending}
        assert int(stub.agent_loop_manager.batches[0]["global_steps"]) == 5
    finally:
        _clear_partition(partition_id)
