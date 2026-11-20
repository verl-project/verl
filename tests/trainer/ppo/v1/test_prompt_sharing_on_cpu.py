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
"""A GRPO prompt is put once into TransferQueue's public partition and linked by its trajectories."""

import uuid

import pytest
import torch
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.experimental.agent_loop.preprocess_cache import PromptPreprocessCache
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _PublicMultiModalInputs
from verl.trainer.ppo.v1.prompt_sharing import (
    PUBLIC_PARTITION,
    prompt_sharing_enabled,
    prompt_sharing_requested,
    public_multi_modal_key,
    publish_public_prompts,
    release_public_prompts,
)
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.skipif(not hasattr(tq, "kv_link"), reason="TransferQueue has no public partition links")

_WorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


class _DummyWorker:
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _compute_score = AgentLoopWorker._compute_score
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    _share_multi_modal_inputs = _WorkerTQ._share_multi_modal_inputs
    _get_multi_modal_inputs = _WorkerTQ._get_multi_modal_inputs
    distillation_enabled = False

    def __init__(self, public_prompt_fields: dict[str, list[str]], shareable_multi_modal_keys=frozenset()):
        self.rollout_config = OmegaConf.create({"topk_log_probs": 0})
        self.processor = None
        self.reward_loop_worker_handles = None
        self.public_prompt_fields = public_prompt_fields
        self.public_multi_modal_inputs = {uid: _PublicMultiModalInputs() for uid in public_prompt_fields}
        self.preprocess_caches = {}
        self.shareable_multi_modal_keys = shareable_multi_modal_keys


def _uid() -> str:
    return uuid.uuid4().hex


def _prompt_batch(uids: list[str], **extra_fields):
    batch = tu.get_tensordict(
        {
            "uid": uids,
            "raw_prompt": [[{"role": "user", "content": f"question {uid}"}] for uid in uids],
            "index": torch.arange(len(uids)),
            **extra_fields,
        }
    )
    tu.assign_non_tensor_data(batch, "global_steps", 1)
    return batch


def _public_keys(uids: list[str]) -> set[str]:
    return set(tq.kv_exists(keys=uids, partition_id=PUBLIC_PARTITION))


def _config(storage_backend: str = "SimpleStorage", share_prompts: bool | None = None) -> OmegaConf:
    agent = {} if share_prompts is None else {"share_prompts": share_prompts}
    return OmegaConf.create(
        {
            "transfer_queue": {"backend": {"storage_backend": storage_backend}},
            "actor_rollout_ref": {"rollout": {"agent": agent}},
        }
    )


def test_prompt_sharing_needs_simple_storage():
    assert prompt_sharing_enabled(_config())
    assert not prompt_sharing_enabled(_config(storage_backend="MooncakeStore"))


def test_share_prompts_switch_turns_off_storage_and_preprocessing_sharing():
    off = _config(share_prompts=False)
    assert not prompt_sharing_enabled(off) and not prompt_sharing_requested(off)
    assert prompt_sharing_requested(_config(storage_backend="MooncakeStore"))


@pytest.mark.asyncio
async def test_trajectories_link_public_prompt_until_all_are_cleared(tq_init):
    uid = _uid()
    batch = _prompt_batch([uid])
    fields = await publish_public_prompts(batch)
    assert set(fields) == {"uid", "raw_prompt", "index"}

    worker = _DummyWorker({uid: fields})
    raw_prompt = [{"role": "user", "content": f"question {uid}"}]
    prompt = {"uid": uid, "raw_prompt": raw_prompt, "index": torch.tensor(0), "global_steps": 1}
    keys = []
    for session_id in range(2):
        output = AgentLoopOutput(
            prompt_ids=[101, 102],
            response_ids=[11 + session_id],
            response_mask=[1],
            num_turns=2,
            metrics=AgentLoopMetrics(),
        )
        await _WorkerTQ._agent_loop_postprocess(worker, output, False, session_id=session_id, **prompt)
        keys.append(f"{uid}_{session_id}_0")

    rows = tq.kv_describe(keys=keys, partition_id="train")
    assert all(rows[key]["links"] == {field: uid for field in fields} for key in keys)
    data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=["uid", "raw_prompt", "index", "responses"])
    assert data.get("uid").tolist() == [uid, uid]
    assert data.get("raw_prompt").tolist() == [raw_prompt] * 2
    assert data["index"].tolist() == [0, 0]

    tq.kv_clear(keys=keys[:1], partition_id="train")
    release_public_prompts()
    assert _public_keys([uid]) == {uid}

    tq.kv_clear(keys=keys[1:], partition_id="train")
    release_public_prompts()
    assert _public_keys([uid]) == set()


@pytest.mark.asyncio
async def test_sessions_with_equal_multi_modal_inputs_link_one_copy(tq_init):
    uid = _uid()
    fields = await publish_public_prompts(_prompt_batch([uid]))
    worker = _DummyWorker({uid: fields})
    image = {"pixel_values": torch.arange(12.0).reshape(3, 4), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    other_image = {"pixel_values": torch.zeros(3, 4), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    prompt = {"uid": uid, "raw_prompt": [{"role": "user", "content": "x"}], "index": torch.tensor(0), "global_steps": 1}
    keys = []
    for session_id, inputs in enumerate([image, image, other_image]):
        worker._compute_multi_modal_inputs = lambda output, input_ids, inputs=inputs: dict(inputs)
        output = AgentLoopOutput(
            prompt_ids=[101, 102], response_ids=[11], response_mask=[1], num_turns=2, metrics=AgentLoopMetrics()
        )
        await _WorkerTQ._agent_loop_postprocess(worker, output, False, session_id=session_id, **prompt)
        keys.append(f"{uid}_{session_id}_0")

    mm_key = public_multi_modal_key(uid)
    rows = tq.kv_describe(keys=keys, partition_id="train")
    assert [rows[key]["links"].get("multi_modal_inputs") for key in keys] == [mm_key, mm_key, None]
    data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=["multi_modal_inputs"])
    pixel_values = [inputs["pixel_values"] for inputs in data.get("multi_modal_inputs").tolist()]
    assert torch.equal(pixel_values[0], image["pixel_values"]) and torch.equal(pixel_values[1], image["pixel_values"])
    assert torch.equal(pixel_values[2], other_image["pixel_values"])

    tq.kv_clear(keys=keys, partition_id="train")
    release_public_prompts()
    assert _public_keys([uid, mm_key]) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("text_dependent", [False, True])
async def test_sessions_with_same_media_compute_multi_modal_inputs_once(tq_init, text_dependent):
    uid = _uid()
    fields = await publish_public_prompts(_prompt_batch([uid]))
    worker = _DummyWorker({uid: fields}, shareable_multi_modal_keys=frozenset({"pixel_values", "image_grid_thw"}))
    worker.preprocess_caches[uid] = PromptPreprocessCache()
    computed = []

    def compute(output, input_ids):
        computed.append(output.multi_modal_data["images"])
        inputs = {"pixel_values": torch.full((3, 4), float(len(output.multi_modal_data["images"])))}
        inputs["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        if text_dependent:
            inputs["token_type_ids"] = torch.zeros_like(input_ids)
        return inputs

    worker._compute_multi_modal_inputs = compute
    image, tool_image = object(), object()
    prompt = {"uid": uid, "raw_prompt": [{"role": "user", "content": "x"}], "index": torch.tensor(0), "global_steps": 1}
    keys = []
    # Sessions copy the cached media lists; the last one appended a tool image.
    for session_id, images in enumerate([[image], [image], [image, tool_image]]):
        output = AgentLoopOutput(
            prompt_ids=[101, 102],
            response_ids=[11 + session_id],
            response_mask=[1],
            multi_modal_data={"images": images},
            num_turns=2,
            metrics=AgentLoopMetrics(),
        )
        await _WorkerTQ._agent_loop_postprocess(worker, output, False, session_id=session_id, **prompt)
        keys.append(f"{uid}_{session_id}_0")

    assert len(computed) == (3 if text_dependent else 2)
    data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=["multi_modal_inputs"])
    pixel_values = [inputs["pixel_values"] for inputs in data.get("multi_modal_inputs").tolist()]
    assert [values[0, 0].item() for values in pixel_values] == [1.0, 1.0, 2.0]

    tq.kv_clear(keys=keys, partition_id="train")
    release_public_prompts([uid])


@pytest.mark.asyncio
async def test_publishing_skips_the_prompts_that_already_have_a_public_key(tq_init):
    """Resuming re-dispatches prompts whose public keys exist, and TransferQueue refuses to rewrite them."""
    shared, private = _uid(), _uid()
    await publish_public_prompts(_prompt_batch([shared]))
    assert _public_keys([shared, private]) == {shared}

    batch = _prompt_batch([shared, private])
    try:
        assert set(await publish_public_prompts(batch)) == {"uid", "raw_prompt", "index"}
        assert _public_keys([shared, private]) == {shared, private}
        data = tq.kv_batch_get(keys=[private], partition_id=PUBLIC_PARTITION, select_fields=["raw_prompt"])
        assert data.get("raw_prompt").tolist() == [[{"role": "user", "content": f"question {private}"}]]
    finally:
        tq.kv_clear(keys=[shared, private], partition_id=PUBLIC_PARTITION)


@pytest.mark.asyncio
async def test_never_linked_public_prompt_is_freed_only_when_given_up(tq_init):
    uids = [_uid(), _uid()]
    await publish_public_prompts(_prompt_batch(uids))
    # A session put the shared multi_modal_inputs of uids[0], then the group failed before linking them.
    mm_key = public_multi_modal_key(uids[0])
    tq.kv_put(key=mm_key, partition_id=PUBLIC_PARTITION, fields={"pixel_values": torch.zeros(2)})
    assert _public_keys([mm_key]) == {mm_key}

    release_public_prompts()
    assert _public_keys(uids) == set(uids)

    release_public_prompts(uids[:1])
    assert _public_keys([*uids, mm_key]) == {uids[1]}

    release_public_prompts(uids[1:])
    assert _public_keys(uids) == set()
