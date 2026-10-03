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

import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq_module
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.protocol import DataProto
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _settle_session_tasks
from verl.trainer.ppo.v1.utils import compute_advantage, compute_advantage_for_multi_trajectories


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


_WorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


class _DummyWorker:
    _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
    _compute_position_ids = AgentLoopWorker._compute_position_ids
    _compute_score = AgentLoopWorker._compute_score
    _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
    distillation_enabled = False

    def __init__(self, topk_log_probs: int):
        self.rollout_config = OmegaConf.create({"topk_log_probs": topk_log_probs})
        self.processor = None
        self.reward_loop_worker_handles = None


def _output(extra_fields: dict) -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102, 103],
        response_ids=[11, 12],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


async def _postprocess(
    monkeypatch,
    worker: _DummyWorker,
    output: AgentLoopOutput | list[AgentLoopOutput],
    validate: bool,
    session_id: int = 0,
):
    captured = {}

    async def async_kv_batch_put(*, keys, fields, tags, partition_id):
        captured["fields"] = fields

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_batch_put", async_kv_batch_put)
    await _WorkerTQ._agent_loop_postprocess(
        worker,
        output,
        validate,
        uid="u0",
        session_id=session_id,
        global_steps=1,
        raw_prompt=[{"role": "user", "content": "hi"}],
    )
    return captured["fields"]


def test_grpo_vectorized_with_copied_session_rewards(monkeypatch):
    # 1. Three sessions. Only the final output of each session has a reward.
    session_0 = [_output({"reward_extra_info": {}})]
    session_1 = [_output({"reward_extra_info": {}})]
    session_2 = [
        _output({"reward_extra_info": {}}),
        _output({"reward_extra_info": {}}),
        _output({"reward_extra_info": {}}),
        _output({"reward_extra_info": {}}),
    ]
    session_0[-1].reward_score = 1.0
    session_1[-1].reward_score = 4.0
    session_2[-1].reward_score = 5.0
    rewards_before_copy = [output.reward_score for output in session_2]
    print("\n1. Session 2 before copying:", rewards_before_copy)
    assert rewards_before_copy == [None, None, None, 5.0]

    # 2. Run real postprocessing. The existing helper only intercepts storage writes.
    worker = _DummyWorker(topk_log_probs=0)
    stored_0 = asyncio.run(_postprocess(monkeypatch, worker, session_0, False, session_id=0))
    stored_1 = asyncio.run(_postprocess(monkeypatch, worker, session_1, False, session_id=1))
    stored_2 = asyncio.run(_postprocess(monkeypatch, worker, session_2, False, session_id=2))
    rewards_after_copy = [output.reward_score for output in session_2]
    print("2. Session 2 after copying: ", rewards_after_copy)
    assert rewards_after_copy == [5.0, 5.0, 5.0, 5.0]

    # 3. Read the actual reward tensors emitted by postprocessing.
    token_rewards = torch.cat([stored_0["rm_scores"], stored_1["rm_scores"], stored_2["rm_scores"]])
    response_mask = torch.cat([stored_0["response_mask"], stored_1["response_mask"], stored_2["response_mask"]])
    print("3. Stored rewards:", token_rewards.sum(dim=1).tolist())
    data = DataProto.from_dict(
        tensors={"token_level_rewards": token_rewards, "response_mask": response_mask},
        non_tensors={"uid": np.array(["u0"] * 6, dtype=object)},
    )
    batch_keys = ["u0_0_0", "u0_1_0", "u0_2_0", "u0_2_1", "u0_2_2", "u0_2_3"]

    # 4. This direct call is what the old early-return branch did: count all six rows.
    row_result = compute_advantage(data, adv_estimator="grpo_vectorized")
    row_advantages = row_result.batch["advantages"][:, 0].clone()
    print("4. Per-row advantages (old path):", row_advantages.tolist())

    # 5. The fixed wrapper selects one final output per session before computing advantages.
    result = compute_advantage_for_multi_trajectories(data, batch_keys, "grpo_vectorized")
    session_advantages = result.batch["advantages"][:, 0]
    print("5. Per-session advantages:      ", session_advantages.tolist())

    # For rewards [1, 4, 5], the mean is 10/3 and the sample std is sqrt(13/3).
    expected_per_output = torch.tensor([-1.120897, 0.320256, 0.800640, 0.800640, 0.800640, 0.800640])
    expected = expected_per_output.unsqueeze(-1) * response_mask
    torch.testing.assert_close(result.batch["advantages"], expected)
    torch.testing.assert_close(result.batch["returns"], expected)


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_stores_unpadded_rollout_topk(monkeypatch):
    output = _output(
        {"response_topk_ids": [[11, 13], [12, 14]], "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]]}
    )

    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), output, validate=False)

    ids, log_probs = fields["rollout_topk_ids"][0], fields["rollout_topk_log_probs"][0]
    assert ids.shape == (5, 2) and ids.dtype == torch.int32
    assert log_probs.shape == (5, 2) and log_probs.dtype == torch.float32
    assert ids[2].tolist() == [11, 13] and ids[3].tolist() == [12, 14]
    assert log_probs[2].tolist() == pytest.approx([-0.1, -2.0]) and log_probs[3].tolist() == pytest.approx([-0.2, -1.5])
    assert ids[0].tolist() == [0, 1] and ids[4].tolist() == [0, 1]
    assert "response_topk_ids" not in fields["extra_fields"][0]


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_skips_rollout_topk_on_validate(monkeypatch):
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), _output({}), validate=True)

    assert "rollout_topk_ids" not in fields.keys()
    assert "rollout_topk_log_probs" not in fields.keys()
