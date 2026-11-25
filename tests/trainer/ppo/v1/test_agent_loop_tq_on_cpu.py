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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf

import verl.experimental.agent_loop.agent_loop as agent_loop_module
import verl.trainer.ppo.v1.agent_loop_tq as agent_loop_tq_module
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopWorkerTQ, _settle_session_tasks, apply_greedy_sampling_params
from verl.utils.tokenizer.deepseek import expand_image_tokens


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


async def _postprocess(monkeypatch, worker: _DummyWorker, output: AgentLoopOutput, validate: bool):
    captured = {}

    async def async_kv_batch_put(*, keys, fields, tags, partition_id):
        captured["fields"] = fields

    monkeypatch.setattr(agent_loop_tq_module.tq, "async_kv_batch_put", async_kv_batch_put)
    await _WorkerTQ._agent_loop_postprocess(
        worker, output, validate, uid="u0", session_id=0, global_steps=1, raw_prompt=[{"role": "user", "content": "hi"}]
    )
    return captured["fields"]


@pytest.mark.asyncio
async def test_agent_loop_tq_keeps_padded_text_routes(monkeypatch):
    output = _output({})
    output.routed_experts = torch.tensor([10, 20, 30], dtype=torch.int16).reshape(-1, 1, 1).numpy()

    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=0), output, validate=False)

    assert fields["routed_experts"][0].flatten().tolist() == [10, 20, 30, 0, 0]


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
    assert fields["rollout_support_complete"][0].tolist() == [True, True, False, False, True]


@pytest.mark.asyncio
async def test_expanded_prompt_realigns_rollout_heads_and_coverage(monkeypatch):
    """Response predictions stay at prompt_length - 1 after adopting the expanded image prompt."""
    prompt = [101, 129264, 129264, 129264, 129264, 103]
    worker = _DummyWorker(topk_log_probs=2)
    worker._compute_multi_modal_inputs = lambda output, ids: {
        "_expanded_input_ids": torch.tensor([prompt + output.response_ids]),
        "_image_source_rows": torch.tensor([0, 1, 5, 6, 7]),
    }
    output = _output(
        {
            "response_topk_ids": [[11, 13], [12, 14]],
            "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]],
            "response_support_complete": [True, False],
        }
    )
    fields = await _postprocess(monkeypatch, worker, output, validate=False)
    ids, log_probs, complete = (
        fields[key][0] for key in ("rollout_topk_ids", "rollout_topk_log_probs", "rollout_support_complete")
    )
    start = len(prompt) - 1
    assert ids.shape == log_probs.shape == (len(prompt) + 2, 2)
    assert ids[start : start + 2].tolist() == [[11, 13], [12, 14]]
    torch.testing.assert_close(log_probs[start : start + 2], torch.tensor([[-0.1, -2.0], [-0.2, -1.5]]))
    assert complete.tolist() == [True] * start + [True, False, True]
    assert ids[:start].tolist() == [[0, 1]] * start
    assert ids[-1].tolist() == [0, 1]
    assert log_probs[:start].exp().tolist() == [[0.5, 0.5]] * start
    torch.testing.assert_close(fields["responses"][0], torch.tensor([11, 12]))


@pytest.mark.asyncio
async def test_tool_image_expansion_realigns_response_metadata_and_routes(monkeypatch):
    worker = _DummyWorker(topk_log_probs=2)

    def compute_multimodal_inputs(output, input_ids):
        grids = torch.tensor([[1, 1], [1, 1]])
        ids, types, source_rows = expand_image_tokens(input_ids, grids, 99, 1)
        return {
            "_expanded_input_ids": ids,
            "image_grid_hws": grids,
            "vision_token_types": types,
            "_image_source_rows": source_rows,
        }

    worker._compute_multi_modal_inputs = compute_multimodal_inputs
    teacher = AsyncMock(return_value=(torch.arange(12).unsqueeze(-1), torch.zeros(12, 1)))
    worker.distillation_enabled = True
    worker.teacher_key = "teacher"
    worker.teacher_server_manager = SimpleNamespace(compute_teacher_logprobs_single=teacher)
    output = AgentLoopOutput(
        prompt_ids=[101, 99, 99, 99, 99, 102],
        response_ids=[11, 99, 12],
        response_mask=[1, 0, 1],
        response_logprobs=[-0.1, 0.0, -0.2],
        routed_experts=torch.tensor([10, 21, 30, 40, 51]).reshape(-1, 1, 1).numpy(),
        reward_score=0.5,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[11, 13], [99, 98], [12, 14]],
            "response_topk_log_probs": [[-0.1, -2.0], [-0.3, -1.0], [-0.2, -1.5]],
            "response_support_complete": [False, True, False],
        },
    )

    fields = await _postprocess(monkeypatch, worker, output, validate=False)

    assert fields["prompts"][0].tolist() == output.prompt_ids
    assert fields["responses"][0].tolist() == [11, 99, 99, 99, 99, 12]
    assert fields["response_mask"][0].tolist() == [1, 0, 0, 0, 0, 1]
    assert fields["rollout_log_probs"][0].tolist() == pytest.approx([-0.1, 0, 0, 0, 0, -0.2])
    assert fields["rm_scores"][0].tolist() == [0, 0, 0, 0, 0, 0.5]
    assert fields["rollout_topk_ids"][0][[5, 6, 7, 10]].tolist() == [[11, 13], [99, 98], [0, 1], [12, 14]]
    assert fields["rollout_support_complete"][0].logical_not().nonzero().flatten().tolist() == [5, 10]
    assert fields["routed_experts"][0].flatten().tolist() == [10, 0, 0, 0, 0, 30, 40, 0, 0, 0, 0, 0]
    assert teacher.call_args.kwargs["sequence_ids"] == output.prompt_ids + [11, 99, 99, 99, 99, 12]
    assert teacher.call_args.kwargs["vision_token_types"] == [-1, 0, 1, 2, 3, -1, -1, 0, 1, 2, 3, -1]
    assert fields["teacher_ids"][0].flatten().tolist() == list(range(12))


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_stores_nested_provider_coverage(monkeypatch):
    output = _output(
        {
            "response_topk_ids": [[11, 13], [12, 14]],
            "response_topk_log_probs": [[0.0, float("-inf")], [-0.2, -1.5]],
            "response_support_complete": [True, False],
        }
    )
    short_output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11],
        response_mask=[1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[11, 13]],
            "response_topk_log_probs": [[-0.1, -2.0]],
            "response_support_complete": [False],
        },
    )
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), [output, short_output], validate=False)
    complete = fields["rollout_support_complete"]
    assert complete.is_nested and complete.dtype == torch.bool
    assert complete[0].tolist() == [True, True, True, False, True]
    assert complete[1].tolist() == [True, False, True]


def test_greedy_reward_baseline_does_not_request_score_centering_heads():
    params = {"top_p": 0.9, "top_k": 64, "temperature": 1.0, "topk_log_probs": 128, "logprobs": True}
    apply_greedy_sampling_params(params)
    assert params == {"top_p": 1.0, "top_k": -1, "temperature": 0, "logprobs": True}


@pytest.mark.asyncio
async def test_agent_loop_tq_postprocess_skips_rollout_topk_on_validate(monkeypatch):
    fields = await _postprocess(monkeypatch, _DummyWorker(topk_log_probs=2), _output({}), validate=True)

    assert "rollout_topk_ids" not in fields.keys()
    assert "rollout_topk_log_probs" not in fields.keys()


@pytest.mark.parametrize("first_span, second_span", [(1, 1), (4, 5), (4, 1), (1, 5)])
def test_deepseek_processor_accepts_compact_and_expanded_images(monkeypatch, first_span, second_span):
    """DeepSeek-V4.1 prompts with one placeholder or a full span per image expand to the same tokens."""
    from transformers import BatchFeature

    worker = object.__new__(_WorkerTQ)
    worker.processor = SimpleNamespace(
        config=SimpleNamespace(
            model_type="deepseek_v41",
            image_token_id=129264,
            vision_config=SimpleNamespace(patch_size=14, downsample_ratio=3),
        )
    )
    monkeypatch.setattr(
        agent_loop_module,
        "build_multimodal_processor_inputs",
        lambda *args, **kwargs: BatchFeature({"image_grid_hws": torch.tensor([[3, 3], [3, 6]])}),
    )
    input_ids = [10] + [129264] * (first_span + second_span) + [12]
    result = worker._compute_multi_modal_inputs(
        SimpleNamespace(multi_modal_data={"images": [object(), object()]}, mm_processor_kwargs={}),
        torch.tensor(input_ids),
    )
    assert result["_expanded_input_ids"].tolist() == [[10] + [129264] * 9 + [12]]
    assert result["vision_token_types"].tolist() == [[-1, 0, 1, 2, 3, 0, 1, 1, 2, 3, -1]]


def test_deepseek_processor_rejects_ambiguous_adjacent_image_boundaries():
    with pytest.raises(ValueError, match="ambiguous compact/expanded span boundaries"):
        expand_image_tokens(torch.tensor([10] + [99] * 5 + [12]), torch.tensor([[1, 1], [1, 1]]), 99, 1)
