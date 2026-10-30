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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop import agent_loop as module
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.workers.rollout.replica import TokenOutput


class RecordingClient:
    def __init__(self, tokens):
        self.tokens = tokens
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        # Matches vLLM with logprobs disabled; the frozen teacher never receives a weight version.
        return TokenOutput(token_ids=self.tokens, extra_fields={"min_global_steps": None, "max_global_steps": None})


def make_worker(monkeypatch, *, multi_modal=None, tokens=None):
    config = OmegaConf.create(
        {
            "distillation": {"rollout_source_selector": {"path": "my_policy.py", "name": "choose"}},
            "data": {},
            "actor_rollout_ref": {"rollout": {"response_length": 4, "ignore_eos": True}},
        }
    )
    loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
    loop.config = config
    loop.rollout_config = config.actor_rollout_ref.rollout
    loop.response_length = 4

    async def parse(messages):
        return multi_modal or {}

    async def build(*args, **kwargs):
        return [101, 102]

    async def merge(prompt, response, *args, **kwargs):
        return SimpleNamespace(token_ids=prompt + response), [1] * len(response), kwargs["assistant_logprobs"]

    loop.process_multi_modal_info = parse
    loop.ct_build_initial_tokens = build
    loop.ct_merge_assistant_token = merge
    loop._get_mm_processor_kwargs = lambda audios: {}
    loop._assert_mm_supported = lambda enabled: None
    loop.build_sglang_video_payload = lambda *args: None
    worker = SimpleNamespace(
        config=config,
        teacher_rollouts=module.is_teacher_rollout_enabled(config),
        rollout_config=config.actor_rollout_ref.rollout,
        llm_client=RecordingClient([11, 12]),
        teacher_client={"default": RecordingClient(tokens or [21, 22, 23, 24])},
        tokenizer=None,
        processor=None,
        hf_model_type=None,
        dataset_cls=None,
        tools=[],
    )

    def instantiate(**kwargs):
        loop.server_manager = kwargs["server_manager"]
        return loop

    async def postprocess(output, validate, **kwargs):
        output.extra_fields["effective_source"] = kwargs.get("rollout_source", "student")
        return output

    worker._agent_loop_postprocess = postprocess
    monkeypatch.setattr(module.hydra.utils, "instantiate", instantiate)
    return worker


async def run(worker, source="teacher", *, validate=False, agent_name="single_turn_agent"):
    return await module.AgentLoopWorker._run_agent_loop(
        worker,
        {"temperature": 0.7 if validate else 1.0, "logprobs": False},
        {"step": 3, "sample_index": 0, "rollout_n": 0, "validate": validate},
        agent_name=agent_name,
        rollout_source=source,
        trace=False,
        raw_prompt=[{"role": "user", "content": "hello"}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens,want", [([21, 22, 23, 24], [21, 22, 23, 24]), ([21], [21]), ([21] * 6, [21] * 4)])
async def test_teacher_uses_response_budget_and_actual_returned_ids(monkeypatch, tokens, want):
    worker = make_worker(monkeypatch, tokens=tokens)
    result = await run(worker)
    assert result.response_ids == want
    assert result.response_mask == [1] * len(want)
    assert not worker.llm_client.calls
    params = worker.teacher_client["default"].calls[0]["sampling_params"]
    assert params == {"temperature": 1.0, "logprobs": False, "max_tokens": 4, "ignore_eos": True}
    assert result.response_logprobs is None
    assert result.routed_experts is None
    assert result.extra_fields["effective_source"] == "teacher"
    assert result.extra_fields["min_global_steps"] is None


@pytest.mark.asyncio
async def test_validation_overrides_stale_teacher_source(monkeypatch):
    worker = make_worker(monkeypatch)
    result = await run(worker, validate=True)
    assert result.response_ids == [11, 12]
    assert not worker.teacher_client["default"].calls
    assert worker.llm_client.calls[0]["sampling_params"]["temperature"] == 0.7
    assert result.extra_fields["effective_source"] == "student"


@pytest.mark.asyncio
async def test_opt_in_rejects_dataset_agent_before_generation(monkeypatch):
    worker = make_worker(monkeypatch)
    with pytest.raises(ValueError, match="single_turn_agent"):
        await run(worker, agent_name="custom_agent")
    assert not worker.llm_client.calls and not worker.teacher_client["default"].calls


@pytest.mark.asyncio
@pytest.mark.parametrize("modality", ["images", "videos", "audios"])
@pytest.mark.parametrize("source", ["student", "teacher"])
async def test_opt_in_rejects_parsed_multimodal_input_before_generation(monkeypatch, modality, source):
    worker = make_worker(monkeypatch, multi_modal={modality: [object()]})
    with pytest.raises(ValueError, match="text-only"):
        await run(worker, source)
    assert not worker.llm_client.calls and not worker.teacher_client["default"].calls
