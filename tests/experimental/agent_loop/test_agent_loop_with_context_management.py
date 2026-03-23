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

from __future__ import annotations

from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, DictConfigWrap
from verl.experimental.agent_loop.agent_loop_with_context_management import SummarizerAgentLoop
from verl.experimental.agent_loop.context_manager import ContextState
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    """Char-level tokenizer mock for deterministic unit tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int] | str:
        del tools, kwargs
        text = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        return self.encode(text)


class _QueuedServerManager:
    """Minimal fake server manager that returns pre-seeded responses in order."""

    def __init__(self, tokenizer: _FakeTokenizer, responses: list[str]):
        self._tokenizer = tokenizer
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del sampling_params, image_data, video_data
        if not self._responses:
            raise AssertionError("No fake response left for _QueuedServerManager.generate().")

        response_text = self._responses.pop(0)
        response_ids = self._tokenizer.encode(response_text)
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), "response_text": response_text})
        return TokenOutput(
            token_ids=response_ids,
            log_probs=[0.0] * len(response_ids),
            num_preempted=0,
        )


def _build_loop(
    *, responses: list[str], max_context_compressions: int = 4
) -> tuple[SummarizerAgentLoop, _FakeTokenizer]:
    """Build a summarizer agent loop with deterministic fake dependencies for unit tests."""

    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 128, "response_length": 256},
                "model": {},
            },
            "data": {"apply_chat_template_kwargs": {}},
        }
    )
    tokenizer = _FakeTokenizer()
    loop = SummarizerAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_QueuedServerManager(tokenizer, responses),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
        max_context_compressions=max_context_compressions,
    )
    return loop, tokenizer


def _build_expected_summary_ids(tokenizer: _FakeTokenizer, summary_text: str) -> list[int]:
    system_prompt_ids = initialize_system_prompt(tokenizer)
    summary_ids = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": summary_text}],
        add_generation_prompt=True,
        tokenize=True,
    )
    return summary_ids[len(system_prompt_ids) :]


def test_summarizer_agent_loop_rejects_negative_max_context_compressions():
    with pytest.raises(ValueError, match="max_context_compressions must be non-negative"):
        _build_loop(responses=["hello"], max_context_compressions=-1)


@pytest.mark.asyncio
async def test_build_output_from_state_handles_empty_response():
    loop, _ = _build_loop(responses=[])
    state = ContextState(
        messages=[{"role": "user", "content": "hi"}],
        trajectory_ids=[1, 2, 3],
        response_mask=[],
        response_logprobs=[],
        metrics=AgentLoopMetrics(),
        extra_fields={"source": "test"},
    )

    output = loop._build_output_from_state(state)

    assert output.prompt_ids == [1, 2, 3]
    assert output.response_ids == []
    assert output.response_mask == []
    assert output.extra_fields["source"] == "test"
    assert output.extra_fields["turn_scores"] == []
    assert output.extra_fields["tool_rewards"] == []


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_returns_multiple_outputs_after_summary_compression():
    summary_text = "<summary>compressed summary</summary>"
    first_response = f"thinking...{summary_text}"
    second_response = "final answer"
    raw_prompt = [{"role": "user", "content": "hello"}]
    loop, tokenizer = _build_loop(responses=[first_response, second_response], max_context_compressions=1)

    outputs = await loop.run(sampling_params={}, raw_prompt=raw_prompt)

    assert len(outputs) == 2
    assert len(loop.server_manager.calls) == 2
    assert loop.server_manager.calls[0]["request_id"] == loop.server_manager.calls[1]["request_id"]

    first_output_text = tokenizer.decode(outputs[0].response_ids)
    second_output_text = tokenizer.decode(outputs[1].response_ids)
    summary_ids = _build_expected_summary_ids(tokenizer, summary_text)

    assert first_output_text == first_response
    assert second_output_text == tokenizer.decode(summary_ids) + second_response
    assert outputs[0].response_mask == [1] * len(outputs[0].response_ids)
    assert outputs[1].response_mask[: len(summary_ids)] == [0] * len(summary_ids)
    assert outputs[1].response_mask[len(summary_ids) :] == [1] * len(tokenizer.encode(second_response))


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_returns_single_output_without_summary():
    loop, tokenizer = _build_loop(responses=["plain final answer"], max_context_compressions=4)

    outputs = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "hello"}])

    assert len(outputs) == 1
    assert tokenizer.decode(outputs[0].response_ids) == "plain final answer"
    assert outputs[0].response_mask == [1] * len(outputs[0].response_ids)


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_respects_zero_max_context_compressions():
    summary_text = "<summary>compressed summary</summary>"
    first_response = f"thinking...{summary_text}"
    loop, tokenizer = _build_loop(responses=[first_response], max_context_compressions=0)

    outputs = await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "hello"}])

    assert len(outputs) == 1
    assert len(loop.server_manager.calls) == 1
    assert tokenizer.decode(outputs[0].response_ids) == first_response


@pytest.mark.asyncio
async def test_summarizer_agent_loop_run_supports_multiple_compressions_until_cap():
    summary1 = "<summary>summary 1</summary>"
    summary2 = "<summary>summary 2</summary>"
    responses = [
        f"step1...{summary1}",
        f"step2...{summary2}",
        "final answer",
    ]
    raw_prompt = [{"role": "user", "content": "hello"}]
    loop, tokenizer = _build_loop(responses=responses, max_context_compressions=2)

    outputs = await loop.run(sampling_params={}, raw_prompt=raw_prompt)

    assert len(outputs) == 3
    assert len(loop.server_manager.calls) == 3
    summary1_ids = _build_expected_summary_ids(tokenizer, summary1)
    summary2_ids = _build_expected_summary_ids(tokenizer, summary2)
    assert tokenizer.decode(outputs[0].response_ids) == responses[0]
    assert tokenizer.decode(outputs[1].response_ids) == tokenizer.decode(summary1_ids) + responses[1]
    assert tokenizer.decode(outputs[2].response_ids) == tokenizer.decode(summary2_ids) + responses[2]
