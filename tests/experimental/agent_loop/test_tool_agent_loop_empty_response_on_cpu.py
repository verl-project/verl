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
"""CPU regression test: an empty first-turn generation must not turn the prompt into the response."""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder
from verl.workers.rollout.replica import TokenOutput

PROMPT_IDS = [11, 12, 13, 14, 15]


class _FixedOutputServer:
    def __init__(self, output: TokenOutput):
        self.output = output

    async def generate(self, request_id: str, *, prompt_ids: list[int], **kwargs: Any) -> TokenOutput:
        del request_id, prompt_ids, kwargs
        return self.output


class _NoToolCallsParser:
    stop_token_ids: list[int] = []

    async def extract_tool_calls(self, response_ids: list[int], tools: list[Any]) -> tuple[str, list[Any]]:
        del response_ids, tools
        return "", []


def _make_loop(output: TokenOutput, response_length: int = 8) -> ToolAgentLoop:
    # Bypass __init__ (it needs a full trainer config); set only what run() reads.
    loop = ToolAgentLoop.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(full_determinism=False)
    loop.processor = None
    loop.mm_processor_kwargs = {}
    loop.continuous_token_builder = ContinuousTokenBuilder(tokenizer=None)
    loop.loop = asyncio.get_running_loop()
    loop.server_manager = _FixedOutputServer(output)
    loop.tool_parser = _NoToolCallsParser()
    loop.tools = {}
    loop.tool_schemas = []
    loop.response_length = response_length
    loop.max_assistant_turns = 0
    loop.max_user_turns = 0
    loop.max_parallel_calls = 1

    async def ct_build_initial_tokens(messages, **kwargs):
        del messages, kwargs
        return list(PROMPT_IDS)

    loop.ct_build_initial_tokens = ct_build_initial_tokens
    return loop


class TestToolAgentLoopEmptyResponseOnCpu(unittest.IsolatedAsyncioTestCase):
    async def _run(self, output: TokenOutput):
        loop = _make_loop(output)
        return await loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "hi"}])

    async def test_empty_first_turn_keeps_prompt_out_of_response(self) -> None:
        # vLLM returns this when a request is aborted before its first token.
        result = await self._run(TokenOutput(token_ids=[], log_probs=None, stop_reason="aborted"))

        assert result.prompt_ids == PROMPT_IDS
        assert result.response_ids == []
        assert result.response_mask == []
        assert result.response_logprobs is None
        assert result.num_turns == 2

    async def test_non_empty_first_turn_splits_prompt_and_response(self) -> None:
        result = await self._run(TokenOutput(token_ids=[21, 22, 23], log_probs=[-0.1, -0.2, -0.3]))

        assert result.prompt_ids == PROMPT_IDS
        assert result.response_ids == [21, 22, 23]
        assert result.response_mask == [1, 1, 1]
        assert result.response_logprobs == [-0.1, -0.2, -0.3]


if __name__ == "__main__":
    unittest.main()
