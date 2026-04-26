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

import json
import re
from dataclasses import dataclass
from typing import Any

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class FakeTokenizer:
    pad_token = "<pad>"

    def __init__(self):
        self.chat_template_calls: list[dict[str, Any]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        return_dict: bool = False,
        **kwargs,
    ) -> list[int] | str:
        del return_dict, kwargs
        rendered = self.render_messages(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        self.chat_template_calls.append(
            {
                "messages": json.loads(json.dumps(messages)),
                "tools": json.loads(json.dumps(tools)) if tools is not None else None,
                "rendered": rendered,
            }
        )
        if not tokenize:
            return rendered
        return self.encode(rendered)

    def render_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        del tools
        rendered_messages = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, sort_keys=True)
            if message.get("role") == "assistant":
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            rendered = f"{message['role']}:{content}"
            if message.get("tool_calls"):
                rendered += f":tool_calls={json.dumps(message['tool_calls'], sort_keys=True)}"
            rendered_messages.append(rendered)
        if add_generation_prompt:
            rendered_messages.append("assistant:")
        return "|".join(rendered_messages)


class FakeServerManager:
    def __init__(self, response_texts: list[str], extra_fields: list[dict[str, Any]] | None = None):
        self.response_texts = list(response_texts)
        self.extra_fields = list(extra_fields) if extra_fields is not None else None
        self.prompt_ids_history: list[list[int]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data
        self.prompt_ids_history.append(list(prompt_ids))
        response_text = self.response_texts.pop(0)
        token_ids = [ord(char) for char in response_text]
        extra_fields = (
            self.extra_fields.pop(0)
            if self.extra_fields is not None
            else {"global_steps": len(self.prompt_ids_history)}
        )
        return TokenOutput(
            token_ids=token_ids,
            log_probs=[0.0] * len(token_ids),
            extra_fields=extra_fields,
        )


@dataclass
class FakeTool:
    name: str = "calculator"

    def __post_init__(self):
        self.tool_schema = OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name=self.name,
                description="Calculator tool",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={"a": OpenAIFunctionPropertySchema(type="integer")},
                    required=["a"],
                ),
            ),
        )

    async def create(self, create_kwargs=None):
        del create_kwargs
        return "instance", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        del instance_id, parameters, kwargs
        return ToolResponse(text="tool result"), 0.0, {}

    async def release(self, instance_id):
        del instance_id


def _make_config(use_inference_chat_template: bool):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 256,
                    "response_length": 512,
                    "multi_turn": {
                        "enable": True,
                        "max_assistant_turns": None,
                        "tool_config_path": None,
                        "max_user_turns": None,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 256,
                        "tool_response_truncate_side": "middle",
                        "use_inference_chat_template": use_inference_chat_template,
                        "tokenization_sanity_check_mode": "strict",
                        "format": "hermes",
                    },
                },
                "model": {},
            },
            "data": {"apply_chat_template_kwargs": {}},
        }
    )


def _make_tool_agent_loop(
    use_inference_chat_template: bool,
    response_texts: list[str],
    extra_fields: list[dict[str, Any]] | None = None,
):
    config = _make_config(use_inference_chat_template)
    tokenizer = FakeTokenizer()
    server_manager = FakeServerManager(response_texts, extra_fields=extra_fields)
    loop = ToolAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )
    tool = FakeTool()
    loop.tools = {tool.name: tool}
    loop.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)]
    return loop, tokenizer, server_manager


@pytest.mark.asyncio
async def test_default_history_uses_flat_prompt_ids_for_generation():
    response_text = "plain answer"
    loop, tokenizer, server_manager = _make_tool_agent_loop(
        use_inference_chat_template=False,
        response_texts=[response_text],
    )
    raw_prompt = [{"role": "user", "content": "hello"}]

    output = await loop.run({}, raw_prompt=raw_prompt)

    expected_prompt = tokenizer.encode(tokenizer.render_messages(raw_prompt))
    assert server_manager.prompt_ids_history == [expected_prompt]
    assert output.response_ids == tokenizer.encode(response_text)
    assert output.response_mask == [1] * len(output.response_ids)
    assert output.extra_fields["use_inference_chat_template"] is False
    assert output.extra_fields["generation_vs_flat_prompt_delta"] == 0
    assert raw_prompt == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_inference_chat_template_uses_generation_messages_without_previous_reasoning():
    first_response = (
        '<think>secret scratchpad</think>visible answer<tool_call>{"name":"calculator","arguments":{"a":3}}</tool_call>'
    )
    second_response = "final response"
    loop, tokenizer, server_manager = _make_tool_agent_loop(
        use_inference_chat_template=True,
        response_texts=[first_response, second_response],
    )
    raw_prompt = [{"role": "user", "content": "hello"}]

    output = await loop.run({}, raw_prompt=raw_prompt)

    assert len(server_manager.prompt_ids_history) == 2
    second_prompt = tokenizer.decode(server_manager.prompt_ids_history[1])
    assert "secret scratchpad" not in second_prompt
    assert "visible answer" in second_prompt
    assert "tool:tool result" in second_prompt
    assert "tool_calls" in second_prompt
    assert "calculator" in second_prompt

    first_response_ids = tokenizer.encode(first_response)
    assert output.response_ids[: len(first_response_ids)] == first_response_ids
    assert output.response_mask[: len(first_response_ids)] == [1] * len(first_response_ids)
    assert 0 in output.response_mask
    assert output.extra_fields["use_inference_chat_template"] is True
    assert output.extra_fields["max_generation_prompt_length"] == len(server_manager.prompt_ids_history[1])
    assert raw_prompt == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_multi_round_generation_keeps_max_global_steps():
    first_response = 'visible answer<tool_call>{"name":"calculator","arguments":{"a":3}}</tool_call>'
    second_response = "final response"
    loop, _, _ = _make_tool_agent_loop(
        use_inference_chat_template=True,
        response_texts=[first_response, second_response],
        extra_fields=[
            {"max_global_steps": 7, "min_global_steps": 7},
            {"max_global_steps": 3, "min_global_steps": 3},
        ],
    )

    output = await loop.run({}, raw_prompt=[{"role": "user", "content": "hello"}])

    assert output.extra_fields["max_global_steps"] == 7


@pytest.mark.asyncio
async def test_inference_chat_template_uses_active_tool_schemas():
    response_text = "plain answer"
    loop, tokenizer, _ = _make_tool_agent_loop(
        use_inference_chat_template=True,
        response_texts=[response_text],
    )
    unused_tool = FakeTool(name="unused")
    loop.tools[unused_tool.name] = unused_tool
    loop.tool_schemas.append(unused_tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True))

    await loop.run(
        {},
        raw_prompt=[{"role": "user", "content": "hello"}],
        extra_info={"tool_selection": ["calculator"]},
    )

    generation_call = tokenizer.chat_template_calls[-1]
    assert [tool["function"]["name"] for tool in generation_call["tools"]] == ["calculator"]
