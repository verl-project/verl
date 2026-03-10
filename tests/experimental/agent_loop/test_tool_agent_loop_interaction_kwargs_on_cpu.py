#!/usr/bin/env python3
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

import json
from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.interactions.base import BaseInteraction
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs: Any,
    ):
        del tools, add_generation_prompt, kwargs
        if not tokenize:
            return "<prompt>"
        # Return different lengths for 1 vs 2 messages so system prompt extraction works.
        return [101, 102, 103] if len(messages) == 1 else [101, 102, 103, 104, 105]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


class _FakeServerManager:
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, prompt_ids, sampling_params, image_data, video_data
        return TokenOutput(token_ids=[11, 12], log_probs=[0.0, 0.0])


class DummyInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config=config)
        self.started: bool = False

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        del kwargs
        self.started = True
        return await super().start_interaction(instance_id=instance_id)

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> tuple[bool, str, float, dict[str, Any]]:
        del instance_id, messages, kwargs
        return True, "done", 0.0, {}


_DUMMY_INTERACTION_CLS = "tests.experimental.agent_loop.test_tool_agent_loop_interaction_kwargs_on_cpu.DummyInteraction"


def _make_loop(*, interaction_config_path: str) -> ToolAgentLoop:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 16,
                    "response_length": 16,
                    "multi_turn": {
                        "max_user_turns": 1,
                        "max_assistant_turns": 1,
                        "max_parallel_calls": 1,
                        "max_tool_response_length": 64,
                        "tool_response_truncate_side": "left",
                        "tool_config_path": None,
                        "format": "hermes",
                        "interaction_config_path": interaction_config_path,
                    },
                }
            },
            "data": {"apply_chat_template_kwargs": {}},
        }
    )

    trainer_config = DictConfigWrap(config)
    dataset_config = DictConfigWrap(config.data)
    return ToolAgentLoop(
        trainer_config=trainer_config,
        server_manager=_FakeServerManager(),
        tokenizer=_FakeTokenizer(),
        processor=None,
        dataset_cls=RLHFDataset,
        dataset_config=dataset_config,
    )


@pytest.mark.asyncio
async def test_tool_agent_loop_allows_missing_interaction_kwargs_on_cpu(tmp_path):
    interaction_config = {
        "interaction": [
            {
                "name": "dummy",
                "class_name": _DUMMY_INTERACTION_CLS,
                "config": {},
            }
        ]
    }
    interaction_path = tmp_path / "interaction.json"
    interaction_path.write_text(json.dumps(interaction_config))

    loop = _make_loop(interaction_config_path=str(interaction_path))
    out = await loop.run(
        sampling_params={},
        raw_prompt=[{"role": "user", "content": "hi"}],
        extra_info={},
    )
    assert out.response_ids == [11, 12]


@pytest.mark.asyncio
async def test_tool_agent_loop_raises_on_unknown_interaction_name_on_cpu(tmp_path):
    interaction_config = {
        "interaction": [
            {
                "name": "dummy",
                "class_name": _DUMMY_INTERACTION_CLS,
                "config": {},
            }
        ]
    }
    interaction_path = tmp_path / "interaction.json"
    interaction_path.write_text(json.dumps(interaction_config))

    loop = _make_loop(interaction_config_path=str(interaction_path))
    with pytest.raises(ValueError, match="Interaction 'nope' not found in interaction_map"):
        await loop.run(
            sampling_params={},
            raw_prompt=[{"role": "user", "content": "hi"}],
            extra_info={"interaction_kwargs": {"name": "nope"}},
        )
