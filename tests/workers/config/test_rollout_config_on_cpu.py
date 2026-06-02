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

import pytest

from verl.workers.config import AgentLoopConfig, MultiTurnConfig, RolloutConfig


def test_vllm_multi_turn_rejects_server_side_tool_parser_args():
    with pytest.raises(ValueError, match="client-side AgentLoop tool parsing"):
        RolloutConfig(
            name="vllm",
            multi_turn=MultiTurnConfig(enable=True),
            engine_kwargs={"vllm": {"enable_auto_tool_choice": True, "tool_call_parser": "hermes"}},
        )


def test_vllm_multi_turn_rejects_server_side_tool_parser_args_from_dict_config():
    with pytest.raises(ValueError, match="client-side AgentLoop tool parsing"):
        RolloutConfig(
            name="vllm",
            multi_turn={"enable": True},
            engine_kwargs={"vllm": {"tool_parser_plugin": "custom_parser.py"}},
        )


def test_vllm_tool_agent_rejects_server_side_tool_parser_args():
    with pytest.raises(ValueError, match="client-side AgentLoop tool parsing"):
        RolloutConfig(
            name="vllm",
            agent=AgentLoopConfig(default_agent_loop="tool_agent"),
            engine_kwargs={"vllm": {"tool_call_parser": "hermes"}},
        )


def test_vllm_function_tool_path_rejects_server_side_tool_parser_args():
    with pytest.raises(ValueError, match="client-side AgentLoop tool parsing"):
        RolloutConfig(
            name="vllm",
            multi_turn=MultiTurnConfig(function_tool_path="tools.py"),
            engine_kwargs={"vllm": {"enable_auto_tool_choice": True}},
        )


def test_vllm_chat_completion_tool_parser_args_still_allowed_without_multi_turn():
    config = RolloutConfig(
        name="vllm",
        multi_turn=MultiTurnConfig(enable=False),
        engine_kwargs={"vllm": {"enable_auto_tool_choice": True, "tool_call_parser": "hermes"}},
    )

    assert config.engine_kwargs["vllm"]["tool_call_parser"] == "hermes"


def test_vllm_multi_turn_allows_unrelated_engine_kwargs():
    config = RolloutConfig(
        name="vllm",
        multi_turn=MultiTurnConfig(enable=True),
        engine_kwargs={"vllm": {"enable_auto_tool_choice": False, "gpu_memory_utilization": 0.75}},
    )

    assert config.engine_kwargs["vllm"]["gpu_memory_utilization"] == 0.75
