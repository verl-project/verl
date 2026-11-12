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

import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl.experimental.agent_loop.agent_loop as agent_loop_module
from verl.experimental.agent_loop import AgentLoopWorker
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.workers.config import RolloutConfig
from verl.workers.config.rollout import AgentLoopConfig, MultiTurnConfig


class CustomSingleTurnAgentLoop(SingleTurnAgentLoop):
    pass


class CustomMultiTurnAgentLoop(SingleTurnAgentLoop):
    merges_context_messages = True


class CustomToolAgentLoop(ToolAgentLoop):
    pass


_CUSTOM_LOOPS = {
    "custom_single_turn": {"_target_": f"{__name__}.CustomSingleTurnAgentLoop"},
    "custom_multi_turn": {"_target_": f"{__name__}.CustomMultiTurnAgentLoop"},
    "custom_tool": {"_target_": f"{__name__}.CustomToolAgentLoop"},
    "unresolvable": {"_target_": f"{__name__}.MissingAgentLoop"},
}


def _worker(monkeypatch, *, default_agent_loop="single_turn_agent", multi_turn=False, tools=(), mode="strict"):
    for name, config in _CUSTOM_LOOPS.items():
        monkeypatch.setitem(agent_loop_module._agent_loop_registry, name, config)
    rollout_config = RolloutConfig(
        name="vllm",
        agent=AgentLoopConfig(default_agent_loop=default_agent_loop),
        multi_turn=MultiTurnConfig(enable=multi_turn, tokenization_sanity_check_mode=mode),
    )

    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create({"data": {"apply_chat_template_kwargs": {"enable_thinking": False}}})
    worker.rollout_config = rollout_config
    worker.model_config = SimpleNamespace(generation_config=None)
    worker.tokenizer = object()
    worker.processor = None
    worker.hf_model_type = "smollm3"
    worker.mm_processor_kwargs = {}
    worker.tools = list(tools)
    return worker


@pytest.fixture
def calls(monkeypatch):
    builder_calls, verify_calls = [], []

    def fake_create(tokenizer, **kwargs):
        builder_calls.append(kwargs)
        return "builder"

    monkeypatch.setattr(agent_loop_module, "create_continuous_token_builder", fake_create)
    monkeypatch.setattr(
        agent_loop_module,
        "verify_continuous_token_builder",
        lambda builder, **kwargs: verify_calls.append((builder, kwargs)),
    )
    return SimpleNamespace(builder=builder_calls, verify=verify_calls)


@pytest.mark.parametrize(
    ("worker_kwargs", "custom_agent_names", "checked"),
    [
        # Single-turn runs never merge context, whatever builder the model falls back to.
        ({}, [], False),
        # A custom single-turn loop from agent_loop_config_path, used as the default or not.
        ({}, ["custom_single_turn"], False),
        ({"default_agent_loop": "custom_single_turn"}, ["custom_single_turn"], False),
        # Datasets route multi-turn samples to the built-in tool agent.
        ({"multi_turn": True}, [], True),
        ({"tools": [object()]}, [], True),
        ({"default_agent_loop": "tool_agent"}, [], True),
        # Custom loops that declare the flag, directly or by subclassing the tool agent.
        ({}, ["custom_multi_turn"], True),
        ({"default_agent_loop": "custom_tool"}, ["custom_tool"], True),
        ({}, ["custom_single_turn", "custom_tool"], True),
        # Names that are not registered are not checked.
        ({"default_agent_loop": "not_registered"}, [], False),
        # Disabling the check skips it even for loops that merge context.
        ({"multi_turn": True, "mode": "disable"}, ["custom_tool"], False),
    ],
)
def test_check_runs_only_for_context_merging_loops(monkeypatch, calls, worker_kwargs, custom_agent_names, checked):
    _worker(monkeypatch, **worker_kwargs)._check_continuous_token_builder(custom_agent_names)
    assert bool(calls.builder) == bool(calls.verify) == checked


def test_check_forwards_worker_settings(monkeypatch, calls):
    _worker(monkeypatch, multi_turn=True, mode="ignore_strippable")._check_continuous_token_builder([])
    assert calls.builder == [
        {
            "hf_model_type": "smollm3",
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {},
            "processor": None,
        }
    ]
    assert calls.verify == [
        ("builder", {"mode": "ignore_strippable", "hf_model_type": "smollm3", "generation_config": None})
    ]


def test_unresolvable_agent_loop_is_checked(monkeypatch, calls, caplog):
    with caplog.at_level(logging.WARNING, logger=agent_loop_module.logger.name):
        _worker(monkeypatch)._check_continuous_token_builder(["unresolvable"])
    assert len(calls.verify) == 1
    assert "Could not resolve agent loop 'unresolvable', assuming it merges context" in caplog.text
