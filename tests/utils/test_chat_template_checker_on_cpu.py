# Copyright 2026 Individual Contributor: JiahaoTanXX
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
"""Offline coverage for reasoning-bearing chat-template checker trajectories."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from scripts import chat_template_checker as checker
from scripts import chat_template_mock_trajectories as trajectories
from verl.utils.tokenizer.continuous_token import ContinuousTokenBuilder

REASONING_TRAJECTORIES = ("singleturnreasoning", "multiturnreasoningtool", "multiturnreasoninguser")

CHAT_TEMPLATE = """{%- for message in messages -%}
{{- '<' + message['role'] + '>' -}}
{%- if message['role'] == 'assistant' and message['reasoning_content'] is defined -%}
{{- '<think>' + message['reasoning_content'] + '</think>' -}}
{%- endif -%}
{{- message['content'] -}}
{%- if message['tool_calls'] is defined -%}
{{- message['tool_calls'] | tojson -}}
{%- endif -%}
{{- eos_token -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- generation_prefix | default('<assistant>') -}}
{%- endif -%}"""


@pytest.fixture
def tokenizer():
    """Use real Transformers/Jinja rendering with a lossless, local byte tokenizer."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={char: index for index, char in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", chat_template=CHAT_TEMPLATE)


def _run_ct(tokenizer, trajectory, **kwargs):
    return checker.run_continuous_token_checks(
        tokenizer,
        trajectory,
        hf_model_type=None,
        model_family="default",
        custom_builder_module=None,
        chat_template_kwargs=kwargs,
    )


@pytest.mark.parametrize("reasoning", [None, "", "推理：2 + 2 = 4."])
def test_single_turn_reasoning_reaches_template(tokenizer, reasoning):
    trajectory = replace(trajectories.SINGLE_TURN_CHAT, assistant_reasoning_content=reasoning)
    assistant = checker._assistant_message_for_single_turn(trajectory)
    assert ("reasoning_content" in assistant) == (reasoning is not None)
    if reasoning is not None:
        assert assistant["reasoning_content"] == reasoning
    ids = checker._render_tokens(
        tokenizer, [*trajectory.raw_prompt, assistant], tools=None, add_generation_prompt=False, chat_template_kwargs={}
    )
    rendered = tokenizer.decode(ids)
    assert ("<think>" in rendered) == (reasoning is not None)
    if reasoning is not None:
        assert f"<think>{reasoning}</think>" in rendered


@pytest.mark.parametrize("reasoning", [None, "", "Check the weather before answering."])
def test_tool_assistant_keeps_reasoning_and_calls(reasoning):
    calls = [{"id": "call_weather", "type": "function", "function": {"name": "get_weather", "arguments": {}}}]
    assistant = trajectories._assistant("Checking.", calls, reasoning_content=reasoning)
    assert assistant["content"] == "Checking."
    assert assistant["tool_calls"] == calls
    assert ("reasoning_content" in assistant) == (reasoning is not None)
    if reasoning is not None:
        assert assistant["reasoning_content"] == reasoning


@pytest.mark.parametrize("name", REASONING_TRAJECTORIES)
def test_reasoning_trajectories_are_checked_without_mutation(tokenizer, name, monkeypatch):
    trajectory = trajectories.get_trajectory(name)
    before = deepcopy(trajectory)
    ct = _run_ct(tokenizer, trajectory)
    recorded_messages = []
    apply_template = tokenizer.apply_chat_template

    def record(messages, **kwargs):
        recorded_messages.append(deepcopy(messages))
        # Exercise defensive copies even with a renderer that mutates its inputs.
        result = apply_template(messages, **kwargs)
        for message in messages:
            if "reasoning_content" in message:
                message["reasoning_content"] = "renderer mutation"
        return result

    monkeypatch.setattr(tokenizer, "apply_chat_template", record)
    messages, tool_call_map = checker._assemble_text_messages(trajectory)
    assistants = [message for message in messages if message["role"] == "assistant"]
    assert assistants and all(message.get("reasoning_content") for message in assistants)
    assert sum(tool_call_map.values()) == (1 if name == "multiturnreasoningtool" else 0)
    raw = checker.run_raw_template_checks(tokenizer, trajectory, chat_template_kwargs={})
    assert len(raw) == (1 if name == "singleturnreasoning" else 3)
    assert all(result.passed for result in raw), raw
    assert len(ct) == (0 if name == "singleturnreasoning" else 1)
    assert all(result.passed for result in ct), ct
    assert trajectory == before
    for assistant in assistants:
        assert any(assistant in batch for batch in recorded_messages)


@pytest.mark.parametrize("name", REASONING_TRAJECTORIES[1:])
def test_checker_detects_corruption_of_reasoning_prefix(tokenizer, name, monkeypatch):
    trajectory = trajectories.get_trajectory(name)

    class CorruptingBuilder(ContinuousTokenBuilder):
        def merge_context_tokens(self, previous_messages, updated_messages, runtime_token_ids, **kwargs):
            result = super().merge_context_tokens(previous_messages, updated_messages, runtime_token_ids, **kwargs)
            text = tokenizer.decode(result.token_ids)
            reasoning = previous_messages[-1]["reasoning_content"]
            assert reasoning in text
            corrupted = text.replace(reasoning, "CORRUPTED REASONING", 1)
            return replace(result, token_ids=tokenizer.encode(corrupted, add_special_tokens=False))

    monkeypatch.setattr(
        checker, "create_continuous_token_builder", lambda *args, **kwargs: CorruptingBuilder(tokenizer)
    )
    results = _run_ct(tokenizer, trajectory)
    assert len(results) == 1
    assert not results[0].passed
    assert "Token mismatch" in results[0].error


@pytest.mark.parametrize("strip_history", [False, True])
def test_cli_distinguishes_raw_warnings_from_ct_mismatches(tokenizer, monkeypatch, capsys, strip_history):
    if strip_history:
        # A follow-up user turn makes the template discard earlier assistant reasoning.
        tokenizer.chat_template = CHAT_TEMPLATE.replace(
            "message['reasoning_content'] is defined",
            "message['reasoning_content'] is defined and messages[-1]['role'] != 'user'",
        )
    kwargs = {"generation_prefix": "<assistant><think>\n"}
    selected = tuple(trajectories.get_trajectory(name) for name in REASONING_TRAJECTORIES)
    args = SimpleNamespace(
        model="offline/reasoning-template",
        template=None,
        allow_download=False,
        chat_template_kwargs=kwargs,
        model_family="default",
        custom_builder_module=None,
        enable_multimodal=False,
        skip_vl=True,
        show_traceback=False,
    )
    monkeypatch.setattr(checker, "parse_args", lambda: args)
    monkeypatch.setattr(checker, "_load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(checker.PretrainedConfig, "get_config_dict", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(checker, "TRAJECTORIES", selected)
    assert checker.main() == (1 if strip_history else 0)
    output = capsys.readouterr().out
    assert "[WARN]" in output
    if strip_history:
        assert "Continuous Token 1/2 passed" in output
        assert "multiturnreasoninguser.merge_at3.user" in output
        assert "Token mismatch" in output
        assert "Verdict: FAIL" in output
    else:
        assert "Continuous Token 2/2 passed" in output
        assert "PASS with raw-prefix warnings" in output


def test_existing_trajectories_keep_their_previous_behavior(tokenizer):
    legacy_names = (
        "singleturnchat",
        "multiturnsingletool",
        "multiturnmultitool",
        "multiturnretryuser",
        "multiturnretrysystem",
    )
    assert len({trajectory.name for trajectory in trajectories.TRAJECTORIES}) == len(trajectories.TRAJECTORIES)
    for name in legacy_names:
        trajectory = trajectories.get_trajectory(name)
        messages, _ = checker._assemble_text_messages(trajectory)
        assert all("reasoning_content" not in message for message in messages)
        results = checker.run_raw_template_checks(tokenizer, trajectory, chat_template_kwargs={})
        results += _run_ct(tokenizer, trajectory)
        assert all(result.passed for result in results), results
