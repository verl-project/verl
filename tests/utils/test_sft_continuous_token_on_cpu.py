# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import copy
import os
from pathlib import Path

import pytest
from transformers import AutoProcessor, AutoTokenizer

import verl.utils.tokenizer.sft_continuous_token as sft_continuous_token_module
from tests.utils.test_continuous_token_on_cpu import (
    _BlockReplacingTemplateProcessor,
    _DeepSeekAssistantTokenizer,
    _DeepSeekV31AssistantTokenizer,
    _Gemma4AssistantTokenizer,
    _Gemma4E4BAssistantTokenizer,
    _GLMAssistantTokenizer,
    _MiniMaxAssistantTokenizer,
    _MiniMaxVLAssistantTokenizer,
    _MockMiniMaxVLAssistantProcessor,
    _MockQwenVLProcessor,
    _MockQwenVLTokenizer,
    _QwenBoundaryTokenizer,
    _RecordingTemplateProcessor,
    _RecordingTemplateTokenizer,
    _TemplateTokenizer,
)
from verl.utils.tokenizer.continuous_token import (
    ContinuousTokenBuilder,
    DeepSeekContinuousTokenBuilder,
    DeepSeekVL2ContinuousTokenBuilder,
    Gemma4ContinuousTokenBuilder,
    Gemma4VLContinuousTokenBuilder,
    GLM46VContinuousTokenBuilder,
    GLMContinuousTokenBuilder,
    GptOssContinuousTokenBuilder,
    KimiVLContinuousTokenBuilder,
    MiniMaxContinuousTokenBuilder,
    MiniMaxVLContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    QwenVLContinuousTokenBuilder,
    VLContinuousTokenBuilder,
)
from verl.utils.tokenizer.deepseek import DeepSeekV4ContinuousTokenBuilder
from verl.utils.tokenizer.sft_continuous_token import (
    _AssistantReconstructor,
    _DeepSeekReconstructor,
    _DeepSeekV4Reconstructor,
    _Gemma4Reconstructor,
    _GLMReconstructor,
    _GptOssReconstructor,
    _KimiVLReconstructor,
    _MiniMaxReconstructor,
    _MiniMaxVLReconstructor,
    _QwenReconstructor,
    _resolve_reconstructor,
    _SFTDeepSeekV4ContinuousTokenBuilder,
    _SFTMiniMaxVLContinuousTokenBuilder,
    adapt_continuous_token_builder_for_sft,
    reconstruct_assistant_tokens,
    validate_sft_tool_support,
)


@pytest.fixture
def minimax_template_builder():
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        bos_token="<beginning_of_sentence>",
        eos_token="<end_of_sentence>",
        additional_special_tokens=["[e~["],
    )
    template = (
        "{% for message in messages %}"
        "{% if keep_responses or message.role != 'function' %}"
        "{{ '<beginning_of_sentence>' }}{{ message.role }}:{{ message.content }}{{ '<end_of_sentence>' }}"
        "{% endif %}{% endfor %}"
        "{% if keep_schemas and tools %}{{ tools | tojson }}{% endif %}"
        "{{ '<beginning_of_sentence>ai' }}"
    )
    tokenizer.chat_template = template

    class Processor(_MockMiniMaxVLAssistantProcessor):
        def apply_chat_template(self, messages, **kwargs):
            return tokenizer.apply_chat_template(messages, **kwargs)

    def build(**kwargs):
        return MiniMaxVLContinuousTokenBuilder(tokenizer, Processor(tokenizer), chat_template_kwargs=kwargs)

    return build


@pytest.mark.parametrize("missing,configured_schema", [("schemas", False), ("schemas", True), ("responses", False)])
def test_minimax_sft_rejects_selected_template_omissions(minimax_template_builder, missing, configured_schema):
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    kwargs = {"keep_schemas": missing != "schemas", "keep_responses": missing != "responses"}
    if missing == "schemas" and configured_schema:
        kwargs["tools"], tools = tools, None
    builder = minimax_template_builder(**kwargs)
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "gold"}]
    if missing == "responses":
        tools = None
        messages.append({"role": "tool", "name": "lookup", "content": "value"})
    original_messages = copy.deepcopy(messages)
    original_kwargs = copy.deepcopy(builder.chat_template_kwargs)
    with pytest.raises(
        ValueError, match=f"MiniMax-VL SFT selected a chat template that does not render tool {missing}"
    ):
        validate_sft_tool_support(builder, messages, tools=tools)
    assert messages == original_messages
    assert builder.chat_template_kwargs == original_kwargs


def test_minimax_sft_rejects_initial_tool_before_rendering(minimax_template_builder, monkeypatch):
    builder = minimax_template_builder(keep_schemas=True, keep_responses=True)
    messages = [
        {"role": "user", "content": "question"},
        {"role": "tool", "name": "lookup", "content": "value"},
        {"role": "assistant", "content": "gold"},
    ]

    def unexpected_render(*args, **kwargs):
        pytest.fail("Initial tool response reached template rendering")

    monkeypatch.setattr(builder.processor, "apply_chat_template", unexpected_render)
    with pytest.raises(ValueError, match="MiniMax-VL SFT does not support initial tool responses"):
        validate_sft_tool_support(builder, messages)


@pytest.mark.parametrize("tool_content", [None, "", "value"])
def test_minimax_sft_accepts_tool_template_and_resolves_response_name(minimax_template_builder, tool_content):
    builder = minimax_template_builder(keep_schemas=True, keep_responses=True)
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call0", "function": {"name": "lookup"}}]},
        {"role": "tool", "tool_call_id": "call0", "content": tool_content},
    ]
    original = copy.deepcopy(messages)
    validate_sft_tool_support(builder, messages, tools=tools)
    assert messages == original


@pytest.mark.parametrize(
    ("builder_cls", "expected_handler"),
    [
        (ContinuousTokenBuilder, _AssistantReconstructor),
        (VLContinuousTokenBuilder, _AssistantReconstructor),
        (GptOssContinuousTokenBuilder, _GptOssReconstructor),
        (QwenContinuousTokenBuilder, _QwenReconstructor),
        (QwenVLContinuousTokenBuilder, _QwenReconstructor),
        (MiniMaxContinuousTokenBuilder, _MiniMaxReconstructor),
        (MiniMaxVLContinuousTokenBuilder, _MiniMaxVLReconstructor),
        (GLMContinuousTokenBuilder, _GLMReconstructor),
        (GLM46VContinuousTokenBuilder, _GLMReconstructor),
        (Gemma4ContinuousTokenBuilder, _Gemma4Reconstructor),
        (Gemma4VLContinuousTokenBuilder, _Gemma4Reconstructor),
        (DeepSeekContinuousTokenBuilder, _DeepSeekReconstructor),
        (DeepSeekV4ContinuousTokenBuilder, _DeepSeekV4Reconstructor),
        (KimiVLContinuousTokenBuilder, _KimiVLReconstructor),
    ],
)
def test_sft_reconstructor_follows_builder_mro(builder_cls, expected_handler):
    builder = object.__new__(builder_cls)

    assert _resolve_reconstructor(builder) is expected_handler


def test_unregistered_builder_subclass_falls_back_to_base_reconstructor():
    class CustomContinuousTokenBuilder(ContinuousTokenBuilder):
        pass

    builder = object.__new__(CustomContinuousTokenBuilder)

    assert _resolve_reconstructor(builder) is _AssistantReconstructor


def test_deepseek_v4_sft_builder_keeps_committed_reasoning_when_drop_thinking_is_enabled():
    tokenizer = _DeepSeekAssistantTokenizer()
    runtime_builder = DeepSeekV4ContinuousTokenBuilder(
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "drop_thinking": True},
        allowed_append_roles={"user"},
    )
    builder = adapt_continuous_token_builder_for_sft(runtime_builder)
    assert builder.chat_template_kwargs == runtime_builder.chat_template_kwargs
    previous_messages = [{"role": "user", "content": "q1"}]
    runtime_ids = builder.build_initial_tokens(previous_messages)
    assistant = {"role": "assistant", "reasoning_content": "reason A", "content": "answer A"}

    assistant_ids = reconstruct_assistant_tokens(
        builder,
        assistant,
        previous_messages=previous_messages,
    )

    reason_ids = tokenizer.encode("reason A", add_special_tokens=False)
    assert assistant_ids[: len(reason_ids)] == reason_ids
    runtime_ids = builder.merge_assistant_tokens(runtime_ids, assistant_ids).token_ids
    previous_messages = [*previous_messages, assistant]
    updated_messages = [*previous_messages, {"role": "user", "content": "q2"}]
    result = builder.merge_non_assistant_tokens(previous_messages, updated_messages, runtime_ids)
    expected_append = tokenizer.encode("<｜User｜>q2<｜Assistant｜><think>", add_special_tokens=False)
    assert result.token_ids == runtime_ids + expected_append
    assert result.appended_token_count == len(expected_append)
    previous_mask = [1] * len(runtime_ids)
    mask, _ = builder.align_response_metadata(result, previous_mask)
    assert mask == previous_mask + [0] * len(expected_append)
    assert builder.build_initial_tokens(updated_messages) == runtime_builder.build_initial_tokens(updated_messages)
    with pytest.raises(ValueError, match="drop_thinking"):
        runtime_builder.merge_non_assistant_tokens(previous_messages, updated_messages, runtime_ids)
    with pytest.raises(ValueError, match="only supports appending roles"):
        builder.tokenize_non_assistant_incremental_messages(
            previous_messages, [*previous_messages, {"role": "system", "content": "policy"}]
        )


@pytest.mark.parametrize("minimax", [False, True], ids=["deepseek_v4", "minimax_vl"])
def test_sft_adapter_refuses_unadapted_custom_subclasses(minimax):
    base_cls = MiniMaxVLContinuousTokenBuilder if minimax else DeepSeekV4ContinuousTokenBuilder

    class CustomBuilder(base_cls):
        pass

    tokenizer = _MiniMaxVLAssistantTokenizer() if minimax else _DeepSeekAssistantTokenizer()
    args = (tokenizer, _MockMiniMaxVLAssistantProcessor(tokenizer)) if minimax else (tokenizer,)
    builder = CustomBuilder(*args)
    with pytest.raises(ValueError, match="SFT cannot automatically adapt custom builder CustomBuilder"):
        adapt_continuous_token_builder_for_sft(builder)


@pytest.mark.parametrize("minimax", [False, True], ids=["deepseek_v4", "minimax_vl"])
def test_sft_adapter_preserves_custom_sft_policy_across_repeated_adaptation(minimax):
    base_cls = _SFTMiniMaxVLContinuousTokenBuilder if minimax else _SFTDeepSeekV4ContinuousTokenBuilder

    class PolicyBuilder(base_cls):
        def __init__(self, *args, blocked_content, **kwargs):
            super().__init__(*args, **kwargs)
            self.blocked_content = blocked_content
            self.seen = []

        def merge_non_assistant_tokens(self, previous_messages, updated_messages, runtime_token_ids, **kwargs):
            content = updated_messages[-1]["content"]
            self.seen.append(content)
            if content == self.blocked_content:
                raise ValueError("Custom append policy refused this message")
            return super().merge_non_assistant_tokens(previous_messages, updated_messages, runtime_token_ids, **kwargs)

    tokenizer = _MiniMaxVLAssistantTokenizer() if minimax else _DeepSeekAssistantTokenizer()
    args = (tokenizer, _MockMiniMaxVLAssistantProcessor(tokenizer)) if minimax else (tokenizer,)
    builder = PolicyBuilder(*args, blocked_content="blocked")
    previous = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "gold"}]
    runtime_ids = builder.build_initial_tokens(previous)
    builder = adapt_continuous_token_builder_for_sft(builder)
    result = builder.merge_non_assistant_tokens(
        previous, [*previous, {"role": "user", "content": "again"}], runtime_ids
    )
    assert result.token_ids[: len(runtime_ids)] == runtime_ids
    assert result.appended_token_count > 0
    builder = adapt_continuous_token_builder_for_sft(builder)
    with pytest.raises(ValueError, match="Custom append policy refused this message"):
        builder.merge_non_assistant_tokens(previous, [*previous, {"role": "user", "content": "blocked"}], runtime_ids)
    assert builder.seen == ["again", "blocked"]


@pytest.mark.parametrize("operation", ["reconstruct", "merge", "handler"])
@pytest.mark.parametrize(
    "unsupported",
    ["schema", "template_schema", "tool_call", "tool_response", "historical_tool_call", "historical_tool_response"],
)
def test_kimi_vl_sft_entries_reject_before_rendering(operation, unsupported, monkeypatch):
    processor = _MockQwenVLProcessor()
    builder = KimiVLContinuousTokenBuilder(_QwenBoundaryTokenizer(), processor)
    message = {"role": "assistant", "content": "gold"}
    previous = [{"role": "user", "content": "question"}]
    tools = None
    if unsupported in {"schema", "template_schema"}:
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        if unsupported == "template_schema":
            builder.chat_template_kwargs["tools"], tools = tools, None
    elif unsupported == "tool_call":
        message["tool_calls"] = [{"function": {"name": "lookup"}}]
    elif unsupported == "tool_response":
        message = {"role": "tool", "content": "value"}
    elif unsupported == "historical_tool_call":
        previous.append({"role": "assistant", "tool_calls": [{"function": {"name": "lookup"}}]})
    else:
        previous.append({"role": "tool", "content": "value"})

    def unexpected_render(*args, **kwargs):
        pytest.fail("Unsupported Kimi-VL SFT tools reached the processor")

    monkeypatch.setattr(processor, "apply_chat_template", unexpected_render)
    with pytest.raises(ValueError, match="Kimi-VL SFT does not support structured"):
        if operation == "reconstruct":
            reconstruct_assistant_tokens(builder, message, tools=tools, previous_messages=previous)
        elif operation == "merge":
            builder.merge_assistant_with_tokenization([1, 2], message, tools=tools, previous_messages=previous)
        else:
            _resolve_reconstructor(builder)(builder).reconstruct(message, tools=tools, previous_messages=previous)


def test_kimi_vl_sft_entries_preserve_plain_turns():
    tokenizer = _QwenBoundaryTokenizer()
    builder = KimiVLContinuousTokenBuilder(tokenizer, _MockQwenVLProcessor(), chat_template_kwargs={"tools": []})
    expected = tokenizer.encode("gold\n", add_special_tokens=False)
    message = {"role": "assistant", "content": "gold", "tool_calls": []}
    previous = [{"role": "user", "content": "question"}]
    assert reconstruct_assistant_tokens(builder, message, tools=[], previous_messages=previous) == expected
    result = builder.merge_assistant_with_tokenization([1, 2], message, tools=[], previous_messages=previous)
    assert result.token_ids == [1, 2, *expected]
    assert result.appended_token_count == len(expected)


@pytest.mark.parametrize("operation", ["reconstruct", "merge", "resolve"])
@pytest.mark.parametrize("structured_tools", [False, True])
def test_deepseek_vl2_sft_entries_refuse_before_rendering(operation, structured_tools, monkeypatch):
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekVL2ContinuousTokenBuilder(tokenizer, object())
    previous = [{"role": "user", "content": "question"}]
    message = {"role": "assistant", "content": "gold"}
    tools = None
    if structured_tools:
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        message["tool_calls"] = [{"function": {"name": "lookup"}}]
        previous.append({"role": "tool", "content": "value"})

    def unexpected_render(*args, **kwargs):
        pytest.fail("Unsupported DeepSeek-VL2 SFT input reached rendering")

    monkeypatch.setattr(builder, "_render_via_processor", unexpected_render)
    monkeypatch.setattr(tokenizer, "apply_chat_template", unexpected_render)
    with pytest.raises(ValueError, match="DeepSeek-VL2 SFT is not supported"):
        if operation == "reconstruct":
            reconstruct_assistant_tokens(builder, message, tools=tools, previous_messages=previous)
        elif operation == "merge":
            builder.merge_assistant_with_tokenization([1, 2], message, tools=tools, previous_messages=previous)
        else:
            _resolve_reconstructor(builder)


@pytest.mark.parametrize("register_base", [False, True], ids=["subclass_handler", "base_handler"])
def test_deepseek_vl2_explicit_reconstructor_precedes_default_refusal(monkeypatch, register_base):
    class CustomDeepSeekVL2Builder(DeepSeekVL2ContinuousTokenBuilder):
        pass

    builder = object.__new__(CustomDeepSeekVL2Builder)
    with pytest.raises(ValueError, match="DeepSeek-VL2 SFT is not supported"):
        _resolve_reconstructor(builder)
    registered_cls = DeepSeekVL2ContinuousTokenBuilder if register_base else CustomDeepSeekVL2Builder
    monkeypatch.setitem(sft_continuous_token_module._RECONSTRUCTORS, registered_cls, _AssistantReconstructor)
    assert _resolve_reconstructor(builder) is _AssistantReconstructor


@pytest.mark.parametrize("operation", ["render", "reconstruct", "merge"])
def test_vl_reconstructor_does_not_mutate_caller_messages(operation):
    builder = QwenVLContinuousTokenBuilder(_MockQwenVLTokenizer(), _BlockReplacingTemplateProcessor())
    messages = [
        {
            "role": "user" if operation == "render" else "assistant",
            "content": [
                {"type": "image_url", "image_url": {"url": "/tmp/a.png"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    expected = copy.deepcopy(messages)
    if operation == "render":
        reconstructor = _resolve_reconstructor(builder)(builder)
        reconstructor._render_text(messages, add_generation_prompt=True, tools=None)
    elif operation == "reconstruct":
        reconstruct_assistant_tokens(builder, messages[0])
    else:
        builder.merge_assistant_with_tokenization([1, 2], messages[0])

    assert messages == expected


def test_base_reconstructor_preserves_empty_tools_semantics_for_text_and_vl():
    messages = [{"role": "user", "content": "question"}]

    text_tokenizer = _RecordingTemplateTokenizer()
    text_builder = ContinuousTokenBuilder(text_tokenizer)
    _resolve_reconstructor(text_builder)(text_builder)._render_text(messages, add_generation_prompt=True, tools=[])
    assert text_tokenizer.calls[-1]["tools"] == []

    vl_processor = _RecordingTemplateProcessor()
    vl_builder = VLContinuousTokenBuilder(_MockQwenVLTokenizer(), vl_processor)
    _resolve_reconstructor(vl_builder)(vl_builder)._render_text(messages, add_generation_prompt=True, tools=[])
    assert vl_processor.template_kwargs[-1]["tools"] is None


@pytest.mark.parametrize("operation", ["reconstruct", "merge"])
def test_default_assistant_encoding_and_public_merge(operation, monkeypatch):
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    message = {"role": "assistant", "content": "gold"}
    original_message = copy.deepcopy(message)
    expected_ids = tokenizer.encode("gold\n", add_special_tokens=False)
    encode = tokenizer.encode
    encoded_texts = []

    def record_encode(text, **kwargs):
        encoded_texts.append(text)
        return encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", record_encode)
    if operation == "reconstruct":
        assert reconstruct_assistant_tokens(builder, message) == expected_ids
    else:
        result = builder.merge_assistant_with_tokenization([10, 20], message)
        assert result.token_ids == [10, 20, *expected_ids]
        assert result.appended_token_count == len(expected_ids)
        assert result.kind == "assistant"

    assert encoded_texts == ["gold\n"]
    assert len(tokenizer.calls) == 2
    assert message not in tokenizer.calls[0]["messages"]
    assert tokenizer.calls[1]["messages"][-1] == message
    assert message == original_message


def test_default_builder_trims_at_first_generated_terminator():
    tokenizer = _TemplateTokenizer()
    tokenizer.eos_token_id = 99
    builder = ContinuousTokenBuilder(tokenizer)

    reconstructor = _resolve_reconstructor(builder)(builder)
    normalized_ids = reconstructor._normalize_ids(
        [10, tokenizer.eos_token_id, 20, tokenizer.eos_token_id, 30],
        {"role": "assistant", "content": "gold"},
    )

    assert normalized_ids == [10, tokenizer.eos_token_id]


def test_gpt_oss_builder_uses_message_specific_assistant_terminators():
    tokenizer = _TemplateTokenizer()
    tokenizer.eos_token_id = 200002
    tokenizer.convert_tokens_to_ids = lambda token: {"<|call|>": 200012}.get(token, 0)
    builder = GptOssContinuousTokenBuilder(tokenizer)

    reconstructor = _resolve_reconstructor(builder)(builder)
    tool_call_ids = reconstructor._normalize_ids(
        [10, 200012, 99],
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function"}]},
    )
    final_answer_ids = reconstructor._normalize_ids(
        [20, tokenizer.eos_token_id, 99],
        {"role": "assistant", "content": "done"},
    )

    assert tool_call_ids == [10, 200012]
    assert final_answer_ids == [20, tokenizer.eos_token_id]


def test_gpt_oss_builder_normalizes_nullable_assistant_fields_for_harmony():
    builder = GptOssContinuousTokenBuilder(_TemplateTokenizer())

    reconstructor = _resolve_reconstructor(builder)(builder)
    rendered_message = reconstructor._prepare_message(
        {
            "role": "assistant",
            "content": None,
            "thinking": None,
            "tool_calls": None,
            "name": None,
        }
    )

    assert rendered_message == {"role": "assistant", "content": ""}


def test_minimax_builder_reconstructs_empty_and_nonempty_reasoning_continuations():
    tokenizer = _MiniMaxAssistantTokenizer()
    builder = MiniMaxContinuousTokenBuilder(tokenizer)

    empty_reasoning_ids = reconstruct_assistant_tokens(builder, {"role": "assistant", "content": "done"})
    reasoning_ids = reconstruct_assistant_tokens(
        builder, {"role": "assistant", "reasoning_content": "reason", "content": "done"}
    )

    assert empty_reasoning_ids == tokenizer.encode("</think>\n\ndone[e~[", add_special_tokens=False)
    assert reasoning_ids == tokenizer.encode("reason\n</think>\n\ndone[e~[", add_special_tokens=False)


def test_minimax_builder_preserves_nested_literal_think_tags():
    tokenizer = _MiniMaxAssistantTokenizer()
    builder = MiniMaxContinuousTokenBuilder(tokenizer)

    assistant_ids = reconstruct_assistant_tokens(
        builder, {"role": "assistant", "content": "<think>I need output the <think> tag</think><think>"}
    )

    assert assistant_ids == tokenizer.encode(
        "I need output the <think> tag\n</think>\n\n<think>[e~[",
        add_special_tokens=False,
    )


@pytest.mark.parametrize(
    ("enable_thinking", "message", "expected_text"),
    [
        (True, {"role": "assistant", "content": "done"}, "</think>done"),
        (
            True,
            {"role": "assistant", "reasoning_content": "reason", "content": "done"},
            "reason</think>done",
        ),
        (
            False,
            {"role": "assistant", "reasoning_content": "hidden", "content": "done"},
            "done",
        ),
    ],
)
def test_glm_builder_reconstructs_thinking_scaffold(enable_thinking, message, expected_text):
    tokenizer = _GLMAssistantTokenizer()
    builder = GLMContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": enable_thinking})

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


def test_glm_builder_drops_embedded_reasoning_from_text_blocks_when_thinking_is_disabled():
    tokenizer = _GLMAssistantTokenizer()
    builder = GLMContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})

    reconstructor = _resolve_reconstructor(builder)(builder)
    rendered_message = reconstructor._prepare_message(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<think>hidden"},
                {"type": "text", "text": "</think>\nanswer"},
            ],
        }
    )

    assert rendered_message == {
        "role": "assistant",
        "reasoning_content": "",
        "content": [{"type": "text", "text": "answer"}],
    }


def test_glm_builder_preserves_nested_literal_think_tags():
    tokenizer = _GLMAssistantTokenizer()
    builder = GLMContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    message = {
        "role": "assistant",
        "content": "<think>I need output the <think> tag</think><think>",
    }

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode(
        "I need output the <think> tag</think><think>",
        add_special_tokens=False,
    )


@pytest.mark.parametrize(
    ("enable_thinking", "message", "expected_text"),
    [
        (False, {"role": "assistant", "content": "done"}, "done<turn|>"),
        (
            True,
            {"role": "assistant", "thinking": "reason", "content": "done"},
            "<|channel>thought\nreason\n<channel|>done<turn|>",
        ),
        (
            True,
            {
                "role": "assistant",
                "thinking": "call reason",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
            },
            "<|channel>thought\ncall reason\n<channel|><|tool_call>call:lookup{}<tool_call|>",
        ),
    ],
)
def test_gemma4_builder_reconstructs_generation_scaffold(enable_thinking, message, expected_text):
    tokenizer = _Gemma4AssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": enable_thinking})

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


@pytest.mark.parametrize("with_tool_call", [False, True])
def test_gemma4_builder_keeps_reasoning_aliases_equivalent(with_tool_call):
    tokenizer = _Gemma4AssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    outputs = {}
    for field in ("thinking", "reasoning_content", "reasoning"):
        message = {"role": "assistant", field: "call reason", "content": "" if with_tool_call else "done"}
        if with_tool_call:
            message["tool_calls"] = [{"type": "function", "function": {"name": "lookup", "arguments": {}}}]
        outputs[field] = reconstruct_assistant_tokens(builder, message)

    # With a tool call the template renders ``reasoning``/``reasoning_content``
    # itself while the verl-only ``thinking`` alias falls back to the manual
    # scaffold branch, so this pins the two branches against each other. Without
    # a tool call the official history template omits reasoning entirely, so all
    # three aliases share the manual branch and this only pins alias consistency.
    assert outputs["reasoning_content"] == outputs["thinking"]
    assert outputs["reasoning"] == outputs["thinking"]


def test_gemma4_builder_does_not_add_trailing_newline_to_empty_reasoning():
    tokenizer = _Gemma4AssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    message = {
        "role": "assistant",
        "thinking": "",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
    }

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode(
        "<|channel>thought\n<channel|><|tool_call>call:lookup{}<tool_call|>",
        add_special_tokens=False,
    )


@pytest.mark.parametrize("trailing_whitespace", ["", "\n", "\n\n", " \n\t"])
@pytest.mark.parametrize("with_tool_call", [False, True])
def test_gemma4_builder_preserves_embedded_thought_whitespace(trailing_whitespace, with_tool_call):
    tokenizer = _Gemma4AssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    thought = f"<|channel>thought\nreason{trailing_whitespace}<channel|>"
    message = {"role": "assistant", "content": thought + ("" if with_tool_call else "done")}
    if with_tool_call:
        message["tool_calls"] = [{"type": "function", "function": {"name": "lookup", "arguments": {}}}]
    expected_tail = "<|tool_call>call:lookup{}<tool_call|>" if with_tool_call else "done<turn|>"

    assert reconstruct_assistant_tokens(builder, message) == tokenizer.encode(
        thought + expected_tail, add_special_tokens=False
    )


@pytest.mark.parametrize("field", ["thinking", "reasoning_content", "reasoning"])
@pytest.mark.parametrize("explicit_reasoning", ["", "override\n"])
def test_gemma4_embedded_thought_keeps_explicit_reasoning_precedence(field, explicit_reasoning):
    tokenizer = _Gemma4AssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    embedded = "<|channel>thought\nembedded\n<channel|>"
    message = {"role": "assistant", "content": embedded + "done", field: explicit_reasoning}
    # Preserve the existing empty-field fallback. Standalone nonempty reasoning
    # still receives the template separator, even if its text ends with a newline.
    expected_thought = f"<|channel>thought\n{explicit_reasoning}\n<channel|>" if explicit_reasoning else embedded

    assert reconstruct_assistant_tokens(builder, message) == tokenizer.encode(
        expected_thought + "done<turn|>", add_special_tokens=False
    )


def test_gemma4_embedded_thought_keeps_disabled_thinking_scaffold():
    tokenizer = _Gemma4AssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})
    message = {"role": "assistant", "content": "<|channel>thought\nreason\n<channel|>done"}

    assert reconstruct_assistant_tokens(builder, message) == tokenizer.encode("done<turn|>", add_special_tokens=False)


def test_gemma4_e4b_builder_uses_template_reasoning_without_duplicate_scaffold():
    tokenizer = _Gemma4E4BAssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})
    message = {
        "role": "assistant",
        "reasoning_content": "call reason",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
    }

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode(
        "<|channel>thought\ncall reason\n<channel|><|tool_call>call:lookup{}<tool_call|>",
        add_special_tokens=False,
    )


@pytest.mark.parametrize("enable_thinking", [True, False])
def test_deepseek_v4_builder_encodes_assistant_with_native_protocol(enable_thinking):
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(
        tokenizer,
        chat_template_kwargs={"enable_thinking": enable_thinking, "drop_thinking": False},
    )

    assistant_ids = reconstruct_assistant_tokens(
        builder, {"role": "assistant", "reasoning_content": "reason", "content": "gold"}
    )

    expected_text = "reason</think>gold<｜end▁of▁sentence｜>" if enable_thinking else "gold<｜end▁of▁sentence｜>"
    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


@pytest.mark.parametrize("task", ["action", "query", "authority", "domain", "title", "read_url"])
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_deepseek_v4_task_merge_encodes_only_gold_continuation(task, enable_thinking, monkeypatch):
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": enable_thinking})
    previous = [{"role": "user", "content": "question", "task": task}]
    original_previous = copy.deepcopy(previous)
    runtime_ids = builder.build_initial_tokens(previous)
    original_runtime = list(runtime_ids)
    message = {"role": "assistant", "reasoning_content": "ignored for a task", "content": "gold"}
    expected_text = "gold<｜end▁of▁sentence｜>"
    expected_ids = tokenizer.encode(expected_text, add_special_tokens=False)
    encode = tokenizer.encode
    encoded_texts = []

    def record_encode(text, **kwargs):
        encoded_texts.append(text)
        return encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", record_encode)
    result = builder.merge_assistant_with_tokenization(runtime_ids, message, previous_messages=previous)
    mask, _ = builder.align_response_metadata(result, [0] * len(runtime_ids))

    assert result.token_ids == original_runtime + expected_ids
    assert mask == [0] * len(original_runtime) + [1] * len(expected_ids)
    assert encoded_texts == [expected_text]
    assert runtime_ids == original_runtime
    assert previous == original_previous


@pytest.mark.parametrize(
    ("previous", "expected_text"),
    [
        ([{"role": "user", "content": "question"}], "reason</think>gold<｜end▁of▁sentence｜>"),
        (
            [
                {"role": "user", "content": "question", "task": "query"},
                {"role": "latest_reminder", "content": "remember"},
            ],
            "reason</think>gold<｜end▁of▁sentence｜>",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {"role": "latest_reminder", "content": "remember", "task": "title"},
            ],
            "gold<｜end▁of▁sentence｜>",
        ),
    ],
)
def test_deepseek_v4_task_scope_is_the_immediate_predecessor(previous, expected_text):
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    message = {"role": "assistant", "reasoning_content": "reason", "content": "gold"}

    assert reconstruct_assistant_tokens(builder, message, previous_messages=previous) == tokenizer.encode(
        expected_text, add_special_tokens=False
    )


def test_deepseek_v4_task_continuation_keeps_native_tool_calls():
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    message = {
        "role": "assistant",
        "reasoning_content": "ignored for a task",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"q": "x"}}}],
    }
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    expected_text = (
        '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="lookup">\n'
        '<｜DSML｜parameter name="q" string="true">x</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>"
    )

    assert reconstruct_assistant_tokens(
        builder, message, tools=tools, previous_messages=[{"role": "user", "content": "question", "task": "action"}]
    ) == tokenizer.encode(expected_text, add_special_tokens=False)


@pytest.mark.parametrize(
    ("enable_thinking", "message", "expected_text"),
    [
        (
            True,
            {"role": "assistant", "content": "<think>I need output the <think> tag</think><think>"},
            "I need output the <think> tag</think><think><｜end▁of▁sentence｜>",
        ),
        (
            False,
            {"role": "assistant", "content": "<think></think><think>"},
            "<think><｜end▁of▁sentence｜>",
        ),
    ],
)
def test_deepseek_v4_builder_preserves_literal_think_tags(enable_thinking, message, expected_text):
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(
        tokenizer,
        chat_template_kwargs={"enable_thinking": enable_thinking, "drop_thinking": False},
    )

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


@pytest.mark.parametrize(
    ("message", "expected_text"),
    [
        (
            {"role": "assistant", "reasoning_content": "reason", "content": "gold"},
            "reason</think>gold<｜end▁of▁sentence｜>",
        ),
        (
            {"role": "assistant", "content": "<think>I need output the <think> tag</think><think>"},
            "I need output the <think> tag</think><think><｜end▁of▁sentence｜>",
        ),
    ],
)
def test_deepseek_v31_builder_reconstructs_thinking_continuation(message, expected_text):
    tokenizer = _DeepSeekV31AssistantTokenizer()
    builder = DeepSeekContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


def test_deepseek_v31_builder_uses_direct_post_tool_assistant_continuation():
    tokenizer = _DeepSeekV31AssistantTokenizer()
    builder = DeepSeekContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": True})
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "name": "lookup", "content": "value"},
    ]

    assistant_ids = reconstruct_assistant_tokens(
        builder,
        {"role": "assistant", "content": "gold"},
        previous_messages=previous_messages,
    )

    assert assistant_ids == tokenizer.encode("gold<｜end▁of▁sentence｜>", add_special_tokens=False)


def test_minimax_vl_builder_extracts_assistant_after_unconditional_scaffold():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)

    assistant_ids = reconstruct_assistant_tokens(builder, {"role": "assistant", "content": "gold"})

    assert assistant_ids == tokenizer.encode("gold<end_of_sentence>", add_special_tokens=False)


@pytest.mark.parametrize(
    ("content", "prefix"),
    [
        ("", ""),
        ("Let me check.", "Let me check."),
        ([{"type": "text", "text": "Let me check."}], "Let me check."),
    ],
)
def test_minimax_vl_builder_reconstructs_structured_assistant_tool_call(content, prefix):
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)

    assistant_ids = reconstruct_assistant_tokens(
        builder,
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        },
    )

    assert assistant_ids == tokenizer.encode(
        prefix + '<function_call>```typescript\nfunctions.lookup({"q":"x"})\n```<end_of_sentence>',
        add_special_tokens=False,
    )


def test_kimi_vl_builder_trims_at_first_im_end_terminator():
    tokenizer = _QwenBoundaryTokenizer()
    builder = KimiVLContinuousTokenBuilder(tokenizer, object())

    reconstructor = _resolve_reconstructor(builder)(builder)
    assistant_ids = reconstructor._normalize_ids(
        [10, tokenizer.im_end_id, 20, tokenizer.im_end_id],
        {"role": "assistant", "content": "gold"},
    )

    assert assistant_ids == [10, tokenizer.im_end_id]


def test_qwen_builder_trims_after_im_end():
    tokenizer = _QwenBoundaryTokenizer()
    tokenizer.eos_token_id = tokenizer.im_end_id
    builder = QwenContinuousTokenBuilder(tokenizer)
    message = {
        "role": "assistant",
        "content": "gold",
    }

    reconstructor = _resolve_reconstructor(builder)(builder)
    normalized_ids = reconstructor._normalize_ids(
        [1, tokenizer.im_end_id, tokenizer.newline_id],
        message,
    )

    assert normalized_ids == [1, tokenizer.im_end_id]


def test_qwen_builder_drops_prepared_reasoning_when_thinking_is_disabled():
    tokenizer = _QwenBoundaryTokenizer()
    builder = QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})

    reconstructor = _resolve_reconstructor(builder)(builder)
    explicit_reasoning = reconstructor._prepare_message(
        {"role": "assistant", "reasoning_content": "hidden", "content": "answer"}
    )
    embedded_reasoning = reconstructor._prepare_message({"role": "assistant", "content": "<think>hidden</think>answer"})

    assert explicit_reasoning == {"role": "assistant", "reasoning_content": "", "content": "answer"}
    assert embedded_reasoning == {"role": "assistant", "reasoning_content": "", "content": "answer"}


@pytest.mark.parametrize(
    ("model_var", "model_name", "use_processor", "block_content", "enable_thinking", "nested_literal"),
    [
        ("VERL_TEST_QWEN25_MODEL", "Qwen2.5-0.5B", False, False, None, False),
        ("VERL_TEST_QWEN25_MODEL", "Qwen2.5-0.5B", False, False, True, False),
        ("VERL_TEST_QWEN3_MODEL", "Qwen3-0.6B", False, False, None, False),
        ("VERL_TEST_QWEN3_MODEL", "Qwen3-0.6B", False, False, True, False),
        ("VERL_TEST_QWEN35_MODEL", "Qwen3.5-0.8B", False, False, True, False),
        ("VERL_TEST_QWEN3_VL_MODEL", "Qwen3-VL-2B-Instruct", True, False, None, False),
        ("VERL_TEST_QWEN3_VL_MODEL", "Qwen3-VL-2B-Instruct", True, False, True, False),
        ("VERL_TEST_QWEN3_VL_MODEL", "Qwen3-VL-2B-Instruct", True, True, None, False),
        ("VERL_TEST_QWEN3_VL_MODEL", "Qwen3-VL-2B-Instruct", True, True, True, False),
        ("VERL_TEST_QWEN3_MODEL", "Qwen3-0.6B", False, False, True, True),
        ("VERL_TEST_QWEN35_MODEL", "Qwen3.5-0.8B", False, False, True, True),
    ],
)
@pytest.mark.parametrize("with_tools", [False, True])
def test_qwen_embedded_reasoning_matches_checkpoint_template(
    model_var, model_name, use_processor, block_content, enable_thinking, nested_literal, with_tools
):
    model_path = Path(os.environ.get(model_var, str(Path.home() / "models" / "Qwen" / model_name)))
    if not model_path.is_dir():
        reason = f"Local tokenizer/processor artifacts are unavailable: {model_path}"
        if os.environ.get("VERL_REQUIRE_LOCAL_MODELS") == "1":
            pytest.fail(reason, pytrace=False)
        pytest.skip(reason)
    owner = (AutoProcessor if use_processor else AutoTokenizer).from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer = owner.tokenizer if use_processor else owner
    kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    builder = (
        QwenVLContinuousTokenBuilder(tokenizer, owner, chat_template_kwargs=kwargs)
        if use_processor
        else QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs=kwargs)
    )
    prompt = [
        {"role": "system", "content": "continuous token synthetic system"},
        {"role": "user", "content": "continuous token synthetic user"},
    ]
    reasoning = "I need output the <think> tag" if nested_literal else "reason"
    answer = "<think>" if nested_literal else "gold"
    content = f"<think>{reasoning}</think>{answer}"
    message = {
        "role": "assistant",
        "content": [{"type": "text", "text": content}] if block_content else content,
    }
    before = copy.deepcopy(message)
    tools = (
        [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
        if with_tools
        else None
    )
    # The raw Qwen3/3.5 template splits at the last opening tag. Explicit
    # reasoning preserves literal tags inside the supplied reasoning and answer.
    oracle_message = {**message, "reasoning_content": reasoning, "content": answer} if nested_literal else message
    prompt_text = owner.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, tools=tools, **kwargs)
    completed_text = owner.apply_chat_template(
        [*prompt, oracle_message], tokenize=False, add_generation_prompt=False, tools=tools, **kwargs
    )
    assert completed_text.startswith(prompt_text)
    continuation = completed_text[len(prompt_text) :].split("<|im_end|>", 1)[0] + "<|im_end|>"
    assert reasoning in continuation and answer in continuation
    assert reconstruct_assistant_tokens(builder, message, tools=tools) == tokenizer.encode(
        continuation, add_special_tokens=False
    )
    assert message == before


def test_minimax_vl_builder_keeps_tool_declarations_in_initial_prompt():
    tokenizer = _MiniMaxVLAssistantTokenizer()

    class Processor(_MockMiniMaxVLAssistantProcessor):
        def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
            rendered = super().apply_chat_template(messages, **kwargs).removesuffix("<beginning_of_sentence>ai\n")
            for tool in tools or []:
                rendered += f"<tool>{tool['name']}</tool>"
            if add_generation_prompt:
                rendered += "<beginning_of_sentence>ai\n"
            return rendered

    builder = adapt_continuous_token_builder_for_sft(MiniMaxVLContinuousTokenBuilder(tokenizer, Processor(tokenizer)))
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    first = [{"role": "user", "content": [{"type": "text", "text": "question"}]}]
    initial = builder.build_initial_tokens(first, tools=tools)
    assert initial == tokenizer.encode(
        "<beginning_of_sentence>user\nquestion<end_of_sentence>\n<tool>lookup</tool><beginning_of_sentence>ai\n",
        add_special_tokens=False,
    )
    previous = [*first, {"role": "assistant", "content": "gold"}]
    runtime_ids = initial + tokenizer.encode("gold<end_of_sentence>", add_special_tokens=False)
    result = builder.merge_non_assistant_tokens(
        previous,
        [*previous, {"role": "user", "content": [{"type": "text", "text": "retry"}]}],
        runtime_ids,
        tools=tools,
    )
    assert result.token_ids == runtime_ids + tokenizer.encode(
        "\n<beginning_of_sentence>user\nretry<end_of_sentence>\n<beginning_of_sentence>ai\n",
        add_special_tokens=False,
    )


@pytest.mark.parametrize("tool_name", ["lookup", 'look"up', "look\\up", "lookup\nnext", "查询"])
@pytest.mark.parametrize("content", [None, "", "sunny", '{"value": 1}'])
def test_minimax_vl_builder_formats_openai_tool_response_as_function_message(tool_name, content):
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = adapt_continuous_token_builder_for_sft(MiniMaxVLContinuousTokenBuilder(tokenizer, processor))
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        }
    ]

    token_ids = builder._tokenize_tool_group(
        [{"role": "tool", "tool_call_id": "call_0", "content": content}],
        previous_messages=previous_messages,
    )

    prefix = tokenizer.encode("<beginning_of_sentence>system function_response=functions\n", add_special_tokens=False)
    suffix = tokenizer.encode("<end_of_sentence>\n", add_special_tokens=False)
    assert token_ids[: len(prefix)] == prefix and token_ids[-len(suffix) :] == suffix
    response = "".join(chr(token) for token in token_ids[len(prefix) : -len(suffix)])
    # The official function template concatenates these fields verbatim, even
    # for non-JSON text; escaping only the name changes its token protocol.
    assert response == '{"name": "' + tool_name + '", "response": ' + (content or "") + "}"


@pytest.mark.parametrize("response_count", [1, 2])
def test_minimax_vl_builder_merges_tool_result_and_fixed_generation_scaffold(response_count):
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = adapt_continuous_token_builder_for_sft(MiniMaxVLContinuousTokenBuilder(tokenizer, processor))
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        },
    ]
    updated_messages = [
        *previous_messages,
        *[{"role": "tool", "tool_call_id": "call_0", "content": '{"value": 1}'} for _ in range(response_count)],
    ]
    runtime_ids = [7, tokenizer.eos_token_id]

    result = builder.merge_non_assistant_tokens(
        previous_messages,
        updated_messages,
        runtime_ids,
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )

    expected_response = tokenizer.encode(
        "<beginning_of_sentence>system function_response=functions\n"
        '{"name": "lookup", "response": {"value": 1}}<end_of_sentence>\n',
        add_special_tokens=False,
    )
    expected_append = expected_response * response_count + builder._vl_scaffold_ids
    assert result.token_ids == runtime_ids + [ord("\n")] + expected_append
    assert result.inserted_token_ids == [ord("\n")]
    assert result.appended_token_count == len(expected_append)


def test_minimax_vl_tool_responses_follow_tokenizer_template(monkeypatch):
    tokenizer = _MiniMaxVLAssistantTokenizer()
    builder = adapt_continuous_token_builder_for_sft(
        MiniMaxVLContinuousTokenBuilder(tokenizer, _MockMiniMaxVLAssistantProcessor(tokenizer))
    )

    def custom_template(messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize and not add_generation_prompt
        assert all(message["role"] == "function" for message in messages)
        rendered = "".join(f"{message['name']}:{message['content'][0]['text']}!" for message in messages)
        return tokenizer.encode(rendered, add_special_tokens=False)

    monkeypatch.setattr(tokenizer, "apply_chat_template", custom_template)
    messages = [
        {"role": "tool", "name": "first", "content": [{"type": "text", "text": "sunny"}]},
        {"role": "tool", "name": "second", "content": None},
    ]
    original = copy.deepcopy(messages)
    result = builder._tokenize_tool_group(messages, previous_messages=[], add_generation_prompt=True)
    assert result == tokenizer.encode("first:sunny!second:!", add_special_tokens=False) + builder._vl_scaffold_ids
    assert messages == original


def test_minimax_sft_adapter_preserves_factory_state_and_restricts_roles():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    runtime_builder = MiniMaxVLContinuousTokenBuilder(
        tokenizer,
        processor,
        chat_template_kwargs={"keep_option": "value"},
        mm_processor_kwargs={"max_pixels": 196},
        allowed_append_roles={"user"},
    )
    builder = adapt_continuous_token_builder_for_sft(runtime_builder)
    assert isinstance(builder, MiniMaxVLContinuousTokenBuilder)
    assert _resolve_reconstructor(builder) is _MiniMaxVLReconstructor
    assert builder.tokenizer is tokenizer and builder.processor is processor
    assert builder.chat_template_kwargs == runtime_builder.chat_template_kwargs
    assert builder.mm_processor_kwargs == runtime_builder.mm_processor_kwargs
    assert adapt_continuous_token_builder_for_sft(builder) is builder
    previous = [{"role": "assistant", "content": "gold"}]
    user = {"role": "user", "content": [{"type": "text", "text": "retry"}]}
    assert builder.tokenize_non_assistant_incremental_messages(previous, [*previous, user])
    with pytest.raises(ValueError, match="only supports appending roles"):
        builder.tokenize_non_assistant_incremental_messages(
            previous, [*previous, {"role": "system", "content": "policy"}]
        )
