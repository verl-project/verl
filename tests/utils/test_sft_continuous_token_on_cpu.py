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

import pytest

from tests.utils.test_continuous_token_on_cpu import (
    _BlockReplacingTemplateProcessor,
    _DeepSeekAssistantTokenizer,
    _DeepSeekV31AssistantTokenizer,
    _Gemma4AssistantTokenizer,
    _Gemma4E4BAssistantTokenizer,
    _GLMAssistantTokenizer,
    _MiniMaxAssistantTokenizer,
    _MiniMaxVLAssistantTokenizer,
    _MockDeepSeekVL2Processor,
    _MockMiniMaxVLAssistantProcessor,
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
    _DeepSeekVL2Reconstructor,
    _Gemma4Reconstructor,
    _GLMReconstructor,
    _GptOssReconstructor,
    _KimiVLReconstructor,
    _MiniMaxReconstructor,
    _MiniMaxVLReconstructor,
    _QwenReconstructor,
    _resolve_reconstructor,
    reconstruct_assistant_tokens,
)


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
        (DeepSeekVL2ContinuousTokenBuilder, _DeepSeekVL2Reconstructor),
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


def test_deepseek_v4_builder_keeps_committed_reasoning_when_drop_thinking_is_enabled():
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "drop_thinking": True},
    )
    previous_messages = [{"role": "user", "content": "q1"}]

    assistant_ids = reconstruct_assistant_tokens(
        builder,
        {"role": "assistant", "reasoning_content": "reason A", "content": "answer A"},
        previous_messages=previous_messages,
    )

    reason_ids = tokenizer.encode("reason A", add_special_tokens=False)
    assert assistant_ids[: len(reason_ids)] == reason_ids


def test_deepseek_vl2_builder_uses_processor_for_text_prompt_and_assistant():
    tokenizer = _DeepSeekAssistantTokenizer()
    processor = _MockDeepSeekVL2Processor(tokenizer)
    builder = DeepSeekVL2ContinuousTokenBuilder(tokenizer, processor)

    assistant_ids = reconstruct_assistant_tokens(builder, {"role": "assistant", "content": "gold"})

    assert assistant_ids == [ord(char) for char in "gold"] + [tokenizer.eos_token_id]
    assert len(processor.calls) == 2
    assert all(force_batchify for _, _, force_batchify, _ in processor.calls)
    assert [inference_mode for _, _, _, inference_mode in processor.calls] == [True, False]


def test_deepseek_vl2_builder_rejects_unsupported_structured_tools():
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekVL2ContinuousTokenBuilder(tokenizer, _MockDeepSeekVL2Processor(tokenizer))

    with pytest.raises(ValueError, match="does not support structured assistant tool calls"):
        reconstruct_assistant_tokens(
            builder,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
            },
        )


def test_kimi_vl_builder_rejects_unsupported_structured_tool_responses():
    builder = KimiVLContinuousTokenBuilder(_QwenBoundaryTokenizer(), object())

    with pytest.raises(ValueError, match="does not support structured assistant tool calls"):
        reconstruct_assistant_tokens(
            builder,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
            },
        )


def test_vl_reconstructor_does_not_mutate_caller_messages():
    builder = QwenVLContinuousTokenBuilder(_MockQwenVLTokenizer(), _BlockReplacingTemplateProcessor())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "/tmp/a.png"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    expected = copy.deepcopy(messages)
    reconstructor = _resolve_reconstructor(builder)(builder)

    reconstructor._render_text(messages, add_generation_prompt=True, tools=None)

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


def test_default_builder_encodes_prepared_assistant_continuation_once():
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    message = {"role": "assistant", "content": "gold"}

    assistant_ids = reconstruct_assistant_tokens(builder, message)

    assert assistant_ids == tokenizer.encode("gold\n", add_special_tokens=False)
    assert len(tokenizer.calls) == 2
    assert all(message not in call["messages"] for call in tokenizer.calls[:1])
    assert tokenizer.calls[1]["messages"][-1] is message


def test_builder_public_api_reconstructs_and_merges_gold_assistant_message():
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)

    result = builder.merge_assistant_with_tokenization(
        [10, 20],
        {"role": "assistant", "content": "gold"},
    )

    assistant_ids = tokenizer.encode("gold\n", add_special_tokens=False)
    assert result.token_ids == [10, 20, *assistant_ids]
    assert result.appended_token_count == len(assistant_ids)
    assert result.kind == "assistant"


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


def test_minimax_vl_builder_reconstructs_structured_assistant_tool_call():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)

    assistant_ids = reconstruct_assistant_tokens(
        builder,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        },
    )

    assert assistant_ids == tokenizer.encode(
        '<function_call>```typescript\nfunctions.lookup({"q":"x"})\n```<end_of_sentence>',
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


def test_qwen_builder_preserves_nested_literal_think_tags_and_trims_after_eos():
    tokenizer = _QwenBoundaryTokenizer()
    tokenizer.eos_token_id = tokenizer.im_end_id
    builder = QwenContinuousTokenBuilder(tokenizer)
    message = {
        "role": "assistant",
        "content": "<think>I need output the <think> tag</think><think>",
    }

    reconstructor = _resolve_reconstructor(builder)(builder)
    rendered_message = reconstructor._prepare_message(message)
    normalized_ids = reconstructor._normalize_ids(
        [1, tokenizer.im_end_id, tokenizer.newline_id],
        message,
    )

    assert rendered_message["reasoning_content"] == "I need output the <think> tag"
    assert rendered_message["content"] == "<think>"
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
