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

import pytest

from verl.utils.continuous_token import (
    ContinuousTokenBuilder,
    Gemma4ContinuousTokenBuilder,
    GLMContinuousTokenBuilder,
    GptOssContinuousTokenBuilder,
    MergeResult,
    MiniMaxContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    ct_align_response_metadata,
)
from verl.utils.continuous_token_wiring import (
    CONTINUOUS_TOKEN_BUILDER_FAMILIES,
    ContinuousTokenModelFamily,
    create_continuous_token_builder,
    get_continuous_token_builder_class,
    infer_continuous_token_model_family,
    list_continuous_token_builder_families,
    resolve_continuous_token_model_family,
)


class _DummyTokenizer:
    name_or_path = "Qwen/Qwen3-8B"


class _InitKwargsTokenizer:
    init_kwargs = {"name_or_path": "MiniMaxAI/MiniMax-M2.7"}


class _TemplateTokenizer:
    name_or_path = "unit-test/default"

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        rendered = "".join(f"<{message['role']}>{message.get('content', '')}\n" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _QwenBoundaryTokenizer(_TemplateTokenizer):
    name_or_path = "Qwen/Qwen3-8B"

    def __init__(self):
        self.im_end_id = 151645
        self.newline_id = 198

    def encode(self, text, add_special_tokens=False):
        if text == "\n":
            return [self.newline_id]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return self.im_end_id
        return 0


class _GLMBoundaryTokenizer(_TemplateTokenizer):
    name_or_path = "zai-org/GLM-4.7-Flash"

    def __init__(self):
        self.observation_id = 151333
        self.user_id = 151336

    def convert_tokens_to_ids(self, token):
        if token == "<|observation|>":
            return self.observation_id
        if token == "<|user|>":
            return self.user_id
        return 0


class _MiniMaxBoundaryTokenizer(_TemplateTokenizer):
    name_or_path = "MiniMaxAI/MiniMax-M2"

    def __init__(self):
        self.eos_id = 200020
        self.newline_id = 10

    def encode(self, text, add_special_tokens=False):
        if text == "\n":
            return [self.newline_id]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def convert_tokens_to_ids(self, token):
        if token == "[e~[":
            return self.eos_id
        return 0


class _Gemma4BoundaryTokenizer(_TemplateTokenizer):
    name_or_path = "google/gemma-4-27b-it"

    def __init__(self):
        self.tool_response_id = 262144

    def convert_tokens_to_ids(self, token):
        if token == "<|tool_response>":
            return self.tool_response_id
        return 0


class _MissingSpecialTokenTokenizer(_TemplateTokenizer):
    def convert_tokens_to_ids(self, token):
        return None


def test_builtin_family_surface():
    assert CONTINUOUS_TOKEN_BUILDER_FAMILIES == (
        "default",
        "qwen",
        "qwen25",
        "qwen3",
        "qwen35",
        "minimax",
        "minimaxm2",
        "minimaxm25",
        "minimaxm27",
        "glm47",
        "glm5",
        "gemma4",
        "gptoss",
    )
    assert list_continuous_token_builder_families() == CONTINUOUS_TOKEN_BUILDER_FAMILIES


@pytest.mark.parametrize(
    ("family", "builder_cls"),
    [
        (ContinuousTokenModelFamily.DEFAULT, ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN25, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN3, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN35, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M2, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M25, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M27, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM47, GLMContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM5, GLMContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GEMMA4, Gemma4ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GPTOSS, GptOssContinuousTokenBuilder),
    ],
)
def test_builtin_family_class_mapping(family, builder_cls):
    assert get_continuous_token_builder_class(family) is builder_cls


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        ("zai-org/GLM-4.7-Flash", ContinuousTokenModelFamily.GLM47),
        ("THUDM/GLM-5-9B-Chat", ContinuousTokenModelFamily.GLM5),
        ("google/gemma-4-27b-it", ContinuousTokenModelFamily.GEMMA4),
        ("openai/gpt-oss-20b", ContinuousTokenModelFamily.GPTOSS),
        ("MiniMaxAI/MiniMax-M2", ContinuousTokenModelFamily.MINIMAX_M2),
        ("MiniMaxAI/MiniMax-M2.5", ContinuousTokenModelFamily.MINIMAX_M25),
        ("MiniMaxAI/MiniMax-M2.7", ContinuousTokenModelFamily.MINIMAX_M27),
        ("MiniMaxAI/MiniMax-Text-01", ContinuousTokenModelFamily.MINIMAX),
        ("Qwen/Qwen3.5-35B-A3B", ContinuousTokenModelFamily.QWEN35),
        ("Qwen/Qwen2.5-7B-Instruct", ContinuousTokenModelFamily.QWEN25),
        ("Qwen/Qwen3-8B", ContinuousTokenModelFamily.QWEN3),
        ("deepseek-ai/DeepSeek-R1", ContinuousTokenModelFamily.DEFAULT),
    ],
)
def test_auto_family_inference(model_path, expected):
    assert infer_continuous_token_model_family(model_path=model_path) == expected


def test_auto_family_inference_uses_tokenizer_name():
    assert infer_continuous_token_model_family(tokenizer=_DummyTokenizer()) == ContinuousTokenModelFamily.QWEN3


def test_auto_family_inference_uses_tokenizer_init_kwargs_name():
    assert infer_continuous_token_model_family(tokenizer=_InitKwargsTokenizer()) == (
        ContinuousTokenModelFamily.MINIMAX_M27
    )


def test_explicit_family_is_not_rewritten():
    assert (
        resolve_continuous_token_model_family(ContinuousTokenModelFamily.DEFAULT, model_path="Qwen/Qwen3-8B")
        == ContinuousTokenModelFamily.DEFAULT
    )
    assert resolve_continuous_token_model_family("qwen_3.5", model_path="deepseek-ai/DeepSeek-R1") == (
        ContinuousTokenModelFamily.QWEN35
    )


def test_auto_family_is_resolved_at_factory_time():
    builder = create_continuous_token_builder(_QwenBoundaryTokenizer(), model_family="auto")
    assert isinstance(builder, QwenContinuousTokenBuilder)


def test_default_builder_creation_forwards_kwargs():
    builder = create_continuous_token_builder(
        _TemplateTokenizer(),
        model_family="default",
        chat_template_kwargs={"enable_thinking": False},
        allowed_append_roles=["tool"],
    )
    assert isinstance(builder, ContinuousTokenBuilder)
    assert builder.chat_template_kwargs == {"enable_thinking": False}
    assert builder.allowed_append_roles == frozenset({"tool"})


def test_default_builder_is_available_from_builtin_registry():
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="default")
    assert isinstance(builder, ContinuousTokenBuilder)


def test_qwen3_builder_inserts_missing_newline_after_im_end():
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="qwen3")

    assert isinstance(builder, QwenContinuousTokenBuilder)
    result = builder._merge_token_ids([1, tokenizer.im_end_id], [2, 3])

    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2, 3]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 2
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = ct_align_response_metadata(
        result,
        [1, 1],
        [-0.1, -0.2],
    )
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0]


def test_qwen35_builder_uses_qwen3_newline_boundary_logic():
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="qwen35")

    assert isinstance(builder, QwenContinuousTokenBuilder)
    result = builder._merge_token_ids([1, tokenizer.im_end_id], [2])

    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 1
    assert result.kind == "non_assistant"


def test_minimax_builder_inserts_missing_newline_after_eos():
    tokenizer = _MiniMaxBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="minimaxm2")

    assert isinstance(builder, MiniMaxContinuousTokenBuilder)
    result = builder._merge_token_ids([1, tokenizer.eos_id], [2, 3])

    assert result.token_ids == [1, tokenizer.eos_id, tokenizer.newline_id, 2, 3]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 2
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = ct_align_response_metadata(
        result,
        [1, 1],
        [-0.1, -0.2],
    )
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0]


def test_glm47_builder_removes_ambiguous_boundary_token():
    tokenizer = _GLMBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="glm47")

    assert isinstance(builder, GLMContinuousTokenBuilder)
    result = builder._merge_token_ids([1, tokenizer.observation_id], [tokenizer.user_id, 2])

    assert result.token_ids == [1, tokenizer.user_id, 2]
    assert result.removed_prefix_token_count == 1
    assert result.appended_token_count == 2
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = ct_align_response_metadata(
        result,
        [1, 1],
        [-0.1, -0.2],
    )
    assert aligned_mask == [1, 0, 0]
    assert aligned_logprobs == [-0.1, 0.0, 0.0]


def test_gemma4_builder_inserts_tool_response_boundary_for_appended_messages():
    tokenizer = _Gemma4BoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="gemma4")
    previous_messages = [{"role": "user", "content": "question"}]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "name": "lookup"}]

    result = builder.merge_tokens(previous_messages, updated_messages, [1, 2, 3])

    assert isinstance(builder, Gemma4ContinuousTokenBuilder)
    assert result.token_ids[:4] == [1, 2, 3, tokenizer.tool_response_id]
    assert result.inserted_token_ids == [tokenizer.tool_response_id]
    assert result.appended_token_count == len(result.token_ids) - 4
    assert result.kind == "non_assistant"


def test_gemma4_builder_does_not_duplicate_existing_tool_response_boundary():
    tokenizer = _Gemma4BoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family=ContinuousTokenModelFamily.GEMMA4)
    previous_messages = [{"role": "user", "content": "question"}]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "name": "lookup"}]

    result = builder.merge_tokens(previous_messages, updated_messages, [1, tokenizer.tool_response_id])

    assert result.token_ids[:2] == [1, tokenizer.tool_response_id]
    assert result.inserted_token_ids == []
    assert result.kind == "non_assistant"


def test_gpt_oss_builder_formats_tool_responses_with_resolved_tool_name():
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    context_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "lookup"},
                }
            ],
        }
    ]
    tool_messages = [{"role": "tool", "tool_call_id": "call_0", "content": [{"type": "text", "text": "ok"}]}]

    token_ids = builder._tokenize_tool_group(tool_messages, context_messages=context_messages)

    expected = '<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>"ok"<|end|>'
    assert isinstance(builder, GptOssContinuousTokenBuilder)
    assert token_ids == [ord(char) for char in expected]


def test_default_builder_merges_append_only_non_assistant_messages():
    tokenizer = _TemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [{"role": "tool", "content": "answer", "tool_call_id": "call_0", "name": "lookup"}]
    runtime_ids = [1, 2, 3]

    result = builder.merge_tokens(old_messages, new_messages, runtime_ids)
    expected_incremental = builder.tokenize_incremental_messages(old_messages, new_messages)

    assert isinstance(result, MergeResult)
    assert result.token_ids == runtime_ids + expected_incremental
    assert result.appended_token_count == len(expected_incremental)
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = ct_align_response_metadata(
        result,
        [1, 1, 1],
        [0.1, 0.2, 0.3],
    )
    assert aligned_mask == [1, 1, 1] + [0] * len(expected_incremental)
    assert aligned_logprobs == [0.1, 0.2, 0.3] + [0.0] * len(expected_incremental)


def test_default_builder_appends_assistant_tokens_to_runtime_stream():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())

    result = builder.append_assistant_tokens([1, 2, 3], [4, 5])

    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.appended_token_count == 2
    assert result.kind == "assistant"
    aligned_mask, aligned_logprobs = ct_align_response_metadata(
        result,
        [0, 1],
        [0.0, -0.1],
        assistant_logprobs=[-0.2, -0.3],
    )
    assert aligned_mask == [0, 1, 1, 1]
    assert aligned_logprobs == [0.0, -0.1, -0.2, -0.3]


def test_assistant_alignment_validates_logprobs():
    result = MergeResult(token_ids=[1, 2, 3], appended_token_count=2, kind="assistant")

    aligned_mask, aligned_logprobs = ct_align_response_metadata(result, [1])
    assert aligned_mask == [1, 1, 1]
    assert aligned_logprobs is None

    with pytest.raises(ValueError, match="response_logprobs is required"):
        ct_align_response_metadata(result, [1], assistant_logprobs=[-0.1, -0.2])

    with pytest.raises(ValueError, match="assistant_logprobs is required"):
        ct_align_response_metadata(result, [1], [0.0])

    with pytest.raises(ValueError, match="assistant_logprobs length must match"):
        ct_align_response_metadata(result, [1], [0.0], assistant_logprobs=[-0.1])


def test_builder_alignment_wrapper_delegates_to_public_helper():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(
        token_ids=[1, 2, 99, 3],
        appended_token_count=1,
        kind="non_assistant",
        inserted_token_ids=[99],
    )
    expected = ct_align_response_metadata(result, [1, 1], [0.1, 0.2])

    assert builder.align_response_metadata(result, [1, 1], [0.1, 0.2]) == expected
    assert ContinuousTokenBuilder.align_response_metadata(result, [1, 1], [0.1, 0.2]) == expected


def test_default_builder_rejects_mutated_message_prefix():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    changed_messages = [{"role": "user", "content": "different"}]

    with pytest.raises(ValueError, match="prefix messages changed"):
        builder.tokenize_incremental_messages(old_messages, changed_messages)


def test_subclass_only_overrides_token_level_merge_hook():
    class BoundaryBuilder(ContinuousTokenBuilder):
        def _merge_token_ids(self, runtime_token_ids, appended_token_ids):
            return MergeResult(
                token_ids=list(runtime_token_ids) + [99] + list(appended_token_ids),
                appended_token_count=len(appended_token_ids),
                kind="non_assistant",
                inserted_token_ids=[99],
            )

    builder = BoundaryBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [{"role": "tool", "content": "answer"}]
    incremental = builder.tokenize_incremental_messages(old_messages, new_messages)

    result = builder.merge_tokens(old_messages, new_messages, [1, 2, 3])

    assert result.token_ids == [1, 2, 3, 99] + incremental
    assert result.appended_token_count == len(incremental)
    assert result.inserted_token_ids == [99]
    assert result.kind == "non_assistant"


def test_non_assistant_alignment_handles_boundary_inserts_and_trims():
    result = MergeResult(
        token_ids=[1, 2, 99, 3, 4],
        appended_token_count=2,
        kind="non_assistant",
        inserted_token_ids=[99],
        removed_prefix_token_count=1,
    )

    aligned_mask, aligned_logprobs = ct_align_response_metadata(
        result,
        [1, 1, 1],
        [0.1, 0.2, 0.3],
    )
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs == [0.1, 0.2, 0.0, 0.0, 0.0]

    aligned_mask, aligned_logprobs = ct_align_response_metadata(result, [1, 1, 1])
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs is None


def test_builder_rejects_unsupported_append_roles():
    builder = ContinuousTokenBuilder(_TemplateTokenizer(), allowed_append_roles=["tool"])

    with pytest.raises(ValueError, match="got 'user'"):
        builder.tokenize_incremental_messages(
            [{"role": "user", "content": "question"}],
            [{"role": "user", "content": "question"}, {"role": "user", "content": "retry"}],
        )


def test_model_specific_builders_validate_required_special_tokens():
    with pytest.raises(ValueError, match="required token '<\\|im_end\\|>'"):
        QwenContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '\\[e~\\['"):
        MiniMaxContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<\\|observation\\|>'"):
        GLMContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<\\|tool_response>'"):
        Gemma4ContinuousTokenBuilder(_MissingSpecialTokenTokenizer())


def test_unknown_family_fails_during_resolution():
    with pytest.raises(ValueError, match="Unknown Continuous Token model_family"):
        create_continuous_token_builder(_DummyTokenizer(), model_family="missing_custom_family")
