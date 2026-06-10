# Copyright 2026 Bytedance Ltd. and/or its affiliates

import pytest

from verl.utils.chat_template import extract_system_prompt_and_generation, initialize_system_prompt


class AppendOnlyTokenizer:
    prefix = [1, 2]
    user_turn = [10, 11]
    assistant_turn = [20]
    generation_prompt = [30]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        assert tokenize
        token_ids = list(self.prefix)
        for message in messages:
            if message["role"] == "user":
                token_ids.extend(self.user_turn)
            elif message["role"] == "assistant":
                token_ids.extend(self.assistant_turn)
            else:
                raise ValueError(f"Unsupported role: {message['role']}")

        if add_generation_prompt:
            token_ids.extend(self.generation_prompt)
        return token_ids


class AlternatingTokenizer(AppendOnlyTokenizer):
    prefix = [101]
    user_turn = [110]
    assistant_turn = [120]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        for prev_message, message in zip(messages, messages[1:], strict=False):
            if prev_message["role"] == message["role"]:
                raise ValueError("Conversation roles must alternate")
        return super().apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )


class AppendOnlyWithFinalTokenTokenizer(AppendOnlyTokenizer):
    final_token = [99]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        token_ids = super().apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=tokenize,
            **kwargs,
        )
        if add_generation_prompt:
            token_ids.extend(self.generation_prompt)
        else:
            token_ids.extend(self.final_token)
        return token_ids


class ConversationFinalTokenTokenizer:
    user_turn = [210, 211]
    assistant_turn = [220, 221]
    final_token = [299]
    generation_prompt = [230]

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
        assert tokenize
        token_ids = []
        for message in messages:
            if message["role"] == "user":
                token_ids.extend(self.user_turn)
            elif message["role"] == "assistant":
                token_ids.extend(self.assistant_turn)
            else:
                raise ValueError(f"Unsupported role: {message['role']}")

        if add_generation_prompt:
            token_ids.extend(self.generation_prompt)
        else:
            token_ids.extend(self.final_token)
        return token_ids


def test_initialize_system_prompt_infers_append_only_prefix():
    assert initialize_system_prompt(AppendOnlyTokenizer()) == AppendOnlyTokenizer.prefix


def test_extract_system_prompt_and_generation_uses_append_only_prefix():
    system_prompt, generation_prompt = extract_system_prompt_and_generation(AppendOnlyTokenizer())

    assert system_prompt == AppendOnlyTokenizer.prefix
    assert generation_prompt == AppendOnlyTokenizer.generation_prompt


def test_initialize_system_prompt_supports_alternating_role_templates():
    assert initialize_system_prompt(AlternatingTokenizer()) == AlternatingTokenizer.prefix


def test_initialize_system_prompt_handles_common_final_tokens():
    assert initialize_system_prompt(AppendOnlyWithFinalTokenTokenizer()) == AppendOnlyWithFinalTokenTokenizer.prefix


@pytest.mark.parametrize("tokenizer_cls", [AppendOnlyWithFinalTokenTokenizer, ConversationFinalTokenTokenizer])
def test_extract_generation_prompt_handles_replaced_final_tokens(tokenizer_cls):
    _, generation_prompt = extract_system_prompt_and_generation(tokenizer_cls())

    assert generation_prompt == tokenizer_cls.generation_prompt


def test_extract_system_prompt_and_generation_supports_alternating_role_templates():
    system_prompt, generation_prompt = extract_system_prompt_and_generation(AlternatingTokenizer())

    assert system_prompt == AlternatingTokenizer.prefix
    assert generation_prompt == AlternatingTokenizer.generation_prompt


@pytest.mark.parametrize("helper", [initialize_system_prompt, extract_system_prompt_and_generation])
def test_system_prompt_inference_ignores_non_append_only_templates(helper):
    result = helper(ConversationFinalTokenTokenizer())
    system_prompt = result[0] if isinstance(result, tuple) else result

    assert system_prompt == []
