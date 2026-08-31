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

import logging
from types import SimpleNamespace

import pytest

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
    MergeResult,
    MiniMaxContinuousTokenBuilder,
    MiniMaxText01ContinuousTokenBuilder,
    MiniMaxVLContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    QwenVLContinuousTokenBuilder,
    VLContinuousTokenBuilder,
)
from verl.utils.tokenizer.continuous_token_wiring import (
    CONTINUOUS_TOKEN_BUILDER_FAMILIES,
    ContinuousTokenModelFamily,
    create_continuous_token_builder,
    get_continuous_token_builder_class,
    infer_continuous_token_model_family,
    list_continuous_token_builder_families,
    resolve_continuous_token_model_family,
)
from verl.utils.tokenizer.deepseek import DeepSeekV4ContinuousTokenBuilder


class _DummyTokenizer:
    name_or_path = "Qwen/Qwen3-8B"


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


class _RecordingTemplateTokenizer(_TemplateTokenizer):
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
                "kwargs": dict(kwargs),
            }
        )
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )


class _NonPrefixStableTokenizer(_TemplateTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        if len(messages) > 1:
            rendered = "mutated-prefix:" + rendered
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
        self.turn_id = 106
        self.tool_call_id = 49

    def convert_tokens_to_ids(self, token):
        if token == "<|tool_response>":
            return self.tool_response_id
        if token == "<turn|>":
            return self.turn_id
        if token == "<tool_call|>":
            return self.tool_call_id
        return 0

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        """Minimal Gemma-style renderer: tool messages become ``<|tool_response>`` blocks
        whose function name is resolved positionally from the latest assistant tool_calls,
        mirroring the real Gemma template enough to exercise the builder's tool path.
        """
        assistant_tool_names: list[str] = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                assistant_tool_names = [
                    tool_call.get("function", {}).get("name", "unknown") for tool_call in message["tool_calls"]
                ]
        rendered = ""
        tool_index = 0
        for message in messages:
            role = message.get("role")
            if role == "tool":
                name = assistant_tool_names[tool_index] if tool_index < len(assistant_tool_names) else "unknown"
                tool_index += 1
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                rendered += f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>'
            else:
                rendered += f"<{role}>{message.get('content', '')}\n"
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _DeepSeekBoundaryTokenizer(_TemplateTokenizer):
    name_or_path = "deepseek-ai/DeepSeek-V3.1"
    unk_token_id = 0

    def __init__(self):
        self.eos_id = 1

    def convert_tokens_to_ids(self, token):
        if token == "<｜end▁of▁sentence｜>":
            return self.eos_id
        return self.unk_token_id

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        """Minimal DeepSeek-style renderer: tool calls are spliced in by string
        concatenation like the real R1 / V3.1 / V3.2 templates (a mapping raises
        TypeError there), and no generation prompt follows a tool message.
        """
        rendered = ""
        last_was_tool = False
        for message in messages:
            role = message.get("role")
            if role == "tool":
                rendered += f"<tool_output_begin>{message.get('content', '')}<tool_output_end>"
                last_was_tool = True
                continue
            last_was_tool = False
            if role == "assistant" and message.get("tool_calls"):
                calls = ""
                for tool_call in message["tool_calls"]:
                    function = tool_call["function"]
                    calls += "<tool_call_begin>" + function["name"] + "<tool_sep>" + function["arguments"]
                    calls += "<tool_call_end>"
                rendered += "<assistant>" + message.get("content", "") + calls + "<eos>"
            else:
                rendered += f"<{role}>{message.get('content', '')}\n"
        if add_generation_prompt and not last_was_tool:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _SpecialTokenTemplateTokenizer(_TemplateTokenizer):
    special_token_ids: dict[str, int] = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        token_ids = []
        special_tokens = sorted(self.special_token_ids, key=len, reverse=True)
        while text:
            matched = next((token for token in special_tokens if text.startswith(token)), None)
            if matched is None:
                token_ids.append(ord(text[0]))
                text = text[1:]
            else:
                token_ids.append(self.special_token_ids[matched])
                text = text[len(matched) :]
        return token_ids

    def convert_tokens_to_ids(self, token):
        return self.special_token_ids.get(token, 0)


class _MiniMaxAssistantTokenizer(_SpecialTokenTemplateTokenizer):
    special_token_ids = {"[e~[": 200020}
    eos_token_id = 200020

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tools, return_dict, kwargs
        rendered = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                rendered += "]~b]ai\n"
                reasoning = message.get("reasoning_content") or ""
                if reasoning:
                    rendered += f"<think>\n{reasoning}\n</think>\n\n"
                rendered += str(message.get("content") or "")
                if message.get("tool_calls"):
                    rendered += "\n<minimax:tool_call>\ncall\n</minimax:tool_call>"
                rendered += "[e~[\n"
            else:
                rendered += f"]~b]{role}\n{message.get('content', '')}[e~[\n"
        if add_generation_prompt:
            rendered += "]~b]ai\n<think>\n"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _MiniMaxText01AssistantTokenizer(_SpecialTokenTemplateTokenizer):
    special_token_ids = {
        "<beginning_of_sentence>": 200100,
        "<end_of_sentence>": 200101,
        "<function_call>": 200102,
    }
    eos_token_id = 200101

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tools, return_dict, kwargs
        rendered = ""
        for message in messages:
            role = message["role"]
            content = message.get("content", [])
            content_text = "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if role == "system":
                rendered += f"<beginning_of_sentence>system ai_setting=assistant\n{content_text}<end_of_sentence>\n"
            elif role == "user":
                rendered += f"<beginning_of_sentence>user name=user\n{content_text}<end_of_sentence>\n"
            elif role == "assistant":
                rendered += f"<beginning_of_sentence>ai name=assistant\n{content_text}<end_of_sentence>\n"
            elif role == "function":
                rendered += (
                    "<beginning_of_sentence>system function_response=functions\n"
                    f'{{"name": "{message["name"]}", "response": {content_text}}}'
                    "<end_of_sentence>\n"
                )
        if add_generation_prompt:
            rendered += "<beginning_of_sentence>ai name=assistant\n"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _MiniMaxText01UnconditionalScaffoldTokenizer(_MiniMaxText01AssistantTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        if not add_generation_prompt:
            rendered += "<beginning_of_sentence>ai name=assistant\n"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _GLMAssistantTokenizer(_SpecialTokenTemplateTokenizer):
    special_token_ids = {"<|observation|>": 151333, "<|user|>": 151336}

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tools, return_dict
        rendered = ""
        for message in messages:
            role = message["role"]
            if role == "assistant":
                rendered += "<|assistant|>"
                reasoning = message.get("reasoning_content") or ""
                rendered += f"<think>{reasoning}</think>" if reasoning else "</think>"
                rendered += str(message.get("content") or "")
                if message.get("tool_calls"):
                    rendered += "<tool_call>lookup</tool_call>"
            else:
                rendered += f"<|{role}|>{message.get('content', '')}"
        if add_generation_prompt:
            rendered += "<|assistant|>" + ("</think>" if kwargs.get("enable_thinking") is False else "<think>")
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _Gemma4AssistantTokenizer(_SpecialTokenTemplateTokenizer):
    tool_response_id = 48
    tool_call_id = 49
    turn_id = 106
    special_token_ids = {
        "<|tool_response>": tool_response_id,
        "<tool_call|>": tool_call_id,
        "<turn|>": turn_id,
    }

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tools, return_dict
        rendered = "<bos>"
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            rendered += f"<|turn>{role}\n"
            if message.get("tool_calls"):
                rendered += "<|tool_call>call:lookup{}<tool_call|>"
            rendered += str(message.get("content") or "")
            rendered += "<turn|>\n"
        if add_generation_prompt:
            rendered += "<|turn>model\n"
            if not kwargs.get("enable_thinking", False):
                rendered += "<|channel>thought\n<channel|>"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _Gemma4E4BAssistantTokenizer(_Gemma4AssistantTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tools, return_dict, kwargs
        rendered = "<bos>"
        for message in messages:
            role = "model" if message["role"] == "assistant" else message["role"]
            rendered += f"<|turn>{role}\n"
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            if message.get("tool_calls") and reasoning:
                rendered += f"<|channel>thought\n{reasoning}\n<channel|>"
            if message.get("tool_calls"):
                rendered += "<|tool_call>call:lookup{}<tool_call|><|tool_response>"
            rendered += str(message.get("content") or "")
            if not message.get("tool_calls"):
                rendered += "<turn|>\n"
        if add_generation_prompt:
            rendered += "<|turn>model\n"
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _DeepSeekAssistantTokenizer(_SpecialTokenTemplateTokenizer):
    name_or_path = "deepseek-ai/DeepSeek-V4-Flash"
    special_token_ids = {
        "<｜begin▁of▁sentence｜>": 0,
        "<｜end▁of▁sentence｜>": 1,
        "<｜User｜>": 2,
        "<｜Assistant｜>": 3,
    }
    eos_token_id = 1
    unk_token_id = -1


class _DeepSeekV31AssistantTokenizer(_SpecialTokenTemplateTokenizer):
    special_token_ids = {"<｜end▁of▁sentence｜>": 100001}
    eos_token_id = 100001
    unk_token_id = -1

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tools, return_dict
        rendered = "<｜begin▁of▁sentence｜>"
        last_role = None
        for message in messages:
            role = message["role"]
            if role == "system":
                rendered += str(message.get("content") or "")
            elif role == "user":
                rendered += "<｜User｜>" + str(message.get("content") or "")
            elif role == "assistant":
                if last_role == "user":
                    rendered += "<｜Assistant｜></think>"
                content = str(message.get("content") or "")
                if last_role != "tool" and "</think>" in content:
                    content = content.split("</think>", 1)[1]
                rendered += content + "<｜end▁of▁sentence｜>"
            elif role == "tool":
                rendered += "<tool>" + str(message.get("content") or "") + "</tool>"
            last_role = role
        if add_generation_prompt and last_role == "user":
            rendered += "<｜Assistant｜>" + ("<think>" if kwargs.get("thinking") else "</think>")
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


class _MockDeepSeekVL2Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def __call__(self, *, conversations, images, force_batchify, inference_mode):
        self.calls.append((conversations, images, force_batchify, inference_mode))
        token_ids = []
        for message in conversations:
            role = message["role"]
            content = message.get("content", "")
            token_ids.extend(ord(char) for char in role)
            token_ids.extend(ord(char) for char in content)
            if role == "<|Assistant|>" and content:
                token_ids.append(self.tokenizer.eos_token_id)
        if inference_mode and token_ids[-1:] == [self.tokenizer.eos_token_id]:
            token_ids = token_ids[:-1]

        class _TokenRow(list):
            def tolist(self):
                return list(self)

        return SimpleNamespace(input_ids=[_TokenRow(token_ids)])


class _MiniMaxVLAssistantTokenizer(_SpecialTokenTemplateTokenizer):
    special_token_ids = {
        "<beginning_of_sentence>": 200100,
        "<end_of_sentence>": 200101,
        "[e~[": 200102,
    }
    eos_token_id = 200101


class _MockMiniMaxVLAssistantProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        del tokenize, add_generation_prompt, tools, return_dict, kwargs
        rendered = ""
        for message in messages:
            role = "ai" if message["role"] == "assistant" else message["role"]
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            rendered += f"<beginning_of_sentence>{role}\n{content}<end_of_sentence>\n"
        # MiniMax-VL-01 appends this even when add_generation_prompt=False.
        return rendered + "<beginning_of_sentence>ai\n"

    def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
        del images, return_tensors, kwargs
        rendered = text[0] if isinstance(text, list | tuple) else text
        return {"input_ids": [self.tokenizer.encode(rendered, add_special_tokens=False)]}


class _MissingSpecialTokenTokenizer(_TemplateTokenizer):
    def convert_tokens_to_ids(self, token):
        return None


class _ListSpecialTokenQwenTokenizer(_QwenBoundaryTokenizer):
    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return [self.im_end_id]
        return super().convert_tokens_to_ids(token)


class _MultiIdSpecialTokenQwenTokenizer(_QwenBoundaryTokenizer):
    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return [self.im_end_id, self.im_end_id + 1]
        return super().convert_tokens_to_ids(token)


class _InvalidSpecialTokenQwenTokenizer(_QwenBoundaryTokenizer):
    def convert_tokens_to_ids(self, token):
        if token == "<|im_end|>":
            return -1
        return super().convert_tokens_to_ids(token)


class _MultiTokenNewlineQwenTokenizer(_QwenBoundaryTokenizer):
    def encode(self, text, add_special_tokens=False):
        if text == "\n":
            return [self.newline_id, self.newline_id + 1]
        return super().encode(text, add_special_tokens=add_special_tokens)


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
        "deepseek",
        "vldefault",
        "qwenvl",
        "qwen25vl",
        "qwen3vl",
        "minimaxvl",
        "gemma4vl",
        "kimivl",
        "glm4v",
        "deepseekvl2",
        "deepseekv4",
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
        (ContinuousTokenModelFamily.MINIMAX, MiniMaxText01ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M2, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M25, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M27, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM47, GLMContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM5, GLMContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GEMMA4, Gemma4ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GPTOSS, GptOssContinuousTokenBuilder),
        (ContinuousTokenModelFamily.DEEPSEEK, DeepSeekContinuousTokenBuilder),
        (ContinuousTokenModelFamily.VL_DEFAULT, VLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN_VL, QwenVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN25_VL, QwenVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN3_VL, QwenVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_VL, MiniMaxVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GEMMA4_VL, Gemma4VLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.KIMI_VL, KimiVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM4V, GLM46VContinuousTokenBuilder),
        (ContinuousTokenModelFamily.DEEPSEEK_VL2, DeepSeekVL2ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.DEEPSEEKV4, DeepSeekV4ContinuousTokenBuilder),
    ],
)
def test_builtin_family_class_mapping(family, builder_cls):
    assert get_continuous_token_builder_class(family) is builder_cls


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("glm4_moe", ContinuousTokenModelFamily.GLM47),
        ("glm_moe_dsa", ContinuousTokenModelFamily.GLM5),
        ("gemma4", ContinuousTokenModelFamily.GEMMA4),
        ("gpt_oss", ContinuousTokenModelFamily.GPTOSS),
        ("minimax_m2", ContinuousTokenModelFamily.MINIMAX_M2),
        ("minimax_text_01", ContinuousTokenModelFamily.MINIMAX),
        ("qwen3_5", ContinuousTokenModelFamily.QWEN35),
        ("qwen3_5_moe", ContinuousTokenModelFamily.QWEN35),
        ("qwen3", ContinuousTokenModelFamily.QWEN3),
        ("qwen3_moe", ContinuousTokenModelFamily.QWEN3),
        ("qwen2", ContinuousTokenModelFamily.QWEN),
        ("deepseek_v2", ContinuousTokenModelFamily.DEEPSEEK),
        ("deepseek_v3", ContinuousTokenModelFamily.DEEPSEEK),
        ("deepseek_v4", ContinuousTokenModelFamily.DEEPSEEKV4),
        # VL families.
        ("qwen2_5_vl", ContinuousTokenModelFamily.QWEN25_VL),
        ("qwen3_vl", ContinuousTokenModelFamily.QWEN3_VL),
        ("qwen3_vl_moe", ContinuousTokenModelFamily.QWEN3_VL),
        ("qwen2_vl", ContinuousTokenModelFamily.QWEN_VL),
        ("minimax_vl_01", ContinuousTokenModelFamily.MINIMAX_VL),
        ("kimi_vl", ContinuousTokenModelFamily.KIMI_VL),
        ("glm4v_moe", ContinuousTokenModelFamily.GLM4V),
        ("deepseek_vl_v2", ContinuousTokenModelFamily.DEEPSEEK_VL2),
    ],
)
def test_auto_family_inference_uses_exact_root_model_type(model_type, expected):
    assert infer_continuous_token_model_family(hf_model_type=model_type) == expected


def test_auto_family_inference_normalizes_hf_model_type():
    assert infer_continuous_token_model_family(hf_model_type=" DeepSeek_V4 ") == (ContinuousTokenModelFamily.DEEPSEEKV4)


def test_auto_family_inference_does_not_guess_unregistered_model_type(caplog):
    with caplog.at_level(logging.WARNING, logger="verl.utils.tokenizer.continuous_token_wiring"):
        family = infer_continuous_token_model_family(hf_model_type="unregistered_wrapper")
    assert family == ContinuousTokenModelFamily.DEFAULT
    assert "unregistered_wrapper" in caplog.text


def test_explicit_family_is_not_rewritten():
    assert (
        resolve_continuous_token_model_family(
            ContinuousTokenModelFamily.DEFAULT,
            hf_model_type="qwen3",
        )
        == ContinuousTokenModelFamily.DEFAULT
    )
    assert resolve_continuous_token_model_family(
        "qwen_3.5",
        hf_model_type="deepseek_v3",
    ) == (ContinuousTokenModelFamily.QWEN35)


def test_unknown_model_type_with_multimodal_processor_resolves_to_vl_default():
    assert (
        infer_continuous_token_model_family(
            hf_model_type="unknown_model",
            has_multimodal_processor=True,
        )
        == ContinuousTokenModelFamily.VL_DEFAULT
    )


def test_auto_family_is_resolved_at_factory_time():
    builder = create_continuous_token_builder(
        _QwenBoundaryTokenizer(),
        hf_model_type="qwen3",
    )
    assert isinstance(builder, QwenContinuousTokenBuilder)


def test_qwen2_auto_uses_qwen_builder_and_newline_boundary_logic():
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, hf_model_type="qwen2")
    result = builder._merge_non_assistant_token_ids([1, tokenizer.im_end_id], [2])

    assert isinstance(builder, QwenContinuousTokenBuilder)
    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2]
    assert result.inserted_token_ids == [tokenizer.newline_id]


def test_unknown_model_with_non_multimodal_processor_uses_default_text_builder(caplog):
    class TextProcessor:
        pass

    with caplog.at_level(logging.WARNING, logger="verl.utils.tokenizer.continuous_token_wiring"):
        builder = create_continuous_token_builder(
            _TemplateTokenizer(),
            hf_model_type="unknown_text_model",
            processor=TextProcessor(),
        )

    assert isinstance(builder, ContinuousTokenBuilder)
    assert "unknown_text_model" in caplog.text
    assert "default" in caplog.text


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


def test_builder_forwards_template_kwargs_and_tools_when_rendering_initial_prompt():
    tokenizer = _RecordingTemplateTokenizer()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    builder = create_continuous_token_builder(
        tokenizer,
        model_family="default",
        chat_template_kwargs={"enable_thinking": False},
    )

    builder.build_initial_tokens([{"role": "user", "content": "question"}], tools=tools)

    assert tokenizer.calls[-1]["add_generation_prompt"] is True
    assert tokenizer.calls[-1]["tools"] is tools
    assert tokenizer.calls[-1]["kwargs"] == {"enable_thinking": False}


def test_default_builder_is_available_from_builtin_registry():
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="default")
    assert isinstance(builder, ContinuousTokenBuilder)


def test_qwen3_builder_inserts_missing_newline_after_im_end():
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="qwen3")

    assert isinstance(builder, QwenContinuousTokenBuilder)
    result = builder._merge_non_assistant_token_ids([1, tokenizer.im_end_id], [2, 3])

    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2, 3]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 2
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
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
    result = builder._merge_non_assistant_token_ids([1, tokenizer.im_end_id], [2])

    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 1
    assert result.kind == "non_assistant"


def test_minimax_builder_inserts_missing_newline_after_eos():
    tokenizer = _MiniMaxBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="minimaxm2")

    assert isinstance(builder, MiniMaxContinuousTokenBuilder)
    result = builder._merge_non_assistant_token_ids([1, tokenizer.eos_id], [2, 3])

    assert result.token_ids == [1, tokenizer.eos_id, tokenizer.newline_id, 2, 3]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 2
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
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
    result = builder._merge_non_assistant_token_ids([1, tokenizer.observation_id], [tokenizer.user_id, 2])

    assert result.token_ids == [1, tokenizer.user_id, 2]
    assert result.removed_prefix_token_count == 1
    assert result.appended_token_count == 2
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1],
        [-0.1, -0.2],
    )
    assert aligned_mask == [1, 0, 0]
    assert aligned_logprobs == [-0.1, 0.0, 0.0]


def test_gemma4_builder_keeps_serialized_tool_response_boundary_for_appended_messages():
    tokenizer = _Gemma4AssistantTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="gemma4")
    previous_messages = [{"role": "user", "content": "question"}]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "name": "lookup"}]

    result = builder.merge_non_assistant_tokens(previous_messages, updated_messages, [1, 2, 3])

    assert isinstance(builder, Gemma4ContinuousTokenBuilder)
    assert result.token_ids[:4] == [1, 2, 3, tokenizer.tool_response_id]
    assert result.inserted_token_ids == []
    assert result.appended_token_count == len(result.token_ids) - 3
    assert result.kind == "non_assistant"


def test_gemma4_builder_does_not_duplicate_existing_tool_response_boundary():
    tokenizer = _Gemma4AssistantTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family=ContinuousTokenModelFamily.GEMMA4)
    previous_messages = [{"role": "user", "content": "question"}]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "name": "lookup"}]

    result = builder.merge_non_assistant_tokens(previous_messages, updated_messages, [1, tokenizer.tool_response_id])

    assert result.token_ids.count(tokenizer.tool_response_id) == 1
    assert result.inserted_token_ids == []
    assert result.kind == "non_assistant"


def test_gemma4_builder_formats_tool_response_by_position_with_warning(caplog):
    builder = create_continuous_token_builder(_Gemma4BoundaryTokenizer(), model_family="gemma4")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
        }
    ]
    tool_messages = [{"role": "tool", "content": "answer"}]

    with caplog.at_level(logging.WARNING):
        token_ids = builder._tokenize_tool_group(
            tool_messages,
            previous_messages=previous_messages,
        )

    expected = '<|tool_response>response:lookup{value:<|"|>answer<|"|>}<tool_response|>'
    assert token_ids == [ord(char) for char in expected]
    assert "resolving a tool response name by position" in caplog.text


def test_deepseek_builder_renders_tool_appends_through_string_concatenating_template():
    tokenizer = _DeepSeekBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="deepseek")
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": {"q": "x"}}}
            ],
        },
    ]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "tool_call_id": "call_0"}]

    # The base synthetic tool call carries a mapping, which this template family cannot splice in.
    with pytest.raises(TypeError):
        ContinuousTokenBuilder(tokenizer).tokenize_non_assistant_incremental_messages(
            previous_messages, updated_messages
        )

    assert isinstance(builder, DeepSeekContinuousTokenBuilder)
    incremental = builder.tokenize_non_assistant_incremental_messages(previous_messages, updated_messages)

    # Only the tool output: DeepSeek templates add no generation prompt after a tool message.
    assert incremental == [ord(char) for char in "<tool_output_begin>answer<tool_output_end>"]


def test_deepseek_builder_synthetic_tool_call_arguments_are_a_json_string():
    builder = DeepSeekContinuousTokenBuilder(_DeepSeekBoundaryTokenizer())
    tool_messages = [
        {"role": "tool", "content": "first", "tool_call_id": "call_0"},
        {"role": "tool", "content": "second", "tool_call_id": "call_1"},
    ]

    synthetic_assistant = builder._synthetic_assistant_for_tools(tool_messages)

    assert [tool_call["id"] for tool_call in synthetic_assistant["tool_calls"]] == ["call_0", "call_1"]
    assert all(tool_call["function"]["arguments"] == "{}" for tool_call in synthetic_assistant["tool_calls"])


def test_gpt_oss_builder_formats_tool_responses_with_resolved_tool_name():
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
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

    token_ids = builder._tokenize_tool_group(tool_messages, previous_messages=previous_messages)

    expected = "<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>ok<|end|>"
    assert isinstance(builder, GptOssContinuousTokenBuilder)
    assert token_ids == [ord(char) for char in expected]


def test_gpt_oss_builder_prefers_tool_message_name_over_context_id():
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "from_context"},
                }
            ],
        }
    ]
    tool_messages = [{"role": "tool", "tool_call_id": "call_0", "name": "from_message", "content": "ok"}]

    token_ids = builder._tokenize_tool_group(tool_messages, previous_messages=previous_messages)

    expected = "<|start|>functions.from_message to=assistant<|channel|>commentary<|message|>ok<|end|>"
    assert token_ids == [ord(char) for char in expected]


def test_gpt_oss_builder_formats_multiple_tool_responses_by_position_with_warning(caplog):
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search"}},
                {"type": "function", "function": {"name": "calculate"}},
            ],
        }
    ]
    tool_messages = [
        {"role": "tool", "content": "hits"},
        {"role": "tool", "content": "42"},
    ]

    with caplog.at_level(logging.WARNING):
        token_ids = builder._tokenize_tool_group(tool_messages, previous_messages=previous_messages)

    expected = (
        "<|start|>functions.search to=assistant<|channel|>commentary<|message|>hits<|end|>"
        "<|start|>functions.calculate to=assistant<|channel|>commentary<|message|>42<|end|>"
    )
    assert token_ids == [ord(char) for char in expected]
    assert "resolving a tool response name by position" in caplog.text


def test_gpt_oss_builder_rejects_ambiguous_positional_tool_name_resolution():
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search"}},
                {"type": "function", "function": {"name": "calculate"}},
            ],
        }
    ]

    with pytest.raises(ValueError, match="cannot resolve tool name by position"):
        builder._tokenize_tool_group([{"role": "tool", "content": "hits"}], previous_messages=previous_messages)

    with pytest.raises(ValueError, match="cannot resolve tool name by position"):
        builder._tokenize_tool_group([{"role": "tool", "content": "fallback"}], previous_messages=[])


def test_gpt_oss_builder_does_not_use_older_assistant_tool_calls_for_position():
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "old_lookup"}}],
        },
        {"role": "assistant", "content": "new answer without tools"},
    ]

    with pytest.raises(ValueError, match="latest assistant has 0 tool calls"):
        builder._tokenize_tool_group([{"role": "tool", "content": "answer"}], previous_messages=previous_messages)


@pytest.mark.parametrize(
    ("builder", "expected_error"),
    [
        (
            create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss"),
            "got 2 tool response messages but the latest assistant has 4 tool calls",
        ),
        (
            create_continuous_token_builder(_Gemma4BoundaryTokenizer(), model_family="gemma4"),
            "got 2 tool response messages but the latest assistant has 4 tool calls",
        ),
    ],
)
def test_strict_tool_name_builders_reject_split_positional_tool_groups(builder, expected_error):
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search"}},
                {"type": "function", "function": {"name": "calculate"}},
                {"type": "function", "function": {"name": "lookup_order"}},
                {"type": "function", "function": {"name": "get_weather"}},
            ],
        },
    ]
    appended_messages = [
        {"role": "tool", "content": "hits"},
        {"role": "tool", "content": "42"},
        {"role": "user", "content": "please continue"},
        {"role": "tool", "content": "order shipped"},
    ]

    with pytest.raises(ValueError, match=expected_error):
        builder.tokenize_non_assistant_incremental_messages(previous_messages, previous_messages + appended_messages)


def test_default_builder_builds_dummy_assistant_from_tool_messages_only():
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    tool_messages = [
        {"role": "tool", "content": "answer", "name": "from_message"},
        {"role": "tool", "content": "fallback"},
    ]

    builder._tokenize_tool_group(tool_messages, previous_messages=[])

    synthetic_assistant = tokenizer.calls[0]["messages"][2]
    assert synthetic_assistant["tool_calls"][0] == {
        "id": "continuous_token_call_0",
        "type": "function",
        "function": {"name": "from_message", "arguments": {}},
    }
    assert synthetic_assistant["tool_calls"][1] == {
        "id": "continuous_token_call_1",
        "type": "function",
        "function": {"name": "continuous_token_tool", "arguments": {}},
    }


def test_default_builder_merges_append_only_non_assistant_messages():
    tokenizer = _TemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [{"role": "tool", "content": "answer", "tool_call_id": "call_0", "name": "lookup"}]
    runtime_ids = [1, 2, 3]

    result = builder.merge_non_assistant_tokens(old_messages, new_messages, runtime_ids)
    expected_incremental = builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages)

    assert isinstance(result, MergeResult)
    assert result.token_ids == runtime_ids + expected_incremental
    assert result.appended_token_count == len(expected_incremental)
    assert result.kind == "non_assistant"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1, 1],
        [0.1, 0.2, 0.3],
    )
    assert aligned_mask == [1, 1, 1] + [0] * len(expected_incremental)
    assert aligned_logprobs == [0.1, 0.2, 0.3] + [0.0] * len(expected_incremental)


def test_default_builder_tokenizes_system_and_user_appends_with_generation_prompt():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "retry"},
    ]

    incremental = builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages)

    expected = "<system>policy\n<user>retry\n<assistant>"
    assert incremental == [ord(char) for char in expected]


def test_default_builder_fuses_generation_prompt_into_last_append_group():
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    old_messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    new_messages = old_messages + [
        {"role": "tool", "content": "first result", "name": "lookup"},
        {"role": "tool", "content": "second result", "name": "lookup"},
    ]

    builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages, tools=tools)

    assert len(tokenizer.calls) == 2
    assert [len(call["messages"]) for call in tokenizer.calls] == [3, 5]
    assert [call["add_generation_prompt"] for call in tokenizer.calls] == [False, True]
    assert all(call["tools"] is tools for call in tokenizer.calls)
    assert all(call["kwargs"] == {"enable_thinking": False} for call in tokenizer.calls)


def test_default_builder_only_fuses_generation_prompt_into_final_append_group():
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "retry"},
    ]

    builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages)

    assert len(tokenizer.calls) == 4
    assert [call["add_generation_prompt"] for call in tokenizer.calls] == [False, False, False, True]


def test_special_builder_can_keep_separate_full_history_generation_prompt():
    class FullHistoryGenerationPromptBuilder(ContinuousTokenBuilder):
        def _should_fuse_generation_prompt_with_last_group(self):
            return False

    tokenizer = _RecordingTemplateTokenizer()
    builder = FullHistoryGenerationPromptBuilder(tokenizer)
    old_messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    new_messages = old_messages + [{"role": "user", "content": "retry"}]

    builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages)

    assert len(tokenizer.calls) == 4
    assert [len(call["messages"]) for call in tokenizer.calls] == [2, 3, len(new_messages), len(new_messages)]
    assert [call["add_generation_prompt"] for call in tokenizer.calls] == [False, False, False, True]


def test_default_builder_does_not_reencode_existing_trajectory_for_generation_prompt():
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    old_messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "already encoded assistant"},
    ]
    new_messages = [*old_messages, {"role": "user", "content": "retry"}]

    builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages)

    rendered_messages = [message for call in tokenizer.calls for message in call["messages"]]
    assert old_messages[0] not in rendered_messages
    assert old_messages[1] not in rendered_messages
    assert new_messages[-1] in rendered_messages


def test_default_builder_rejects_multi_message_user_or_system_groups():
    class BadGroupingBuilder(ContinuousTokenBuilder):
        def _iter_append_groups(self, appended_messages):
            return [appended_messages]

    builder = BadGroupingBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [
        {"role": "user", "content": "retry"},
        {"role": "user", "content": "more context"},
    ]

    with pytest.raises(ValueError, match="expects one 'user' message per append group"):
        builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages)


def test_default_builder_appends_assistant_tokens_to_runtime_stream():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())

    result = builder.merge_assistant_tokens([1, 2, 3], [4, 5])

    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.appended_token_count == 2
    assert result.kind == "assistant"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [0, 1],
        [0.0, -0.1],
        assistant_logprobs=[-0.2, -0.3],
    )
    assert aligned_mask == [0, 1, 1, 1]
    assert aligned_logprobs == [0.0, -0.1, -0.2, -0.3]


def test_default_builder_encodes_prepared_assistant_continuation_once():
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    message = {"role": "assistant", "content": "gold"}

    assistant_ids = builder.tokenize_assistant_message(message)

    assert assistant_ids == tokenizer.encode("gold\n", add_special_tokens=False)
    assert len(tokenizer.calls) == 2
    assert all(message not in call["messages"] for call in tokenizer.calls[:1])
    assert tokenizer.calls[1]["messages"][-1] is message


def test_default_builder_trims_at_first_generated_terminator():
    tokenizer = _TemplateTokenizer()
    tokenizer.eos_token_id = 99
    builder = ContinuousTokenBuilder(tokenizer)

    normalized_ids = builder._normalize_assistant_token_ids(
        [10, tokenizer.eos_token_id, 20, tokenizer.eos_token_id, 30],
        {"role": "assistant", "content": "gold"},
    )

    assert normalized_ids == [10, tokenizer.eos_token_id]


def test_gpt_oss_builder_uses_message_specific_assistant_terminators():
    tokenizer = _TemplateTokenizer()
    tokenizer.eos_token_id = 200002
    tokenizer.convert_tokens_to_ids = lambda token: {"<|call|>": 200012}.get(token, 0)
    builder = GptOssContinuousTokenBuilder(tokenizer)

    tool_call_ids = builder._normalize_assistant_token_ids(
        [10, 200012, 99],
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function"}]},
    )
    final_answer_ids = builder._normalize_assistant_token_ids(
        [20, tokenizer.eos_token_id, 99],
        {"role": "assistant", "content": "done"},
    )

    assert tool_call_ids == [10, 200012]
    assert final_answer_ids == [20, tokenizer.eos_token_id]


def test_gpt_oss_builder_normalizes_nullable_assistant_fields_for_harmony():
    builder = GptOssContinuousTokenBuilder(_TemplateTokenizer())

    rendered_message = builder._prepare_assistant_message_for_render(
        {
            "role": "assistant",
            "content": None,
            "thinking": None,
            "tool_calls": None,
            "name": None,
        }
    )

    assert rendered_message == {"role": "assistant", "content": ""}


def test_minimax_text01_builder_encodes_plain_and_structured_tool_call_continuations():
    tokenizer = _MiniMaxText01AssistantTokenizer()
    builder = MiniMaxText01ContinuousTokenBuilder(tokenizer)

    plain_ids = builder.tokenize_assistant_message({"role": "assistant", "content": "gold"})
    tool_call_ids = builder.tokenize_assistant_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        }
    )

    assert plain_ids == tokenizer.encode("gold<end_of_sentence>", add_special_tokens=False)
    assert tool_call_ids == tokenizer.encode(
        '<function_call>```typescript\nfunctions.lookup({"q":"x"})\n```<end_of_sentence>',
        add_special_tokens=False,
    )


def test_minimax_text01_builder_merges_openai_tool_response():
    tokenizer = _MiniMaxText01AssistantTokenizer()
    builder = MiniMaxText01ContinuousTokenBuilder(tokenizer)
    previous_messages = [
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
        }
    ]
    updated_messages = [
        *previous_messages,
        {"role": "tool", "tool_call_id": "call_0", "content": '{"value": 1}'},
    ]
    runtime_ids = [7, tokenizer.eos_token_id]

    result = builder.merge_non_assistant_tokens(previous_messages, updated_messages, runtime_ids)

    expected_response = tokenizer.encode(
        "<beginning_of_sentence>system function_response=functions\n"
        '{"name": "lookup", "response": {"value": 1}}<end_of_sentence>\n',
        add_special_tokens=False,
    )
    assert result.token_ids == runtime_ids + [ord("\n")] + expected_response + builder._generation_scaffold_ids


def test_minimax_text01_builder_prepares_structured_tool_history():
    tokenizer = _MiniMaxText01AssistantTokenizer()
    builder = MiniMaxText01ContinuousTokenBuilder(tokenizer)
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": '{"value": 1}'},
    ]

    rendered = builder._render_text(messages, add_generation_prompt=True)

    assert '<function_call>```typescript\nfunctions.lookup({"q":"x"})\n```' in rendered
    assert 'function_response=functions\n{"name": "lookup", "response": {"value": 1}}' in rendered


def test_minimax_text01_builder_normalizes_unconditional_generation_scaffold():
    tokenizer = _MiniMaxText01UnconditionalScaffoldTokenizer()
    builder = MiniMaxText01ContinuousTokenBuilder(tokenizer)
    previous_messages = [{"role": "assistant", "content": "gold"}]
    updated_messages = [*previous_messages, {"role": "user", "content": "retry"}]
    runtime_ids = [7, tokenizer.eos_token_id]

    assistant_ids = builder.tokenize_assistant_message({"role": "assistant", "content": "gold"})
    user_ids = builder._tokenize_single_non_tool({"role": "user", "content": "retry"}, add_generation_prompt=True)
    result = builder.merge_non_assistant_tokens(previous_messages, updated_messages, runtime_ids)

    assert builder._should_fuse_generation_prompt_with_last_group() is False
    assert assistant_ids == tokenizer.encode("gold<end_of_sentence>", add_special_tokens=False)
    assert user_ids == tokenizer.encode(
        "<beginning_of_sentence>user name=user\nretry<end_of_sentence>\n",
        add_special_tokens=False,
    )
    assert result.token_ids == runtime_ids + [ord("\n")] + user_ids + builder._generation_scaffold_ids


def test_minimax_builder_reconstructs_empty_and_nonempty_reasoning_continuations():
    tokenizer = _MiniMaxAssistantTokenizer()
    builder = MiniMaxContinuousTokenBuilder(tokenizer)

    empty_reasoning_ids = builder.tokenize_assistant_message({"role": "assistant", "content": "done"})
    reasoning_ids = builder.tokenize_assistant_message(
        {"role": "assistant", "reasoning_content": "reason", "content": "done"}
    )

    assert empty_reasoning_ids == tokenizer.encode("</think>\n\ndone[e~[", add_special_tokens=False)
    assert reasoning_ids == tokenizer.encode("reason\n</think>\n\ndone[e~[", add_special_tokens=False)


def test_minimax_builder_preserves_nested_literal_think_tags():
    tokenizer = _MiniMaxAssistantTokenizer()
    builder = MiniMaxContinuousTokenBuilder(tokenizer)

    assistant_ids = builder.tokenize_assistant_message(
        {"role": "assistant", "content": "<think>I need output the <think> tag</think><think>"}
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

    assistant_ids = builder.tokenize_assistant_message(message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


def test_glm_builder_drops_embedded_reasoning_from_text_blocks_when_thinking_is_disabled():
    tokenizer = _GLMAssistantTokenizer()
    builder = GLMContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})

    rendered_message = builder._prepare_assistant_message_for_render(
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

    assistant_ids = builder.tokenize_assistant_message(message)

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
            "<|channel>thought\nreason<channel|>done<turn|>",
        ),
        (
            True,
            {
                "role": "assistant",
                "thinking": "call reason",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
            },
            "<|channel>thought\ncall reason<channel|><|tool_call>call:lookup{}<tool_call|>",
        ),
    ],
)
def test_gemma4_builder_reconstructs_generation_scaffold(enable_thinking, message, expected_text):
    tokenizer = _Gemma4AssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": enable_thinking})

    assistant_ids = builder.tokenize_assistant_message(message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


def test_gemma4_e4b_builder_uses_template_reasoning_without_duplicate_scaffold():
    tokenizer = _Gemma4E4BAssistantTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})
    message = {
        "role": "assistant",
        "reasoning_content": "call reason",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
    }

    assistant_ids = builder.tokenize_assistant_message(message)

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

    assistant_ids = builder.tokenize_assistant_message(
        {"role": "assistant", "reasoning_content": "reason", "content": "gold"}
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

    assistant_ids = builder.tokenize_assistant_message(message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


def test_deepseek_v4_builder_keeps_committed_reasoning_when_drop_thinking_is_enabled():
    tokenizer = _DeepSeekAssistantTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(
        tokenizer,
        chat_template_kwargs={"enable_thinking": True, "drop_thinking": True},
    )
    previous_messages = [{"role": "user", "content": "q1"}]
    runtime_ids = builder.build_initial_tokens(previous_messages)
    assistant_ids = builder.tokenize_assistant_message(
        {"role": "assistant", "reasoning_content": "reason A", "content": "answer A"},
        previous_messages=previous_messages,
    )
    runtime_ids = builder.merge_assistant_tokens(runtime_ids, assistant_ids).token_ids
    previous_messages = [
        *previous_messages,
        {"role": "assistant", "reasoning_content": "reason A", "content": "answer A"},
    ]

    result = builder.merge_non_assistant_tokens(
        previous_messages,
        [*previous_messages, {"role": "user", "content": "q2"}],
        runtime_ids,
    )

    assert result.token_ids[: len(runtime_ids)] == runtime_ids
    reason_ids = tokenizer.encode("reason A", add_special_tokens=False)
    assert assistant_ids[: len(reason_ids)] == reason_ids


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

    assistant_ids = builder.tokenize_assistant_message(message)

    assert assistant_ids == tokenizer.encode(expected_text, add_special_tokens=False)


def test_deepseek_builder_serializes_synthetic_tool_arguments():
    builder = DeepSeekContinuousTokenBuilder(_DeepSeekAssistantTokenizer())

    synthetic_assistant = builder._synthetic_assistant_for_tools(
        [{"role": "tool", "name": "lookup", "content": "value"}]
    )

    assert synthetic_assistant["tool_calls"][0]["function"]["arguments"] == "{}"


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

    assistant_ids = builder.tokenize_assistant_message(
        {"role": "assistant", "content": "gold"},
        previous_messages=previous_messages,
    )

    assert assistant_ids == tokenizer.encode("gold<｜end▁of▁sentence｜>", add_special_tokens=False)


def test_deepseek_vl2_builder_uses_processor_for_text_prompt_and_assistant():
    tokenizer = _DeepSeekAssistantTokenizer()
    processor = _MockDeepSeekVL2Processor(tokenizer)
    builder = DeepSeekVL2ContinuousTokenBuilder(tokenizer, processor)

    initial_ids = builder.build_initial_tokens([{"role": "user", "content": "question"}])
    assistant_ids = builder.tokenize_assistant_message({"role": "assistant", "content": "gold"})

    assert initial_ids
    assert assistant_ids == [ord(char) for char in "gold"] + [tokenizer.eos_token_id]
    assert len(processor.calls) == 3
    assert all(force_batchify for _, _, force_batchify, _ in processor.calls)
    assert [inference_mode for _, _, _, inference_mode in processor.calls] == [True, True, False]


def test_deepseek_vl2_builder_preserves_native_system_role():
    tokenizer = _DeepSeekAssistantTokenizer()
    processor = _MockDeepSeekVL2Processor(tokenizer)
    builder = DeepSeekVL2ContinuousTokenBuilder(tokenizer, processor)

    builder.build_initial_tokens(
        [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "question"},
        ]
    )

    conversation = processor.calls[0][0]
    assert conversation[0] == {"role": "system", "content": "policy"}
    assert conversation[1]["role"] == "<|User|>"


def test_deepseek_vl2_builder_rejects_unsupported_structured_tools():
    tokenizer = _DeepSeekAssistantTokenizer()
    processor = _MockDeepSeekVL2Processor(tokenizer)
    builder = DeepSeekVL2ContinuousTokenBuilder(tokenizer, processor)

    with pytest.raises(ValueError, match="does not support structured assistant tool calls"):
        builder.tokenize_assistant_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
            }
        )

    previous_messages = [{"role": "user", "content": "question"}]
    with pytest.raises(ValueError, match="does not support tool response messages"):
        builder.merge_non_assistant_tokens(
            previous_messages,
            [*previous_messages, {"role": "tool", "content": "value"}],
            [1, 2, 3],
        )


def test_deepseek_vl2_builder_rejects_processor_output_that_rewrites_runtime_prefix():
    tokenizer = _DeepSeekAssistantTokenizer()
    processor = _MockDeepSeekVL2Processor(tokenizer)
    builder = DeepSeekVL2ContinuousTokenBuilder(tokenizer, processor)
    previous_messages = [{"role": "user", "content": "question"}]

    with pytest.raises(ValueError, match="does not preserve the runtime prefix"):
        builder.merge_non_assistant_tokens(
            previous_messages,
            [*previous_messages, {"role": "user", "content": "retry"}],
            [999],
        )


def test_minimax_vl_builder_extracts_assistant_after_unconditional_scaffold():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)

    assistant_ids = builder.tokenize_assistant_message({"role": "assistant", "content": "gold"})

    assert assistant_ids == tokenizer.encode("gold<end_of_sentence>", add_special_tokens=False)


def test_minimax_vl_builder_keeps_generation_scaffold_separate_for_user_append():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)
    previous_messages = [{"role": "assistant", "content": "gold"}]
    updated_messages = [*previous_messages, {"role": "user", "content": "retry"}]
    runtime_ids = [7, tokenizer.eos_token_id]

    user_ids = builder._tokenize_single_non_tool({"role": "user", "content": "retry"}, add_generation_prompt=True)
    result = builder.merge_non_assistant_tokens(previous_messages, updated_messages, runtime_ids)

    assert builder._should_fuse_generation_prompt_with_last_group() is False
    assert result.token_ids == runtime_ids + [ord("\n")] + user_ids + builder._vl_scaffold_ids


def test_minimax_vl_builder_formats_openai_tool_response_as_function_message():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
    ]

    token_ids = builder._tokenize_tool_group(
        [{"role": "tool", "tool_call_id": "call_0", "content": '{"value": 1}'}],
        previous_messages=previous_messages,
    )

    assert token_ids == tokenizer.encode(
        "<beginning_of_sentence>system function_response=functions\n"
        '{"name": "lookup", "response": {"value": 1}}<end_of_sentence>\n',
        add_special_tokens=False,
    )


def test_minimax_vl_builder_reconstructs_structured_assistant_tool_call():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)

    assistant_ids = builder.tokenize_assistant_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        }
    )

    assert assistant_ids == tokenizer.encode(
        '<function_call>```typescript\nfunctions.lookup({"q":"x"})\n```<end_of_sentence>',
        add_special_tokens=False,
    )


def test_minimax_vl_builder_merges_tool_result_and_fixed_generation_scaffold():
    tokenizer = _MiniMaxVLAssistantTokenizer()
    processor = _MockMiniMaxVLAssistantProcessor(tokenizer)
    builder = MiniMaxVLContinuousTokenBuilder(tokenizer, processor)
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
        {"role": "tool", "tool_call_id": "call_0", "content": '{"value": 1}'},
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
    assert result.token_ids == runtime_ids + [ord("\n")] + expected_response + builder._vl_scaffold_ids
    assert result.inserted_token_ids == [ord("\n")]
    assert result.appended_token_count == len(expected_response) + len(builder._vl_scaffold_ids)


def test_kimi_vl_builder_rejects_unsupported_structured_tool_responses():
    builder = KimiVLContinuousTokenBuilder(_QwenBoundaryTokenizer(), object())

    with pytest.raises(ValueError, match="does not support structured tool schemas"):
        builder.build_initial_tokens(
            [{"role": "user", "content": "question"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )

    with pytest.raises(ValueError, match="does not support structured assistant tool calls"):
        builder.tokenize_assistant_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
            }
        )

    with pytest.raises(ValueError, match="does not support structured tool response messages"):
        builder.build_initial_tokens(
            [
                {"role": "user", "content": "question"},
                {"role": "tool", "content": "value"},
            ]
        )

    with pytest.raises(ValueError, match="does not support structured tool responses"):
        builder._tokenize_tool_group(
            [{"role": "tool", "name": "lookup", "content": "value"}],
            previous_messages=[],
            add_generation_prompt=True,
        )


def test_kimi_vl_builder_trims_at_first_im_end_terminator():
    tokenizer = _QwenBoundaryTokenizer()
    builder = KimiVLContinuousTokenBuilder(tokenizer, object())

    assistant_ids = builder._normalize_assistant_token_ids(
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

    rendered_message = builder._prepare_assistant_message_for_render(message)
    normalized_ids = builder._normalize_assistant_token_ids(
        [1, tokenizer.im_end_id, tokenizer.newline_id],
        message,
    )

    assert rendered_message["reasoning_content"] == "I need output the <think> tag"
    assert rendered_message["content"] == "<think>"
    assert normalized_ids == [1, tokenizer.im_end_id]


def test_qwen_builder_drops_prepared_reasoning_when_thinking_is_disabled():
    tokenizer = _QwenBoundaryTokenizer()
    builder = QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})

    explicit_reasoning = builder._prepare_assistant_message_for_render(
        {"role": "assistant", "reasoning_content": "hidden", "content": "answer"}
    )
    embedded_reasoning = builder._prepare_assistant_message_for_render(
        {"role": "assistant", "content": "<think>hidden</think>answer"}
    )

    assert explicit_reasoning == {"role": "assistant", "reasoning_content": "", "content": "answer"}
    assert embedded_reasoning == {"role": "assistant", "reasoning_content": "", "content": "answer"}


def test_assistant_alignment_validates_logprobs():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(token_ids=[1, 2, 3], appended_token_count=2, kind="assistant")

    aligned_mask, aligned_logprobs = builder.align_response_metadata(result, [1])
    assert aligned_mask == [1, 1, 1]
    assert aligned_logprobs is None

    with pytest.raises(ValueError, match="response_logprobs is required"):
        builder.align_response_metadata(result, [1], assistant_logprobs=[-0.1, -0.2])

    with pytest.raises(ValueError, match="assistant_logprobs is required"):
        builder.align_response_metadata(result, [1], [0.0])

    with pytest.raises(ValueError, match="assistant_logprobs length must match"):
        builder.align_response_metadata(result, [1], [0.0], assistant_logprobs=[-0.1])


def test_builder_align_response_metadata_handles_inserted_boundary_tokens():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(
        token_ids=[1, 2, 99, 3],
        appended_token_count=1,
        kind="non_assistant",
        inserted_token_ids=[99],
    )

    aligned_mask, aligned_logprobs = builder.align_response_metadata(result, [1, 1], [0.1, 0.2])

    assert aligned_mask == [1, 1, 0, 0]
    assert aligned_logprobs == [0.1, 0.2, 0.0, 0.0]


def test_alignment_rejects_unknown_merge_kind():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(token_ids=[1], appended_token_count=0, kind="unknown")

    with pytest.raises(ValueError, match="Unknown Continuous Token merge kind"):
        builder.align_response_metadata(result, [1])


def test_default_builder_rejects_mutated_message_prefix():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    changed_messages = [{"role": "user", "content": "different"}]

    with pytest.raises(ValueError, match="prefix messages changed"):
        builder.tokenize_non_assistant_incremental_messages(old_messages, changed_messages)

    with pytest.raises(ValueError, match="updated_messages is shorter"):
        builder.tokenize_non_assistant_incremental_messages(old_messages, [])


def test_default_builder_returns_empty_delta_when_no_message_is_appended():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    messages = [{"role": "user", "content": "question"}]

    assert builder.tokenize_non_assistant_incremental_messages(messages, messages) == []


def test_default_builder_rejects_non_prefix_stable_template_deltas():
    builder = ContinuousTokenBuilder(_NonPrefixStableTokenizer())

    with pytest.raises(ValueError, match="token-id suffix diff failed"):
        builder.render_delta_token_id(
            [{"role": "user", "content": "question"}],
            [{"role": "tool", "content": "answer"}],
            add_generation_prompt=True,
        )


def test_subclass_only_overrides_token_level_merge_hook():
    class BoundaryBuilder(ContinuousTokenBuilder):
        def _merge_non_assistant_token_ids(self, runtime_token_ids, appended_token_ids):
            return MergeResult(
                token_ids=list(runtime_token_ids) + [99] + list(appended_token_ids),
                appended_token_count=len(appended_token_ids),
                kind="non_assistant",
                inserted_token_ids=[99],
            )

    builder = BoundaryBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [{"role": "tool", "content": "answer"}]
    incremental = builder.tokenize_non_assistant_incremental_messages(old_messages, new_messages)

    result = builder.merge_non_assistant_tokens(old_messages, new_messages, [1, 2, 3])

    assert result.token_ids == [1, 2, 3, 99] + incremental
    assert result.appended_token_count == len(incremental)
    assert result.inserted_token_ids == [99]
    assert result.kind == "non_assistant"


def test_non_assistant_alignment_handles_boundary_inserts_and_trims():
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(
        token_ids=[1, 2, 99, 3, 4],
        appended_token_count=2,
        kind="non_assistant",
        inserted_token_ids=[99],
        removed_prefix_token_count=1,
    )

    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1, 1],
        [0.1, 0.2, 0.3],
    )
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs == [0.1, 0.2, 0.0, 0.0, 0.0]

    aligned_mask, aligned_logprobs = builder.align_response_metadata(result, [1, 1, 1])
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs is None


def test_builder_rejects_unsupported_append_roles():
    builder = ContinuousTokenBuilder(_TemplateTokenizer(), allowed_append_roles=["tool"])

    with pytest.raises(ValueError, match="got 'user'"):
        builder.tokenize_non_assistant_incremental_messages(
            [{"role": "user", "content": "question"}],
            [{"role": "user", "content": "question"}, {"role": "user", "content": "retry"}],
        )

    with pytest.raises(ValueError, match="Unsupported Continuous Token append roles"):
        ContinuousTokenBuilder(_TemplateTokenizer(), allowed_append_roles=["assistant"])


def test_model_specific_builders_validate_required_special_tokens():
    with pytest.raises(ValueError, match="required token '<\\|im_end\\|>'"):
        QwenContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '\\[e~\\['"):
        MiniMaxContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<end_of_sentence>'"):
        MiniMaxText01ContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<\\|observation\\|>'"):
        GLMContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<\\|tool_response>'"):
        Gemma4ContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<｜end▁of▁sentence｜>'"):
        DeepSeekV4ContinuousTokenBuilder(_MissingSpecialTokenTokenizer())


def test_model_specific_builders_validate_special_token_id_shape():
    builder = QwenContinuousTokenBuilder(_ListSpecialTokenQwenTokenizer())
    assert builder._merge_non_assistant_token_ids([1, builder._im_end_id], [2]).token_ids == [
        1,
        builder._im_end_id,
        198,
        2,
    ]

    with pytest.raises(ValueError, match="returned multiple ids"):
        QwenContinuousTokenBuilder(_MultiIdSpecialTokenQwenTokenizer())

    with pytest.raises(ValueError, match="returned invalid id"):
        QwenContinuousTokenBuilder(_InvalidSpecialTokenQwenTokenizer())

    with pytest.raises(ValueError, match="Expected Qwen newline"):
        QwenContinuousTokenBuilder(_MultiTokenNewlineQwenTokenizer())


def test_unknown_family_fails_during_resolution():
    with pytest.raises(ValueError, match="Unknown Continuous Token model_family"):
        create_continuous_token_builder(_DummyTokenizer(), model_family="missing_custom_family")


@pytest.mark.parametrize("model_family", ["", "   ", None])
def test_empty_family_fails_during_resolution(model_family):
    with pytest.raises(ValueError, match="model_family must be a non-empty string"):
        resolve_continuous_token_model_family(model_family)


# =============================================================================
# Multimodal (VL) continuous token builders, base-class MM hooks, and VL wiring
# (merged from the former tests/utils/test_continuous_token_mm_on_cpu.py)
# =============================================================================


class TestMergeResultTokenFields:
    """Verify MergeResult stays token-only and works with VL builders."""

    def test_default_values_text_only(self):
        result = MergeResult(token_ids=[1, 2, 3], appended_token_count=2)
        assert result.token_ids == [1, 2, 3]
        assert result.appended_token_count == 2
        assert result.kind == "non_assistant"

    def test_backward_compat_construction(self):
        result = MergeResult(
            token_ids=[10, 20, 30],
            appended_token_count=1,
            kind="assistant",
            inserted_token_ids=[99],
            removed_prefix_token_count=0,
        )
        assert result.token_ids == [10, 20, 30]
        assert result.kind == "assistant"
        assert result.inserted_token_ids == [99]

    def test_frozen_immutability(self):
        """MergeResult should remain frozen (no assignment after construction)."""
        result = MergeResult(token_ids=[1], appended_token_count=0)
        with pytest.raises(AttributeError):
            result.token_ids = [2]  # type: ignore[misc]


class TestBaseClassMMHooks:
    """Verify base class MM hooks behave correctly (NotImplementedError / False)."""

    def setup_method(self):
        """Create a minimal mock tokenizer for base class instantiation."""

        class MockTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2, 3]

        self.builder = ContinuousTokenBuilder(MockTokenizer())

    def test_supports_multimodal_default_false(self):
        """Base class should return False for supports_multimodal."""
        assert ContinuousTokenBuilder.supports_multimodal() is False
        assert self.builder.supports_multimodal() is False

    def test_render_tokens_with_mm_raises(self):
        """Base class render_tokens_with_mm should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not implement render_tokens_with_mm"):
            self.builder.render_tokens_with_mm(
                messages=[{"role": "user", "content": "hi"}],
                images=["fake_image.png"],
            )

    def test_supports_multimodal_classmethod(self):
        """supports_multimodal should be callable as classmethod without instance."""
        assert ContinuousTokenBuilder.supports_multimodal() is False

    def test_subclass_can_override_supports_multimodal(self):
        """A VL subclass that overrides supports_multimodal should return True."""

        class FakeVLBuilder(ContinuousTokenBuilder):
            @classmethod
            def supports_multimodal(cls) -> bool:
                return True

        assert FakeVLBuilder.supports_multimodal() is True


class TestMultimodalMergeResultWithExistingSubclasses:
    """Ensure existing text subclass _merge_token_ids still produce valid MergeResult."""

    def test_qwen_merge_still_works(self):
        """QwenContinuousTokenBuilder merge should produce token-only MergeResult."""
        from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

        class MockQwenTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                if token == "<|im_end|>":
                    return 151645
                return 0

        builder = QwenContinuousTokenBuilder(MockQwenTokenizer())
        # Simulate: prefix ends with <|im_end|>, appended is [10, 20]
        result = builder._merge_non_assistant_token_ids([100, 200, 151645], [10, 20])
        assert result.token_ids == [100, 200, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]
        assert result.appended_token_count == 2


# =============================================================================
# Tests for VL subclasses
# =============================================================================


class TestQwenVLContinuousTokenBuilder:
    """Test QwenVL vision token handling."""

    def setup_method(self):
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        class MockQwenVLTokenizer:
            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockImageProcessor:
            merge_size = 2

        class MockProcessor:
            image_processor = MockImageProcessor()

        self.tokenizer = MockQwenVLTokenizer()
        self.processor = MockProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_supports_multimodal(self):
        assert self.builder.supports_multimodal() is True

    def test_merge_inherits_qwen_newline_patch(self):
        """VL builder should still insert newline after im_end (from QwenBuilder)."""
        result = self.builder._merge_non_assistant_token_ids([100, 151645], [10, 20])
        assert result.token_ids == [100, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]


# =============================================================================
# Tests for wiring factory with VL families
# =============================================================================


class TestWiringVLFactory:
    """Test that create_continuous_token_builder handles VL families correctly."""

    def test_vl_family_requires_processor(self):
        """VL families should raise if processor not provided."""
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        with pytest.raises(ValueError, match="requires a processor"):
            create_continuous_token_builder(
                MockTokenizer(),
                model_family="qwen25vl",
            )

    def test_vl_family_succeeds_with_processor(self):
        """VL families should instantiate correctly with processor provided."""
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockImageProcessor:
            merge_size = 2

        class MockProcessor:
            image_processor = MockImageProcessor()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            model_family="qwen25vl",
            processor=MockProcessor(),
        )
        assert isinstance(builder, QwenVLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True

    def test_vl_family_inferred_from_model_type_with_processor(self):
        """A registered VL model_type resolves to its processor-backed builder."""
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                return [198] if text == "\n" else [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                return {"<|im_end|>": 151645}.get(token, 0)

        class MockProcessor:
            image_processor = type("IP", (), {"merge_size": 2})()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            hf_model_type="qwen2_5_vl",
            processor=MockProcessor(),
        )
        assert isinstance(builder, QwenVLContinuousTokenBuilder)

    def test_unknown_model_with_processor_falls_back_to_default_vl(self, caplog):
        """Unrecognized model + multimodal processor -> default VL builder, with a warning."""
        from verl.utils.tokenizer.continuous_token import VLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "acme/foobar-7b-instruct"

        class MockProcessor:
            image_processor = type("IP", (), {"merge_size": 2})()

        with caplog.at_level(logging.WARNING, logger="verl.utils.tokenizer.continuous_token_wiring"):
            builder = create_continuous_token_builder(
                MockTokenizer(),
                hf_model_type="acme_model",
                processor=MockProcessor(),
            )
        assert isinstance(builder, VLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True
        assert "acme_model" in caplog.text
        assert "vldefault" in caplog.text

    def test_gemma4_unified_with_processor_upgrades_to_vl(self):
        """Gemma4 (unified checkpoint, no vl marker) + processor -> Gemma4 VL builder."""
        from verl.utils.tokenizer.continuous_token import Gemma4VLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "google/gemma-4-27b-it"

            def convert_tokens_to_ids(self, token):
                return {"<|tool_response>": 12345}.get(token, 0)

        class MockProcessor:
            image_processor = type("IP", (), {"merge_size": 2})()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            hf_model_type="gemma4",
            processor=MockProcessor(),
        )
        assert isinstance(builder, Gemma4VLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True

    def test_qwen35_unified_with_processor_upgrades_to_vl(self):
        """Qwen3.5 (unified checkpoint, no vl marker) + processor -> Qwen VL builder."""
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "Qwen/Qwen3.5-35B-A3B"

            def encode(self, text, add_special_tokens=False):
                return [198] if text == "\n" else [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockProcessor:
            image_processor = type("IP", (), {"merge_size": 2})()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            hf_model_type="qwen3_5_moe",
            processor=MockProcessor(),
        )
        assert isinstance(builder, QwenVLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True

    def test_text_specific_family_with_processor_raises(self):
        """A recognized text-only family paired with a multimodal processor is a misconfiguration."""
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            name_or_path = "Qwen/Qwen3-8B"

        class MockProcessor:
            image_processor = type("IP", (), {"merge_size": 2})()

        with pytest.raises(ValueError, match="multimodal processor was provided"):
            create_continuous_token_builder(
                MockTokenizer(),
                hf_model_type="qwen3",
                processor=MockProcessor(),
            )


# =============================================================================
# Integration tests: VL builder build_initial_tokens + merge_non_assistant_tokens end-to-end
# =============================================================================


class _MockQwenVLProcessor:
    """Faithful-ish mock of a Qwen2.5-VL processor's two-step render.

    Mirrors how the real processor works so incremental renders stay prefix-stable:

    1. ``apply_chat_template`` renders the message list to text, emitting an
       ``<|image_pad|>`` placeholder *in place* wherever an image content block
       appears (never at some fixed offset).
    2. ``__call__`` tokenizes that text (each char -> its ``ord``) and expands each
       ``<|image_pad|>`` placeholder in place into a vision span
       (``<|vision_start|>`` + 4 ``<|image_pad|>`` pads + ``<|vision_end|>``),
       simulating merge_size=2 on a 1x4x4 grid -> 4 pad tokens per image.

    Because a newly appended turn (and its placeholder) lands at the end of the
    text and is expanded in place, ``render(prefix)`` is always a token prefix of
    ``render(prefix + new_turn)``.
    """

    _IMAGE_PLACEHOLDER = "<|image_pad|>"

    class _ImageProcessor:
        merge_size = 2

    image_processor = _ImageProcessor()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, return_dict=False, **kwargs
    ):
        parts: list[str] = []
        for message in messages:
            parts.append(f"<{message.get('role')}>")
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        parts.append(self._IMAGE_PLACEHOLDER)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
            else:
                parts.append(str(content))
            parts.append("\n")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
        rendered = text[0] if isinstance(text, list | tuple) else (text or "")
        segments = rendered.split(self._IMAGE_PLACEHOLDER)
        num_images = len(segments) - 1

        token_ids: list[int] = []
        for index, segment in enumerate(segments):
            token_ids.extend(ord(char) for char in segment)
            if index < num_images:
                # Expand this image's placeholder in place: vision_start + 4 pads + vision_end
                token_ids.extend([151652, 151655, 151655, 151655, 151655, 151653])

        result = {"input_ids": [token_ids]}
        if num_images > 0:
            # pixel_values dim0 = raw patches (t*h*w = 1*4*4 = 16 per image)
            import numpy as np

            result["pixel_values"] = np.zeros((num_images * 16, 3, 14, 14), dtype=np.float32)
            # image_grid_thw: each image is (1, 4, 4)
            result["image_grid_thw"] = np.array([[1, 4, 4]] * num_images, dtype=np.int64)

        return result


class _MockQwenVLTokenizer:
    """Mock tokenizer for VL integration tests."""

    name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    def encode(self, text, add_special_tokens=False):
        if text == "\n":
            return [198]
        return [1000, 1001, 1002]

    def convert_tokens_to_ids(self, token):
        mapping = {
            "<|im_end|>": 151645,
            "<|im_start|>": 151644,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
            "<|image_pad|>": 151655,
            "<|observation|>": 151333,
            "<|user|>": 151336,
            "<|begin_of_image|>": 151700,
            "<|end_of_image|>": 151701,
            "<|media_start|>": 151800,
            "<|media_end|>": 151801,
        }
        return mapping.get(token, 0)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        """Simple mock chat template."""
        tokens = [151644]  # im_start
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        tokens.extend([151652, 151655, 151653])  # vision placeholder
                    elif isinstance(block, dict) and block.get("type") == "text":
                        tokens.extend([1000, 1001])
            else:
                tokens.extend([1000, 1001, 1002])
            tokens.append(151645)  # im_end
        if add_generation_prompt:
            tokens.append(151644)  # im_start for assistant
        if not tokenize:
            return "mock_text_render"
        return tokens


class TestQwenVLBuildInitialTokens:
    """Integration test for QwenVL build_initial_tokens with images."""

    def setup_method(self):
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_build_initial_no_images(self):
        """Without images, should use text-only path."""
        messages = [{"role": "user", "content": "Hello"}]
        token_ids = self.builder.build_initial_tokens(messages)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)

    def test_build_initial_with_images(self):
        """With images, should use processor-expanded token IDs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "fake_image.png"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        token_ids = self.builder.build_initial_tokens(messages, images=["fake_image.png"])
        assert isinstance(token_ids, list)
        assert token_ids.count(151655) == 4


class TestQwenVLMergeNonAssistantTokens:
    """Integration test for QwenVL merge_non_assistant_tokens with images in appended messages."""

    def setup_method(self):
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_merge_no_new_images(self):
        """Without new images in appended messages, should use text-only merge."""
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        result = self.builder.merge_non_assistant_tokens(previous, updated, runtime_ids)
        assert isinstance(result, MergeResult)
        assert result.kind == "non_assistant"

    def test_merge_with_new_images(self):
        """With new images in appended messages, should merge processor-expanded token IDs."""
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "new_image.png"},
                    {"type": "text", "text": "Look at this"},
                ],
            },
        ]
        # Simulate runtime token state
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        result = self.builder.merge_non_assistant_tokens(previous, updated, runtime_ids)
        assert isinstance(result, MergeResult)
        assert result.kind == "non_assistant"
        assert 151655 in result.token_ids

    def test_merge_with_new_images_rejects_non_prefix_processor_output(self):
        """Incremental rendering should fail fast if the processor output is not append-only."""

        class BadPrefixProcessor(_MockQwenVLProcessor):
            def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
                result = super().__call__(text=text, images=images, return_tensors=return_tensors, **kwargs)
                # Corrupt only the image-bearing (full) render so it diverges from the
                # image-free prefix render, breaking the append-only prefix invariant.
                if images:
                    result["input_ids"][0][0] = 9999
                return result

        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        builder = QwenVLContinuousTokenBuilder(self.tokenizer, BadPrefixProcessor())
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "new_image.png"},
                    {"type": "text", "text": "Look at this"},
                ],
            },
        ]
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        with pytest.raises(ValueError, match="suffix diff failed"):
            builder.merge_non_assistant_tokens(previous, updated, runtime_ids)


@pytest.mark.parametrize(
    "builder_name",
    [
        "GLM46VContinuousTokenBuilder",
        "KimiVLContinuousTokenBuilder",
    ],
)
def test_other_vl_builders_reject_non_prefix_processor_output(builder_name):
    """All VL builders should validate the append-only prefix invariant during merge."""

    class BadPrefixProcessor(_MockQwenVLProcessor):
        def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
            result = super().__call__(text=text, images=images, return_tensors=return_tensors, **kwargs)
            # Corrupt only the image-bearing (full) render so it diverges from the
            # image-free prefix render, breaking the append-only prefix invariant.
            if images:
                result["input_ids"][0][0] = 9999
            return result

    import verl.utils.tokenizer.continuous_token as continuous_token

    builder_cls = getattr(continuous_token, builder_name)
    builder = builder_cls(_MockQwenVLTokenizer(), BadPrefixProcessor())
    previous = [{"role": "user", "content": "Hi"}]
    updated = [
        {"role": "user", "content": "Hi"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "new_image.png"},
                {"type": "text", "text": "Look at this"},
            ],
        },
    ]
    runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
    with pytest.raises(ValueError, match="suffix diff failed"):
        builder.merge_non_assistant_tokens(previous, updated, runtime_ids)


# =============================================================================
# Tests: chat_template_kwargs / mm_processor_kwargs are wired to the VL builder
# at construction time AND actually take effect at render time.
# =============================================================================


class _ConfigurablePadProcessor(_MockQwenVLProcessor):
    """VL processor whose per-image pad count is driven by the ``pads_per_image``
    mm kwarg, mirroring how real ``max_pixels``/``min_pixels`` change how many
    vision tokens an image expands into. Also records the kwargs each call
    receives so tests can assert they were forwarded verbatim.
    """

    def __init__(self):
        self.call_kwargs: list[dict] = []

    def __call__(self, *, text=None, images=None, return_tensors=None, pads_per_image=4, **kwargs):
        self.call_kwargs.append({"pads_per_image": pads_per_image, **kwargs})
        rendered = text[0] if isinstance(text, list | tuple) else (text or "")
        segments = rendered.split(self._IMAGE_PLACEHOLDER)
        num_images = len(segments) - 1

        token_ids: list[int] = []
        for index, segment in enumerate(segments):
            token_ids.extend(ord(char) for char in segment)
            if index < num_images:
                token_ids.append(151652)
                token_ids.extend([151655] * pads_per_image)
                token_ids.append(151653)

        result = {"input_ids": [token_ids]}
        if num_images > 0:
            import numpy as np

            result["pixel_values"] = np.zeros((num_images * 16, 3, 14, 14), dtype=np.float32)
            result["image_grid_thw"] = np.array([[1, 4, 4]] * num_images, dtype=np.int64)
        return result


class _RecordingTemplateProcessor(_MockQwenVLProcessor):
    """VL processor that records the kwargs its ``apply_chat_template`` receives."""

    def __init__(self):
        self.template_kwargs: list[dict] = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, return_dict=False, **kwargs
    ):
        self.template_kwargs.append(dict(kwargs))
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )


def test_vl_builder_creation_forwards_chat_template_and_mm_processor_kwargs():
    """create_continuous_token_builder must store both kwarg dicts on a VL builder."""
    builder = create_continuous_token_builder(
        _MockQwenVLTokenizer(),
        model_family="qwen25vl",
        processor=_MockQwenVLProcessor(),
        chat_template_kwargs={"enable_thinking": False},
        mm_processor_kwargs={"max_pixels": 12345, "min_pixels": 3136},
    )

    assert isinstance(builder, QwenVLContinuousTokenBuilder)
    assert builder.chat_template_kwargs == {"enable_thinking": False}
    assert builder.mm_processor_kwargs == {"max_pixels": 12345, "min_pixels": 3136}


def test_text_builder_creation_ignores_mm_processor_kwargs():
    """mm_processor_kwargs is multimodal-only: a text builder must not carry it."""
    builder = create_continuous_token_builder(
        _TemplateTokenizer(),
        model_family="default",
        mm_processor_kwargs={"max_pixels": 12345},
    )

    assert isinstance(builder, ContinuousTokenBuilder)
    assert not hasattr(builder, "mm_processor_kwargs")


def test_vl_builder_forwards_mm_processor_kwargs_to_processor_call_at_render():
    """mm_processor_kwargs must be forwarded verbatim into the processor call."""
    processor = _ConfigurablePadProcessor()
    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        processor,
        mm_processor_kwargs={"pads_per_image": 3, "max_pixels": 999},
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "x.png"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]

    builder.build_initial_tokens(messages, images=["x.png"])

    assert processor.call_kwargs
    assert processor.call_kwargs[-1]["pads_per_image"] == 3
    assert processor.call_kwargs[-1]["max_pixels"] == 999


def test_vl_builder_mm_processor_kwargs_actually_change_rendered_token_count():
    """Different mm_processor_kwargs must produce a different number of vision pad
    tokens, proving the kwargs genuinely take effect (not merely stored)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "x.png"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]

    small = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(), _ConfigurablePadProcessor(), mm_processor_kwargs={"pads_per_image": 2}
    )
    large = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(), _ConfigurablePadProcessor(), mm_processor_kwargs={"pads_per_image": 6}
    )

    small_ids = small.build_initial_tokens(messages, images=["x.png"])
    large_ids = large.build_initial_tokens(messages, images=["x.png"])

    assert small_ids.count(151655) == 2
    assert large_ids.count(151655) == 6


def test_vl_builder_forwards_chat_template_kwargs_to_processor_template():
    """chat_template_kwargs must reach the processor's apply_chat_template (VL path),
    not just the tokenizer path exercised by the text-only builder test."""
    processor = _RecordingTemplateProcessor()
    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        processor,
        chat_template_kwargs={"enable_thinking": False},
    )

    builder.build_initial_tokens([{"role": "user", "content": "question"}])

    assert processor.template_kwargs
    assert processor.template_kwargs[-1].get("enable_thinking") is False


def test_vl_builder_folds_processor_sampling_rate_into_mm_processor_kwargs():
    """A processor exposing feature_extractor.sampling_rate should have that value
    folded into mm_processor_kwargs so audio renders stay aligned."""

    class _AudioProcessor(_MockQwenVLProcessor):
        feature_extractor = type("FE", (), {"sampling_rate": 16000})()

    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        _AudioProcessor(),
        mm_processor_kwargs={"max_pixels": 111},
    )

    assert builder.mm_processor_kwargs["sampling_rate"] == 16000
    assert builder.mm_processor_kwargs["max_pixels"] == 111


def test_vl_builder_preserves_explicit_sampling_rate_over_processor_default():
    """An explicit sampling_rate in mm_processor_kwargs must not be overwritten by
    the processor's feature_extractor default."""

    class _AudioProcessor(_MockQwenVLProcessor):
        feature_extractor = type("FE", (), {"sampling_rate": 16000})()

    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        _AudioProcessor(),
        mm_processor_kwargs={"sampling_rate": 24000},
    )

    assert builder.mm_processor_kwargs["sampling_rate"] == 24000
