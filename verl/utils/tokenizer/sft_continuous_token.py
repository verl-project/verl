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
"""SFT-only reconstruction of gold assistant messages for Continuous Token.

Continuous Token runtime builders operate on token IDs produced by rollout.
Multi-turn SFT instead starts from structured gold assistant messages.  This
module is the package-local adapter between those two representations; it does
not add an assistant-message API to the shared runtime builders.  It
intentionally reuses package-private protocol helpers that are shared by SFT
and rollout rather than defining a second copy of those rules.
"""

from __future__ import annotations

from typing import Any

from .chat_template import apply_chat_template
from .continuous_token import (
    _DUMMY_TOOL_NAME,
    _SYNTHETIC_SYSTEM_MESSAGE,
    _SYNTHETIC_USER_MESSAGE,
    ContinuousTokenBuilder,
    DeepSeekContinuousTokenBuilder,
    DeepSeekVL2ContinuousTokenBuilder,
    Gemma4ContinuousTokenBuilder,
    GLMContinuousTokenBuilder,
    GptOssContinuousTokenBuilder,
    KimiVLContinuousTokenBuilder,
    MiniMaxContinuousTokenBuilder,
    MiniMaxText01ContinuousTokenBuilder,
    MiniMaxVLContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    VLContinuousTokenMixin,
    _copy_messages_for_template,
    _prepare_minimax_legacy_assistant_message,
    _stringify_tool_content,
    require_token_id,
)
from .deepseek import (
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    DeepSeekV4ContinuousTokenBuilder,
    encode_messages,
)
from .tokenizer import normalize_token_ids


class _AssistantReconstructor:
    def __init__(self, builder: ContinuousTokenBuilder):
        self.builder = builder

    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        del previous_messages
        self._require_assistant(message)

        rendered_message = self._prepare_message(message)
        synthetic_prompt = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE]
        prompt_text = self._render_text(synthetic_prompt, add_generation_prompt=True, tools=tools)
        completed_text = self._render_text(
            [*synthetic_prompt, rendered_message],
            add_generation_prompt=False,
            tools=tools,
        )
        if not completed_text.startswith(prompt_text):
            raise ValueError(
                "Continuous Token assistant encoding requires the generation prompt to be a text prefix "
                "of the completed assistant turn"
            )

        assistant_text = completed_text[len(prompt_text) :]
        assistant_token_ids = normalize_token_ids(
            self.builder.tokenizer.encode(assistant_text, add_special_tokens=False)
        )
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)

    @staticmethod
    def _require_assistant(message: dict[str, Any]) -> None:
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )

    def _prepare_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return message

    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        template_owner = (
            self.builder.processor if isinstance(self.builder, VLContinuousTokenMixin) else self.builder.tokenizer
        )
        template_kwargs = dict(self.builder.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools
        rendered = apply_chat_template(
            template_owner,
            _copy_messages_for_template(messages) if isinstance(self.builder, VLContinuousTokenMixin) else messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        if not isinstance(rendered, str):
            owner = "processor chat template" if isinstance(self.builder, VLContinuousTokenMixin) else "chat template"
            raise TypeError(f"Expected {owner} to render str, got {type(rendered).__name__}")
        return rendered

    def _render_turn_deltas(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, str]:
        synthetic_prompt = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE]
        context_text = self._render_text(synthetic_prompt, add_generation_prompt=False, tools=tools)
        prompt_text = self._render_text(synthetic_prompt, add_generation_prompt=True, tools=tools)
        completed_text = self._render_text(
            [*synthetic_prompt, message],
            add_generation_prompt=False,
            tools=tools,
        )
        if not prompt_text.startswith(context_text) or not completed_text.startswith(context_text):
            raise ValueError(
                "Continuous Token assistant encoding requires both generation and completed renders "
                "to preserve the fixed synthetic context"
            )
        return prompt_text[len(context_text) :], completed_text[len(context_text) :]

    def _normalize_ids(self, assistant_token_ids: list[int], message: dict[str, Any]) -> list[int]:
        terminator_ids = self._terminator_ids(message)
        if not terminator_ids:
            return list(assistant_token_ids)
        for index, token_id in enumerate(assistant_token_ids):
            if token_id in terminator_ids:
                return list(assistant_token_ids[: index + 1])
        raise ValueError(
            "Continuous Token assistant token-id suffix does not contain an accepted terminator "
            f"{sorted(terminator_ids)}; tail={assistant_token_ids[-16:]}"
        )

    def _terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        eos_token_id = getattr(self.builder.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            return {eos_token_id}
        if isinstance(eos_token_id, list | tuple | set):
            return {int(token_id) for token_id in eos_token_id if token_id is not None}
        if eos_token_id is None:
            return set()
        raise TypeError(f"Unsupported eos_token_id type: {type(eos_token_id)!r}")


class _GptOssReconstructor(_AssistantReconstructor):
    def _prepare_message(self, message: dict[str, Any]) -> dict[str, Any]:
        rendered_message = {key: value for key, value in message.items() if value is not None}
        if message.get("content") is None:
            rendered_message["content"] = ""
        return rendered_message

    def _terminator_ids(self, message: dict[str, Any]) -> set[int]:
        if message.get("tool_calls"):
            return {require_token_id(self.builder.tokenizer, "<|call|>")}
        return super()._terminator_ids(message)


class _QwenReconstructor(_AssistantReconstructor):
    def _prepare_message(self, message: dict[str, Any]) -> dict[str, Any]:
        enable_thinking = self.builder.chat_template_kwargs.get("enable_thinking")
        if isinstance(message.get("reasoning_content"), str):
            if enable_thinking is not False:
                return message
            rendered_message = dict(message)
            rendered_message["reasoning_content"] = ""
            return rendered_message
        content = message.get("content")
        content_is_text_blocks = isinstance(content, list) and all(
            isinstance(block, dict) and block.get("type") == "text" for block in content
        )
        if content_is_text_blocks:
            content_text = "".join(str(block.get("text", "")) for block in content)
        elif isinstance(content, str):
            content_text = content
        else:
            return message
        if not content_text.startswith("<think>") or "</think>" not in content_text:
            return message

        reasoning_content, answer_content = content_text[len("<think>") :].split("</think>", 1)
        rendered_message = dict(message)
        rendered_message["reasoning_content"] = reasoning_content if enable_thinking is not False else ""
        rendered_message["content"] = (
            [{"type": "text", "text": answer_content}] if content_is_text_blocks else answer_content
        )
        return rendered_message


class _MiniMaxText01Reconstructor(_AssistantReconstructor):
    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        del tools
        return super().reconstruct(
            _prepare_minimax_legacy_assistant_message(message),
            tools=None,
            previous_messages=previous_messages,
        )

    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        return self.builder._render_text(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        return {self.builder._eos_id}


class _MiniMaxReconstructor(_AssistantReconstructor):
    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        del previous_messages
        self._require_assistant(message)
        rendered_message = self._prepare_message(message)
        prompt_delta, completed_delta = self._render_turn_deltas(rendered_message, tools=tools)
        assistant_header = "]~b]ai\n"
        think_open = "<think>\n"
        if prompt_delta != assistant_header + think_open or not completed_delta.startswith(assistant_header):
            raise ValueError("Continuous Token MiniMax assistant scaffold does not match the supported protocol")

        completed_body = completed_delta[len(assistant_header) :]
        if completed_body.startswith(think_open):
            assistant_text = completed_body[len(think_open) :]
        else:
            assistant_text = "</think>\n\n" + completed_body
        assistant_token_ids = normalize_token_ids(
            self.builder.tokenizer.encode(assistant_text, add_special_tokens=False)
        )
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)

    def _prepare_message(self, message: dict[str, Any]) -> dict[str, Any]:
        if isinstance(message.get("reasoning_content"), str):
            return message
        content = message.get("content")
        content_is_text_blocks = isinstance(content, list) and all(
            isinstance(block, dict) and block.get("type") == "text" for block in content
        )
        if content_is_text_blocks:
            content_text = "".join(str(block.get("text", "")) for block in content)
        elif isinstance(content, str):
            content_text = content
        else:
            return message
        if not content_text.startswith("<think>") or "</think>" not in content_text:
            return message

        reasoning_content, answer_content = content_text[len("<think>") :].split("</think>", 1)
        rendered_message = dict(message)
        rendered_message["reasoning_content"] = reasoning_content
        rendered_message["content"] = (
            [{"type": "text", "text": answer_content}] if content_is_text_blocks else answer_content
        )
        return rendered_message

    def _terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        return {self.builder._eos_id}


class _GLMReconstructor(_AssistantReconstructor):
    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        del previous_messages
        self._require_assistant(message)
        rendered_message = self._prepare_message(message)
        prompt_delta, completed_delta = self._render_turn_deltas(rendered_message, tools=tools)
        assistant_header = "<|assistant|>"
        if not prompt_delta.startswith(assistant_header) or not completed_delta.startswith(assistant_header):
            raise ValueError("Continuous Token GLM assistant scaffold does not match the supported protocol")

        if completed_delta.startswith(prompt_delta):
            assistant_text = completed_delta[len(prompt_delta) :]
        elif prompt_delta == assistant_header + "<think>":
            assistant_text = completed_delta[len(assistant_header) :]
        else:
            raise ValueError("Continuous Token GLM completed assistant turn does not extend its generation scaffold")

        assistant_token_ids = normalize_token_ids(
            self.builder.tokenizer.encode(assistant_text, add_special_tokens=False)
        )
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)

    def _prepare_message(self, message: dict[str, Any]) -> dict[str, Any]:
        enable_thinking = self.builder.chat_template_kwargs.get("enable_thinking")
        if isinstance(message.get("reasoning_content"), str):
            if enable_thinking is not False:
                return message
            rendered_message = dict(message)
            rendered_message["reasoning_content"] = ""
            return rendered_message

        content = message.get("content")
        content_is_text_blocks = isinstance(content, list) and all(
            isinstance(block, dict) and block.get("type") == "text" for block in content
        )
        if content_is_text_blocks:
            content_text = "".join(str(block.get("text", "")) for block in content)
        elif isinstance(content, str):
            content_text = content
        else:
            if enable_thinking is not False:
                return message
            rendered_message = dict(message)
            rendered_message["reasoning_content"] = ""
            return rendered_message
        if not content_text.startswith("<think>") or "</think>" not in content_text:
            if enable_thinking is not False:
                return message
            rendered_message = dict(message)
            rendered_message["reasoning_content"] = ""
            return rendered_message

        reasoning_content, answer_content = content_text[len("<think>") :].split("</think>", 1)
        rendered_message = dict(message)
        rendered_message["reasoning_content"] = reasoning_content if enable_thinking is not False else ""
        answer_content = answer_content.lstrip("\n")
        rendered_message["content"] = (
            [{"type": "text", "text": answer_content}] if content_is_text_blocks else answer_content
        )
        return rendered_message

    def _normalize_ids(self, assistant_token_ids: list[int], message: dict[str, Any]) -> list[int]:
        normalized_ids = list(assistant_token_ids)
        observation_id = self.builder._observation_id
        if message.get("tool_calls") and (not normalized_ids or normalized_ids[-1] != observation_id):
            normalized_ids.append(observation_id)
        return normalized_ids


class _Gemma4Reconstructor(_AssistantReconstructor):
    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        del previous_messages
        self._require_assistant(message)

        rendered_message = dict(message)
        reasoning = next(
            (
                value
                for key in ("thinking", "reasoning_content", "reasoning")
                if isinstance((value := message.get(key)), str)
            ),
            "",
        )
        content = _stringify_tool_content(message.get("content", ""))
        thought_open = "<|channel>thought\n"
        thought_close = "<channel|>"
        if content.startswith(thought_open) and thought_close in content:
            embedded_reasoning, content = content[len(thought_open) :].split(thought_close, 1)
            if not reasoning:
                reasoning = embedded_reasoning
            rendered_message["content"] = content

        prompt_delta, completed_delta = self._render_turn_deltas(rendered_message, tools=tools)
        assistant_header = "<|turn>model\n"
        if not prompt_delta.startswith(assistant_header) or not completed_delta.startswith(assistant_header):
            raise ValueError("Continuous Token Gemma4 assistant scaffold does not match the supported protocol")
        completed_body = completed_delta[len(assistant_header) :]
        prompt_scaffold = prompt_delta[len(assistant_header) :]
        empty_thought_scaffold = thought_open + thought_close

        if prompt_scaffold == empty_thought_scaffold:
            assistant_text = completed_body
        elif not prompt_scaffold:
            if completed_body.startswith(thought_open):
                assistant_text = completed_body
            elif reasoning or self.builder.chat_template_kwargs.get("enable_thinking", False):
                assistant_text = thought_open + reasoning + thought_close + completed_body
            else:
                assistant_text = completed_body
        else:
            raise ValueError("Continuous Token Gemma4 generation prompt has an unsupported scaffold")

        assistant_token_ids = normalize_token_ids(
            self.builder.tokenizer.encode(assistant_text, add_special_tokens=False)
        )
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)

    def _normalize_ids(self, assistant_token_ids: list[int], message: dict[str, Any]) -> list[int]:
        token = "<tool_call|>" if message.get("tool_calls") else "<turn|>"
        terminator_id = require_token_id(self.builder.tokenizer, token)
        for index, token_id in enumerate(assistant_token_ids):
            if token_id == terminator_id:
                return list(assistant_token_ids[: index + 1])
        raise ValueError(
            "Continuous Token Gemma4 assistant token-id suffix does not contain the expected terminator "
            f"{terminator_id}; tail={assistant_token_ids[-16:]}"
        )


class _DeepSeekReconstructor(_AssistantReconstructor):
    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        template_kwargs = dict(self.builder.chat_template_kwargs)
        if "enable_thinking" in template_kwargs and "thinking" not in template_kwargs:
            template_kwargs["thinking"] = template_kwargs["enable_thinking"]
        rendered = apply_chat_template(
            self.builder.tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **template_kwargs,
        )
        if not isinstance(rendered, str):
            raise TypeError(f"Expected chat template to render str, got {type(rendered).__name__}")
        return rendered

    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        self._require_assistant(message)
        if previous_messages and previous_messages[-1].get("role") == "tool":
            return self._reconstruct_after_tool(message, tools=tools)

        prompt_delta, completed_delta = self._render_turn_deltas(message, tools=tools)
        if completed_delta.startswith(prompt_delta):
            assistant_text = completed_delta[len(prompt_delta) :]
        else:
            think_open = "<think>"
            think_close = "</think>"
            completed_prefix = prompt_delta[: -len(think_open)] + think_close
            if not prompt_delta.endswith(think_open) or not completed_delta.startswith(completed_prefix):
                raise ValueError(
                    "Continuous Token DeepSeek completed assistant turn does not extend a supported generation scaffold"
                )

            reasoning = next(
                (value for key in ("reasoning_content", "reasoning") if isinstance((value := message.get(key)), str)),
                "",
            )
            content = _stringify_tool_content(message.get("content", ""))
            if not reasoning and content.startswith(think_open) and think_close in content:
                reasoning = content[len(think_open) :].split(think_close, 1)[0]
            assistant_header = completed_prefix[: -len(think_close)]
            assistant_text = reasoning + completed_delta[len(assistant_header) :]

        assistant_token_ids = normalize_token_ids(
            self.builder.tokenizer.encode(assistant_text, add_special_tokens=False)
        )
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)

    def _reconstruct_after_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        synthetic_tool_message = {
            "role": "tool",
            "content": "continuous token synthetic tool response",
            "tool_call_id": "continuous_token_call_0",
            "name": _DUMMY_TOOL_NAME,
        }
        synthetic_context = [
            _SYNTHETIC_SYSTEM_MESSAGE,
            _SYNTHETIC_USER_MESSAGE,
            self.builder._synthetic_assistant_for_tools([synthetic_tool_message]),
            synthetic_tool_message,
        ]
        prompt_text = self._render_text(synthetic_context, add_generation_prompt=True, tools=tools)
        completed_text = self._render_text(
            [*synthetic_context, message],
            add_generation_prompt=False,
            tools=tools,
        )
        if not completed_text.startswith(prompt_text):
            raise ValueError(
                "Continuous Token DeepSeek post-tool assistant encoding requires the generation prompt "
                "to prefix the completed turn"
            )
        assistant_text = completed_text[len(prompt_text) :]
        assistant_token_ids = normalize_token_ids(
            self.builder.tokenizer.encode(assistant_text, add_special_tokens=False)
        )
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)

    def _terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        return {self.builder._eos_id}


class _DeepSeekV4Reconstructor(_AssistantReconstructor):
    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None,
    ) -> str:
        return encode_messages(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            add_bos_token=True,
            enable_thinking=self.builder._enable_thinking,
            drop_thinking=self.builder._drop_thinking,
            reasoning_effort=self.builder._reasoning_effort,
        )

    def _prepare_message(self, message: dict[str, Any]) -> dict[str, Any]:
        if isinstance(message.get("reasoning_content"), str) or isinstance(message.get("reasoning"), str):
            return message

        content = message.get("content")
        if isinstance(content, list):
            if not all(isinstance(part, dict) and part.get("type") == "text" for part in content):
                return message
            content_text = "".join(str(part.get("text", "")) for part in content)
            content_is_text_blocks = True
        elif isinstance(content, str):
            content_text = content
            content_is_text_blocks = False
        else:
            return message
        if not content_text.startswith(THINK_START_TOKEN) or THINK_END_TOKEN not in content_text:
            return message

        reasoning, answer = content_text[len(THINK_START_TOKEN) :].split(THINK_END_TOKEN, 1)
        rendered_message = dict(message)
        rendered_message["reasoning_content"] = reasoning if self.builder._enable_thinking else ""
        rendered_message["content"] = [{"type": "text", "text": answer}] if content_is_text_blocks else answer
        return rendered_message

    def _terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        return {self.builder._eos_id}


class _MiniMaxVLReconstructor(_MiniMaxText01Reconstructor):
    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        del tools, previous_messages
        self._require_assistant(message)
        rendered_message = _prepare_minimax_legacy_assistant_message(message)
        synthetic_prompt = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE]
        context_ids = self.builder._render_tokens(synthetic_prompt, add_generation_prompt=False, tools=None)
        prompt_ids = self.builder._render_tokens(synthetic_prompt, add_generation_prompt=True, tools=None)
        completed_ids = self.builder._render_tokens(
            [*synthetic_prompt, rendered_message],
            add_generation_prompt=False,
            tools=None,
        )
        if prompt_ids[: len(context_ids)] != context_ids or completed_ids[: len(context_ids)] != context_ids:
            raise ValueError("Continuous Token MiniMax-VL assistant renders do not preserve the fixed context")
        if prompt_ids[len(context_ids) :] != self.builder._vl_scaffold_ids:
            raise ValueError("Continuous Token MiniMax-VL generation prompt has an unsupported scaffold")
        completed_delta = completed_ids[len(context_ids) :]
        if completed_delta[: len(self.builder._vl_scaffold_ids)] != self.builder._vl_scaffold_ids:
            raise ValueError("Continuous Token MiniMax-VL completed turn does not start with its assistant scaffold")
        assistant_token_ids = completed_delta[len(self.builder._vl_scaffold_ids) :]
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)


class _KimiVLReconstructor(_AssistantReconstructor):
    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        self.builder._reject_tools(tools)
        self.builder._reject_structured_messages([message])
        return super().reconstruct(message, tools=None, previous_messages=previous_messages)

    def _normalize_ids(self, assistant_token_ids: list[int], message: dict[str, Any]) -> list[int]:
        im_end_id = self.builder.tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_token_id = getattr(self.builder.tokenizer, "unk_token_id", None)
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id != unk_token_id:
            for index, token_id in enumerate(assistant_token_ids):
                if token_id == im_end_id:
                    return list(assistant_token_ids[: index + 1])
        return super()._normalize_ids(assistant_token_ids, message)


class _DeepSeekVL2Reconstructor(_DeepSeekReconstructor):
    def reconstruct(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
        previous_messages: list[dict[str, Any]] | None,
    ) -> list[int]:
        del previous_messages
        self._require_assistant(message)
        if tools:
            raise ValueError("DeepSeek-VL2 Continuous Token does not support tool schemas")
        if message.get("tool_calls"):
            raise ValueError("DeepSeek-VL2 Continuous Token does not support structured assistant tool calls")

        synthetic_prompt = [_SYNTHETIC_USER_MESSAGE]
        prompt_token_ids = self.builder._render_via_processor(
            synthetic_prompt,
            [],
        )
        conversation, images = self.builder._to_vl2_conversation(
            [*synthetic_prompt, message],
            [],
            add_generation_prompt=False,
        )
        completed = self.builder.processor.__call__(
            conversations=conversation,
            images=images,
            force_batchify=True,
            inference_mode=False,
        )
        completed_token_ids = normalize_token_ids(completed.input_ids[0].tolist())
        if completed_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
            raise ValueError(
                "Continuous Token assistant encoding requires the processor generation prompt to be a token-id "
                "prefix of the completed assistant turn"
            )
        assistant_token_ids = completed_token_ids[len(prompt_token_ids) :]
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_ids(assistant_token_ids, message)


_RECONSTRUCTORS: dict[type[ContinuousTokenBuilder], type[_AssistantReconstructor]] = {
    ContinuousTokenBuilder: _AssistantReconstructor,
    GptOssContinuousTokenBuilder: _GptOssReconstructor,
    QwenContinuousTokenBuilder: _QwenReconstructor,
    MiniMaxText01ContinuousTokenBuilder: _MiniMaxText01Reconstructor,
    MiniMaxContinuousTokenBuilder: _MiniMaxReconstructor,
    GLMContinuousTokenBuilder: _GLMReconstructor,
    Gemma4ContinuousTokenBuilder: _Gemma4Reconstructor,
    DeepSeekContinuousTokenBuilder: _DeepSeekReconstructor,
    DeepSeekV4ContinuousTokenBuilder: _DeepSeekV4Reconstructor,
    MiniMaxVLContinuousTokenBuilder: _MiniMaxVLReconstructor,
    KimiVLContinuousTokenBuilder: _KimiVLReconstructor,
    DeepSeekVL2ContinuousTokenBuilder: _DeepSeekVL2Reconstructor,
}


def _resolve_reconstructor(builder: ContinuousTokenBuilder) -> type[_AssistantReconstructor]:
    for cls in type(builder).__mro__:
        handler = _RECONSTRUCTORS.get(cls)
        if handler is not None:
            return handler
    return _AssistantReconstructor


def reconstruct_assistant_tokens(
    builder: ContinuousTokenBuilder,
    message: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    previous_messages: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Convert one prepared gold assistant message to rollout-shaped token IDs."""
    handler = _resolve_reconstructor(builder)(builder)
    return handler.reconstruct(message, tools=tools, previous_messages=previous_messages)
