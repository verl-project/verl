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
"""Continuous Token builder implementations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from .chat_template import apply_chat_template
from .tokenizer import build_multimodal_processor_inputs, normalize_token_ids

_SUPPORTED_APPEND_ROLES = frozenset({"tool", "user", "system"})
_SYNTHETIC_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": "continuous token synthetic system"}
_SYNTHETIC_USER_MESSAGE: dict[str, Any] = {"role": "user", "content": "continuous token synthetic user"}
_ASSISTANT_REASONING_CONTENT: str = "reasoning"
_DUMMY_TOOL_NAME = "continuous_token_tool"
MergeKind = Literal["assistant", "non_assistant"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeResult:
    """Merged runtime tokens plus the edits callers need to align metadata.

    ``token_ids`` is the updated runtime token stream. The other fields describe
    how the stream changed at the merge junction: ``inserted_token_ids`` are
    CT-created boundary tokens, ``appended_token_count`` counts newly appended
    assistant or non-assistant tokens excluding those inserted boundary tokens,
    and ``removed_prefix_token_count`` counts stale prefix tokens dropped before
    appending. Boundary tokens are not model-generated and therefore must not
    carry loss or model logprobs.
    """

    token_ids: list[int]
    appended_token_count: int
    kind: MergeKind = "non_assistant"
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class ContinuousTokenBuilder:
    """Build and update continuous-token runtime prompts for multi-turn rollouts.

    This class exposes two API layers:

    AgentLoop-facing runtime APIs:
        ``build_initial_tokens`` renders the first prompt, ``merge_non_assistant_tokens``
        merges append-only tool/user/system messages, ``merge_assistant_tokens``
        appends model-generated assistant tokens, and ``align_response_metadata``
        applies the recorded token edits to masks/logprobs.

    SFT-facing API:
        ``tokenize_assistant_message`` converts one prepared gold assistant
        message into the same continuation-token shape produced by rollout.

    Developer extension APIs:
        Model-specific builders should subclass this class and keep the runtime
        API contracts above stable. Chat template specific behavior belongs in hooks
        such as ``_tokenize_tool_group``, ``_tokenize_single_non_tool``,
        ``_should_fuse_generation_prompt_with_last_group``,
        ``_tokenize_generation_prompt_delta``, ``_assistant_terminator_ids``, and
        ``_merge_non_assistant_token_ids``. ``render_delta_token_id`` is the shared
        suffix-diff helper those hooks can reuse.
    """

    allowed_append_roles: frozenset[str] = _SUPPORTED_APPEND_ROLES

    def __init__(
        self,
        tokenizer: Any,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        allowed_append_roles: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        # Text-only base: no processor / mm_processor_kwargs. All multimodal state
        # (processor, mm_processor_kwargs, sampling-rate defaults) lives in the VL
        # layer so a text builder never carries multimodal parameters it cannot use.
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        if allowed_append_roles is not None:
            allowed_roles = frozenset(allowed_append_roles)
            unknown_roles = allowed_roles - _SUPPORTED_APPEND_ROLES
            if unknown_roles:
                raise ValueError(f"Unsupported Continuous Token append roles: {sorted(unknown_roles)}")
            self.allowed_append_roles = allowed_roles

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        # Text-only builders ignore multimodal inputs; VL builders override this.
        return self._render_tokens(messages, add_generation_prompt=True, tools=tools)

    def tokenize_non_assistant_incremental_messages(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages) :]
        if not appended_messages:
            return []
        incremental_ids: list[int] = []

        groups = self._iter_append_groups(appended_messages)
        fuse_generation_prompt = self._should_fuse_generation_prompt_with_last_group()

        for index, group in enumerate(groups):
            add_generation_prompt = fuse_generation_prompt and index == len(groups) - 1
            role = group[0].get("role")
            if role == "tool":
                incremental_ids.extend(
                    self._tokenize_tool_group(
                        group,
                        previous_messages=previous_messages,
                        tools=tools,
                        add_generation_prompt=add_generation_prompt,
                    )
                )
            elif role in {"user", "system"}:
                # System appends can represent retry/control messages; unsupported templates will fail in suffix diff.
                if len(group) != 1:
                    raise ValueError(
                        f"Continuous Token expects one {role!r} message per append group, got {len(group)}"
                    )
                incremental_ids.extend(
                    self._tokenize_single_non_tool(
                        group[0],
                        tools=tools,
                        add_generation_prompt=add_generation_prompt,
                    )
                )
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")

        if not fuse_generation_prompt:
            incremental_ids.extend(self._tokenize_generation_prompt_delta(updated_messages, tools=tools))
        return incremental_ids

    def merge_non_assistant_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        appended_ids = self.tokenize_non_assistant_incremental_messages(
            previous_messages, updated_messages, tools=tools
        )
        return self._merge_non_assistant_token_ids(runtime_token_ids, appended_ids)

    def merge_assistant_tokens(self, runtime_token_ids: list[int], assistant_token_ids: list[int]) -> MergeResult:
        """Merge model-generated assistant tokens into the runtime token stream."""
        merged_token_ids = list(runtime_token_ids) + list(assistant_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(assistant_token_ids),
            kind="assistant",
        )

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Encode one prepared SFT assistant message into rollout-shaped tokens.

        Rollout engines already provide generated assistant token IDs and should
        continue to call :meth:`merge_assistant_tokens` directly. SFT datasets,
        however, start from structured assistant messages. This helper renders
        the assistant message exactly once behind a fixed synthetic prompt and
        removes that prompt at the token boundary. No token from the real
        trajectory prefix is decoded or re-encoded.

        Chat templates may emit whitespace after the assistant stop token even
        though a rollout server stops at that token. The returned suffix is
        therefore normalized to the runtime stop shape before it is merged.
        """
        del previous_messages
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )

        rendered_message = self._prepare_assistant_message_for_render(message)
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

        # Encode only the model-produced continuation. Encoding prompt+response
        # together would let the tokenizer merge across their boundary and would
        # no longer match rollout, where the prompt tokens already exist.
        assistant_text = completed_text[len(prompt_text) :]
        assistant_token_ids = normalize_token_ids(self.tokenizer.encode(assistant_text, add_special_tokens=False))
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    def _prepare_assistant_message_for_render(self, message: dict[str, Any]) -> dict[str, Any]:
        """Normalize one prepared message before extracting its generated text."""
        return message

    def _render_assistant_turn_deltas(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str]:
        """Render generation/completion deltas behind one fixed context.

        Unlike :meth:`tokenize_assistant_message`, this helper does not require
        the generation prompt to prefix the completed turn. Model builders with
        template-specific generation scaffolds can compare the two deltas and
        reconstruct the rollout continuation explicitly.
        """
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

    def _normalize_assistant_token_ids(
        self,
        assistant_token_ids: list[int],
        message: dict[str, Any],
    ) -> list[int]:
        """Trim template-only tokens after the final generated stop token."""
        terminator_ids = self._assistant_terminator_ids(message)
        if not terminator_ids:
            return list(assistant_token_ids)
        # A rollout server stops as soon as it generates any configured stop
        # token. Mirror that first-stop behavior when reconstructing gold SFT
        # continuations, including the uncommon case where the serialized
        # content itself contains a special terminator token.
        for index, token_id in enumerate(assistant_token_ids):
            if token_id in terminator_ids:
                return list(assistant_token_ids[: index + 1])
        raise ValueError(
            "Continuous Token assistant token-id suffix does not contain an accepted terminator "
            f"{sorted(terminator_ids)}; tail={assistant_token_ids[-16:]}"
        )

    def _assistant_terminator_ids(self, message: dict[str, Any]) -> set[int]:
        """Return valid generation stop tokens for one assistant message.

        Most model families terminate every assistant continuation with EOS.
        Builders whose protocol has message-dependent stops (for example, a
        distinct tool-call terminator) override this hook.
        """
        del message
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            return {eos_token_id}
        elif isinstance(eos_token_id, list | tuple | set):
            return {int(token_id) for token_id in eos_token_id if token_id is not None}
        elif eos_token_id is None:
            return set()
        raise TypeError(f"Unsupported eos_token_id type: {type(eos_token_id)!r}")

    def _merge_non_assistant_token_ids(
        self, runtime_token_ids: list[int], appended_token_ids: list[int]
    ) -> MergeResult:
        """Merge runtime prefix tokens and appended non-assistant tokens.

        Model-specific builders usually override this hook for boundary handling,
        such as inserting or trimming tokens at the prefix/appended-token junction.
        """
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
        )

    def _render_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        tokenized = apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.chat_template_kwargs,
        )
        return normalize_token_ids(tokenized)

    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        rendered = apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.chat_template_kwargs,
        )
        if not isinstance(rendered, str):
            raise TypeError(f"Expected chat template to render str, got {type(rendered).__name__}")
        return rendered

    def render_delta_token_id(
        self,
        prefix_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Render prefix/full prompts as token IDs and return the token-level suffix."""
        prefix_token_ids = self._render_tokens(prefix_messages, add_generation_prompt=False, tools=tools)
        full_token_ids = self._render_tokens(
            prefix_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if full_token_ids[: len(prefix_token_ids)] != prefix_token_ids:
            roles = [message.get("role") for message in appended_messages] or ["generation_prompt"]
            raise ValueError(f"Continuous Token token-id suffix diff failed for roles: {roles}")
        return full_token_ids[len(prefix_token_ids) :]

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        synthetic_assistant = self._synthetic_assistant_for_tools(tool_messages)
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant],
            tool_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE],
            [message],
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        """Whether the last append-group render should also add the generation prompt.

        Builders whose chat template derives the generation scaffold from more than
        the appended group may return ``False`` and implement
        :meth:`_tokenize_generation_prompt_delta` using the required history.
        """
        return True

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Tokenize a generation prompt without re-encoding trajectory messages."""
        if not updated_messages:
            raise ValueError("Continuous Token requires messages before a generation prompt")

        if updated_messages[-1].get("role") == "tool":
            tool_group_start = len(updated_messages) - 1
            while tool_group_start > 0 and updated_messages[tool_group_start - 1].get("role") == "tool":
                tool_group_start -= 1
            tool_messages = updated_messages[tool_group_start:]
            synthetic_tool_messages = [
                {
                    "role": "tool",
                    "content": "continuous token synthetic tool response",
                    "tool_call_id": f"continuous_token_call_{index}",
                    "name": _DUMMY_TOOL_NAME,
                }
                for index in range(len(tool_messages))
            ]
            synthetic_context = [
                _SYNTHETIC_SYSTEM_MESSAGE,
                _SYNTHETIC_USER_MESSAGE,
                self._synthetic_assistant_for_tools(synthetic_tool_messages),
                *synthetic_tool_messages,
            ]
        else:
            # Generation scaffolds for supported user/system appends are
            # content-independent. A fixed valid context prevents this delta
            # probe from encoding any real trajectory message a second time.
            synthetic_context = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE]

        return self.render_delta_token_id(
            synthetic_context,
            [],
            add_generation_prompt=True,
            tools=tools,
        )

    def _iter_append_groups(self, appended_messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(appended_messages):
            role = appended_messages[index].get("role")
            if role == "tool":
                end = index + 1
                while end < len(appended_messages) and appended_messages[end].get("role") == "tool":
                    end += 1
                groups.append(appended_messages[index:end])
                index = end
            else:
                groups.append([appended_messages[index]])
                index += 1
        return groups

    def _assert_append_only(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
    ) -> None:
        if len(updated_messages) < len(previous_messages):
            raise ValueError("Continuous Token messages must be append-only; updated_messages is shorter")
        if updated_messages[: len(previous_messages)] != previous_messages:
            raise ValueError("Continuous Token messages must be append-only; prefix messages changed")
        for message in updated_messages[len(previous_messages) :]:
            role = message.get("role")
            if role not in self.allowed_append_roles:
                raise ValueError(
                    f"Continuous Token only supports appending roles {sorted(self.allowed_append_roles)}, got {role!r}"
                )

    def _synthetic_assistant_for_tools(
        self,
        tool_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tool_calls = []
        for index, tool_message in enumerate(tool_messages):
            tool_call = {
                "id": _tool_call_id_or_dummy(tool_message, index),
                "type": "function",
                "function": {
                    "name": _tool_message_name_or_dummy(tool_message),
                    "arguments": {},
                },
            }
            tool_calls.append(tool_call)
        return {
            "role": "assistant",
            "content": "",
            "reasoning_content": _ASSISTANT_REASONING_CONTENT,
            "tool_calls": tool_calls,
        }

    def align_response_metadata(
        self,
        merge_result: MergeResult,
        response_mask: list[int],
        response_logprobs: list[float] | None = None,
        *,
        assistant_logprobs: list[float] | None = None,
    ) -> tuple[list[int], list[float] | None]:
        """Align response masks and logprobs after a Continuous Token merge.

        ``MergeResult`` records token edits at the runtime-prefix boundary. This
        method applies the same edits to response-side metadata: trimming
        metadata for removed prefix tokens, assigning zero mask/logprob to
        inserted boundary or non-assistant tokens, and assigning assistant
        mask/logprobs to appended assistant tokens.
        """
        aligned_mask = list(response_mask)
        aligned_logprobs = list(response_logprobs) if response_logprobs is not None else None
        if aligned_logprobs is None and assistant_logprobs is not None:
            raise ValueError("response_logprobs is required when assistant_logprobs is provided")

        # If merge trimmed tokens from the current prefix, trim their metadata too.
        if merge_result.removed_prefix_token_count:
            aligned_mask = aligned_mask[: -merge_result.removed_prefix_token_count]
            if aligned_logprobs is not None:
                aligned_logprobs = aligned_logprobs[: -merge_result.removed_prefix_token_count]

        # Boundary tokens are added by CT itself, so they get mask/logprob 0.
        inserted_token_count = len(merge_result.inserted_token_ids)
        aligned_mask += [0] * inserted_token_count
        if aligned_logprobs is not None:
            aligned_logprobs += [0.0] * inserted_token_count

        # Assistant tokens get mask 1 and their logprobs; tool/user/system tokens get mask/logprob 0.
        if merge_result.kind == "assistant":
            aligned_mask += [1] * merge_result.appended_token_count
            if aligned_logprobs is not None:
                if assistant_logprobs is None:
                    if merge_result.appended_token_count:
                        raise ValueError("assistant_logprobs is required for assistant Continuous Token alignment")
                    assistant_logprobs = []
                if len(assistant_logprobs) != merge_result.appended_token_count:
                    raise ValueError(
                        "assistant_logprobs length must match appended assistant token count, "
                        f"got {len(assistant_logprobs)} and {merge_result.appended_token_count}"
                    )
                aligned_logprobs += list(assistant_logprobs)
        elif merge_result.kind == "non_assistant":
            aligned_mask += [0] * merge_result.appended_token_count
            if aligned_logprobs is not None:
                aligned_logprobs += [0.0] * merge_result.appended_token_count
        else:
            raise ValueError(f"Unknown Continuous Token merge kind: {merge_result.kind!r}")

        return aligned_mask, aligned_logprobs

    # === Multimodal hooks (VL subclasses override these) ===

    @classmethod
    def supports_multimodal(cls) -> bool:
        """Whether this builder handles vision inputs.

        VL subclasses override this to return True. Used by the wiring layer
        to decide whether to pass images through the CT pipeline.
        """
        return False

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Render messages with images through the processor.

        Unlike ``_render_tokens`` which uses only the tokenizer, this method
        invokes the full multimodal processor so image placeholders are expanded
        into the same token IDs the rollout backend will consume. VL subclasses apply
        their ``self.mm_processor_kwargs`` (min/max pixels, sampling rate, ...) captured
        at construction; the text base does not implement this method.

        Args:
            messages: OpenAI-format message list with image content items.
            images: List of PIL images (or paths), one per image content item.
            add_generation_prompt: Whether to append the generation prompt.

        Returns:
            Token IDs rendered by the multimodal processor. Pixel tensors are
            intentionally not returned here; final multimodal tensors are built
            from the full image list during agent-loop postprocessing.

        Raises:
            NotImplementedError: Unless overridden by a VL subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement render_tokens_with_mm.")


class GptOssContinuousTokenBuilder(ContinuousTokenBuilder):
    """GPT-OSS tool-response formatting."""

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        # Tool responses are encoded directly rather than through a chat-template
        # suffix diff, so keep generation-prompt derivation as a separate step.
        return False

    def _prepare_assistant_message_for_render(self, message: dict[str, Any]) -> dict[str, Any]:
        # Arrow-backed SFT datasets can materialize absent optional struct
        # fields as null. Harmony checks key presence for ``tool_calls`` and
        # ``thinking``, so remove null optionals and normalize nullable content
        # before invoking the template. Keep this normalization in the builder
        # as well as the dataset boundary for direct SFT callers.
        rendered_message = {key: value for key, value in message.items() if value is not None}
        if message.get("content") is None:
            rendered_message["content"] = ""
        return rendered_message

    def _assistant_terminator_ids(self, message: dict[str, Any]) -> set[int]:
        # Harmony final answers stop at <|return|> (the tokenizer EOS), while
        # assistant tool calls stop at <|call|> and contain no EOS token.
        if message.get("tool_calls"):
            return {require_token_id(self.tokenizer, "<|call|>")}
        return super()._assistant_terminator_ids(message)

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        del tools, add_generation_prompt
        response_text = "".join(
            self._format_tool_response(
                tool_message,
                _resolve_required_tool_name(
                    tool_message,
                    index,
                    tool_messages,
                    previous_messages,
                ),
            )
            for index, tool_message in enumerate(tool_messages)
        )
        return self.tokenizer.encode(response_text, add_special_tokens=False)

    @staticmethod
    def _format_tool_response(tool_message: dict[str, Any], tool_name: str) -> str:
        content = _stringify_tool_content(tool_message.get("content", ""))
        return f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{content}<|end|>"


class QwenContinuousTokenBuilder(ContinuousTokenBuilder):
    """Qwen ChatML boundary handling.

    Qwen2.5, Qwen3, and Qwen3.5 templates render ``<|im_end|>\n`` after a turn,
    while generation may stop at ``<|im_end|>``. When the runtime prefix ends
    there, insert the missing newline before appending non-assistant tokens.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected Qwen newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._im_end_id = require_token_id(tokenizer, "<|im_end|>")

    def _prepare_assistant_message_for_render(self, message: dict[str, Any]) -> dict[str, Any]:
        """Preserve literal nested think tags in raw Qwen assistant output.

        Qwen's template uses the last opening think tag when it infers
        reasoning_content from content, which drops text when reasoning itself
        mentions a literal think tag. Split only the outermost leading block and
        pass the fields explicitly so every nested/generated tag survives.
        """
        enable_thinking = self.chat_template_kwargs.get("enable_thinking")
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

    def _merge_non_assistant_token_ids(
        self, runtime_token_ids: list[int], appended_token_ids: list[int]
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            inserted_token_ids=inserted_token_ids,
        )


class MiniMaxText01ContinuousTokenBuilder(ContinuousTokenBuilder):
    """MiniMax-Text-01 ``<beginning_of_sentence>`` protocol."""

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected MiniMax-Text-01 newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._eos_id = require_token_id(tokenizer, "<end_of_sentence>")
        self._generation_scaffold_text = "<beginning_of_sentence>ai name=assistant\n"
        self._generation_scaffold_ids = normalize_token_ids(
            tokenizer.encode(self._generation_scaffold_text, add_special_tokens=False)
        )

    @staticmethod
    def _prepare_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return _prepare_minimax_legacy_messages(messages)

    def _render_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # Encode the normalized text ourselves so Text-01 template revisions
        # that differ only in whether they honor ``add_generation_prompt`` have
        # identical Continuous Token boundaries.
        rendered = self._render_text(
            messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        return normalize_token_ids(self.tokenizer.encode(rendered, add_special_tokens=False))

    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        rendered = apply_chat_template(
            self.tokenizer,
            self._prepare_messages(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=[tool.get("function", tool) for tool in tools] if tools else None,
            **self.chat_template_kwargs,
        )
        if not isinstance(rendered, str):
            raise TypeError(f"Expected chat template to render str, got {type(rendered).__name__}")
        # Some published Text-01 templates unconditionally append the next
        # assistant header. Normalize those revisions to the same contract as
        # templates that honor ``add_generation_prompt``.
        if not add_generation_prompt and rendered.endswith(self._generation_scaffold_text):
            rendered = rendered[: -len(self._generation_scaffold_text)]
        return rendered

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # Tool declarations sit after messages in the Text-01 template and do
        # not affect the generated continuation. Omitting them from this fixed
        # probe prevents their position from invalidating the prefix boundary.
        del tools
        return super().tokenize_assistant_message(
            _prepare_minimax_legacy_assistant_message(message),
            tools=None,
            previous_messages=previous_messages,
        )

    def _assistant_terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        return {self._eos_id}

    def _merge_non_assistant_token_ids(
        self, runtime_token_ids: list[int], appended_token_ids: list[int]
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            inserted_token_ids=inserted_token_ids,
        )

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
        response_text = _format_minimax_legacy_tool_responses(tool_messages, previous_messages)
        return normalize_token_ids(self.tokenizer.encode(response_text, add_special_tokens=False))

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
        return self._render_tokens([message], add_generation_prompt=False, tools=None)

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
        if not updated_messages:
            raise ValueError("Continuous Token requires messages before a generation prompt")
        return list(self._generation_scaffold_ids)


class MiniMaxContinuousTokenBuilder(ContinuousTokenBuilder):
    """MiniMax boundary handling.

    MiniMax templates render ``[e~[\n`` after a turn, while generation may stop
    at ``[e~[``. When the runtime prefix ends there, insert the missing newline
    before appending non-assistant tokens.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected MiniMax newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._eos_id = require_token_id(tokenizer, "[e~[")

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Reconstruct the response behind MiniMax's always-open think prompt."""
        del previous_messages
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )
        rendered_message = self._prepare_assistant_message_for_render(message)
        prompt_delta, completed_delta = self._render_assistant_turn_deltas(rendered_message, tools=tools)
        assistant_header = "]~b]ai\n"
        think_open = "<think>\n"
        if prompt_delta != assistant_header + think_open or not completed_delta.startswith(assistant_header):
            raise ValueError("Continuous Token MiniMax assistant scaffold does not match the supported protocol")

        completed_body = completed_delta[len(assistant_header) :]
        if completed_body.startswith(think_open):
            assistant_text = completed_body[len(think_open) :]
        else:
            # The full-history template omits an empty reasoning block, but a
            # rollout prompt has already emitted ``<think>\n``. The model must
            # therefore generate the closing side before its visible answer.
            assistant_text = "</think>\n\n" + completed_body
        assistant_token_ids = normalize_token_ids(self.tokenizer.encode(assistant_text, add_special_tokens=False))
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    def _prepare_assistant_message_for_render(self, message: dict[str, Any]) -> dict[str, Any]:
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

    def _assistant_terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        return {self._eos_id}

    def _merge_non_assistant_token_ids(
        self, runtime_token_ids: list[int], appended_token_ids: list[int]
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            inserted_token_ids=inserted_token_ids,
        )


class GLMContinuousTokenBuilder(ContinuousTokenBuilder):
    """GLM observation/user boundary handling.

    ``<|observation|>`` and ``<|user|>`` can be both assistant stop tokens and
    next-message start tokens. If the runtime prefix ends with either, remove
    that token before appending the next non-assistant segment.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._observation_id = require_token_id(tokenizer, "<|observation|>")
        self._user_id = require_token_id(tokenizer, "<|user|>")
        self._ambiguous_boundary_ids = {self._observation_id, self._user_id}

    def _prepare_assistant_message_for_render(self, message: dict[str, Any]) -> dict[str, Any]:
        enable_thinking = self.chat_template_kwargs.get("enable_thinking")
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

    def _merge_non_assistant_token_ids(
        self, runtime_token_ids: list[int], appended_token_ids: list[int]
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        removed_prefix_token_count = 0
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]
            removed_prefix_token_count = 1
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            removed_prefix_token_count=removed_prefix_token_count,
        )

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Handle GLM's empty-reasoning scaffold without rewriting the prompt."""
        del previous_messages
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )
        rendered_message = self._prepare_assistant_message_for_render(message)
        prompt_delta, completed_delta = self._render_assistant_turn_deltas(rendered_message, tools=tools)
        assistant_header = "<|assistant|>"
        if not prompt_delta.startswith(assistant_header) or not completed_delta.startswith(assistant_header):
            raise ValueError("Continuous Token GLM assistant scaffold does not match the supported protocol")

        if completed_delta.startswith(prompt_delta):
            assistant_text = completed_delta[len(prompt_delta) :]
        elif prompt_delta == assistant_header + "<think>":
            # With empty reasoning the history template emits only </think>, so
            # the completed render is not prefixed by the generation prompt's
            # opening <think>. That opening token is already in runtime tokens.
            assistant_text = completed_delta[len(assistant_header) :]
        else:
            raise ValueError("Continuous Token GLM completed assistant turn does not extend its generation scaffold")

        assistant_token_ids = normalize_token_ids(self.tokenizer.encode(assistant_text, add_special_tokens=False))
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    def _normalize_assistant_token_ids(
        self,
        assistant_token_ids: list[int],
        message: dict[str, Any],
    ) -> list[int]:
        # GLM templates do not terminate assistant text with EOS. Tool-call
        # generation instead stops on <|observation|>, which is also the opening
        # boundary of the following tool response.
        normalized_ids = list(assistant_token_ids)
        if message.get("tool_calls") and (not normalized_ids or normalized_ids[-1] != self._observation_id):
            normalized_ids.append(self._observation_id)
        return normalized_ids


class Gemma4ContinuousTokenBuilder(ContinuousTokenBuilder):
    """Gemma4 tool-response boundary handling."""

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._tool_response_id = require_token_id(tokenizer, "<|tool_response>")
        self._turn_id = require_token_id(tokenizer, "<turn|>")
        self._tool_call_id = require_token_id(tokenizer, "<tool_call|>")

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Reconstruct a Gemma4 response behind its thinking-mode scaffold."""
        del previous_messages
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )

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

        prompt_delta, completed_delta = self._render_assistant_turn_deltas(rendered_message, tools=tools)
        assistant_header = "<|turn>model\n"
        if not prompt_delta.startswith(assistant_header) or not completed_delta.startswith(assistant_header):
            raise ValueError("Continuous Token Gemma4 assistant scaffold does not match the supported protocol")
        completed_body = completed_delta[len(assistant_header) :]
        prompt_scaffold = prompt_delta[len(assistant_header) :]
        empty_thought_scaffold = thought_open + thought_close

        if prompt_scaffold == empty_thought_scaffold:
            # Standard 26B/31B non-thinking prompts already contain the complete
            # empty thought channel, so only the visible completion is generated.
            assistant_text = completed_body
        elif not prompt_scaffold:
            # Thinking prompts and E4B prompts end immediately after the model
            # header. E4B may already serialize reasoning for tool calls; other
            # templates drop it from history and require reconstruction here.
            if completed_body.startswith(thought_open):
                assistant_text = completed_body
            elif reasoning or self.chat_template_kwargs.get("enable_thinking", False):
                assistant_text = thought_open + reasoning + thought_close + completed_body
            else:
                assistant_text = completed_body
        else:
            raise ValueError("Continuous Token Gemma4 generation prompt has an unsupported scaffold")

        assistant_token_ids = normalize_token_ids(self.tokenizer.encode(assistant_text, add_special_tokens=False))
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    def _normalize_assistant_token_ids(
        self,
        assistant_token_ids: list[int],
        message: dict[str, Any],
    ) -> list[int]:
        terminator_id = self._tool_call_id if message.get("tool_calls") else self._turn_id
        for index, token_id in enumerate(assistant_token_ids):
            if token_id == terminator_id:
                return list(assistant_token_ids[: index + 1])
        raise ValueError(
            "Continuous Token Gemma4 assistant token-id suffix does not contain the expected terminator "
            f"{terminator_id}; tail={assistant_token_ids[-16:]}"
        )

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        # Gemma4's generation scaffold depends on the preceding message type and
        # is handled by the model-specific hook below.
        return False

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        del tools, add_generation_prompt
        response_parts = []
        for index, tool_message in enumerate(tool_messages):
            resolved_name = _resolve_required_tool_name(
                tool_message,
                index,
                tool_messages,
                previous_messages,
            )
            content = _stringify_tool_content(tool_message.get("content", ""))
            response_parts.append(
                f'<|tool_response>response:{resolved_name}{{value:<|"|>{content}<|"|>}}<tool_response|>'
            )
        return normalize_token_ids(self.tokenizer.encode("".join(response_parts), add_special_tokens=False))

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        last_message = updated_messages[-1]
        if last_message.get("role") not in {"user", "system"}:
            return []
        return super()._tokenize_generation_prompt_delta(updated_messages, tools=tools)

    def merge_non_assistant_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        appended_token_ids = self.tokenize_non_assistant_incremental_messages(
            previous_messages, updated_messages, tools=tools
        )
        appended_messages = updated_messages[len(previous_messages) :]

        prefix = list(runtime_token_ids)
        appended_token_ids = list(appended_token_ids)
        inserted_token_ids: list[int] = []
        # Gemma's tool block opens with <|tool_response>. The synthetic-prefix
        # suffix diff attributes that boundary token to the diffed-away assistant
        # turn, so it is missing from ``appended_token_ids``; re-insert it at the
        # junction. Guard against double insertion in case the prefix already ends
        # with it or the diff happens to retain it.
        if appended_messages and appended_messages[0].get("role") == "tool":
            if prefix[-1:] == [self._tool_response_id] and appended_token_ids[:1] == [self._tool_response_id]:
                appended_token_ids = appended_token_ids[1:]
            elif prefix[-1:] != [self._tool_response_id] and appended_token_ids[:1] != [self._tool_response_id]:
                prefix.append(self._tool_response_id)
                inserted_token_ids.append(self._tool_response_id)

        return MergeResult(
            token_ids=prefix + appended_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            inserted_token_ids=inserted_token_ids,
        )


def require_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not define required token {token!r}")
    if isinstance(token_id, list):
        if len(token_id) != 1:
            raise ValueError(f"Tokenizer returned multiple ids for required token {token!r}: {token_id!r}")
        token_id = token_id[0]
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Tokenizer returned invalid id for required token {token!r}: {token_id!r}")
    return token_id


def _stringify_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def _prepare_minimax_legacy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI text/tool messages to MiniMax-01's legacy block schema."""
    prepared_messages = []
    for message_index, message in enumerate(messages):
        prepared_message = (
            _prepare_minimax_legacy_assistant_message(message) if message.get("role") == "assistant" else dict(message)
        )
        if message.get("role") == "tool":
            tool_group_start = message_index
            while tool_group_start > 0 and messages[tool_group_start - 1].get("role") == "tool":
                tool_group_start -= 1
            tool_group_end = message_index + 1
            while tool_group_end < len(messages) and messages[tool_group_end].get("role") == "tool":
                tool_group_end += 1
            tool_group = messages[tool_group_start:tool_group_end]
            prepared_message["role"] = "function"
            prepared_message["name"] = _resolve_required_tool_name(
                message,
                message_index - tool_group_start,
                tool_group,
                messages[:tool_group_start],
            )
        content = prepared_message.get("content")
        if isinstance(content, str):
            prepared_message["content"] = [{"type": "text", "text": content}]
        elif content is None:
            prepared_message["content"] = []
        prepared_messages.append(prepared_message)
    return prepared_messages


def _prepare_minimax_legacy_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct MiniMax-01's textual function-call continuation."""
    if not message.get("tool_calls") or _stringify_tool_content(message.get("content", "")):
        return message

    call_parts = []
    for tool_call in message["tool_calls"]:
        function = tool_call.get("function", tool_call)
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        call_parts.append(f"<function_call>```typescript\nfunctions.{name}({arguments})\n```")
    rendered_message = dict(message)
    rendered_message["content"] = "".join(call_parts)
    return rendered_message


def _format_minimax_legacy_tool_responses(
    tool_messages: list[dict[str, Any]],
    previous_messages: list[dict[str, Any]],
) -> str:
    response_parts = []
    for index, tool_message in enumerate(tool_messages):
        resolved_name = _resolve_required_tool_name(
            tool_message,
            index,
            tool_messages,
            previous_messages,
        )
        content = _stringify_tool_content(tool_message.get("content", ""))
        response_parts.append(
            "<beginning_of_sentence>system function_response=functions\n"
            f'{{"name": "{resolved_name}", "response": {content}}}'
            "<end_of_sentence>\n"
        )
    return "".join(response_parts)


def _tool_message_name_or_dummy(tool_message: dict[str, Any]) -> str:
    if tool_message.get("name"):
        return str(tool_message["name"])
    return _DUMMY_TOOL_NAME


def _tool_call_id_or_dummy(tool_message: dict[str, Any], index: int) -> Any:
    if tool_message.get("tool_call_id") is not None:
        return tool_message["tool_call_id"]
    return f"continuous_token_call_{index}"


def _latest_assistant_tool_call_names(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str | None]]:
    tool_names_by_id: dict[str, str] = {}
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            return tool_names_by_id, []
        positional_tool_names: list[str | None] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                positional_tool_names.append(None)
                continue
            name = _tool_call_function_name(tool_call)
            positional_tool_names.append(name)
            tool_call_id = tool_call.get("id")
            if name is not None and tool_call_id is not None:
                tool_names_by_id.setdefault(str(tool_call_id), name)
        return tool_names_by_id, positional_tool_names
    return tool_names_by_id, []


def _resolve_required_tool_name(
    tool_message: dict[str, Any],
    index: int,
    tool_messages: list[dict[str, Any]],
    previous_messages: list[dict[str, Any]],
) -> str:
    if tool_message.get("name"):
        return str(tool_message["name"])

    tool_names_by_id, positional_tool_names = _latest_assistant_tool_call_names(previous_messages)
    tool_call_id = tool_message.get("tool_call_id")
    if tool_call_id is not None and str(tool_call_id) in tool_names_by_id:
        return tool_names_by_id[str(tool_call_id)]

    if len(tool_messages) != len(positional_tool_names):
        raise ValueError(
            "Continuous Token cannot resolve tool name by position: "
            f"got {len(tool_messages)} tool response messages but the latest assistant has "
            f"{len(positional_tool_names)} tool calls"
        )
    if index >= len(positional_tool_names) or positional_tool_names[index] is None:
        raise ValueError(
            "Continuous Token cannot resolve tool name by position: "
            f"assistant tool call at index {index} has no function name"
        )

    # ToolAgentLoop uses asyncio.gather and appends responses in the original
    # tool-call order, so positional matching is safe for its full response
    # batches. Black-box agent loops may return responses in another order; they
    # must provide tool message name or tool_call_id instead of relying on this.
    logger.warning(
        "Continuous Token is resolving a tool response name by position; this is only safe when "
        "tool responses are appended in the same order as the latest assistant tool_calls"
    )
    return positional_tool_names[index]


def _tool_call_function_name(tool_call: dict[str, Any]) -> str | None:
    function = tool_call.get("function")
    if isinstance(function, dict) and function.get("name") is not None:
        return str(function["name"])
    return None


class DeepSeekContinuousTokenBuilder(ContinuousTokenBuilder):
    """DeepSeek V3/R1 boundary handling.

    DeepSeek uses direct concatenation at boundaries (no separator between
    ``<|end_of_sentence|>`` and the next role marker). The subclass validates
    key special tokens use correct Unicode (fullwidth vertical line U+FF5C
    and lower one-eighth block U+2581) to catch encoding regressions early.
    """

    # DeepSeek special tokens use fullwidth vertical line and lower one-eighth block
    _EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    _BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    _USER_TOKEN = "<\uff5cUser\uff5c>"
    _ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        if "enable_thinking" in self.chat_template_kwargs and "thinking" not in self.chat_template_kwargs:
            self.chat_template_kwargs = {
                **self.chat_template_kwargs,
                "thinking": self.chat_template_kwargs["enable_thinking"],
            }
        # EOS is the only token guaranteed across V2/V3/R1
        self._eos_id = require_token_id(tokenizer, self._EOS_TOKEN)
        # V3/R1-specific tokens — lookup but tolerate absence (V2-Lite has none)
        self._bos_id = self._optional_token_id(tokenizer, self._BOS_TOKEN)
        self._user_id = self._optional_token_id(tokenizer, self._USER_TOKEN)
        self._assistant_id = self._optional_token_id(tokenizer, self._ASSISTANT_TOKEN)

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Handle both classic DeepSeek and V3.1 thinking scaffolds."""
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )

        if previous_messages and previous_messages[-1].get("role") == "tool":
            return self._tokenize_assistant_after_tool(message, tools=tools)

        prompt_delta, completed_delta = self._render_assistant_turn_deltas(message, tools=tools)
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
            # V3.1's completed render already contains the closing tag and its
            # visible answer/tool call, but drops reasoning. The generation
            # prompt contains the opening tag, so put only the missing reasoning
            # in front of that completed suffix.
            assistant_header = completed_prefix[: -len(think_close)]
            assistant_text = reasoning + completed_delta[len(assistant_header) :]

        assistant_token_ids = normalize_token_ids(self.tokenizer.encode(assistant_text, add_special_tokens=False))
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    def _tokenize_assistant_after_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> list[int]:
        """Encode the direct continuation used after a DeepSeek tool result."""
        synthetic_tool_message = {
            "role": "tool",
            "content": "continuous token synthetic tool response",
            "tool_call_id": "continuous_token_call_0",
            "name": _DUMMY_TOOL_NAME,
        }
        synthetic_context = [
            _SYNTHETIC_SYSTEM_MESSAGE,
            _SYNTHETIC_USER_MESSAGE,
            self._synthetic_assistant_for_tools([synthetic_tool_message]),
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
        assistant_token_ids = normalize_token_ids(self.tokenizer.encode(assistant_text, add_special_tokens=False))
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    def _assistant_terminator_ids(self, message: dict[str, Any]) -> set[int]:
        del message
        return {self._eos_id}

    def _synthetic_assistant_for_tools(
        self,
        tool_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        synthetic_assistant = super()._synthetic_assistant_for_tools(tool_messages)
        # DeepSeek V2/V3 templates concatenate arguments directly and therefore
        # require serialized JSON rather than the dict accepted by ChatML-style
        # templates.
        for tool_call in synthetic_assistant["tool_calls"]:
            tool_call["function"]["arguments"] = "{}"
        return synthetic_assistant

    @staticmethod
    def _optional_token_id(tokenizer: Any, token: str) -> int | None:
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk:
            return None
        return int(token_id)

    def _merge_non_assistant_token_ids(
        self, runtime_token_ids: list[int], appended_token_ids: list[int]
    ) -> MergeResult:
        # Direct concatenation — DeepSeek template has no inter-turn separator
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
        )


# =============================================================================
# Multimodal (VL) subclasses
# =============================================================================


class VLContinuousTokenMixin:
    """Shared processor-backed logic for vision-language continuous token builders.

    Provides the multimodal workflow (image extraction, processor rendering,
    incremental dummy+trim encoding) common to all VL builders. Subclasses
    combine this mixin with a text-family builder (e.g. QwenContinuousTokenBuilder)
    via Python MRO so that boundary handling like Qwen's newline insertion or
    GLM's observation/user trim still applies through ``_merge_non_assistant_token_ids``.
    """

    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        # Processor kwargs (e.g. max_pixels, do_pan_and_scan) that control how media
        # expands into tokens. Constant for the builder's whole lifetime, so renders
        # (initial prompt and incremental tool/user) stay aligned.
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        # Fold in the processor's audio sampling rate (a static processor property) so
        # mm_processor_kwargs is complete. Image-only processors have no
        # feature_extractor -> no-op.
        if "sampling_rate" not in self.mm_processor_kwargs:
            sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                self.mm_processor_kwargs = {**self.mm_processor_kwargs, "sampling_rate": int(sampling_rate)}

    @classmethod
    def supports_multimodal(cls) -> bool:
        return True

    def _extract_images_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Extract image references from OpenAI-style content blocks."""
        images: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                        image_ref = block.get("image")
                        if not image_ref:
                            image_url = block.get("image_url")
                            if isinstance(image_url, dict):
                                image_ref = image_url.get("url")
                            elif isinstance(image_url, str):
                                image_ref = image_url
                        if image_ref is not None:
                            images.append(image_ref)
        return images

    def _extract_videos_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Extract video references from OpenAI-style content blocks."""
        videos: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "video":
                        video_ref = block.get("video")
                        if video_ref is not None:
                            videos.append(video_ref)
        return videos

    def _extract_audios_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Extract audio references from OpenAI-style content blocks."""
        audios: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio":
                        audio_ref = block.get("audio")
                        if audio_ref is None:
                            audio_ref = block.get("audio_url")
                        if audio_ref is not None:
                            audios.append(audio_ref)
        return audios

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Render messages through the processor (full render with all media)."""
        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools

        # Render the chat template through the processor (not the tokenizer) so the
        # placeholder text matches the legacy rollout path. VL models may ship a
        # processor chat template that differs from the tokenizer one, so it is
        # necessary to use the processor chat template for VL models.
        text = apply_chat_template(
            self.processor,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        # Processor kwargs are the builder-level constant captured at construction,
        # so initial-prompt and incremental (tool/user) renders stay aligned.
        proc_kwargs = dict(self.mm_processor_kwargs or {})
        processor_output = build_multimodal_processor_inputs(
            self.processor,
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            audio=audios if audios else None,
            mm_processor_kwargs=proc_kwargs if proc_kwargs else None,
        )
        return normalize_token_ids(processor_output["input_ids"])

    def _render_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Render messages to token IDs through the processor (VL override).

        Routes the base text renderer through the processor chat template +
        processor call so list-of-blocks content is handled and vision
        placeholders are expanded into per-image pad tokens. Media references are
        extracted from ``messages`` themselves.
        """
        return self.render_tokens_with_mm(
            messages,
            self._extract_images_from_messages(messages),
            videos=self._extract_videos_from_messages(messages),
            audios=self._extract_audios_from_messages(messages),
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _render_text(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools
        rendered = apply_chat_template(
            self.processor,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        if not isinstance(rendered, str):
            raise TypeError(f"Expected processor chat template to render str, got {type(rendered).__name__}")
        return rendered

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        return self.render_tokens_with_mm(
            messages,
            images,
            videos=videos,
            audios=audios,
            add_generation_prompt=True,
            tools=tools,
        )


class VLContinuousTokenBuilder(VLContinuousTokenMixin, ContinuousTokenBuilder):
    """Generic vision-language builder used as the default for VL models that
    have no model-specific builder.

    Combines the shared processor-backed VL rendering (from the mixin) with the
    base, family-agnostic boundary handling (from ContinuousTokenBuilder).
    """


class QwenVLContinuousTokenBuilder(VLContinuousTokenMixin, QwenContinuousTokenBuilder):
    """Qwen Vision-Language: Qwen ChatML newline patch + VL processor logic.

    Handles Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and Qwen3-VL-MoE.
    """


class MiniMaxVLContinuousTokenBuilder(VLContinuousTokenMixin, MiniMaxText01ContinuousTokenBuilder):
    """MiniMax-VL-01 legacy sentence protocol + VL processor logic.

    MiniMax-VL-01's *processor* chat template ignores ``add_generation_prompt`` and
    unconditionally appends an assistant scaffold ``<beginning_of_sentence>ai
    name=assistant\\n`` after every render. That breaks Continuous Token's
    append-only / suffix-diff invariant (``render(prefix)`` is no longer a token
    prefix of ``render(prefix + msg)``). We normalize the template here: strip the
    auto-appended scaffold when ``add_generation_prompt=False`` and keep it when
    ``True`` (where it legitimately is the generation prompt).
    """

    def __init__(self, tokenizer: Any, processor: Any, **kwargs: Any):
        super().__init__(tokenizer, processor, **kwargs)
        self._vl_scaffold_ids = self._compute_generation_scaffold_ids()

    def _compute_generation_scaffold_ids(self) -> list[int]:
        """Extract the assistant scaffold across published template revisions."""
        prepared_messages = self._prepare_vl_messages([_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE])
        without_prompt_ids = VLContinuousTokenMixin.render_tokens_with_mm(
            self,
            prepared_messages,
            [],
            add_generation_prompt=False,
        )
        with_prompt_ids = VLContinuousTokenMixin.render_tokens_with_mm(
            self,
            prepared_messages,
            [],
            add_generation_prompt=True,
        )

        # Revised templates may honor ``add_generation_prompt``. In that case
        # the generation scaffold is the ordinary suffix delta.
        if with_prompt_ids[: len(without_prompt_ids)] == without_prompt_ids and len(with_prompt_ids) > len(
            without_prompt_ids
        ):
            return with_prompt_ids[len(without_prompt_ids) :]

        # The original published template ignores the flag and appends the same
        # scaffold to both renders. Find that unconditional final sentence.
        if with_prompt_ids != without_prompt_ids:
            raise ValueError("MiniMax-VL generation-prompt renders have an unsupported boundary")
        bos_id = require_token_id(self.tokenizer, "<beginning_of_sentence>")
        bos_positions = [i for i, token_id in enumerate(with_prompt_ids) if token_id == bos_id]
        if not bos_positions:
            raise ValueError("MiniMax-VL scaffold detection failed: no <beginning_of_sentence> token")
        scaffold = with_prompt_ids[bos_positions[-1] :]
        if not scaffold:
            raise ValueError("MiniMax-VL scaffold detection produced an empty scaffold")
        return scaffold

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # Strip an unconditional scaffold when no generation prompt was
        # requested; templates that honor the flag have no matching tail here.
        token_ids = super().render_tokens_with_mm(
            self._prepare_vl_messages(messages),
            images,
            videos=videos,
            audios=audios,
            add_generation_prompt=add_generation_prompt,
            tools=[tool.get("function", tool) for tool in tools] if tools else None,
        )
        scaffold = self._vl_scaffold_ids
        if token_ids[-len(scaffold) :] == scaffold and not add_generation_prompt:
            token_ids = token_ids[: -len(scaffold)]
        return token_ids

    @staticmethod
    def _prepare_vl_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Adapt OpenAI text/tool messages to MiniMax-VL's block schema."""
        return _prepare_minimax_legacy_messages(messages)

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
        response_text = _format_minimax_legacy_tool_responses(tool_messages, previous_messages)
        return normalize_token_ids(self.tokenizer.encode(response_text, add_special_tokens=False))

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # MiniMax-VL places tool declarations after conversation messages, so
        # probing with them would move already-committed declarations. A single
        # user/system turn is independently serializable through the processor.
        del tools
        return self._render_tokens([message], add_generation_prompt=False, tools=None)

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
        if not updated_messages:
            raise ValueError("Continuous Token requires messages before a generation prompt")
        return list(self._vl_scaffold_ids)

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Extract the VL assistant body after its unconditional scaffold."""
        del tools, previous_messages
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )
        rendered_message = self._prepare_vl_assistant_message(message)
        synthetic_prompt = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE]
        context_ids = self._render_tokens(synthetic_prompt, add_generation_prompt=False, tools=None)
        prompt_ids = self._render_tokens(synthetic_prompt, add_generation_prompt=True, tools=None)
        completed_ids = self._render_tokens(
            [*synthetic_prompt, rendered_message],
            add_generation_prompt=False,
            tools=None,
        )
        if prompt_ids[: len(context_ids)] != context_ids or completed_ids[: len(context_ids)] != context_ids:
            raise ValueError("Continuous Token MiniMax-VL assistant renders do not preserve the fixed context")
        if prompt_ids[len(context_ids) :] != self._vl_scaffold_ids:
            raise ValueError("Continuous Token MiniMax-VL generation prompt has an unsupported scaffold")
        completed_delta = completed_ids[len(context_ids) :]
        if completed_delta[: len(self._vl_scaffold_ids)] != self._vl_scaffold_ids:
            raise ValueError("Continuous Token MiniMax-VL completed turn does not start with its assistant scaffold")
        assistant_token_ids = completed_delta[len(self._vl_scaffold_ids) :]
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    @staticmethod
    def _prepare_vl_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
        return _prepare_minimax_legacy_assistant_message(message)


class Gemma4VLContinuousTokenBuilder(VLContinuousTokenMixin, Gemma4ContinuousTokenBuilder):
    """Gemma4 (unified) vision-language: Gemma4 ``<|tool_response>`` boundary handling
    + VL processor rendering.

    Gemma4 is a unified text+vision architecture, so the same checkpoint serves
    both modalities. The mixin routes user/system/assistant rendering through the
    multimodal processor chat template, while tool-response boundary handling is
    inherited from :class:`Gemma4ContinuousTokenBuilder`.
    """


class GLM46VContinuousTokenBuilder(VLContinuousTokenMixin, GLMContinuousTokenBuilder):
    """GLM-4.6V: GLM observation/user trim + VL processor logic."""


class KimiVLContinuousTokenBuilder(VLContinuousTokenMixin, ContinuousTokenBuilder):
    """Kimi-VL (MoonViT): direct concatenation + VL processor logic."""

    @staticmethod
    def _reject_tools(tools: list[dict[str, Any]] | None) -> None:
        if tools:
            raise ValueError("Kimi-VL Continuous Token does not support structured tool schemas")

    @staticmethod
    def _reject_structured_messages(messages: list[dict[str, Any]]) -> None:
        if any(message.get("role") == "tool" for message in messages):
            raise ValueError("Kimi-VL Continuous Token does not support structured tool response messages")
        if any(message.get("role") == "assistant" and message.get("tool_calls") for message in messages):
            raise ValueError("Kimi-VL Continuous Token does not support structured assistant tool calls")

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        self._reject_tools(tools)
        self._reject_structured_messages(messages)
        return super().build_initial_tokens(
            messages,
            tools=None,
            images=images,
            videos=videos,
            audios=audios,
        )

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        self._reject_tools(tools)
        self._reject_structured_messages([message])
        return super().tokenize_assistant_message(
            message,
            tools=None,
            previous_messages=previous_messages,
        )

    def merge_non_assistant_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        self._reject_tools(tools)
        self._reject_structured_messages(updated_messages)
        return super().merge_non_assistant_tokens(
            previous_messages,
            updated_messages,
            runtime_token_ids,
            tools=None,
        )

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tool_messages, previous_messages, tools
        raise ValueError("Kimi-VL Continuous Token does not support structured tool responses")

    def _normalize_assistant_token_ids(
        self,
        assistant_token_ids: list[int],
        message: dict[str, Any],
    ) -> list[int]:
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id != unk_token_id:
            for index, token_id in enumerate(assistant_token_ids):
                if token_id == im_end_id:
                    return list(assistant_token_ids[: index + 1])
        return super()._normalize_assistant_token_ids(assistant_token_ids, message)


class DeepSeekVL2ContinuousTokenBuilder(DeepSeekContinuousTokenBuilder):
    """DeepSeek-VL2 continuous token builder.

    VL2 uses its own DeepseekVLV2Processor that handles both conversation
    formatting and image token expansion in a single __call__. It does NOT
    support standard apply_chat_template, so all rendering goes through the
    processor directly.

    The processor produces stable prefixes: full_render[:len(prev)] == prev,
    so we use full render + prefix diff (like the original CT approach).
    """

    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        # VL2 renders through DeepseekVLV2Processor directly and does not consume
        # mm_processor_kwargs, but it is stored for API symmetry with other VL builders.
        self.mm_processor_kwargs = mm_processor_kwargs or {}

    @classmethod
    def supports_multimodal(cls) -> bool:
        return True

    def _extract_images_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Extract image references from content blocks."""
        images: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                        image_ref = block.get("image")
                        if not image_ref:
                            image_url = block.get("image_url")
                            if isinstance(image_url, dict):
                                image_ref = image_url.get("url")
                            elif isinstance(image_url, str):
                                image_ref = image_url
                        if image_ref is not None:
                            images.append(image_ref)
        return images

    def _to_vl2_conversation(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        add_generation_prompt: bool = True,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Convert OpenAI-style messages to VL2 conversation format."""
        conv: list[dict[str, Any]] = []
        img_idx = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                msg_images: list[Any] = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype in ("image", "image_url") and img_idx < len(images):
                            parts.append("<image>")
                            msg_images.append(images[img_idx])
                            img_idx += 1
                        elif btype == "text":
                            parts.append(block.get("text", ""))
                content = "".join(parts)
            else:
                msg_images = []

            if role == "user":
                conv.append({"role": "<|User|>", "content": content, "images": msg_images})
            elif role == "assistant":
                conv.append({"role": "<|Assistant|>", "content": content})
            elif role == "system":
                conv.append({"role": "system", "content": content})
            else:
                raise ValueError(f"DeepSeek-VL2 Continuous Token does not support message role {role!r}")

        if add_generation_prompt:
            if not conv or conv[-1].get("role") != "<|Assistant|>" or conv[-1].get("content"):
                conv.append({"role": "<|Assistant|>", "content": ""})
        return conv, images

    def _render_via_processor(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        """Render messages through DeepseekVLV2Processor."""
        conv, all_images = self._to_vl2_conversation(messages, images, add_generation_prompt)
        # The official processor removes its final EOS in inference mode. A
        # generation prompt needs that behavior, while a completed gold
        # assistant turn must retain EOS as a generated continuation token.
        out = self.processor.__call__(
            conversations=conv,
            images=all_images,
            force_batchify=True,
            inference_mode=add_generation_prompt,
        )
        return normalize_token_ids(out.input_ids[0].tolist())

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        if images is None:
            images = self._extract_images_from_messages(messages)
        if tools:
            raise ValueError("DeepSeek-VL2 Continuous Token does not support tool schemas")
        return self._render_via_processor(messages, images, add_generation_prompt=True)

    def tokenize_assistant_message(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        previous_messages: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Extract one assistant continuation through DeepseekVLV2Processor."""
        del previous_messages
        if message.get("role") != "assistant":
            raise ValueError(
                f"Continuous Token assistant encoding requires role='assistant', got {message.get('role')!r}"
            )
        if tools:
            raise ValueError("DeepSeek-VL2 Continuous Token does not support tool schemas")
        if message.get("tool_calls"):
            raise ValueError("DeepSeek-VL2 Continuous Token does not support structured assistant tool calls")

        synthetic_prompt = [_SYNTHETIC_USER_MESSAGE]
        prompt_token_ids = self._render_via_processor(synthetic_prompt, [], add_generation_prompt=True)
        completed_token_ids = self._render_via_processor(
            [*synthetic_prompt, message],
            [],
            add_generation_prompt=False,
        )
        if completed_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
            raise ValueError(
                "Continuous Token assistant encoding requires the processor generation prompt to be a token-id "
                "prefix of the completed assistant turn"
            )
        assistant_token_ids = completed_token_ids[len(prompt_token_ids) :]
        if not assistant_token_ids:
            raise ValueError("Continuous Token assistant encoding produced an empty token-id suffix")
        return self._normalize_assistant_token_ids(assistant_token_ids, message)

    def merge_non_assistant_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        """Merge tokens: always use processor + prefix diff for VL2.

        VL2 tokenizer has no chat_template, so all rendering goes through
        the processor. Prefix stability is guaranteed by the processor.
        """
        self._assert_append_only(previous_messages, updated_messages)
        if tools:
            raise ValueError("DeepSeek-VL2 Continuous Token does not support tool schemas")
        if any(message.get("role") == "tool" for message in updated_messages[len(previous_messages) :]):
            raise ValueError("DeepSeek-VL2 Continuous Token does not support tool response messages")

        # Always use full render + prefix diff (VL2 has no apply_chat_template)
        all_images = self._extract_images_from_messages(updated_messages)
        full_token_ids = self._render_via_processor(updated_messages, all_images, add_generation_prompt=True)

        prefix_len = len(runtime_token_ids)
        if full_token_ids[:prefix_len] != list(runtime_token_ids):
            raise ValueError("DeepSeek-VL2 Continuous Token processor output does not preserve the runtime prefix")
        appended_token_ids = full_token_ids[prefix_len:]
        return self._merge_non_assistant_token_ids(runtime_token_ids, appended_token_ids)
