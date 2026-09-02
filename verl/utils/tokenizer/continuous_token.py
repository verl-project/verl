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

_SUPPORTED_APPEND_ROLES = frozenset({"tool", "user", "system", "assistant"})
_SYNTHETIC_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": "continuous token synthetic system"}
_SYNTHETIC_USER_MESSAGE: dict[str, Any] = {"role": "user", "content": "continuous token synthetic user"}
_ASSISTANT_REASONING_CONTENT: str = "reasoning"
_DUMMY_TOOL_NAME = "continuous_token_tool"
MergeKind = Literal["assistant", "context"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeResult:
    """Merged runtime tokens plus the edits callers need to align metadata.

    ``token_ids`` is the updated runtime token stream. The other fields describe
    how the stream changed at the merge junction: ``inserted_token_ids`` are
    CT-created boundary tokens, ``appended_token_count`` counts newly appended
    model-generated assistant or context tokens excluding those inserted boundary tokens,
    and ``removed_prefix_token_count`` counts stale prefix tokens dropped before
    appending. Boundary tokens are not model-generated and therefore must not
    carry loss or model logprobs.
    """

    token_ids: list[int]
    appended_token_count: int
    kind: MergeKind = "context"
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class ContinuousTokenBuilder:
    """Build and update continuous-token runtime prompts for multi-turn rollouts.

    This class exposes two API layers:

    AgentLoop-facing runtime APIs:
        ``build_initial_tokens`` renders the first prompt, ``merge_context_tokens``
        merges append-only context messages, ``merge_assistant_tokens``
        appends model-generated assistant tokens, and ``align_response_metadata``
        applies the recorded token edits to masks/logprobs.

    Developer extension APIs:
        Model-specific builders should subclass this class and keep the runtime
        API contracts above stable. Chat template specific behavior belongs in hooks
        such as ``_tokenize_tool_group``, ``_tokenize_single_non_tool``,
        ``_merge_context_group``,
        ``_should_fuse_generation_prompt_with_last_group``,
        ``_tokenize_generation_prompt_delta``, and ``_merge_context_token_ids``.
        ``render_delta_token_id`` is the shared suffix-diff helper those hooks can reuse.
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

    def tokenize_context_incremental_messages(
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
        processed_messages = list(previous_messages)

        groups = self._iter_append_groups(appended_messages)
        fuse_generation_prompt = self._should_fuse_generation_prompt_with_last_group()

        for index, group in enumerate(groups):
            add_generation_prompt = fuse_generation_prompt and index == len(groups) - 1
            role = group[0].get("role")
            if role == "tool":
                group_token_ids = self._tokenize_tool_group(
                    group,
                    previous_messages=processed_messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            elif role in {"user", "system", "assistant"}:
                # System appends can represent retry/control messages; unsupported templates will fail in suffix diff.
                if len(group) != 1:
                    raise ValueError(
                        f"Continuous Token expects one {role!r} message per append group, got {len(group)}"
                    )
                if role == "assistant" and index == len(groups) - 1:
                    raise ValueError("Continuous Token context incremental messages cannot end with assistant")
                group_token_ids = self._tokenize_single_non_tool(
                    group[0],
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")
            self._merge_context_group(
                incremental_ids,
                group_token_ids,
                group=group,
                processed_messages=processed_messages,
                tools=tools,
            )
            processed_messages.extend(group)

        if not fuse_generation_prompt:
            incremental_ids.extend(self._tokenize_generation_prompt_delta(updated_messages, tools=tools))
        return incremental_ids

    def _merge_context_group(
        self,
        incremental_ids: list[int],
        group_token_ids: list[int],
        *,
        group: list[dict[str, Any]],
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        """Merge one encoded context group into the incremental token stream."""
        incremental_ids.extend(group_token_ids)

    def merge_context_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        appended_messages = updated_messages[len(previous_messages) :]
        appended_ids = self.tokenize_context_incremental_messages(previous_messages, updated_messages, tools=tools)
        return self._merge_context_token_ids(
            runtime_token_ids,
            appended_ids,
            previous_messages=previous_messages,
            appended_messages=appended_messages,
        )

    def merge_assistant_tokens(self, runtime_token_ids: list[int], assistant_token_ids: list[int]) -> MergeResult:
        """Merge model-generated assistant tokens into the runtime token stream."""
        merged_token_ids = list(runtime_token_ids) + list(assistant_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(assistant_token_ids),
            kind="assistant",
        )

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        """Merge runtime prefix tokens and appended context tokens.

        Model-specific builders usually override this hook for boundary handling,
        such as inserting or trimming tokens at the prefix/appended-token junction.
        ``merge_context_tokens`` also supplies message-level boundary context as
        keyword arguments for builders whose token edits depend on message roles.
        """
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="context",
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
        """Tokenize the tokens added only by ``add_generation_prompt=True``."""
        return self.render_delta_token_id(updated_messages, [], add_generation_prompt=True, tools=tools)

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
        inserted boundary or context tokens, and assigning model-generated assistant
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

        # Model-generated assistant tokens get mask 1 and their logprobs; context tokens get mask/logprob 0.
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
        elif merge_result.kind == "context":
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

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            del add_generation_prompt, tools
            # Context incremental messages currently reject a final assistant turn,
            # so a plain GPT-OSS assistant here is always historical context. Keep
            # its analysis/reasoning channel before the final channel, and close
            # both with <|end|> rather than the training-only final <|return|>.
            # If final assistant context is supported in the future, this function may need change.
            return self.tokenizer.encode(self._format_plain_assistant_context(message), add_special_tokens=False)
        return super()._tokenize_single_non_tool(
            message,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        # Tool responses are encoded directly rather than through a chat-template
        # suffix diff, so keep generation-prompt derivation as a separate step.
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
        json_content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        return f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{json_content}<|end|>"

    @staticmethod
    def _format_plain_assistant_context(message: dict[str, Any]) -> str:
        thinking = message.get("thinking")
        if thinking is None:
            thinking = message.get("reasoning_content")
        if thinking is None:
            thinking = message.get("reasoning")

        rendered = ""
        if thinking:
            thinking_text = _stringify_tool_content(thinking)
            rendered += f"<|start|>assistant<|channel|>analysis<|message|>{thinking_text}<|end|>"
        content = _stringify_tool_content(message.get("content", ""))
        return rendered + f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"


class QwenContinuousTokenBuilder(ContinuousTokenBuilder):
    """Qwen ChatML boundary handling.

    Qwen2, Qwen2.5, Qwen3, and Qwen3.5 templates render ``<|im_end|>\n`` after a turn,
    while generation may stop at ``<|im_end|>``. When the runtime prefix ends
    there, insert the missing newline before appending context tokens.

    Note: when a Qwen3/Qwen3.5 context assistant message has no thinking, grouped
    CT encoding renders that assistant in a synthetic context and may produce an
    empty ``<think></think>`` block. Rendering that assistant together with the
    following context messages through the full chat template does not produce
    the empty thinking block.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected Qwen newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._im_end_id = require_token_id(tokenizer, "<|im_end|>")

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            inserted_token_ids=inserted_token_ids,
        )


class MiniMaxContinuousTokenBuilder(ContinuousTokenBuilder):
    """MiniMax boundary handling.

    MiniMax templates render ``[e~[\n`` after a turn, while generation may stop
    at ``[e~[``. When the runtime prefix ends there, insert the missing newline
    before appending context tokens.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected MiniMax newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._eos_id = require_token_id(tokenizer, "[e~[")

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            inserted_token_ids=inserted_token_ids,
        )


class GLMContinuousTokenBuilder(ContinuousTokenBuilder):
    """GLM observation/user boundary handling.

    ``<|observation|>`` and ``<|user|>`` can be both assistant stop tokens and
    next-message start tokens. If the runtime prefix ends with either, remove
    that token before appending the next context segment.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._observation_id = require_token_id(tokenizer, "<|observation|>")
        self._user_id = require_token_id(tokenizer, "<|user|>")
        self._ambiguous_boundary_ids = {self._observation_id, self._user_id}

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        prefix = list(runtime_token_ids)
        removed_prefix_token_count = 0
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]
            removed_prefix_token_count = 1
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            removed_prefix_token_count=removed_prefix_token_count,
        )


class Gemma4ContinuousTokenBuilder(ContinuousTokenBuilder):
    """Gemma4 tool-response boundary handling."""

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._tool_response_id = require_token_id(tokenizer, "<|tool_response>")
        turn_separator_ids = normalize_token_ids(tokenizer.encode("<turn|>\n", add_special_tokens=False))
        if not turn_separator_ids:
            raise ValueError("Expected Gemma4 '<turn|>\\n' turn separator to tokenize to at least one token")
        self._turn_separator_ids = turn_separator_ids

    def tokenize_context_incremental_messages(
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
        processed_messages = list(previous_messages)

        groups = self._iter_append_groups(appended_messages)
        for group_index, group in enumerate(groups):
            role = group[0].get("role")
            if role == "tool":
                incremental_ids.extend(
                    self._tokenize_tool_group(
                        group,
                        previous_messages=processed_messages,
                        tools=tools,
                    )
                )
            elif role in {"user", "system", "assistant"}:
                if len(group) != 1:
                    raise ValueError(
                        f"Continuous Token expects one {role!r} message per append group, got {len(group)}"
                    )
                if role == "assistant" and group_index == len(groups) - 1:
                    raise ValueError("Continuous Token context incremental messages cannot end with assistant")
                if role == "user" and self._needs_user_turn_separator(
                    incremental_ids,
                    processed_messages=processed_messages,
                    tools=tools,
                ):
                    incremental_ids.extend(self._turn_separator_ids)
                incremental_ids.extend(
                    self._tokenize_single_non_tool(
                        group[0],
                        add_generation_prompt=False,
                        tools=tools,
                    )
                )
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")
            processed_messages.extend(group)

        incremental_ids.extend(self._tokenize_generation_prompt_delta(updated_messages, tools=tools))
        return incremental_ids

    def _needs_user_turn_separator(
        self,
        incremental_ids: list[int],
        *,
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        if not incremental_ids and not processed_messages:
            return False
        return not self._context_prefix_ends_with_turn_separator(
            incremental_ids,
            processed_messages=processed_messages,
            tools=tools,
        )

    def _context_prefix_ends_with_turn_separator(
        self,
        incremental_ids: list[int],
        *,
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        separator_ids = self._turn_separator_ids
        if incremental_ids:
            return incremental_ids[-len(separator_ids) :] == separator_ids
        if not processed_messages:
            return False
        rendered_ids = self._render_tokens(processed_messages, add_generation_prompt=False, tools=tools)
        return rendered_ids[-len(separator_ids) :] == separator_ids

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
        del add_generation_prompt
        tool_calls = []
        rendered_tool_messages = []
        for index, tool_message in enumerate(tool_messages):
            call_id = _tool_call_id_or_dummy(tool_message, index)
            resolved_name = _resolve_required_tool_name(
                tool_message,
                index,
                tool_messages,
                previous_messages,
            )
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": resolved_name,
                        "arguments": {},
                    },
                }
            )
            rendered_message = dict(tool_message)
            rendered_message["tool_call_id"] = call_id
            if not rendered_message.get("name"):
                rendered_message["name"] = resolved_name
            rendered_tool_messages.append(rendered_message)
        synthetic_assistant = {
            "role": "assistant",
            "content": "",
            "reasoning_content": _ASSISTANT_REASONING_CONTENT,
            "tool_calls": tool_calls,
        }
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant],
            rendered_tool_messages,
            tools=tools,
        )

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        last_message = updated_messages[-1]
        if last_message.get("role") not in {"user", "system"}:
            return []
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, last_message],
            [],
            add_generation_prompt=True,
            tools=tools,
        )

    def merge_context_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        appended_messages = updated_messages[len(previous_messages) :]

        # Gemma4 renders an assistant message after a tool response as a
        # continuation of the same model turn. Appending that assistant via the
        # generic synthetic system/user context would incorrectly add a second
        # model-turn header. Compute the exact tail edit from the real rolled-back
        # message prefix instead. This is still a tail-only edit and is reported in
        # MergeResult so response metadata is trimmed and masked consistently.
        # TODO: this full-template tail edit currently follows Gemma4's historical
        # assistant behavior and drops thinking from appended context assistant
        # messages when later context turns exist. Find an encoding strategy that
        # both preserves assistant thinking and keeps the assistant/tool/user
        # boundary identical to applying the Gemma4 chat template to the full
        # context.
        if any(message.get("role") == "assistant" for message in appended_messages):
            self._assert_append_only(previous_messages, updated_messages)
            previous_rendered_ids = self._render_tokens(
                previous_messages,
                add_generation_prompt=False,
                tools=tools,
            )
            updated_rendered_ids = self._render_tokens(
                updated_messages,
                add_generation_prompt=True,
                tools=tools,
            )
            common_prefix_length = 0
            for previous_id, updated_id in zip(previous_rendered_ids, updated_rendered_ids, strict=False):
                if previous_id != updated_id:
                    break
                common_prefix_length += 1

            removed_prefix_token_count = len(previous_rendered_ids) - common_prefix_length
            removed_rendered_ids = previous_rendered_ids[common_prefix_length:]
            if removed_prefix_token_count > len(runtime_token_ids):
                raise ValueError("Gemma4 assistant context tail edit exceeds the runtime token prefix")
            if removed_rendered_ids and runtime_token_ids[-removed_prefix_token_count:] != removed_rendered_ids:
                raise ValueError("Gemma4 assistant context tail does not match the rolled-back runtime prefix")

            retained_runtime_ids = (
                runtime_token_ids[:-removed_prefix_token_count]
                if removed_prefix_token_count
                else list(runtime_token_ids)
            )
            appended_token_ids = updated_rendered_ids[common_prefix_length:]
            return MergeResult(
                token_ids=retained_runtime_ids + appended_token_ids,
                appended_token_count=len(appended_token_ids),
                kind="context",
                removed_prefix_token_count=removed_prefix_token_count,
            )

        appended_token_ids = self.tokenize_context_incremental_messages(
            previous_messages, updated_messages, tools=tools
        )

        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        # Gemma's tool block opens with <|tool_response>. The synthetic-prefix
        # suffix diff attributes that boundary token to the diffed-away assistant
        # turn, so it is missing from ``appended_token_ids``; re-insert it at the
        # junction. Guard against double insertion in case the prefix already ends
        # with it or the diff happens to retain it.
        if (
            appended_messages
            and appended_messages[0].get("role") == "tool"
            and prefix[-1:] != [self._tool_response_id]
            and appended_token_ids[:1] != [self._tool_response_id]
        ):
            prefix.append(self._tool_response_id)
            inserted_token_ids.append(self._tool_response_id)

        return MergeResult(
            token_ids=prefix + appended_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="context",
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
    _TOOL_OUTPUTS_END_TOKEN = "<\uff5ctool\u2581outputs\u2581end\uff5c>"

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        # EOS is the only token guaranteed across V2/V3/R1
        self._eos_id = require_token_id(tokenizer, self._EOS_TOKEN)
        # V3/R1-specific tokens — lookup but tolerate absence (V2-Lite has none)
        self._bos_id = self._optional_token_id(tokenizer, self._BOS_TOKEN)
        self._user_id = self._optional_token_id(tokenizer, self._USER_TOKEN)
        self._assistant_id = self._optional_token_id(tokenizer, self._ASSISTANT_TOKEN)
        self._tool_outputs_end_ids = normalize_token_ids(
            tokenizer.encode(self._TOOL_OUTPUTS_END_TOKEN, add_special_tokens=False)
        )
        if not self._tool_outputs_end_ids:
            raise ValueError(f"Expected DeepSeek {self._TOOL_OUTPUTS_END_TOKEN!r} to tokenize to at least one token")

    @staticmethod
    def _optional_token_id(tokenizer: Any, token: str) -> int | None:
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk:
            return None
        return int(token_id)

    def _synthetic_assistant_for_tools(
        self,
        tool_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # The R1 / V3.1 / V3.2 templates splice ``tool['function']['arguments']`` into
        # the prompt by string concatenation, so a mapping raises TypeError there. The
        # synthetic tool call only exists to be diffed away; carry its arguments as the
        # JSON string those templates expect.
        synthetic_assistant = super()._synthetic_assistant_for_tools(tool_messages)
        for tool_call in synthetic_assistant["tool_calls"]:
            tool_call["function"]["arguments"] = "{}"
        return synthetic_assistant

    def tokenize_context_incremental_messages(
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
        processed_messages = list(previous_messages)

        groups = self._iter_append_groups(appended_messages)
        fuse_generation_prompt = self._should_fuse_generation_prompt_with_last_group()

        for index, group in enumerate(groups):
            add_generation_prompt = fuse_generation_prompt and index == len(groups) - 1
            role = group[0].get("role")
            if role == "tool":
                incremental_ids.extend(
                    self._tokenize_tool_group(
                        group,
                        previous_messages=processed_messages,
                        tools=tools,
                        add_generation_prompt=add_generation_prompt,
                    )
                )
            elif role in {"user", "system", "assistant"}:
                if len(group) != 1:
                    raise ValueError(
                        f"Continuous Token expects one {role!r} message per append group, got {len(group)}"
                    )
                if role == "assistant" and index == len(groups) - 1:
                    raise ValueError("Continuous Token context incremental messages cannot end with assistant")
                if role == "user":
                    self._strip_trailing_tool_outputs_end(incremental_ids)
                incremental_ids.extend(
                    self._tokenize_single_non_tool(
                        group[0],
                        add_generation_prompt=add_generation_prompt,
                        tools=tools,
                    )
                )
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")
            processed_messages.extend(group)

        if not fuse_generation_prompt:
            incremental_ids.extend(self._tokenize_generation_prompt_delta(updated_messages, tools=tools))
        return incremental_ids

    def merge_context_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        appended_token_ids = self.tokenize_context_incremental_messages(
            previous_messages, updated_messages, tools=tools
        )
        appended_messages = updated_messages[len(previous_messages) :]

        prefix = list(runtime_token_ids)
        removed_prefix_token_count = 0
        if appended_messages and appended_messages[0].get("role") == "user":
            removed_prefix_token_count = self._strip_trailing_tool_outputs_end(prefix)

        return MergeResult(
            token_ids=prefix + appended_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="context",
            removed_prefix_token_count=removed_prefix_token_count,
        )

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        # Direct concatenation — DeepSeek template has no inter-turn separator.
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="context",
        )

    def _strip_trailing_tool_outputs_end(self, token_ids: list[int]) -> int:
        suffix = self._tool_outputs_end_ids
        if token_ids[-len(suffix) :] != suffix:
            return 0
        del token_ids[-len(suffix) :]
        return len(suffix)


# =============================================================================
# Multimodal (VL) subclasses
# =============================================================================


class VLContinuousTokenMixin:
    """Shared processor-backed logic for vision-language continuous token builders.

    Provides the multimodal workflow (image extraction, processor rendering,
    incremental dummy+trim encoding) common to all VL builders. Subclasses
    combine this mixin with a text-family builder (e.g. QwenContinuousTokenBuilder)
    via Python MRO so that boundary handling like Qwen's newline insertion or
    GLM's observation/user trim still applies through ``_merge_context_token_ids``.
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


class MiniMaxVLContinuousTokenBuilder(VLContinuousTokenMixin, MiniMaxContinuousTokenBuilder):
    """MiniMax-VL (e.g. MiniMax-VL-01): MiniMax ``[e~[`` newline patch + VL processor logic.

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
        # MiniMax-VL-01 uses ``<end_of_sentence>`` as its turn terminator, not the
        # MiniMax-Text-01 ``[e~[`` token the base builder resolves. Repoint the EOS
        # so the boundary-newline reinsertion in ``_merge_context_token_ids``
        # fires for VL turns.
        self._eos_id = require_token_id(tokenizer, "<end_of_sentence>")
        self._vl_scaffold_ids = self._compute_generation_scaffold_ids()

    def _compute_generation_scaffold_ids(self) -> list[int]:
        """Extract the unconditional trailing assistant scaffold token ids.

        Rendered via the processor chat template (bypassing this class's override),
        the scaffold is the final ``<beginning_of_sentence>...`` block, i.e. every
        token from the last ``<beginning_of_sentence>`` to the end.
        """
        text = apply_chat_template(
            self.processor,
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE],
            tokenize=False,
            add_generation_prompt=False,
            **self.chat_template_kwargs,
        )
        ids = normalize_token_ids(
            build_multimodal_processor_inputs(self.processor, text=[text], images=None)["input_ids"]
        )
        bos_id = require_token_id(self.tokenizer, "<beginning_of_sentence>")
        bos_positions = [i for i, t in enumerate(ids) if t == bos_id]
        if not bos_positions:
            raise ValueError("MiniMax-VL scaffold detection failed: no <beginning_of_sentence> token")
        scaffold = ids[bos_positions[-1] :]
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
        # The processor template always appends the scaffold; strip it unless a
        # generation prompt was requested, restoring append-only rendering.
        token_ids = super().render_tokens_with_mm(
            messages,
            images,
            videos=videos,
            audios=audios,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        scaffold = self._vl_scaffold_ids
        if token_ids[-len(scaffold) :] == scaffold and not add_generation_prompt:
            token_ids = token_ids[: -len(scaffold)]
        return token_ids


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
                conv.append({"role": "<|User|>", "content": content, "images": []})

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
        out = self.processor.__call__(conversations=conv, images=all_images, force_batchify=True)
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
        if not images:
            return self._render_tokens(messages, add_generation_prompt=True, tools=tools)
        return self._render_via_processor(messages, images, add_generation_prompt=True)

    def merge_context_tokens(
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

        # Always use full render + prefix diff (VL2 has no apply_chat_template)
        all_images = self._extract_images_from_messages(updated_messages)
        full_token_ids = self._render_via_processor(updated_messages, all_images, add_generation_prompt=True)

        prefix_len = len(runtime_token_ids)
        appended_token_ids = full_token_ids[prefix_len:]
        return self._merge_context_token_ids(runtime_token_ids, appended_token_ids)
