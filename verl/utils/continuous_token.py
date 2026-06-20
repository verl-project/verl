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
from typing import Any, Literal, Sequence

logger = logging.getLogger(__name__)

_SUPPORTED_APPEND_ROLES = frozenset({"tool", "user", "system"})
_SYNTHETIC_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": "continuous token synthetic system"}
_SYNTHETIC_USER_MESSAGE: dict[str, Any] = {"role": "user", "content": "continuous token synthetic user"}
_ASSISTANT_REASONING_CONTENT: str = "reasoning"
MergeKind = Literal["assistant", "non_assistant"]


@dataclass(frozen=True)
class MergeResult:
    """Token merge result with the token-level delta needed by callers.

    ``inserted_token_ids`` are CT-created boundary tokens at the merge junction.
    They are not model-generated tokens and therefore must not carry loss or
    model logprobs. Generated assistant tokens are represented by
    ``appended_token_count`` with ``kind="assistant"``.
    """

    token_ids: list[int]
    appended_token_count: int
    kind: MergeKind = "non_assistant"
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0

    # --- Multimodal fields (added for VL continuous token support) ---
    pixel_values: Any = None
    """Preprocessed pixel tensor(s) for all images in the merged sequence.
    Shape is model-dependent. None when text-only or images unchanged."""

    image_grid_thw: list[tuple[int, int, int]] = field(default_factory=list)
    """Per-image (temporal, height_grid, width_grid) after processor rescaling.
    Empty list when text-only. Length equals number of images in merged sequence."""

    image_token_spans: list[tuple[int, int]] = field(default_factory=list)
    """(start, end) half-open index pairs locating each image's pad tokens in
    token_ids. Empty list when text-only. Length == len(image_grid_thw)."""

    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra kwargs for the processor / rollout engine (e.g. min_pixels).
    Empty dict when unused."""


def ct_align_response_metadata(
    merge_result: MergeResult,
    response_mask: list[int],
    response_logprobs: list[float] | None = None,
    *,
    assistant_logprobs: list[float] | None = None,
) -> tuple[list[int], list[float] | None]:
    """Align response masks and logprobs after a Continuous Token merge.

    ``MergeResult`` describes token edits at the runtime-prefix boundary. This
    helper mirrors those edits for response-side metadata so downstream agent
    loops can reuse Continuous Token without depending on ``AgentLoopBase``.
    """
    aligned_mask = list(response_mask)
    aligned_logprobs = list(response_logprobs) if response_logprobs is not None else None
    if aligned_logprobs is None and assistant_logprobs is not None:
        raise ValueError("response_logprobs is required when assistant_logprobs is provided")

    if merge_result.removed_prefix_token_count:
        aligned_mask = aligned_mask[: -merge_result.removed_prefix_token_count]
        if aligned_logprobs is not None:
            aligned_logprobs = aligned_logprobs[: -merge_result.removed_prefix_token_count]

    # Inserted tokens are CT-created boundary tokens, not model-generated tokens.
    inserted_token_count = len(merge_result.inserted_token_ids)
    aligned_mask += [0] * inserted_token_count
    if aligned_logprobs is not None:
        aligned_logprobs += [0.0] * inserted_token_count

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


class ContinuousTokenBuilder:
    """Continuous Token builder for runtime prefix reuse."""

    allowed_append_roles: frozenset[str] = _SUPPORTED_APPEND_ROLES

    def __init__(
        self,
        tokenizer: Any,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        allowed_append_roles: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        if allowed_append_roles is not None:
            allowed_roles = frozenset(allowed_append_roles)
            unknown_roles = allowed_roles - _SUPPORTED_APPEND_ROLES
            if unknown_roles:
                raise ValueError(f"Unsupported Continuous Token append roles: {sorted(unknown_roles)}")
            self.allowed_append_roles = allowed_roles

    @staticmethod
    def align_response_metadata(
        merge_result: MergeResult,
        response_mask: list[int],
        response_logprobs: list[float] | None = None,
        *,
        assistant_logprobs: list[float] | None = None,
    ) -> tuple[list[int], list[float] | None]:
        return ct_align_response_metadata(
            merge_result,
            response_mask,
            response_logprobs,
            assistant_logprobs=assistant_logprobs,
        )

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return self._render_tokens(messages, add_generation_prompt=True, tools=tools)

    def tokenize_incremental_messages(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages) :]
        incremental_ids: list[int] = []
        processed_messages: list[dict[str, Any]] = []

        for group in self._iter_append_groups(appended_messages):
            role = group[0].get("role")
            if role == "tool":
                incremental_ids.extend(
                    self._tokenize_tool_group(
                        group,
                        context_messages=previous_messages + processed_messages,
                        tools=tools,
                    )
                )
            elif role in {"user", "system"}:
                # System appends can represent retry/control messages; unsupported templates will fail in suffix diff.
                incremental_ids.extend(self._tokenize_single_non_tool(group[0], tools=tools))
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")
            processed_messages.extend(group)

        incremental_ids.extend(
            self.render_delta_token_id(updated_messages, [], add_generation_prompt=True, tools=tools)
        )
        return incremental_ids

    def merge_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        appended_ids = self.tokenize_incremental_messages(previous_messages, updated_messages, tools=tools)
        return self._merge_token_ids(runtime_token_ids, appended_ids)

    def append_assistant_tokens(self, runtime_token_ids: list[int], assistant_token_ids: list[int]) -> MergeResult:
        """Append model-generated assistant tokens to the runtime token stream."""
        merged_token_ids = list(runtime_token_ids) + list(assistant_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(assistant_token_ids),
            kind="assistant",
        )

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
        """Merge runtime prefix tokens and appended tokens.

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
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import normalize_token_ids

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
        context_messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        synthetic_assistant = self._synthetic_assistant_for_tools(tool_messages, context_messages=context_messages)
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant],
            tool_messages,
            tools=tools,
        )

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE],
            [message],
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
        *,
        context_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tool_names_by_id, positional_tool_names = _assistant_tool_call_names(context_messages or [])
        tool_calls = []
        for index, tool_message in enumerate(tool_messages):
            tool_call = {
                "type": "function",
                "function": {
                    "name": _resolve_tool_name(tool_message, index, tool_names_by_id, positional_tool_names),
                    "arguments": {},
                },
            }
            if tool_message.get("tool_call_id") is not None:
                tool_call["id"] = tool_message["tool_call_id"]
            tool_calls.append(tool_call)
        return {
            "role": "assistant",
            "content": "",
            "reasoning_content": _ASSISTANT_REASONING_CONTENT,
            "tool_calls": tool_calls,
        }


    # === Multimodal hooks (VL subclasses override these) ===

    @classmethod
    def supports_multimodal(cls) -> bool:
        """Whether this builder handles vision inputs.

        VL subclasses override this to return True. Used by the wiring layer
        to decide whether to pass images through the CT pipeline.
        """
        return False

    def count_vision_tokens(self, image_grid_thw_row: tuple[int, int, int]) -> int:
        """Compute how many pad tokens one image occupies given its grid dims.

        Args:
            image_grid_thw_row: (temporal, grid_h, grid_w) for a single image,
                as returned by the processor's image preprocessing.

        Returns:
            Number of image placeholder tokens for this image.

        Raises:
            NotImplementedError: Unless overridden by a VL subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement count_vision_tokens. "
            "Override supports_multimodal() and this method for VL models."
        )

    def extract_vision_placeholders(self, token_ids: Sequence[int]) -> list[tuple[int, int]]:
        """Find all (start, end) spans of vision placeholder tokens.

        Args:
            token_ids: Full token ID sequence to scan.

        Returns:
            List of [start, end) index pairs, one per image, in appearance order.

        Raises:
            NotImplementedError: Unless overridden by a VL subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement extract_vision_placeholders."
        )

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        add_generation_prompt: bool = True,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Render messages with images through the processor.

        Unlike ``_render_tokens`` which uses only the tokenizer, this method
        invokes the full multimodal processor to produce both token IDs and
        pixel tensors.

        Args:
            messages: OpenAI-format message list with image content items.
            images: List of PIL images (or paths), one per image content item.
            add_generation_prompt: Whether to append the generation prompt.
            mm_processor_kwargs: Extra kwargs for the processor (min/max pixels).

        Returns:
            (token_ids, mm_extras) where mm_extras contains at minimum:
                - "pixel_values": processed pixel tensor
                - "image_grid_thw": list of (t, h, w) tuples per image
            Additional model-specific keys may be present.

        Raises:
            NotImplementedError: Unless overridden by a VL subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement render_tokens_with_mm."
        )


class GptOssContinuousTokenBuilder(ContinuousTokenBuilder):
    """GPT-OSS tool-response formatting."""

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        context_messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
        tool_names_by_id, positional_tool_names = _assistant_tool_call_names(context_messages or [])
        response_text = "".join(
            self._format_tool_response(
                tool_message,
                _resolve_tool_name(tool_message, index, tool_names_by_id, positional_tool_names),
            )
            for index, tool_message in enumerate(tool_messages)
        )
        return self.tokenizer.encode(response_text, add_special_tokens=False)

    @staticmethod
    def _format_tool_response(tool_message: dict[str, Any], tool_name: str) -> str:
        content = json.dumps(_stringify_tool_content(tool_message.get("content", "")), ensure_ascii=False)
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
        self._im_end_id = _require_token_id(tokenizer, "<|im_end|>")

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
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
        self._eos_id = _require_token_id(tokenizer, "[e~[")

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
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
        self._observation_id = _require_token_id(tokenizer, "<|observation|>")
        self._user_id = _require_token_id(tokenizer, "<|user|>")
        self._ambiguous_boundary_ids = {self._observation_id, self._user_id}

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
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


class Gemma4ContinuousTokenBuilder(ContinuousTokenBuilder):
    """Gemma4 tool-response boundary handling."""

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._tool_response_id = _require_token_id(tokenizer, "<|tool_response>")

    def merge_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        appended_token_ids = self.tokenize_incremental_messages(previous_messages, updated_messages, tools=tools)
        appended_messages = updated_messages[len(previous_messages) :]

        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if appended_messages and prefix[-1:] != [self._tool_response_id]:
            prefix.append(self._tool_response_id)
            inserted_token_ids.append(self._tool_response_id)

        return MergeResult(
            token_ids=prefix + appended_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
            inserted_token_ids=inserted_token_ids,
        )


def _require_token_id(tokenizer: Any, token: str) -> int:
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


def _assistant_tool_call_names(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    tool_names_by_id: dict[str, str] = {}
    positional_tool_names: list[str] = []
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        names: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = _tool_call_function_name(tool_call)
            if name is None:
                continue
            names.append(name)
            tool_call_id = tool_call.get("id")
            if tool_call_id is not None:
                tool_names_by_id.setdefault(str(tool_call_id), name)
        if names and not positional_tool_names:
            positional_tool_names = names
    return tool_names_by_id, positional_tool_names


def _resolve_tool_name(
    tool_message: dict[str, Any],
    index: int,
    tool_names_by_id: dict[str, str],
    positional_tool_names: list[str],
) -> str:
    tool_call_id = tool_message.get("tool_call_id")
    if tool_call_id is not None and str(tool_call_id) in tool_names_by_id:
        return tool_names_by_id[str(tool_call_id)]
    if index < len(positional_tool_names):
        return positional_tool_names[index]
    if tool_message.get("name"):
        return str(tool_message["name"])
    return "continuous_token_tool"


def _tool_call_function_name(tool_call: dict[str, Any]) -> str | None:
    function = tool_call.get("function")
    if isinstance(function, dict) and function.get("name") is not None:
        return str(function["name"])
    return None


# =============================================================================
# New model-family text subclasses
# =============================================================================


class MiMoContinuousTokenBuilder(ContinuousTokenBuilder):
    """Xiaomi MiMo ChatML boundary handling.

    MiMo uses Qwen2Tokenizer with identical ChatML format. Behavior matches
    QwenContinuousTokenBuilder (insert newline after <|im_end|>) but is kept
    as an independent class for structural clarity and future divergence.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected MiMo newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._im_end_id = _require_token_id(tokenizer, "<|im_end|>")

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
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


class DeepSeekContinuousTokenBuilder(ContinuousTokenBuilder):
    """DeepSeek V3/R1 boundary handling.

    DeepSeek uses direct concatenation at boundaries (no separator between
    <|end_of_sentence|> and the next role marker). The subclass validates
    the EOS token uses correct Unicode (fullwidth | U+FF5C and lower-eighth
    block _ U+2581) to catch common encoding bugs early.
    """

    # DeepSeek special tokens use fullwidth vertical line and lower one-eighth block
    _EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._eos_id = _require_token_id(tokenizer, self._EOS_TOKEN)

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
        # DeepSeek has no intervening token between EOS and next role marker.
        # Direct concatenation is correct behavior.
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
        )


class KimiContinuousTokenBuilder(ContinuousTokenBuilder):
    """Kimi (Moonshot) three-part turn boundary handling.

    Kimi uses a unique three-part turn structure:
        <|im_{role}|>{role_text}<|im_middle|>{content}<|im_end|>
    with no whitespace/newline between turns. When the runtime prefix ends
    with <|im_end|>, no separator is inserted (unlike Qwen which needs \\n).
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._im_end_id = _require_token_id(tokenizer, "<|im_end|>")

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
        # Kimi concatenates directly after <|im_end|> — no separator needed.
        # The next turn's <|im_{role}|> token follows immediately.
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
        )


class Nemotron4ContinuousTokenBuilder(ContinuousTokenBuilder):
    """NVIDIA Nemotron-4 (extra_id style) boundary handling.

    Nemotron-4 uses <extra_id_0> for system and <extra_id_1> for user/assistant/tool
    role markers. There is no explicit end-of-turn token — turns are implicitly
    bounded by the next role marker. Direct concatenation is correct.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        # Validate the role marker tokens exist
        self._extra_id_0 = _require_token_id(tokenizer, "<extra_id_0>")
        self._extra_id_1 = _require_token_id(tokenizer, "<extra_id_1>")

    def _merge_token_ids(self, runtime_token_ids: list[int], appended_token_ids: list[int]) -> MergeResult:
        # No end-of-turn delimiter in Nemotron-4; direct concatenation.
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="non_assistant",
        )


# =============================================================================
# Multimodal (VL) subclasses — Phase 1
# =============================================================================


class QwenVLContinuousTokenBuilder(QwenContinuousTokenBuilder):
    """Qwen Vision-Language continuous token builder.

    Handles Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and Qwen3-VL-MoE. Inherits
    the ChatML newline boundary patch from QwenContinuousTokenBuilder and
    adds vision token handling (Tier 1 Wrapper pattern):
        <|vision_start|> + N * <|image_pad|> + <|vision_end|>

    The __init__ resolves model-variant differences (patch_size 14 vs 16,
    spatial_merge_size source) via the processor/config.
    """

    def __init__(self, tokenizer: Any, processor: Any, *, model_type: str = "qwen2_5_vl", **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        self.model_type = model_type

        # Vision special token IDs
        self._vision_start_id = _require_token_id(tokenizer, "<|vision_start|>")
        self._vision_end_id = _require_token_id(tokenizer, "<|vision_end|>")
        self._image_pad_id = _require_token_id(tokenizer, "<|image_pad|>")

        # Spatial merge size — determines how many pad tokens per image
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "merge_size"):
            self._spatial_merge_size = int(processor.image_processor.merge_size)
        else:
            # Fallback: try model config
            self._spatial_merge_size = 2

    @classmethod
    def supports_multimodal(cls) -> bool:
        return True

    def count_vision_tokens(self, image_grid_thw_row: tuple[int, int, int]) -> int:
        """Compute image_pad count: t * (h // merge) * (w // merge)."""
        t, h, w = image_grid_thw_row
        merge = self._spatial_merge_size
        return t * (h // merge) * (w // merge)

    def extract_vision_placeholders(self, token_ids: Sequence[int]) -> list[tuple[int, int]]:
        """Find all <|vision_start|>...<|vision_end|> spans in token sequence."""
        spans: list[tuple[int, int]] = []
        i = 0
        n = len(token_ids)
        while i < n:
            if token_ids[i] == self._vision_start_id:
                j = i + 1
                while j < n and token_ids[j] != self._vision_end_id:
                    j += 1
                if j < n:
                    spans.append((i + 1, j))
                else:
                    logger.warning("Unmatched <|vision_start|> at position %d", i)
                i = j + 1
            else:
                i += 1
        return spans

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        add_generation_prompt: bool = True,
        mm_processor_kwargs: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Render messages through the Qwen VL processor."""
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools

        text = apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        proc_kwargs = dict(mm_processor_kwargs or {})
        processor_output = build_multimodal_processor_inputs(
            self.processor,
            text=text,
            images=images if images else None,
            mm_processor_kwargs=proc_kwargs if proc_kwargs else None,
        )

        token_ids = normalize_token_ids(processor_output["input_ids"])

        mm_extras: dict[str, Any] = {}
        if "pixel_values" in processor_output:
            mm_extras["pixel_values"] = processor_output["pixel_values"]
        if "image_grid_thw" in processor_output:
            grid_thw = processor_output["image_grid_thw"]
            if hasattr(grid_thw, "tolist"):
                mm_extras["image_grid_thw"] = [tuple(row) for row in grid_thw.tolist()]
            else:
                mm_extras["image_grid_thw"] = list(grid_thw)
        if proc_kwargs:
            mm_extras["mm_processor_kwargs"] = proc_kwargs

        return token_ids, mm_extras

    def _extract_images_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Extract image references from message content blocks."""
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

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Build initial tokens with multimodal processor support.

        Stores mm_extras in self._last_mm_extras for caller retrieval.
        Falls back to text-only rendering if no images are present.
        """
        images = self._extract_images_from_messages(messages)
        if not images:
            # No images — use text-only path (inherited from base)
            self._last_mm_extras: dict[str, Any] = {}
            return self._render_tokens(messages, add_generation_prompt=True, tools=tools)

        token_ids, mm_extras = self.render_tokens_with_mm(
            messages, images, add_generation_prompt=True, tools=tools
        )
        self._last_mm_extras = mm_extras
        return token_ids

    def merge_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        """Merge tokens with multimodal awareness.

        If new images appear in the appended messages, re-renders the full
        message list through the processor and slices out the delta mm_extras.
        Otherwise falls back to text-only incremental merge.
        """
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages):]

        new_images = self._extract_images_from_messages(appended_messages)
        if not new_images:
            # No new images — use text-only incremental merge (inherited)
            appended_ids = self.tokenize_incremental_messages(
                previous_messages, updated_messages, tools=tools
            )
            result = self._merge_token_ids(runtime_token_ids, appended_ids)
            return result

        # New images present — re-render full message list through processor
        all_images = self._extract_images_from_messages(updated_messages)
        prev_images = self._extract_images_from_messages(previous_messages)

        full_token_ids, full_mm_extras = self.render_tokens_with_mm(
            updated_messages, all_images, add_generation_prompt=True, tools=tools
        )

        # Slice mm_extras: only the delta (new images) portion
        delta_mm_extras = self._slice_mm_delta(
            prev_image_count=len(prev_images),
            full_mm_extras=full_mm_extras,
        )

        # Compute appended tokens via prefix diff
        prefix_len = len(runtime_token_ids)
        appended_token_ids = full_token_ids[prefix_len:]

        # Apply boundary handling (newline after im_end from QwenBuilder)
        merge_result = self._merge_token_ids(runtime_token_ids, appended_token_ids)

        # Populate MM fields — only spans for new images to align with delta pixel_values
        all_spans = self.extract_vision_placeholders(merge_result.token_ids)
        image_token_spans = all_spans[len(prev_images):]

        return MergeResult(
            token_ids=merge_result.token_ids,
            appended_token_count=merge_result.appended_token_count,
            kind=merge_result.kind,
            inserted_token_ids=merge_result.inserted_token_ids,
            removed_prefix_token_count=merge_result.removed_prefix_token_count,
            pixel_values=delta_mm_extras.get("pixel_values"),
            image_grid_thw=delta_mm_extras.get("image_grid_thw", []),
            image_token_spans=image_token_spans,
            mm_processor_kwargs=delta_mm_extras.get("mm_processor_kwargs", {}),
        )

    def _slice_mm_delta(
        self,
        prev_image_count: int,
        full_mm_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Slice full mm_extras to only include new (delta) images.

        Uses image_grid_thw length to determine image boundaries in pixel_values.
        """
        grid_thw = full_mm_extras.get("image_grid_thw", [])
        pixel_values = full_mm_extras.get("pixel_values")

        if prev_image_count == 0:
            # All images are new
            return full_mm_extras

        if prev_image_count >= len(grid_thw):
            # No new images in mm_extras (shouldn't happen if we got here)
            return {}

        # Slice grid_thw
        delta_grid_thw = grid_thw[prev_image_count:]

        # Slice pixel_values: dim0 is raw patches (t*h*w), NOT merged tokens
        if pixel_values is not None:
            prev_patch_count = sum(
                row[0] * row[1] * row[2] for row in grid_thw[:prev_image_count]
            )
            if hasattr(pixel_values, "__getitem__"):
                delta_pixel_values = pixel_values[prev_patch_count:]
            else:
                logger.warning(
                    "pixel_values type %s does not support slicing; delta will be None",
                    type(pixel_values).__name__,
                )
                delta_pixel_values = None
        else:
            delta_pixel_values = None

        return {
            "pixel_values": delta_pixel_values,
            "image_grid_thw": delta_grid_thw,
        }


class MiMoVLContinuousTokenBuilder(MiMoContinuousTokenBuilder):
    """Xiaomi MiMo-VL continuous token builder.

    MiMo-VL uses Qwen2_5_VLForConditionalGeneration with identical vision
    token structure. Inherits MiMo's ChatML boundary handling and adds
    the same Tier 1 Wrapper vision logic as QwenVL.

    Kept as independent class for structural clarity.
    """

    def __init__(self, tokenizer: Any, processor: Any, *, model_type: str = "mimo_vl", **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        self.model_type = model_type

        # Vision special token IDs (same as Qwen)
        self._vision_start_id = _require_token_id(tokenizer, "<|vision_start|>")
        self._vision_end_id = _require_token_id(tokenizer, "<|vision_end|>")
        self._image_pad_id = _require_token_id(tokenizer, "<|image_pad|>")

        # Spatial merge size
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "merge_size"):
            self._spatial_merge_size = int(processor.image_processor.merge_size)
        else:
            self._spatial_merge_size = 2

    @classmethod
    def supports_multimodal(cls) -> bool:
        return True

    def count_vision_tokens(self, image_grid_thw_row: tuple[int, int, int]) -> int:
        """Compute image_pad count: t * (h // merge) * (w // merge)."""
        t, h, w = image_grid_thw_row
        merge = self._spatial_merge_size
        return t * (h // merge) * (w // merge)

    def extract_vision_placeholders(self, token_ids: Sequence[int]) -> list[tuple[int, int]]:
        """Find all <|vision_start|>...<|vision_end|> spans."""
        spans: list[tuple[int, int]] = []
        i = 0
        n = len(token_ids)
        while i < n:
            if token_ids[i] == self._vision_start_id:
                j = i + 1
                while j < n and token_ids[j] != self._vision_end_id:
                    j += 1
                if j < n:
                    spans.append((i + 1, j))
                else:
                    logger.warning("Unmatched <|vision_start|> at position %d", i)
                i = j + 1
            else:
                i += 1
        return spans

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        add_generation_prompt: bool = True,
        mm_processor_kwargs: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Render messages through the MiMo-VL processor (Qwen2.5-VL compatible)."""
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools

        flat_messages = self._flatten_multimodal_content(messages)
        text = apply_chat_template(
            self.tokenizer,
            flat_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        proc_kwargs = dict(mm_processor_kwargs or {})
        processor_output = build_multimodal_processor_inputs(
            self.processor,
            text=text,
            images=images if images else None,
            mm_processor_kwargs=proc_kwargs if proc_kwargs else None,
        )

        token_ids = normalize_token_ids(processor_output["input_ids"])

        mm_extras: dict[str, Any] = {}
        if "pixel_values" in processor_output:
            mm_extras["pixel_values"] = processor_output["pixel_values"]
        if "image_grid_thw" in processor_output:
            grid_thw = processor_output["image_grid_thw"]
            if hasattr(grid_thw, "tolist"):
                mm_extras["image_grid_thw"] = [tuple(row) for row in grid_thw.tolist()]
            else:
                mm_extras["image_grid_thw"] = list(grid_thw)
        if proc_kwargs:
            mm_extras["mm_processor_kwargs"] = proc_kwargs

        return token_ids, mm_extras

    def _extract_images_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        """Extract image references from message content blocks."""
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

    def _flatten_multimodal_content(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten list-format content to string with vision placeholders for MiMo-VL's template."""
        flat: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                flat.append(msg)
                continue
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype in ("image", "image_url"):
                    parts.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif btype == "video":
                    parts.append("<|vision_start|><|video_pad|><|vision_end|>")
                elif btype == "text":
                    parts.append(block.get("text", ""))
            flat.append({**msg, "content": "".join(parts)})
        return flat

    def _render_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        flat = self._flatten_multimodal_content(messages)
        return super()._render_tokens(flat, add_generation_prompt=add_generation_prompt, tools=tools)

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Build initial tokens with multimodal processor support."""
        images = self._extract_images_from_messages(messages)
        if not images:
            self._last_mm_extras: dict[str, Any] = {}
            return self._render_tokens(messages, add_generation_prompt=True, tools=tools)

        token_ids, mm_extras = self.render_tokens_with_mm(
            messages, images, add_generation_prompt=True, tools=tools
        )
        self._last_mm_extras = mm_extras
        return token_ids

    def merge_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        """Merge tokens with multimodal awareness."""
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages):]

        new_images = self._extract_images_from_messages(appended_messages)
        if not new_images:
            appended_ids = self.tokenize_incremental_messages(
                previous_messages, updated_messages, tools=tools
            )
            return self._merge_token_ids(runtime_token_ids, appended_ids)

        all_images = self._extract_images_from_messages(updated_messages)
        prev_images = self._extract_images_from_messages(previous_messages)

        full_token_ids, full_mm_extras = self.render_tokens_with_mm(
            updated_messages, all_images, add_generation_prompt=True, tools=tools
        )

        delta_mm_extras = self._slice_mm_delta(
            prev_image_count=len(prev_images),
            full_mm_extras=full_mm_extras,
        )

        prefix_len = len(runtime_token_ids)
        appended_token_ids = full_token_ids[prefix_len:]
        merge_result = self._merge_token_ids(runtime_token_ids, appended_token_ids)
        all_spans = self.extract_vision_placeholders(merge_result.token_ids)
        image_token_spans = all_spans[len(prev_images):]

        return MergeResult(
            token_ids=merge_result.token_ids,
            appended_token_count=merge_result.appended_token_count,
            kind=merge_result.kind,
            inserted_token_ids=merge_result.inserted_token_ids,
            removed_prefix_token_count=merge_result.removed_prefix_token_count,
            pixel_values=delta_mm_extras.get("pixel_values"),
            image_grid_thw=delta_mm_extras.get("image_grid_thw", []),
            image_token_spans=image_token_spans,
            mm_processor_kwargs=delta_mm_extras.get("mm_processor_kwargs", {}),
        )

    def _slice_mm_delta(
        self,
        prev_image_count: int,
        full_mm_extras: dict[str, Any],
    ) -> dict[str, Any]:
        """Slice full mm_extras to only include new (delta) images."""
        grid_thw = full_mm_extras.get("image_grid_thw", [])
        pixel_values = full_mm_extras.get("pixel_values")

        if prev_image_count == 0:
            return full_mm_extras
        if prev_image_count >= len(grid_thw):
            return {}

        delta_grid_thw = grid_thw[prev_image_count:]

        if pixel_values is not None:
            prev_patch_count = sum(
                row[0] * row[1] * row[2] for row in grid_thw[:prev_image_count]
            )
            if hasattr(pixel_values, "__getitem__"):
                delta_pixel_values = pixel_values[prev_patch_count:]
            else:
                logger.warning(
                    "pixel_values type %s does not support slicing; delta will be None",
                    type(pixel_values).__name__,
                )
                delta_pixel_values = None
        else:
            delta_pixel_values = None

        return {
            "pixel_values": delta_pixel_values,
            "image_grid_thw": delta_grid_thw,
        }
