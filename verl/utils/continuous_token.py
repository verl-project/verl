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

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

logger = logging.getLogger(__name__)

_SUPPORTED_APPEND_ROLES = frozenset({"tool", "user", "system"})
_SYNTHETIC_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": "continuous token synthetic system"}
_SYNTHETIC_USER_MESSAGE: dict[str, Any] = {"role": "user", "content": "continuous token synthetic user"}
_ASSISTANT_REASONING_CONTENT: str = "reasoning"
_DUMMY_TOOL_NAME = "continuous_token_tool"
MergeKind = Literal["assistant", "non_assistant"]


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

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
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

        for group in self._iter_append_groups(appended_messages):
            role = group[0].get("role")
            if role == "tool":
                incremental_ids.extend(
                    self._tokenize_tool_group(
                        group,
                        previous_messages=previous_messages,
                        tools=tools,
                    )
                )
            elif role in {"user", "system"}:
                # System appends can represent retry/control messages; unsupported templates will fail in suffix diff.
                if len(group) != 1:
                    raise ValueError(
                        f"Continuous Token expects one {role!r} message per append group, got {len(group)}"
                    )
                incremental_ids.extend(self._tokenize_single_non_tool(group[0], tools=tools))
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")

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
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        synthetic_assistant = self._synthetic_assistant_for_tools(tool_messages)
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
        raise NotImplementedError(f"{type(self).__name__} does not implement extract_vision_placeholders.")

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        add_generation_prompt: bool = True,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> list[int]:
        """Render messages with images through the processor.

        Unlike ``_render_tokens`` which uses only the tokenizer, this method
        invokes the full multimodal processor so image placeholders are expanded
        into the same token IDs the rollout backend will consume.

        Args:
            messages: OpenAI-format message list with image content items.
            images: List of PIL images (or paths), one per image content item.
            add_generation_prompt: Whether to append the generation prompt.
            mm_processor_kwargs: Extra kwargs for the processor (min/max pixels).

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

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
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
        self._im_end_id = _require_token_id(tokenizer, "<|im_end|>")

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
        self._observation_id = _require_token_id(tokenizer, "<|observation|>")
        self._user_id = _require_token_id(tokenizer, "<|user|>")
        self._ambiguous_boundary_ids = {self._observation_id, self._user_id}

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


class Gemma4ContinuousTokenBuilder(ContinuousTokenBuilder):
    """Gemma4 tool-response boundary handling."""

    def __init__(self, tokenizer: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self._tool_response_id = _require_token_id(tokenizer, "<|tool_response>")

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        del tools
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
        return f'<|tool_response>response:{tool_name}{{value:<|"|>{content}<|"|>}}<tool_response|>'

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


def _token_suffix_after_prefix(
    prefix_token_ids: Sequence[int],
    full_token_ids: Sequence[int],
    *,
    context: str,
) -> list[int]:
    """Return the token suffix after asserting the prefix is unchanged."""
    prefix = list(prefix_token_ids)
    full = list(full_token_ids)
    if full[: len(prefix)] != prefix:
        raise ValueError(f"Continuous Token token-id suffix diff failed for {context}")
    return full[len(prefix) :]


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


# =============================================================================
# Multimodal (VL) subclasses — Phase 1
# =============================================================================


class VLContinuousTokenMixin:
    """Shared processor-backed logic for vision-language continuous token builders.

    Provides the multimodal workflow (image extraction, processor rendering,
    incremental dummy+trim encoding) common to all VL builders. Subclasses
    combine this mixin with a text-family builder (e.g. QwenContinuousTokenBuilder)
    via Python MRO so that boundary handling like Qwen's newline insertion or
    GLM's observation/user trim still applies through ``_merge_non_assistant_token_ids``.

    Subclasses must define class attributes:
        vision_start_token: str — e.g. "<|vision_start|>"
        vision_end_token: str — e.g. "<|vision_end|>"
        merge_size_attr: str = "merge_size" — processor.image_processor attribute name

    Optional hooks:
        _prepare_mm_messages(messages) — preprocess messages (e.g. flatten content)
    """

    vision_start_token: str = ""
    vision_end_token: str = ""
    merge_size_attr: str = "merge_size"

    def __init__(self, tokenizer: Any, processor: Any, **kwargs: Any):
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        self._vision_start_id = _require_token_id(tokenizer, self.vision_start_token)
        self._vision_end_id = _require_token_id(tokenizer, self.vision_end_token)
        self._spatial_merge_size = self._resolve_spatial_merge_size(processor)

    def _resolve_spatial_merge_size(self, processor: Any) -> int:
        ip = getattr(processor, "image_processor", None)
        if ip is None:
            return 2
        value = getattr(ip, self.merge_size_attr, None)
        if value is None:
            return 2
        if isinstance(value, (list, tuple)):
            value = value[0]
        return int(value)

    @classmethod
    def supports_multimodal(cls) -> bool:
        return True

    def count_vision_tokens(self, image_grid_thw_row: tuple[int, int, int]) -> int:
        t, h, w = image_grid_thw_row
        merge = self._spatial_merge_size
        return t * (h // merge) * (w // merge)

    def extract_vision_placeholders(self, token_ids: Sequence[int]) -> list[tuple[int, int]]:
        """Find all <vision_start>...<vision_end> spans (exclusive of markers)."""
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
                    logger.warning(
                        "Unmatched %s at position %d", self.vision_start_token, i
                    )
                i = j + 1
            else:
                i += 1
        return spans

    def _prepare_mm_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Hook for subclass message preprocessing. Default: pass through."""
        return messages

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

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        add_generation_prompt: bool = True,
        mm_processor_kwargs: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Render messages through the processor (full render with all images)."""
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools

        prepared = self._prepare_mm_messages(messages)
        text = apply_chat_template(
            self.tokenizer, prepared, tokenize=False,
            add_generation_prompt=add_generation_prompt, **template_kwargs,
        )

        proc_kwargs = dict(mm_processor_kwargs or {})
        processor_output = build_multimodal_processor_inputs(
            self.processor, text=text, images=images if images else None,
            mm_processor_kwargs=proc_kwargs if proc_kwargs else None,
        )
        return normalize_token_ids(processor_output["input_ids"])

    def _render_incremental_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Render incremental messages via dummy+trim (single processor call)."""
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools

        prefix_msgs = [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE]
        prefix_text = apply_chat_template(
            self.tokenizer, prefix_msgs, tokenize=False,
            add_generation_prompt=False, **template_kwargs,
        )
        prefix_token_ids = normalize_token_ids(
            self.tokenizer.encode(prefix_text, add_special_tokens=False)
        )

        prepared = self._prepare_mm_messages(messages)
        full_text = apply_chat_template(
            self.tokenizer, prefix_msgs + prepared, tokenize=False,
            add_generation_prompt=True, **template_kwargs,
        )
        processor_output = build_multimodal_processor_inputs(
            self.processor, text=full_text, images=images if images else None,
        )

        all_ids = normalize_token_ids(processor_output["input_ids"])
        return _token_suffix_after_prefix(
            prefix_token_ids, all_ids,
            context="multimodal synthetic prefix",
        )

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        images = self._extract_images_from_messages(messages)
        if not images:
            return self._render_tokens(messages, add_generation_prompt=True, tools=tools)
        return self.render_tokens_with_mm(messages, images, add_generation_prompt=True, tools=tools)

    def merge_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        """Merge tokens with multimodal awareness.

        If new images appear, uses single processor call (dummy+trim) to get
        incremental token_ids, then applies text-family boundary handling via
        ``_merge_non_assistant_token_ids`` (provided by parent class through MRO).
        """
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages):]

        new_images = self._extract_images_from_messages(appended_messages)
        if not new_images:
            appended_ids = self.tokenize_non_assistant_incremental_messages(
                previous_messages, updated_messages, tools=tools
            )
            return self._merge_non_assistant_token_ids(runtime_token_ids, appended_ids)

        appended_token_ids = self._render_incremental_with_mm(
            appended_messages, new_images, tools=tools
        )
        return self._merge_non_assistant_token_ids(runtime_token_ids, appended_token_ids)


class QwenVLContinuousTokenBuilder(VLContinuousTokenMixin, QwenContinuousTokenBuilder):
    """Qwen Vision-Language: Qwen ChatML newline patch + VL processor logic.

    Handles Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and Qwen3-VL-MoE.
    """

    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"
    merge_size_attr = "merge_size"


class MiMoVLContinuousTokenBuilder(VLContinuousTokenMixin, QwenContinuousTokenBuilder):
    """MiMo-VL: shares Qwen2.5-VL architecture, but template needs content flattening.

    The MiMo chat template only handles string content, so list-format multimodal
    content blocks must be flattened into placeholder strings before rendering.
    """

    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"
    merge_size_attr = "merge_size"

    def _prepare_mm_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._flatten_multimodal_content(messages)

    def _flatten_multimodal_content(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert list-content to string with vision placeholders for MiMo-VL template."""
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


class GLM4VContinuousTokenBuilder(VLContinuousTokenMixin, GLMContinuousTokenBuilder):
    """GLM-4V / GLM-4.5-VL: GLM observation/user trim + VL processor logic."""

    vision_start_token = "<|begin_of_image|>"
    vision_end_token = "<|end_of_image|>"
    merge_size_attr = "merge_size"


class KimiVLContinuousTokenBuilder(VLContinuousTokenMixin, ContinuousTokenBuilder):
    """Kimi-VL (MoonViT): direct concatenation + VL processor logic.

    Uses <|media_start|>/<|media_end|> wrappers and merge_kernel_size attribute.
    """

    vision_start_token = "<|media_start|>"
    vision_end_token = "<|media_end|>"
    merge_size_attr = "merge_kernel_size"
