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

from dataclasses import dataclass, field
from typing import Any, Literal

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

        for group in self._iter_append_groups(appended_messages):
            role = group[0].get("role")
            if role == "tool":
                incremental_ids.extend(self._tokenize_tool_group(group, tools=tools))
            elif role in {"user", "system"}:
                # System appends can represent retry/control messages; unsupported templates will fail in suffix diff.
                incremental_ids.extend(self._tokenize_single_non_tool(group[0], tools=tools))
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")

        incremental_ids.extend(self.render_delta_token_id(updated_messages, [], add_generation_prompt=True, tools=tools))
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

    def _synthetic_assistant_for_tools(self, tool_messages: list[dict[str, Any]]) -> dict[str, Any]:
        tool_calls = []
        for tool_message in tool_messages:
            tool_call = {
                "type": "function",
                "function": {
                    "name": tool_message.get("name") or "continuous_token_tool",
                    "arguments": {},
                },
            }
            if tool_message.get("tool_call_id") is not None:
                tool_call["id"] = tool_message["tool_call_id"]
            tool_calls.append(tool_call)
        return {"role": "assistant", "content": "", "reasoning_content": _ASSISTANT_REASONING_CONTENT, "tool_calls": tool_calls}


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
