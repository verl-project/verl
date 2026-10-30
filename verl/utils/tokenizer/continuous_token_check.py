# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Check that Continuous Token merges reproduce the chat template at turn boundaries.

Continuous Token keeps the generated token ids and appends each tool/user turn as a
token delta that is rendered against a synthetic context and joined to the runtime
stream by a model-specific boundary hook. Nothing at runtime compares the merged
stream with what the chat template renders for the real conversation, so a builder
that does not match the template (for example a ``model_type`` that falls back to the
default builder and drops the turn separator) trains on token sequences the template
never produces, without any error. This module replays a few fixed conversations
through a builder and compares every merge boundary with a single-pass render.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from .continuous_token import ContinuousTokenBuilder, MergeResult

logger = logging.getLogger(__name__)

SANITY_CHECK_MODES = ("strict", "ignore_strippable", "disable")

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get today's weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name."}},
                "required": ["city"],
            },
        },
    }
]


def _tool_call(call_id: str, city: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": {"city": city}}}


def _tool_result(call_id: str, city: str, condition: str) -> dict[str, Any]:
    content = json.dumps({"city": city, "condition": condition})
    return {"role": "tool", "tool_call_id": call_id, "name": "get_weather", "content": content}


_SYSTEM = {"role": "system", "content": "You are a concise weather assistant."}
_USER = {"role": "user", "content": "Which city has the best weather today: Seattle, Portland or Boise?"}
_FIRST_ASSISTANT = {
    "role": "assistant",
    "content": "I will check Seattle first.",
    "tool_calls": [_tool_call("call_0", "Seattle")],
}

# Every assistant turn is followed by the context run an agent loop would append after it.
_CHECK_CONVERSATIONS: dict[str, tuple[dict[str, Any], ...]] = {
    "tool": (
        _SYSTEM,
        _USER,
        _FIRST_ASSISTANT,
        _tool_result("call_0", "Seattle", "rain"),
        {
            "role": "assistant",
            "content": "Now Portland and Boise.",
            "tool_calls": [_tool_call("call_1", "Portland"), _tool_call("call_2", "Boise")],
        },
        _tool_result("call_1", "Portland", "cloudy"),
        _tool_result("call_2", "Boise", "sunny"),
    ),
    "user": (
        _SYSTEM,
        _USER,
        _FIRST_ASSISTANT,
        {"role": "user", "content": "The weather tool is down. Answer from what you know."},
        {"role": "assistant", "content": "Boise is usually the sunniest of the three."},
        {"role": "user", "content": "Thanks. What about tomorrow?"},
    ),
}
_REASONING = "The user wants the city with the best weather."


@dataclass(frozen=True)
class BoundaryResult:
    """Outcome of one merge boundary.

    ``status`` is ``"match"`` when the merged stream equals the single-pass render,
    ``"history"`` when only turns before the boundary differ (the template re-renders
    past turns), ``"boundary"`` when the tokens the builder produced at the boundary
    differ, and ``"skipped"`` when the template or builder could not render the case.
    """

    case: str
    status: Literal["match", "history", "boundary", "skipped"]
    detail: str = ""
    whitespace_only: bool = False


def check_continuous_token_builder(
    builder: ContinuousTokenBuilder, *, stop_token_ids: set[int]
) -> list[BoundaryResult]:
    """Replay the check conversations through ``builder`` and classify every merge boundary.

    For each assistant turn followed by a context run, the runtime stream is simulated
    as the single-pass prompt plus the rendered assistant turn cut after the model's
    stop token, the builder merges the context run into it, and the result is compared
    with a single-pass render of the conversation through that run.
    """
    results = []
    for name, conversation in _check_cases():
        for start, end in _merge_boundaries(conversation):
            roles = "_".join(message["role"] for message in conversation[start:end])
            case = f"{name}.merge_at{start}.{roles}"
            results.append(_check_boundary(builder, conversation, start, end, stop_token_ids, case))
    return results


def verify_continuous_token_builder(
    builder: ContinuousTokenBuilder, *, mode: str, hf_model_type: str | None, generation_config: Any | None
) -> None:
    """Raise when ``builder`` disagrees with the chat template at a turn boundary.

    ``mode`` is ``actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode``.
    Differences confined to earlier turns are expected for templates that rewrite
    history and only warn.
    """
    if mode not in SANITY_CHECK_MODES:
        raise ValueError(f"tokenization_sanity_check_mode must be one of {SANITY_CHECK_MODES}, got {mode!r}")
    if mode == "disable":
        return

    stop_token_ids = _stop_token_ids(builder.tokenizer, generation_config)
    results = check_continuous_token_builder(builder, stop_token_ids=stop_token_ids)
    builder_desc = f"{type(builder).__name__} (config.json model_type={hf_model_type!r})"

    mismatches = [result for result in results if result.status == "boundary"]
    if mode == "ignore_strippable":
        tolerated = [result for result in mismatches if result.whitespace_only]
        mismatches = [result for result in mismatches if not result.whitespace_only]
        if tolerated:
            logger.warning(
                "Continuous Token builder %s differs from the chat template in whitespace at multi-turn "
                "boundaries (tolerated by ignore_strippable):\n%s",
                builder_desc,
                _format_results(tolerated),
            )
    if mismatches:
        raise ValueError(
            f"Continuous Token builder {builder_desc} does not reproduce the chat template at multi-turn "
            "boundaries, so multi-turn rollouts train on token sequences the template never produces:\n"
            f"{_format_results(mismatches)}\n"
            "Register a builder for this model_type in verl/utils/tokenizer/continuous_token_wiring.py, or set "
            "actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable to run anyway."
        )

    history = [result for result in results if result.status == "history"]
    if history:
        logger.warning(
            "The chat template re-renders earlier turns once later messages are appended (for example it drops "
            "<think> blocks from history). Continuous Token keeps the generated tokens, so multi-turn rollouts "
            "train on contexts that differ from a single-pass render of the same conversation:\n%s",
            _format_results(history[:1]),
        )

    compared = sum(result.status != "skipped" for result in results)
    if compared == 0:
        logger.warning(
            "Could not verify Continuous Token builder %s against the chat template:\n%s",
            builder_desc,
            _format_results(results),
        )
    else:
        logger.info(
            "Checked Continuous Token builder %s against the chat template at %d multi-turn boundaries "
            "(%d exact matches, %d could not be rendered).",
            builder_desc,
            compared,
            sum(result.status == "match" for result in results),
            len(results) - compared,
        )


def _check_cases():
    """Yield each check conversation as written and with ``reasoning_content`` on its assistant turns.

    Thinking templates render an assistant turn consistently with their generation prompt
    (which opens a reasoning block) only when the turn carries reasoning, while
    non-thinking templates and hybrid templates with thinking disabled need turns without it.
    """
    for name, conversation in _CHECK_CONVERSATIONS.items():
        yield name, conversation
        yield (
            f"{name}_reasoning",
            tuple(
                {**message, "reasoning_content": _REASONING} if message["role"] == "assistant" else message
                for message in conversation
            ),
        )


def _merge_boundaries(messages: tuple[dict[str, Any], ...]) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of the non-assistant runs that follow an assistant turn."""
    roles = [message["role"] for message in messages]
    boundaries = []
    for start in range(1, len(roles)):
        if roles[start - 1] == "assistant" and roles[start] != "assistant":
            end = next((index for index in range(start, len(roles)) if roles[index] == "assistant"), len(roles))
            boundaries.append((start, end))
    return boundaries


def _check_boundary(
    builder: ContinuousTokenBuilder,
    conversation: tuple[dict[str, Any], ...],
    start: int,
    end: int,
    stop_token_ids: set[int],
    case: str,
) -> BoundaryResult:
    # Renderers may mutate messages (e.g. parse tool-call arguments), so each case gets its own copy.
    messages = json.loads(json.dumps(conversation))
    try:
        # Single-pass renders go through the builder's own first-prompt path, which builders with a
        # custom encoder or processor override; only the assistant turn needs the raw template.
        prompt_ids = builder.build_initial_tokens(messages[: start - 1], tools=_TOOLS)
        turn_ids = builder._render_tokens(messages[:start], add_generation_prompt=False, tools=_TOOLS)
        expected_ids = builder.build_initial_tokens(messages[:end], tools=_TOOLS)
        runtime_ids = _simulate_generation(prompt_ids, turn_ids, expected_ids, stop_token_ids)
        merge_result = builder.merge_context_tokens(messages[:start], messages[:end], runtime_ids, tools=_TOOLS)
    except Exception as exc:
        return BoundaryResult(case, "skipped", f"{type(exc).__name__}: {exc}")
    return _classify(builder.tokenizer, case, runtime_ids, merge_result, expected_ids)


def _simulate_generation(
    prompt_ids: list[int], turn_ids: list[int], expected_ids: list[int], stop_token_ids: set[int]
) -> list[int]:
    """Return the runtime stream after the model generated the assistant turn and stopped.

    Generation stops at the stop token, so template text rendered after it (such as
    Qwen's trailing newline) is never part of the runtime stream.
    """
    if turn_ids[: len(prompt_ids)] == prompt_ids:
        for index in range(len(turn_ids) - 1, len(prompt_ids) - 1, -1):
            if turn_ids[index] in stop_token_ids:
                return turn_ids[: index + 1]
        # Templates such as GLM's close an assistant turn with the next message's role token,
        # which the model emits as its stop token.
        next_ids = expected_ids[len(turn_ids) : len(turn_ids) + 1]
        if expected_ids[: len(turn_ids)] == turn_ids and next_ids and next_ids[0] in stop_token_ids:
            return turn_ids + next_ids
    raise ValueError("could not locate the assistant stop token in the template render")


def _classify(
    tokenizer: Any, case: str, runtime_ids: list[int], merge_result: MergeResult, expected_ids: list[int]
) -> BoundaryResult:
    merged_ids = merge_result.token_ids
    if merged_ids == expected_ids:
        return BoundaryResult(case, "match")

    # Anchor on the last runtime token the merge kept: from there on, the stream is what
    # the builder produced for the junction and the appended messages, so it must be the
    # tail of the single-pass render even when the template re-renders earlier turns.
    anchor = max(len(runtime_ids) - merge_result.removed_prefix_token_count - 1, 0)
    tail = merged_ids[anchor:]
    offset = max(len(expected_ids) - len(tail), 0)
    if expected_ids[offset:] == tail:
        index = _first_difference(expected_ids, merged_ids)
        return BoundaryResult(case, "history", _describe(tokenizer, expected_ids, index, merged_ids, index))

    index = _first_difference(expected_ids[offset:], tail)
    whitespace_only = _squash(tokenizer.decode(expected_ids)).endswith(_squash(tokenizer.decode(tail)))
    detail = _describe(tokenizer, expected_ids, offset + index, merged_ids, anchor + index)
    return BoundaryResult(case, "boundary", detail, whitespace_only)


def _stop_token_ids(tokenizer: Any, generation_config: Any | None) -> set[int]:
    stop_token_ids: set[int] = set()
    for eos in (getattr(generation_config, "eos_token_id", None), getattr(tokenizer, "eos_token_id", None)):
        if isinstance(eos, int):
            stop_token_ids.add(eos)
        elif isinstance(eos, list | tuple):
            stop_token_ids.update(token_id for token_id in eos if isinstance(token_id, int))
    return stop_token_ids


def _first_difference(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    return next((index for index in range(limit) if left[index] != right[index]), limit)


def _squash(text: str) -> str:
    return "".join(text.split())


def _describe(
    tokenizer: Any, expected_ids: list[int], expected_index: int, actual_ids: list[int], actual_index: int
) -> str:
    def window(ids: list[int], index: int) -> str:
        return repr(tokenizer.decode(ids[max(index - 12, 0) : index + 12]))

    return f"template: {window(expected_ids, expected_index)}\n    builder:  {window(actual_ids, actual_index)}"


def _format_results(results: list[BoundaryResult]) -> str:
    return "\n".join(f"  {result.case}:\n    {result.detail}" for result in results)
