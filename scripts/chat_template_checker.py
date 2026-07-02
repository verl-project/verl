#!/usr/bin/env python3
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
"""Check chat-template append-only.

The checker runs the mock trajectories in
``scripts.chat_template_mock_trajectories`` through two layers:

1. Check 1: raw template prefix diagnostics at token-id level. This is a quick checker that
   indicates whether applying the raw chat template to a prefix produces token
   IDs that remain a prefix after later messages are rendered.

   Note: Failures in this diagnostic are warnings, not final verdict failures.
   Continuous Token does not strictly require the model chat template to be
   globally append-only. A raw diagnostic warning means users should check
   whether non-assistant incremental messages are still append-only under the
   builder's dummy context. The default dummy message construction in
   ContinuousTokenBuilder is designed for this; for example, a non-empty
   reasoning_content in synthetic assistant messages can make Qwen3-style
   templates append-only for the incremental non-assistant extraction step even
   when the original full conversation template is not globally append-only.

2. Check 2: production-shaped Continuous Token builder checks at token level. This layer
   verifies the builder's merge-boundary handling one boundary at a time. For
   each point where a non-assistant run (tool/user/system) follows an assistant
   turn, it renders the prefix up to and including that assistant turn through
   the chat template, trims it to the runtime stop shape (as if generation
   stopped at EOS), then drives the builder's merge_non_assistant_tokens over the
   appended run. The merged token IDs must match a single-pass render of the same
   messages with a trailing generation prompt. Because the prefix (including
   assistant turns) is rendered directly, assistant output tokens are never
   reconstructed, so a template that is not append-only surfaces as a plain token
   mismatch instead of an error. Trajectories with no non-assistant run after the
   assistant turn (single-turn chat) have no boundary and are covered by Check 1.

TODO(@gxlvera (Xiaole Guo)): add coverage for trajectories carrying
reasoning_content.

Examples:

    python scripts/chat_template_checker.py --model Qwen/Qwen3-0.6B
    python scripts/chat_template_checker.py --model zai-org/GLM-4.7-Flash --allow-download
    python scripts/chat_template_checker.py --model Qwen/Qwen3-0.6B --template /path/to/chat_template.jinja
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.chat_template_mock_trajectories import (  # noqa: E402
    TRAJECTORIES,
    MockTrajectory,
    SingleTurnTrajectory,
    ToolAgentTrajectory,
    VLMockTrajectory,
    VLSingleTurnTrajectory,
    VLToolTrajectory,
    build_vl_trajectories,
)
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids  # noqa: E402
from verl.utils.tokenizer.chat_template import apply_chat_template  # noqa: E402
from verl.utils.tokenizer.continuous_token_wiring import (  # noqa: E402
    CONTINUOUS_TOKEN_BUILDER_FAMILIES,
    create_continuous_token_builder,
    get_continuous_token_builder_class,
    resolve_continuous_token_model_family,
)

CheckLayer = Literal["raw-template", "continuous-token"]


@dataclass(frozen=True)
class CheckResult:
    layer: CheckLayer
    case_name: str
    passed: bool
    error: str | None = None


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _set_pad_token(tokenizer) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_tokenizer(model: str, *, local_files_only: bool, template_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=local_files_only,
        use_fast=True,
    )
    _set_pad_token(tokenizer)
    if template_path:
        tokenizer.chat_template = Path(template_path).read_text()
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat_template")
    return tokenizer


def _tools_for(trajectory: MockTrajectory) -> list[dict[str, Any]] | None:
    if isinstance(trajectory, ToolAgentTrajectory):
        return trajectory.tool_schemas()
    return None


def _initial_messages(trajectory: MockTrajectory) -> list[dict[str, Any]]:
    return [_clone(message) for message in trajectory.raw_prompt]


def _assistant_message_for_single_turn(trajectory: SingleTurnTrajectory) -> dict[str, Any]:
    return {"role": "assistant", "content": trajectory.assistant_response}


def _render_tokens(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    tokenized = apply_chat_template(
        tokenizer,
        _clone(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        tools=_clone(tools),
        **chat_template_kwargs,
    )
    return normalize_token_ids(tokenized)


def _render_ids(
    tokenizer,
    processor,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    """Render ground-truth token IDs using the renderer the *model* uses.

    Whether rendering goes through the processor or the bare tokenizer is a
    property of the model, not of the trajectory: a VL model (processor present)
    always renders through its processor chat template + image expansion — the
    same path its Continuous Token builder and the rollout backend use — while a
    text model renders through the tokenizer. This keeps the ground truth aligned
    with the builder for both text-only and image-carrying trajectories.
    """
    if processor is not None:
        return _render_processor_ids(
            processor,
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
    return _render_tokens(
        tokenizer,
        messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs=chat_template_kwargs,
    )


def _token_repr(tokenizer, token_id: int | None) -> str:
    if tokenizer is None or token_id is None:
        return "?"
    try:
        return repr(tokenizer.decode([token_id]))
    except Exception:
        return "?"


def _diff_context(
    tokenizer,
    left_ids: list[int],
    right_ids: list[int],
    mismatch: int,
    *,
    left_label: str,
    right_label: str,
    radius: int = 16,
) -> str:
    """Render a decoded window around the first diverging token, so a failure
    shows human-readable text (e.g. ``<think>`` vs ``<tool_call>``) rather than
    bare token IDs. Returns an empty string when no decoder is available."""
    if tokenizer is None:
        return ""
    start = max(0, mismatch - radius)
    stop = mismatch + radius
    label_width = max(len(left_label), len(right_label))
    try:
        left_text = tokenizer.decode(left_ids[start:stop])
        right_text = tokenizer.decode(right_ids[start:stop])
    except Exception:
        return ""
    return (
        f"\n    {left_label:<{label_width}} [{start}:{stop}]: {left_text!r}"
        f"\n    {right_label:<{label_width}} [{start}:{stop}]: {right_text!r}"
    )


def _token_prefix_error(prefix_ids: list[int], full_ids: list[int], *, tokenizer=None) -> str | None:
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return None
    limit = min(len(prefix_ids), len(full_ids))
    mismatch = next((idx for idx in range(limit) if prefix_ids[idx] != full_ids[idx]), limit)
    prefix_value = prefix_ids[mismatch] if mismatch < len(prefix_ids) else None
    full_value = full_ids[mismatch] if mismatch < len(full_ids) else None
    summary = (
        f"Token prefix mismatch at index {mismatch}: "
        f"prefix_len={len(prefix_ids)}, full_len={len(full_ids)}, "
        f"prefix={prefix_value} ({_token_repr(tokenizer, prefix_value)}), "
        f"full={full_value} ({_token_repr(tokenizer, full_value)})"
    )
    return summary + _diff_context(tokenizer, prefix_ids, full_ids, mismatch, left_label="prefix", right_label="full")


def _token_mismatch_error(expected: list[int], actual: list[int], *, tokenizer=None) -> str | None:
    if actual == expected:
        return None
    limit = min(len(expected), len(actual))
    mismatch = next((idx for idx in range(limit) if expected[idx] != actual[idx]), limit)
    expected_value = expected[mismatch] if mismatch < len(expected) else None
    actual_value = actual[mismatch] if mismatch < len(actual) else None
    summary = (
        f"Token mismatch at index {mismatch}: "
        f"expected_len={len(expected)}, actual_len={len(actual)}, "
        f"expected={expected_value} ({_token_repr(tokenizer, expected_value)}), "
        f"actual={actual_value} ({_token_repr(tokenizer, actual_value)})"
    )
    return summary + _diff_context(tokenizer, expected, actual, mismatch, left_label="expected", right_label="actual")


def _tokenizer_name(tokenizer) -> str:
    name = getattr(tokenizer, "name_or_path", None)
    if name:
        return str(name)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict) and init_kwargs.get("name_or_path"):
        return str(init_kwargs["name_or_path"])
    return ""


def _is_glm_tokenizer(tokenizer) -> bool:
    tokenizer_name = _tokenizer_name(tokenizer).lower()
    compact_name = "".join(char for char in tokenizer_name if char.isalnum())
    # GLM text families (GLM-4.7 / GLM-5) and the GLM-4V vision families
    # (GLM-4V / GLM-4.1V / GLM-4.5V, a.k.a. GLM-4.1-VL / GLM-4.5-VL) share the
    # same assistant-turn stop shape: the full chat-template render omits a
    # trailing EOS after assistant content, and boundaries use
    # ``<|observation|>`` / ``<|user|>``.
    return any(
        marker in tokenizer_name
        for marker in (
            "glm-4.7",
            "glm_4.7",
            "glm-5",
            "glm_5",
            "glm-4v",
            "glm-4.1v",
            "glm-4.5v",
            "glm-4.6v",
            "glm-4.1-vl",
            "glm-4.5-vl",
            "glm-4.6-vl",
        )
    ) or any(marker in compact_name for marker in ("glm47", "glm5", "glm4v", "glm41v", "glm45v", "glm46v"))


def _warn_glm_tool_template_limitations(model_name: str) -> None:
    """Warn about GLM vision templates that cannot faithfully render tool turns.

    - GLM-4V / GLM-4.1V templates have no ``tool`` role branch at all, so any
      ``role: "tool"`` message is silently dropped. Multi-turn tool trajectories
      are therefore unrepresentable and any tool-based check is meaningless.
    - GLM-4.5V handles a ``tool`` role, but its list-content branch stringifies
      each part (``{{ tr.output if tr.output is defined else tr }}``) instead of
      expanding images, so an image-bearing tool response leaks the live PIL
      object ``repr`` (with a non-deterministic memory address) into the text and
      breaks append-only. GLM-4.6V fixed this (proper ``<|image|>`` expansion).
    """
    name = model_name.lower()
    compact = "".join(char for char in name if char.isalnum())
    is_glm4v = any(m in name for m in ("glm-4v", "glm-4.1v", "glm-4.1-vl")) or any(
        m in compact for m in ("glm4v", "glm41v")
    )
    is_glm45v = ("glm-4.5v" in name) or ("glm-4.5-vl" in name) or ("glm45v" in compact)
    if is_glm4v:
        print(
            "WARNING: GLM-4V / GLM-4.1V chat templates have no 'tool' role branch, so tool "
            "messages are silently dropped. Multi-turn tool trajectories (with or without images) "
            "cannot be rendered correctly for this model; use GLM-4.6V for tool + image support."
        )
    elif is_glm45v:
        print(
            "WARNING: GLM-4.5V handles the 'tool' role but does not expand images inside tool "
            "responses (list parts are stringified), so an image-bearing tool response leaks a "
            "non-deterministic PIL repr and breaks append-only. Multi-turn tool trajectories with "
            "images will fail for this model; use GLM-4.6V for tool + image support."
        )


def _tokenizer_eos_token_ids(tokenizer) -> set[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, list | tuple | set):
        return {int(token_id) for token_id in eos_token_id if token_id is not None}
    raise TypeError(f"Unsupported eos_token_id type: {type(eos_token_id)!r}")


def _require_single_token_id(tokenizer, token: str) -> int:
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


def _truncate_after_final_eos(
    tokenizer,
    token_ids: list[int],
    *,
    has_tool_calls: bool,
) -> list[int]:
    """Approximate the runtime token stream returned by generation.

    Full chat-template renders can include template whitespace after the model's
    stop token. The real rollout server usually stops at EOS/stop, so the CT
    boundary check trims to that shape before appending non-assistant messages.

    ``has_tool_calls`` is passed explicitly rather than read off the assistant
    message because VL templates embed tool calls as raw assistant text (the
    structured ``tool_calls`` field is dropped), so the message alone cannot tell
    us whether the turn was a tool call.
    """

    eos_token_ids = _tokenizer_eos_token_ids(tokenizer)
    eos_token_ids.update(_kimi_assistant_end_token_ids(tokenizer))
    if not token_ids:
        raise ValueError("Assistant output token-id suffix is empty")

    # GLM assistant turns are not terminated by EOS in the template render; the
    # canonical stop after a tool call is <|observation|>, otherwise the next
    # role token is the boundary. Mirror real GLM rollout by appending
    # <|observation|> for tool-call turns.
    if _is_glm_tokenizer(tokenizer):
        if has_tool_calls:
            return token_ids + [_require_single_token_id(tokenizer, "<|observation|>")]
        return token_ids

    if eos_token_ids:
        if token_ids[-1] in eos_token_ids:
            return token_ids
        for index in range(len(token_ids) - 1, -1, -1):
            if token_ids[index] in eos_token_ids:
                return token_ids[: index + 1]

    raise ValueError(
        "Assistant output token-id suffix does not contain eos_token_id "
        f"{sorted(eos_token_ids)}; tail={token_ids[-16:]}"
    )


def _record_raw_prefix_check(
    results: list[CheckResult],
    *,
    case_name: str,
    tokenizer,
    processor,
    prefix_messages: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    prefix_add_generation_prompt: bool,
    full_add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
) -> None:
    try:
        prefix_ids = _render_ids(
            tokenizer,
            processor,
            prefix_messages,
            tools=tools,
            add_generation_prompt=prefix_add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        full_ids = _render_ids(
            tokenizer,
            processor,
            full_messages,
            tools=tools,
            add_generation_prompt=full_add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        error = _token_prefix_error(prefix_ids, full_ids, tokenizer=tokenizer)
        results.append(CheckResult("raw-template", case_name, error is None, error))
    except Exception as exc:
        results.append(CheckResult("raw-template", case_name, False, f"{type(exc).__name__}: {exc}"))


def run_raw_template_checks(
    tokenizer,
    trajectory: MockTrajectory,
    *,
    processor: Any | None = None,
    chat_template_kwargs: dict[str, Any],
) -> list[CheckResult]:
    """Run direct chat-template prefix diagnostics without Continuous Token logic.

    Each case renders the trajectory boundary twice with ``tokenize=True``:
    first as the current prefix, then as the later full message list. The check
    passes only when the prefix token IDs are exactly a prefix of the full token
    IDs. This is a raw append-only smoke test for the template itself. Failures
    are diagnostic because this does not exercise the builder's dummy contexts
    or boundary merge patches.
    """

    results: list[CheckResult] = []
    tools = _tools_for(trajectory)
    messages = _initial_messages(trajectory)

    if isinstance(trajectory, SingleTurnTrajectory):
        assistant = _assistant_message_for_single_turn(trajectory)
        _record_raw_prefix_check(
            results,
            case_name=f"{trajectory.name}.assistant_turn1",
            tokenizer=tokenizer,
            processor=processor,
            prefix_messages=messages,
            full_messages=messages + [assistant],
            prefix_add_generation_prompt=True,
            full_add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        return results

    for turn_index, step in enumerate(trajectory.steps, start=1):
        assistant = _clone(step.assistant)
        messages_with_assistant = messages + [assistant]
        _record_raw_prefix_check(
            results,
            case_name=f"{trajectory.name}.assistant_turn{turn_index}",
            tokenizer=tokenizer,
            processor=processor,
            prefix_messages=messages,
            full_messages=messages_with_assistant,
            prefix_add_generation_prompt=True,
            full_add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        appended_messages = [_clone(message) for message in step.appended_messages]
        if appended_messages:
            roles = "_".join(message.get("role", "unknown") for message in appended_messages)
            _record_raw_prefix_check(
                results,
                case_name=f"{trajectory.name}.append_turn{turn_index}.{roles}",
                tokenizer=tokenizer,
                processor=processor,
                prefix_messages=messages_with_assistant,
                full_messages=messages_with_assistant + appended_messages,
                prefix_add_generation_prompt=False,
                full_add_generation_prompt=True,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
        messages = messages_with_assistant + appended_messages

    return results


def _append_ct_result(
    results: list[CheckResult],
    *,
    case_name: str,
    expected_ids: list[int],
    actual_ids: list[int],
    tokenizer=None,
) -> None:
    error = _token_mismatch_error(expected_ids, actual_ids, tokenizer=tokenizer)
    results.append(CheckResult("continuous-token", case_name, error is None, error))


def _create_builder_or_error(
    tokenizer,
    *,
    trajectory_name: str,
    model: str,
    model_family: str,
    custom_builder_module: str | None,
    chat_template_kwargs: dict[str, Any],
    processor: Any | None,
):
    """Instantiate the CT builder, returning ``(builder, None)`` or ``(None, error)``."""
    try:
        if custom_builder_module:
            importlib.import_module(custom_builder_module)
        builder = create_continuous_token_builder(
            tokenizer,
            model_family=model_family,
            model_path=model,
            tokenizer_name_or_path=model,
            chat_template_kwargs=chat_template_kwargs,
            processor=processor,
        )
        return builder, None
    except Exception as exc:
        return None, CheckResult(
            "continuous-token", f"{trajectory_name}.builder", False, f"{type(exc).__name__}: {exc}"
        )


def _ct_merge_boundaries(messages: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Enumerate merge boundaries as ``(n, m)`` spans.

    A boundary exists wherever a non-assistant run follows an assistant turn:
    ``messages[n-1]`` is the assistant turn, and ``messages[n:m]`` is the maximal
    run of non-assistant messages (``m`` stops at the next assistant turn or the
    end). ``messages[:n]`` is the pretokenized prefix; ``messages[:m]`` is the
    prefix plus the appended run whose merge we verify.
    """
    boundaries: list[tuple[int, int]] = []
    i = 1
    while i < len(messages):
        if messages[i - 1].get("role") == "assistant" and messages[i].get("role") != "assistant":
            m = i
            while m < len(messages) and messages[m].get("role") != "assistant":
                m += 1
            boundaries.append((i, m))
            i = m
        else:
            i += 1
    return boundaries


def _run_ct_merge_checks(
    *,
    tokenizer,
    processor: Any | None,
    builder,
    trajectory_name: str,
    messages: list[dict[str, Any]],
    assistant_has_tool_calls: dict[int, bool],
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
    clone,
) -> list[CheckResult]:
    """Per-boundary Continuous Token merge check (production-shaped, no diff extraction).

    For each ``(n, m)`` boundary the prefix ``messages[:n]`` (which ends at an
    assistant turn) is rendered through the model's chat template and trimmed to
    the runtime stop shape — exactly the pretokenized prefix production would
    hold. The builder then merges the appended non-assistant run ``messages[n:m]``
    via ``merge_non_assistant_tokens``, and the merged stream must equal a
    single-pass render of ``messages[:m]`` with a trailing generation prompt.

    Because the prefix is rendered (not reconstructed from a prompt/full diff),
    no assistant output tokens are ever extracted, so thinking-model generation
    prompts that open a ``<think>`` block never raise here; a template that is
    not append-only surfaces as a plain token mismatch instead.
    """
    results: list[CheckResult] = []
    for n, m in _ct_merge_boundaries(messages):
        roles = "_".join(msg.get("role", "unknown") for msg in messages[n:m])
        case_name = f"{trajectory_name}.merge_at{n}.{roles}"
        prefix_msgs = [clone(msg) for msg in messages[:n]]
        full_msgs = [clone(msg) for msg in messages[:m]]
        has_tool_calls = assistant_has_tool_calls.get(n - 1, False)
        try:
            prefix_ids = _render_ids(
                tokenizer,
                processor,
                prefix_msgs,
                tools=tools,
                add_generation_prompt=False,
                chat_template_kwargs=chat_template_kwargs,
            )
            prefix_ids = _truncate_after_final_eos(tokenizer, prefix_ids, has_tool_calls=has_tool_calls)
            merge_result = builder.merge_non_assistant_tokens(prefix_msgs, full_msgs, prefix_ids, tools=tools)
            expected_ids = _render_ids(
                tokenizer,
                processor,
                full_msgs,
                tools=tools,
                add_generation_prompt=True,
                chat_template_kwargs=chat_template_kwargs,
            )
        except Exception as exc:
            results.append(CheckResult("continuous-token", case_name, False, f"{type(exc).__name__}: {exc}"))
            continue
        _append_ct_result(
            results,
            case_name=case_name,
            expected_ids=expected_ids,
            actual_ids=merge_result.token_ids,
            tokenizer=tokenizer,
        )
    return results


def _assemble_text_messages(trajectory: MockTrajectory) -> tuple[list[dict[str, Any]], dict[int, bool]]:
    """Flatten a text trajectory to a message list + per-assistant tool-call map."""
    messages = _initial_messages(trajectory)
    tool_calls: dict[int, bool] = {}
    if isinstance(trajectory, SingleTurnTrajectory):
        assistant = _assistant_message_for_single_turn(trajectory)
        tool_calls[len(messages)] = bool(assistant.get("tool_calls"))
        messages.append(assistant)
        return messages, tool_calls
    for step in trajectory.steps:
        assistant = _clone(step.assistant)
        tool_calls[len(messages)] = bool(assistant.get("tool_calls"))
        messages.append(assistant)
        messages.extend(_clone(message) for message in step.appended_messages)
    return messages, tool_calls


def run_continuous_token_checks(
    tokenizer,
    trajectory: MockTrajectory,
    *,
    model: str,
    model_family: str,
    custom_builder_module: str | None,
    chat_template_kwargs: dict[str, Any],
    processor: Any | None = None,
) -> list[CheckResult]:
    """Run per-boundary Continuous Token merge checks for a text trajectory.

    Mirrors production's incremental path: a pretokenized prefix (up to and
    including an assistant turn) plus a builder-driven merge of the following
    non-assistant run. See :func:`_run_ct_merge_checks` for the per-boundary
    contract. Single-turn trajectories have no post-assistant run and thus no
    merge boundary; the raw prefix diagnostics cover them instead.
    """
    builder, error = _create_builder_or_error(
        tokenizer,
        trajectory_name=trajectory.name,
        model=model,
        model_family=model_family,
        custom_builder_module=custom_builder_module,
        chat_template_kwargs=chat_template_kwargs,
        processor=processor,
    )
    if error is not None:
        return [error]

    messages, assistant_has_tool_calls = _assemble_text_messages(trajectory)
    return _run_ct_merge_checks(
        tokenizer=tokenizer,
        processor=processor,
        builder=builder,
        trajectory_name=trajectory.name,
        messages=messages,
        assistant_has_tool_calls=assistant_has_tool_calls,
        tools=_tools_for(trajectory),
        chat_template_kwargs=chat_template_kwargs,
        clone=_clone,
    )


# =============================================================================
# Vision-language (VL) checks
#
# The text checks above render ground truth through the *tokenizer* chat
# template. VL models must render through the *processor* chat template instead:
# the tokenizer template cannot expand image placeholders, and some VL tokenizer
# templates cannot render list-of-blocks content at all (and can even disagree
# with the processor template on tool injection). Images are passed alongside
# the text so vision placeholders expand into the same per-image pad-token spans
# the rollout backend consumes. These helpers mirror the processor-based
# rendering that the VL Continuous Token builder uses internally, so the CT
# reconstruction is compared against the template output the model actually sees.
# =============================================================================


def _clone_mm(value: Any) -> Any:
    # VL messages embed PIL images, which are not JSON-serializable, so the
    # text-path json clone cannot be reused here.
    return copy.deepcopy(value)


def _extract_images(messages: list[dict[str, Any]]) -> list[Any]:
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    image = block.get("image")
                    if image is not None:
                        images.append(image)
    return images


def _render_processor_ids(
    processor,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    """Render messages to token IDs through the multimodal processor.

    Mirrors ``VLContinuousTokenMixin.render_tokens_with_mm``: render text via the
    processor chat template (so tool/vision handling matches the runtime path),
    then expand image placeholders through the processor with the extracted
    images.
    """
    messages = _clone_mm(messages)
    text = apply_chat_template(
        processor,
        messages,
        tools=_clone(tools),
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **chat_template_kwargs,
    )
    images = _extract_images(messages)
    output = build_multimodal_processor_inputs(
        processor,
        text=[text],
        images=images if images else None,
    )
    return normalize_token_ids(output["input_ids"])


def _kimi_assistant_end_token_ids(tokenizer) -> set[int]:
    if "kimi" not in _tokenizer_name(tokenizer).lower():
        return set()
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(token_id, int) and token_id >= 0:
        return {token_id}
    return set()


def _infer_tool_parser(model: str) -> str:
    normalized = model.lower()
    compact = "".join(char for char in normalized if char.isalnum())
    if "kimi" in normalized:
        return "kimi"
    if "glm" in compact:
        return "glm"
    # Qwen VL / MiMo VL and other VL families use the Hermes tool format.
    return "hermes"


def _vl_tools_for(trajectory: VLMockTrajectory) -> list[dict[str, Any]] | None:
    if isinstance(trajectory, VLToolTrajectory):
        return trajectory.tool_schemas
    return None


def _vl_assistant_steps(
    trajectory: VLMockTrajectory,
    *,
    tool_parser: str,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], bool]]:
    """Return ``(assistant_message, appended_messages, has_tool_calls)`` per turn."""
    if isinstance(trajectory, VLSingleTurnTrajectory):
        return [(trajectory.assistant_message(), [], False)]
    steps: list[tuple[dict[str, Any], list[dict[str, Any]], bool]] = []
    for step in trajectory.steps:
        assistant = step.assistant_message(tool_parser)
        appended = _clone_mm(step.appended_messages)
        steps.append((assistant, appended, bool(step.calls)))
    return steps


def run_raw_template_checks_vl(
    processor,
    trajectory: VLMockTrajectory,
    *,
    tool_parser: str,
    chat_template_kwargs: dict[str, Any],
) -> list[CheckResult]:
    """Processor-based raw append-only diagnostics for a VL trajectory."""
    results: list[CheckResult] = []
    tokenizer = processor.tokenizer
    tools = _vl_tools_for(trajectory)
    messages = _clone_mm(list(trajectory.raw_prompt))
    steps = _vl_assistant_steps(trajectory, tool_parser=tool_parser)
    for turn_index, (assistant, appended, _has_tool_calls) in enumerate(steps, start=1):
        messages_with_assistant = messages + [assistant]
        _record_raw_prefix_check(
            results,
            case_name=f"{trajectory.name}.assistant_turn{turn_index}",
            tokenizer=tokenizer,
            processor=processor,
            prefix_messages=messages,
            full_messages=messages_with_assistant,
            prefix_add_generation_prompt=True,
            full_add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        if appended:
            roles = "_".join(message.get("role", "unknown") for message in appended)
            _record_raw_prefix_check(
                results,
                case_name=f"{trajectory.name}.append_turn{turn_index}.{roles}",
                tokenizer=tokenizer,
                processor=processor,
                prefix_messages=messages_with_assistant,
                full_messages=messages_with_assistant + appended,
                prefix_add_generation_prompt=False,
                full_add_generation_prompt=True,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
        messages = messages_with_assistant + appended
    return results


def _assemble_vl_messages(
    trajectory: VLMockTrajectory, *, tool_parser: str
) -> tuple[list[dict[str, Any]], dict[int, bool]]:
    """Flatten a VL trajectory to a message list + per-assistant tool-call map.

    VL assistant turns embed tool calls as raw text (no structured ``tool_calls``
    field), so the tool-call flag is taken from the trajectory step rather than
    read back off the message.
    """
    messages = _clone_mm(list(trajectory.raw_prompt))
    tool_calls: dict[int, bool] = {}
    for assistant, appended, has_tool_calls in _vl_assistant_steps(trajectory, tool_parser=tool_parser):
        tool_calls[len(messages)] = has_tool_calls
        messages.append(assistant)
        messages.extend(appended)
    return messages, tool_calls


def run_continuous_token_checks_vl(
    processor,
    tokenizer,
    trajectory: VLMockTrajectory,
    *,
    model: str,
    model_family: str,
    custom_builder_module: str | None,
    tool_parser: str,
    chat_template_kwargs: dict[str, Any],
) -> list[CheckResult]:
    """Per-boundary VL Continuous Token merge check.

    Same per-boundary contract as :func:`run_continuous_token_checks`, but every
    render goes through the processor (image placeholders expand to the same
    per-image pad spans the runtime consumes) and the VL builder drives the
    merge. Single-turn (image + one assistant) trajectories have no post-assistant
    run and are covered by the raw prefix diagnostics.
    """
    builder, error = _create_builder_or_error(
        tokenizer,
        trajectory_name=trajectory.name,
        model=model,
        model_family=model_family,
        custom_builder_module=custom_builder_module,
        chat_template_kwargs=chat_template_kwargs,
        processor=processor,
    )
    if error is not None:
        return [error]

    messages, assistant_has_tool_calls = _assemble_vl_messages(trajectory, tool_parser=tool_parser)
    return _run_ct_merge_checks(
        tokenizer=tokenizer,
        processor=processor,
        builder=builder,
        trajectory_name=trajectory.name,
        messages=messages,
        assistant_has_tool_calls=assistant_has_tool_calls,
        tools=_vl_tools_for(trajectory),
        chat_template_kwargs=chat_template_kwargs,
        clone=_clone_mm,
    )


def _print_results(title: str, results: list[CheckResult], *, failed_status: str = "FAIL") -> None:
    print(title)
    max_name_len = max((len(result.case_name) for result in results), default=0)
    for result in results:
        status = "PASS" if result.passed else failed_status
        line = f"  [{status}] {result.case_name:<{max_name_len}}"
        error_lines = result.error.splitlines() if result.error else []
        if error_lines:
            first_line = error_lines[0]
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            line += f"  -- {first_line}"
        print(line)
        # For any non-passing case, print the full decoded diff window (the
        # summary line is truncated in the table above, but the context lines
        # let the user see exactly what diverged, e.g. <think> vs <tool_call>).
        if not result.passed and len(error_lines) > 1:
            for extra in error_lines[1:]:
                print(f"    {extra.strip()}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a model chat template against verl mock trajectories and Continuous Token.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local tokenizer path.")
    parser.add_argument("--template", help="Optional local .jinja chat template override.")
    parser.add_argument(
        "--model-family",
        default="auto",
        help=(
            "Continuous Token builder family. Default: auto. "
            f"Built-ins: {', '.join(CONTINUOUS_TOKEN_BUILDER_FAMILIES)}."
        ),
    )
    parser.add_argument(
        "--custom-builder-module",
        default=None,
        help="Optional Python module to import before creating a custom Continuous Token builder.",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        metavar="JSON",
        help="Extra kwargs forwarded to apply_chat_template, e.g. '{\"enable_thinking\": false}'.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow AutoTokenizer to download missing tokenizer files. Default requires local cache.",
    )
    parser.add_argument(
        "--enable-multimodal",
        action="store_true",
        help="Load an AutoProcessor for multimodal (VL) builder families. Requires the model to have a processor.",
    )
    parser.add_argument(
        "--skip-vl",
        action="store_true",
        help="Skip the image-carrying VL trajectories even when a processor is loaded (run text trajectories only).",
    )
    parser.add_argument(
        "--show-traceback",
        action="store_true",
        help="Print full traceback if tokenizer loading or setup fails.",
    )
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    chat_template_kwargs = dict(args.chat_template_kwargs or {})

    processor = None
    try:
        tokenizer = _load_tokenizer(args.model, local_files_only=not args.allow_download, template_path=args.template)
        resolved_family = resolve_continuous_token_model_family(
            args.model_family,
            model_path=args.model,
            tokenizer=tokenizer,
            tokenizer_name_or_path=args.model,
        )

        # Load processor for VL families
        builder_cls = get_continuous_token_builder_class(resolved_family)
        needs_processor = hasattr(builder_cls, "supports_multimodal") and builder_cls.supports_multimodal()
        if args.enable_multimodal or needs_processor:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                args.model,
                trust_remote_code=True,
                local_files_only=not args.allow_download,
            )
            print(f"Processor loaded:      {type(processor).__name__}")
        elif needs_processor and not args.enable_multimodal:
            print(
                f"WARNING: Model family {resolved_family!r} requires a processor for multimodal. "
                f"Pass --enable-multimodal to load one, or text-only checks will be run."
            )
    except Exception as exc:
        print(f"Failed to initialize checker: {type(exc).__name__}: {exc}")
        if args.show_traceback:
            print(traceback.format_exc())
        return 2

    source_desc = f"template override: {args.template}" if args.template else f"tokenizer chat_template: {args.model}"
    print(f"Template source:       {source_desc}")
    print(f"Model:                 {args.model}")
    print(f"Continuous family:     {resolved_family} (requested: {args.model_family})")
    if chat_template_kwargs:
        print(f"Chat template kwargs:  {chat_template_kwargs}")
    print(f"Trajectories:          {len(TRAJECTORIES)}")
    _warn_glm_tool_template_limitations(args.model)
    print()

    raw_results: list[CheckResult] = []
    ct_results: list[CheckResult] = []
    for trajectory in TRAJECTORIES:
        raw_results.extend(
            run_raw_template_checks(
                tokenizer,
                trajectory,
                processor=processor,
                chat_template_kwargs=chat_template_kwargs,
            )
        )
        ct_results.extend(
            run_continuous_token_checks(
                tokenizer,
                trajectory,
                model=args.model,
                model_family=args.model_family,
                custom_builder_module=args.custom_builder_module,
                chat_template_kwargs=chat_template_kwargs,
                processor=processor,
            )
        )

    _print_results("Raw template prefix diagnostics (text):", raw_results, failed_status="WARN")
    _print_results("Continuous Token checks (text):", ct_results)

    # Vision-language trajectories: only run when a processor is available, since
    # image expansion and the VL builder both require it.
    if processor is not None and not args.skip_vl:
        try:
            vl_trajectories = list(build_vl_trajectories().values())
        except Exception as exc:
            print(f"WARNING: skipping VL trajectories (could not build them): {type(exc).__name__}: {exc}")
            vl_trajectories = []
        if vl_trajectories:
            tool_parser = _infer_tool_parser(args.model)
            print(f"VL trajectories:       {len(vl_trajectories)} (tool_parser={tool_parser})")
            print()
            for trajectory in vl_trajectories:
                raw_results.extend(
                    run_raw_template_checks_vl(
                        processor,
                        trajectory,
                        tool_parser=tool_parser,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                )
                ct_results.extend(
                    run_continuous_token_checks_vl(
                        processor,
                        tokenizer,
                        trajectory,
                        model=args.model,
                        model_family=args.model_family,
                        custom_builder_module=args.custom_builder_module,
                        tool_parser=tool_parser,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                )
            vl_raw = [result for result in raw_results if result.case_name.startswith("vl_")]
            vl_ct = [result for result in ct_results if result.case_name.startswith("vl_")]
            _print_results("Raw template prefix diagnostics (VL):", vl_raw, failed_status="WARN")
            _print_results("Continuous Token checks (VL):", vl_ct)

    raw_passed = sum(result.passed for result in raw_results)
    raw_failed = len(raw_results) - raw_passed
    ct_passed = sum(result.passed for result in ct_results)
    ct_failed = len(ct_results) - ct_passed
    print(
        "Results: "
        f"raw diagnostics {raw_passed}/{len(raw_results)} passed ({raw_failed} warnings), "
        f"Continuous Token {ct_passed}/{len(ct_results)} passed"
    )
    if ct_failed:
        print("Verdict: FAIL - Continuous Token builder is not safe for these trajectories")
        return 1
    if raw_failed:
        print(
            "Verdict: PASS with raw-prefix warnings - raw template is not globally append-only, "
            "but Continuous Token checks passed"
        )
        return 0

    print("Verdict: PASS - chat template passed raw prefix diagnostics and Continuous Token checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
