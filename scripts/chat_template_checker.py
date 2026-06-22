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

1. raw template prefix diagnostics at token-id level. This is a quick checker that
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

2. production-shaped Continuous Token builder checks at token level. This layer
   incrementally rebuilds the runtime token stream turn by turn with Continuous
   Token logic, then compares the final assembled runtime IDs with directly
   applying the chat template to the complete message list with tokenization.
   If the final message is an assistant output, the full render is trimmed to
   the runtime stop shape after the final EOS/stop token. Its main purpose is
   to verify the builder's merge-boundary handling.

Examples:

    python scripts/chat_template_checker.py --model Qwen/Qwen3-0.6B
    python scripts/chat_template_checker.py --model zai-org/GLM-4.7-Flash --allow-download
    python scripts/chat_template_checker.py --model Qwen/Qwen3-0.6B --template /path/to/chat_template.jinja
"""

from __future__ import annotations

import argparse
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
)
from verl.utils.chat_template import apply_chat_template  # noqa: E402
from verl.utils.continuous_token_wiring import (  # noqa: E402
    CONTINUOUS_TOKEN_BUILDER_FAMILIES,
    create_continuous_token_builder,
    get_continuous_token_builder_class,
    resolve_continuous_token_model_family,
)
from verl.utils.tokenizer import normalize_token_ids  # noqa: E402

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


def _token_prefix_error(prefix_ids: list[int], full_ids: list[int]) -> str | None:
    if full_ids[: len(prefix_ids)] == prefix_ids:
        return None
    limit = min(len(prefix_ids), len(full_ids))
    mismatch = next((idx for idx in range(limit) if prefix_ids[idx] != full_ids[idx]), limit)
    prefix_value = prefix_ids[mismatch] if mismatch < len(prefix_ids) else None
    full_value = full_ids[mismatch] if mismatch < len(full_ids) else None
    return (
        f"Token prefix mismatch at index {mismatch}: "
        f"prefix_len={len(prefix_ids)}, full_len={len(full_ids)}, "
        f"prefix={prefix_value}, full={full_value}"
    )


def _token_mismatch_error(expected: list[int], actual: list[int]) -> str | None:
    if actual == expected:
        return None
    limit = min(len(expected), len(actual))
    mismatch = next((idx for idx in range(limit) if expected[idx] != actual[idx]), limit)
    expected_value = expected[mismatch] if mismatch < len(expected) else None
    actual_value = actual[mismatch] if mismatch < len(actual) else None
    return (
        f"Token mismatch at index {mismatch}: "
        f"expected_len={len(expected)}, actual_len={len(actual)}, "
        f"expected={expected_value}, actual={actual_value}"
    )


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
    return any(marker in tokenizer_name for marker in ("glm-4.7", "glm_4.7", "glm-5", "glm_5")) or any(
        marker in compact_name for marker in ("glm47", "glm5")
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
    assistant_message: dict[str, Any],
) -> list[int]:
    """Approximate the runtime token stream returned by generation.

    Full chat-template renders can include template whitespace after the model's
    stop token. The real rollout server usually stops at EOS/stop, so the CT
    boundary check trims to that shape before appending non-assistant messages.
    """

    eos_token_ids = _tokenizer_eos_token_ids(tokenizer)
    if not token_ids:
        raise ValueError("Assistant output token-id suffix is empty")
    if eos_token_ids:
        if token_ids[-1] in eos_token_ids:
            return token_ids
        for index in range(len(token_ids) - 1, -1, -1):
            if token_ids[index] in eos_token_ids:
                return token_ids[: index + 1]

    if _is_glm_tokenizer(tokenizer):
        if assistant_message.get("tool_calls"):
            return token_ids + [_require_single_token_id(tokenizer, "<|observation|>")]
        return token_ids

    raise ValueError(
        "Assistant output token-id suffix does not contain eos_token_id "
        f"{sorted(eos_token_ids)}; tail={token_ids[-16:]}"
    )


def _extract_assistant_output_ids(
    tokenizer,
    prefix_messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
) -> list[int]:
    prompt_ids = _render_tokens(
        tokenizer,
        prefix_messages,
        tools=tools,
        add_generation_prompt=True,
        chat_template_kwargs=chat_template_kwargs,
    )
    full_ids = _render_tokens(
        tokenizer,
        prefix_messages + [_clone(assistant_message)],
        tools=tools,
        add_generation_prompt=False,
        chat_template_kwargs=chat_template_kwargs,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Assistant output token-id suffix diff failed")
    return _truncate_after_final_eos(
        tokenizer,
        full_ids[len(prompt_ids) :],
        assistant_message=assistant_message,
    )


def _record_raw_prefix_check(
    results: list[CheckResult],
    *,
    case_name: str,
    tokenizer,
    prefix_messages: list[dict[str, Any]],
    full_messages: list[dict[str, Any]],
    prefix_add_generation_prompt: bool,
    full_add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any],
) -> None:
    try:
        prefix_ids = _render_tokens(
            tokenizer,
            prefix_messages,
            tools=tools,
            add_generation_prompt=prefix_add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        full_ids = _render_tokens(
            tokenizer,
            full_messages,
            tools=tools,
            add_generation_prompt=full_add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        error = _token_prefix_error(prefix_ids, full_ids)
        results.append(CheckResult("raw-template", case_name, error is None, error))
    except Exception as exc:
        results.append(CheckResult("raw-template", case_name, False, f"{type(exc).__name__}: {exc}"))


def run_raw_template_checks(
    tokenizer,
    trajectory: MockTrajectory,
    *,
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
) -> None:
    error = _token_mismatch_error(expected_ids, actual_ids)
    results.append(CheckResult("continuous-token", case_name, error is None, error))


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
    """Run an end-to-end Continuous Token reconstruction check.

    The checker builds the runtime token stream the same way production would:
    create the initial prompt, append each assistant output suffix, and merge
    each appended non-assistant turn through the selected builder. It performs a
    single final assertion per trajectory: the fully assembled runtime token IDs
    must match directly applying the chat template to the complete final message
    list with ``tokenize=True``. When the final message is an assistant output,
    the full render is normalized to the runtime stop shape by trimming template
    tokens after the final EOS/stop token. This targets CT merge-boundary bugs
    without treating intermediate prompt states as separate pass/fail cases.
    """

    results: list[CheckResult] = []
    tools = _tools_for(trajectory)
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
    except Exception as exc:
        return [
            CheckResult(
                "continuous-token",
                f"{trajectory.name}.builder",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        ]

    case_name = f"{trajectory.name}.full_trajectory"
    messages = _initial_messages(trajectory)
    try:
        runtime_ids = builder.build_initial_tokens(messages, tools=tools)
        final_messages = messages
    except Exception as exc:
        results.append(
            CheckResult(
                "continuous-token",
                case_name,
                False,
                f"initial_prompt: {type(exc).__name__}: {exc}",
            )
        )
        return results

    if isinstance(trajectory, SingleTurnTrajectory):
        assistant_steps = ((_assistant_message_for_single_turn(trajectory), ()),)
    else:
        assistant_steps = tuple(
            (_clone(step.assistant), tuple(_clone(message) for message in step.appended_messages))
            for step in trajectory.steps
        )

    for turn_index, (assistant, appended) in enumerate(assistant_steps, start=1):
        try:
            assistant_ids = _extract_assistant_output_ids(
                tokenizer,
                messages,
                assistant,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            )
            merge_result = builder.merge_assistant_tokens(runtime_ids, assistant_ids)
            runtime_ids = merge_result.token_ids
        except Exception as exc:
            results.append(
                CheckResult(
                    "continuous-token",
                    case_name,
                    False,
                    f"assistant_turn{turn_index}: {type(exc).__name__}: {exc}",
                )
            )
            return results

        messages_with_assistant = messages + [assistant]
        final_messages = messages_with_assistant
        appended_messages = list(appended)
        if appended_messages:
            roles = "_".join(message.get("role", "unknown") for message in appended_messages)
            next_messages = messages_with_assistant + appended_messages
            try:
                merge_result = builder.merge_non_assistant_tokens(
                    messages_with_assistant, next_messages, runtime_ids, tools=tools
                )
                runtime_ids = merge_result.token_ids
            except Exception as exc:
                results.append(
                    CheckResult(
                        "continuous-token",
                        case_name,
                        False,
                        f"append_turn{turn_index}.{roles}: {type(exc).__name__}: {exc}",
                    )
                )
                return results
            final_messages = next_messages
        messages = messages_with_assistant + appended_messages

    try:
        final_is_assistant = final_messages and final_messages[-1].get("role") == "assistant"
        expected_final_ids = _render_tokens(
            tokenizer,
            final_messages,
            tools=tools,
            add_generation_prompt=not final_is_assistant,
            chat_template_kwargs=chat_template_kwargs,
        )
        if final_is_assistant:
            expected_final_ids = _truncate_after_final_eos(
                tokenizer,
                expected_final_ids,
                assistant_message=final_messages[-1],
            )
        _append_ct_result(
            results,
            case_name=case_name,
            expected_ids=expected_final_ids,
            actual_ids=runtime_ids,
        )
    except Exception as exc:
        results.append(
            CheckResult(
                "continuous-token",
                case_name,
                False,
                f"final_full_render: {type(exc).__name__}: {exc}",
            )
        )

    return results


def _print_results(title: str, results: list[CheckResult], *, failed_status: str = "FAIL") -> None:
    print(title)
    max_name_len = max((len(result.case_name) for result in results), default=0)
    for result in results:
        status = "PASS" if result.passed else failed_status
        line = f"  [{status}] {result.case_name:<{max_name_len}}"
        if result.error:
            first_line = result.error.splitlines()[0]
            if len(first_line) > 120:
                first_line = first_line[:117] + "..."
            line += f"  -- {first_line}"
        print(line)
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
    print()

    raw_results: list[CheckResult] = []
    ct_results: list[CheckResult] = []
    for trajectory in TRAJECTORIES:
        raw_results.extend(run_raw_template_checks(tokenizer, trajectory, chat_template_kwargs=chat_template_kwargs))
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

    _print_results("Raw template prefix diagnostics:", raw_results, failed_status="WARN")
    _print_results("Continuous Token checks:", ct_results)

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
