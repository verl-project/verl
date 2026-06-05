"""Instruction-following rewards for IFEvalG, IFEval, and IFBench."""

from __future__ import annotations

import ast
import json
from typing import Any

from . import instructions_registry


_COT_END_MARKERS = ("</think>", "<|inner_suffix|>")


def _get_instruction_cls(instruction_key: str):
    if instruction_key in instructions_registry.INSTRUCTION_DICT:
        return instructions_registry.INSTRUCTION_DICT[instruction_key]

    from .ifbench import instructions_registry as ifbench_instructions_registry

    if instruction_key in ifbench_instructions_registry.INSTRUCTION_DICT:
        return ifbench_instructions_registry.INSTRUCTION_DICT[instruction_key]

    raise KeyError(f"Unknown instruction-following constraint: {instruction_key}")


def _parse_constraint(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        if not value:
            raise ValueError("Empty instruction-following constraint list.")
        return _parse_constraint(value[0])

    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = json.loads(text)
        return _parse_constraint(parsed)

    raise TypeError(
        f"Unsupported instruction-following constraint type: {type(value)!r}"
    )


def _extract_answer(prediction: str) -> str:
    marker_ends = [
        marker_index + len(marker)
        for marker in _COT_END_MARKERS
        if (marker_index := prediction.rfind(marker)) != -1
    ]
    if not marker_ends:
        return prediction.strip()
    return prediction[max(marker_ends) :].strip()


def _constraint_from_inputs(
    ground_truth: Any, extra_info: dict[str, Any] | None
) -> dict[str, Any]:
    candidates = [ground_truth]
    if extra_info:
        candidates.extend(
            extra_info.get(key)
            for key in ("constraint", "ground_truth", "label")
            if extra_info.get(key) is not None
        )

    last_error: Exception | None = None
    for candidate in candidates:
        if candidate is None or candidate == "":
            continue
        try:
            constraint = _parse_constraint(candidate)
        except (json.JSONDecodeError, SyntaxError, ValueError, TypeError) as exc:
            last_error = exc
            continue
        if "instruction_id" in constraint and "kwargs" in constraint:
            return constraint

    if last_error is not None:
        raise ValueError(
            "Could not parse instruction-following constraint."
        ) from last_error
    raise ValueError("No instruction-following constraint was provided.")


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    constraint_dict = _constraint_from_inputs(ground_truth, extra_info)
    answer = _extract_answer(solution_str)
    instruction_keys = constraint_dict["instruction_id"]
    args_list = constraint_dict["kwargs"]

    if isinstance(instruction_keys, str):
        instruction_keys = [instruction_keys]
    if isinstance(args_list, dict) or args_list is None:
        args_list = [args_list]
    if len(instruction_keys) != len(args_list):
        raise ValueError(
            "Instruction-following constraint has mismatched instruction_id "
            f"and kwargs lengths: {len(instruction_keys)} != {len(args_list)}"
        )

    rewards = []
    for instruction_key, args in zip(instruction_keys, args_list, strict=True):
        if args is None:
            args = {}
        args = {key: value for key, value in args.items() if value is not None}

        instruction_cls = _get_instruction_cls(instruction_key)
        instruction_instance = instruction_cls(instruction_key)
        instruction_instance.build_description(**args)
        rewards.append(
            float(
                bool(solution_str.strip())
                and instruction_instance.check_following(answer)
            )
        )

    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


__all__ = ["compute_score"]
