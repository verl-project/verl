#!/usr/bin/env python3
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
"""Extend a context probe parquet to a target token length.

Pre-tokenized rows cycle ``input_ids`` and token-aligned columns. Message-only
rows cycle complete user/assistant messages until their text reaches the
character length estimated from ``--target-length``. The message-only estimate
uses 0.33 tokens per character; it preserves complete turns but is not an exact
tokenizer measurement. Repeated content is a synthetic stress signal, not a
meaningful language-modeling sample.

Usage::

    python scripts/tools/extend_ctxprobe_parquet.py \
        --source /data/ctxprobe/trainreal1m.parquet \
        --target-length 1500000 \
        --output /data/ctxprobe/trainreal1_5m.parquet
"""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOKEN_ALIGNED_COLUMNS = ("loss_mask", "teacher_ids", "teacher_logprobs")
TOKENS_PER_CHARACTER = 0.33


def _is_missing(value: Any) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _as_list(value: Any, *, column: str, row_number: int) -> list[Any]:
    if _is_missing(value):
        raise ValueError(f"row {row_number} column {column!r} is null")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"row {row_number} column {column!r} is not a sequence")
    values = list(value)
    if not values:
        raise ValueError(f"row {row_number} column {column!r} is empty")
    return values


def _cycle_to_length(values: list[Any], target_length: int) -> np.ndarray:
    array = np.asarray(values)
    return np.resize(array, (target_length, *array.shape[1:]))


def _message_text_characters(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        return sum(_message_text_characters(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(_message_text_characters(item) for item in value)
    return 0


def _message_characters(message: dict[str, Any], *, row_number: int, message_number: int) -> int:
    if "content" not in message:
        raise ValueError(f"row {row_number} message {message_number} has no 'content'")
    return _message_text_characters(message["content"])


def _validate_messages(messages: list[Any], *, row_number: int) -> tuple[list[dict[str, Any]], int]:
    if not messages:
        raise ValueError(f"row {row_number} column 'messages' is empty")
    if any(not isinstance(message, dict) for message in messages):
        raise TypeError(f"row {row_number} column 'messages' must contain message objects")

    system_prefix: list[dict[str, Any]] = []
    if messages[0].get("role") == "system":
        system_prefix.append(messages[0])
        messages = messages[1:]
    if not messages:
        raise ValueError(f"row {row_number} messages must contain a user/assistant conversation")

    expected_role = "user"
    for message_number, message in enumerate(messages, start=len(system_prefix)):
        role = message.get("role")
        if role != expected_role:
            raise ValueError(f"row {row_number} message {message_number} has role {role!r}; expected {expected_role!r}")
        _message_characters(message, row_number=row_number, message_number=message_number)
        expected_role = "assistant" if expected_role == "user" else "user"
    if expected_role != "user":
        raise ValueError(f"row {row_number} messages must end with an assistant message to allow safe repetition")

    prefix_characters = sum(
        _message_characters(message, row_number=row_number, message_number=message_number)
        for message_number, message in enumerate(system_prefix)
    )
    conversation_characters = sum(
        _message_characters(message, row_number=row_number, message_number=message_number)
        for message_number, message in enumerate(messages, start=len(system_prefix))
    )
    if conversation_characters == 0:
        raise ValueError(f"row {row_number} conversation contains no text characters to estimate tokens")
    return system_prefix + messages, prefix_characters


def _cycle_messages_to_characters(
    messages: list[dict[str, Any]], target_characters: int, *, row_number: int
) -> list[Any]:
    messages, prefix_characters = _validate_messages(messages, row_number=row_number)
    system_count = 1 if messages[0].get("role") == "system" else 0
    conversation = messages[system_count:]
    generated = copy.deepcopy(messages)
    character_count = prefix_characters + sum(
        _message_characters(message, row_number=row_number, message_number=message_number)
        for message_number, message in enumerate(conversation, start=system_count)
    )
    message_number = len(messages)

    while (
        character_count < target_characters or len(generated) < system_count + 2 or generated[-1].get("role") == "user"
    ):
        message = conversation[(message_number - system_count) % len(conversation)]
        generated.append(copy.deepcopy(message))
        character_count += _message_characters(message, row_number=row_number, message_number=message_number)
        message_number += 1

    return generated


def extend_dataframe(dataframe: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """Extend pre-tokenized rows or message-only rows to the requested length."""
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if dataframe.empty:
        raise ValueError("source parquet must contain at least one row")
    if "input_ids" not in dataframe.columns:
        if "messages" not in dataframe.columns:
            raise ValueError("source parquet must contain either 'input_ids' or 'messages'")
        if any(column in dataframe.columns for column in (*TOKEN_ALIGNED_COLUMNS, "position_ids")):
            raise ValueError("message-only rows cannot contain token-aligned columns without 'input_ids'")
        target_characters = math.ceil(target_length / TOKENS_PER_CHARACTER)
        result = dataframe.copy().reset_index(drop=True)
        result["messages"] = result["messages"].astype("object")
        for row_number, index in enumerate(result.index):
            messages = _as_list(result.at[index, "messages"], column="messages", row_number=row_number)
            result.at[index, "messages"] = _cycle_messages_to_characters(
                messages, target_characters, row_number=row_number
            )
        return result
    if ("teacher_ids" in dataframe.columns) != ("teacher_logprobs" in dataframe.columns):
        raise ValueError("source parquet must contain both 'teacher_ids' and 'teacher_logprobs' or neither")

    result = dataframe.copy().reset_index(drop=True)
    for column in ("input_ids", *TOKEN_ALIGNED_COLUMNS, "position_ids"):
        if column in result.columns:
            result[column] = result[column].astype("object")

    for row_number, index in enumerate(result.index):
        input_ids = _as_list(result.at[index, "input_ids"], column="input_ids", row_number=row_number)
        source_length = len(input_ids)
        result.at[index, "input_ids"] = _cycle_to_length(input_ids, target_length)

        if "teacher_ids" in result.columns and (
            _is_missing(result.at[index, "teacher_ids"]) != _is_missing(result.at[index, "teacher_logprobs"])
        ):
            raise ValueError(f"row {row_number} must provide both teacher_ids and teacher_logprobs")

        for column in TOKEN_ALIGNED_COLUMNS:
            if column not in result.columns or _is_missing(result.at[index, column]):
                continue
            values = _as_list(result.at[index, column], column=column, row_number=row_number)
            if len(values) != source_length:
                raise ValueError(
                    f"row {row_number} column {column!r} has {len(values)} entries, but input_ids has {source_length}"
                )
            result.at[index, column] = _cycle_to_length(values, target_length)

        if "position_ids" in result.columns and not _is_missing(result.at[index, "position_ids"]):
            position_ids = _as_list(result.at[index, "position_ids"], column="position_ids", row_number=row_number)
            if len(position_ids) != source_length:
                raise ValueError(
                    f"row {row_number} column 'position_ids' has {len(position_ids)} entries, "
                    f"but input_ids has {source_length}"
                )
            result.at[index, "position_ids"] = np.arange(target_length, dtype=np.int64)

    lengths = result["input_ids"].map(len)
    if not (lengths == target_length).all():
        raise AssertionError(f"generated input_ids lengths are not all {target_length}: {lengths.tolist()}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing context probe parquet")
    parser.add_argument(
        "--target-length", type=int, required=True, help="Target token length for every row (estimated for messages)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output parquet path")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output parquet")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.target_length <= 0:
        raise SystemExit("--target-length must be positive")
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"source parquet does not exist: {source}")
    if source == output:
        raise SystemExit("--source and --output must be different paths")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists (pass --overwrite to replace): {output}")

    dataframe = pd.read_parquet(source)
    if "input_ids" not in dataframe.columns and "messages" in dataframe.columns:
        target_characters = math.ceil(args.target_length / TOKENS_PER_CHARACTER)
        print(
            f"messages-only estimate: {args.target_length:,} target tokens / {TOKENS_PER_CHARACTER:g} tokens/char "
            f"= {target_characters:,} target characters per row "
            f"(~{target_characters * TOKENS_PER_CHARACTER:,.0f} tokens)"
        )
    result = extend_dataframe(dataframe, args.target_length)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    if "input_ids" in result.columns:
        print(f"wrote {len(result)} rows x {args.target_length} tokens to {output}")
    else:
        actual_characters = [
            sum(
                _message_characters(message, row_number=row_number, message_number=message_number)
                for message_number, message in enumerate(messages)
            )
            for row_number, messages in enumerate(result["messages"])
        ]
        estimated_tokens = [round(characters * TOKENS_PER_CHARACTER) for characters in actual_characters]
        print(
            f"wrote {len(result)} messages-only rows to {output}; generated characters "
            f"{min(actual_characters):,}-{max(actual_characters):,}, estimated tokens "
            f"{min(estimated_tokens):,}-{max(estimated_tokens):,}"
        )


if __name__ == "__main__":
    main()
