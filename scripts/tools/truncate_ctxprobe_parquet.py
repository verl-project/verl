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
"""Truncate message text in a context probe parquet to a token budget.

Rows and messages are consumed in order. The final retained message is
truncated at its text-character budget; later messages and rows are omitted.
Tokens per character is an explicit estimate, not tokenizer output. The
0.3867 default is the measured ratio used to produce the DSV4.1 1.55M profile.

Usage::

    python scripts/tools/truncate_ctxprobe_parquet.py \
        --source /data/ctxprobe/trainreal1_5m.parquet \
        --target-tokens 1550000 \
        --tokens-per-character 0.3867 \
        --output /data/ctxprobe/trainreal1_55m.parquet
"""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

MEASURED_TOKENS_PER_CHARACTER = 0.3867


def _text_character_count(value: Any) -> int:
    if hasattr(value, "tolist"):
        return _text_character_count(value.tolist())
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        if "text" in value:
            return _text_character_count(value["text"])
        return sum(_text_character_count(item) for item in value.values() if isinstance(item, dict | list | tuple))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(_text_character_count(item) for item in value)
    return 0


def _truncate_text(value: Any, character_budget: int) -> tuple[Any, int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        retained = value[:character_budget]
        return retained, len(retained)
    if isinstance(value, dict):
        result = copy.deepcopy(value)
        if "text" in value:
            result["text"], retained_characters = _truncate_text(value["text"], character_budget)
            return result, retained_characters
        remaining = character_budget
        retained_characters = 0
        for key, item in value.items():
            if not isinstance(item, dict | list | tuple):
                continue
            result[key], count = _truncate_text(item, remaining)
            retained_characters += count
            remaining -= count
        return result, retained_characters
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        result = []
        remaining = character_budget
        retained_characters = 0
        for item in value:
            truncated, count = _truncate_text(item, remaining)
            result.append(truncated)
            retained_characters += count
            remaining -= count
            if remaining == 0:
                break
        return result, retained_characters
    return copy.deepcopy(value), 0


def truncate_messages_dataframe(
    dataframe: pd.DataFrame, target_tokens: int, tokens_per_character: float
) -> pd.DataFrame:
    """Retain the first messages up to a character budget estimated from tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if not math.isfinite(tokens_per_character) or tokens_per_character <= 0:
        raise ValueError("tokens_per_character must be a finite positive number")
    if dataframe.empty:
        raise ValueError("source parquet must contain at least one row")
    if "messages" not in dataframe.columns:
        raise ValueError("source parquet must contain a 'messages' column")

    character_budget = math.floor(target_tokens / tokens_per_character)
    result = dataframe.iloc[0:0].copy()
    result["messages"] = result["messages"].astype("object")
    remaining = character_budget
    for row_index, row in dataframe.iterrows():
        messages = row["messages"]
        if hasattr(messages, "tolist"):
            messages = messages.tolist()
        if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
            raise TypeError(f"row {row_index} column 'messages' is not a sequence")

        retained_messages = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError(f"row {row_index} column 'messages' must contain message objects")
            if "content" not in message:
                raise ValueError(f"row {row_index} message has no 'content'")
            message_characters = _text_character_count(message["content"])
            if message_characters <= remaining:
                retained_messages.append(copy.deepcopy(message))
                remaining -= message_characters
                if remaining == 0:
                    break
                continue

            truncated_content, retained_characters = _truncate_text(message["content"], remaining)
            if retained_characters:
                truncated_message = copy.deepcopy(message)
                truncated_message["content"] = truncated_content
                retained_messages.append(truncated_message)
                remaining -= retained_characters
            break

        if retained_messages:
            retained_row = row.copy()
            retained_row["messages"] = retained_messages
            result.loc[len(result)] = retained_row
        if remaining == 0 or len(retained_messages) < len(messages):
            break

    retained_characters = character_budget - remaining
    if retained_characters != min(
        character_budget,
        sum(_text_character_count(message["content"]) for messages in result["messages"] for message in messages),
    ):
        raise AssertionError("truncated message character count is inconsistent with the budget")
    return result.reset_index(drop=True)


def truncate_messages_parquet(
    source: Path,
    output: Path,
    *,
    target_tokens: int,
    tokens_per_character: float = MEASURED_TOKENS_PER_CHARACTER,
) -> pd.DataFrame:
    """Read, truncate and write a messages parquet; return the generated rows."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source parquet does not exist: {source}")
    if source == output:
        raise ValueError("source and output must be different paths")

    result = truncate_messages_dataframe(pd.read_parquet(source), target_tokens, tokens_per_character)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source parquet with a messages column")
    parser.add_argument(
        "--target-tokens", type=int, required=True, help="Estimated token budget for all retained messages"
    )
    parser.add_argument(
        "--tokens-per-character",
        type=float,
        default=MEASURED_TOKENS_PER_CHARACTER,
        help=(
            "Measured tokens per text character (default: 0.3867, the observed DSV4.1 data ratio); "
            "character budget is floor(target tokens / ratio)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Truncated output parquet path")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output parquet")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists (pass --overwrite to replace): {args.output}")
    try:
        result = truncate_messages_parquet(
            args.source,
            args.output,
            target_tokens=args.target_tokens,
            tokens_per_character=args.tokens_per_character,
        )
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    character_count = sum(
        _text_character_count(message["content"]) for messages in result["messages"] for message in messages
    )
    print(
        f"wrote {len(result)} rows and {character_count:,} text characters to {args.output}; "
        f"estimated {character_count * args.tokens_per_character:,.0f} tokens "
        f"at {args.tokens_per_character:g} tokens/character"
    )


if __name__ == "__main__":
    main()
