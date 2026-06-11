"""Utilities for adapting LEXam MCQ samples for RL preprocessing."""

from __future__ import annotations

import ast
import re
from typing import Any

SRC = "LEXam_mcq"


def extract_meta(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep source metadata fields that are not part of the verifier target."""
    return {
        key: value
        for key, value in sample.items()
        if key not in {"question", "choices", "gold"}
    }


def parse_choices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(choice).strip() for choice in value]
    if isinstance(value, tuple):
        return [str(choice).strip() for choice in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = re.findall(r"'(.*?)'", value)
        if isinstance(parsed, list):
            return [str(choice).strip() for choice in parsed]
    raise ValueError(f"Cannot parse LEXam choices: {value!r}")


def roman_statement_answers(question: str, choice_text: str) -> list[str]:
    """Expand Roman numeral choices to their source statements.

    Adapted from AlignedDragon/data_parse_apertus convert_LEXam_mcq.py. The
    original converter used this to format SFT responses; for RL we keep the
    expansion as metadata while the verifiable target remains the answer label.
    """
    prefixes = re.findall(r"\b[ivx]+\b", choice_text.lower())
    if not prefixes:
        return []

    alternatives = "|".join(re.escape(prefix) for prefix in sorted(prefixes, key=len, reverse=True))
    pattern = rf"^(?:{alternatives})\..*"
    answers = re.findall(pattern, question, re.MULTILINE | re.IGNORECASE)
    return answers or prefixes


def join_statement_answers(answers: list[str], language: str) -> str:
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]

    conjunction = "und" if language == "de" else "and"
    return f"{', '.join(answers[:-1])} {conjunction} {answers[-1]}"


def lexam_answer_text(question: str, choices: list[str], answer_index: int, language: str) -> str:
    right_choice = choices[answer_index]
    if right_choice.lower() in {"none of the statements", "keine der aussagen"}:
        return right_choice

    statement_answers = roman_statement_answers(question, right_choice)
    return join_statement_answers(statement_answers, language) or right_choice


def normalize_lexam_mcq_sample(sample: dict[str, Any]) -> dict[str, Any]:
    choices = parse_choices(sample["choices"])
    answer_index = int(sample["gold"])
    if answer_index < 0 or answer_index >= len(choices):
        raise ValueError(f"LEXam gold index out of range: {answer_index}")

    language = str(sample.get("language") or "").strip()
    return {
        "question": str(sample["question"]).strip(),
        "choices": choices,
        "answer_index": answer_index,
        "answer_text": lexam_answer_text(
            str(sample["question"]), choices, answer_index, language
        ),
        "metadata": extract_meta(sample),
    }

if __name__ == "__main__":
    import datasets
    import json


    ds = datasets.load_dataset("LEXam-Benchmark/LEXam", "mcq_4_choices", split="test")
    ds = ds.filter(lambda sample: sample["language"] == "en")
    i = 1
    print("Question sample:")
    print(ds[i]["question"])
    print("\nChoices sample:")
    print(ds[i]["choices"])
    print("\n\n\n")
    normalized_ds = ds.map(normalize_lexam_mcq_sample)
    
    print("Normalized sample:")
    print(normalized_ds[i]["question"])
    print("\nChoices sample:")
    print(normalized_ds[i]["choices"])
    print(normalized_ds[i]["answer_text"])
    print(normalized_ds[i])
    

    