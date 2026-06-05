from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sympy


_GSM8K_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*(.+)", re.IGNORECASE)
_REASONING_BLOCK_RE = re.compile(r"REASONING:\s*(.*?)(?:\n\s*FINAL_ANSWER:|\Z)", re.IGNORECASE | re.DOTALL)
_UNIT_SUFFIX_RE = re.compile(
    r"\b(?:dollars?|cents?|eggs?|apple?s?|hours?|minutes?|days?|weeks?|months?|years?|meters?|feet|foot|cm|km|gallons?|ounces?|clips?)\b",
    re.IGNORECASE,
)
_WORD_NUMBER_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}


@dataclass(frozen=True)
class ParsedAnswer:
    extracted_answer: str | None
    normalized_answer: str | None
    reason: str


def sanitize_answer_surface(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    text = text.strip(" \t\r\n'\"`")
    text = re.sub(r"\s+", " ", text)
    return text.strip() or None


def normalize_numeric_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = sanitize_answer_surface(value)
    if text is None:
        return None
    lower = text.lower()
    if lower in _WORD_NUMBER_MAP:
        text = _WORD_NUMBER_MAP[lower]
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = _UNIT_SUFFIX_RE.sub("", text).strip()
    percent = False
    if text.endswith("%"):
        percent = True
        text = text[:-1].strip()
    text = re.sub(r"\s+", "", text)
    if not text:
        return None
    try:
        expr = sympy.sympify(text, evaluate=True)
    except Exception:
        return None
    if expr.free_symbols:
        return None
    simplified = sympy.nsimplify(expr)
    if percent:
        simplified = simplified / 100
    if simplified.is_Rational:
        if simplified.q == 1:
            return str(int(simplified))
        return f"{int(simplified.p)}/{int(simplified.q)}"
    numeric = simplified.evalf()
    if numeric == int(numeric):
        return str(int(numeric))
    return format(float(numeric), ".12g")


def values_equal(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return bool(sympy.simplify(sympy.sympify(left) - sympy.sympify(right)) == 0)
    except Exception:
        return False


def parse_gold_answer(answer_text: str) -> ParsedAnswer:
    matches = _GSM8K_ANSWER_RE.findall(answer_text)
    if not matches:
        return ParsedAnswer(extracted_answer=None, normalized_answer=None, reason="missing_hash_answer")
    extracted = sanitize_answer_surface(matches[-1])
    return ParsedAnswer(
        extracted_answer=extracted,
        normalized_answer=normalize_numeric_text(extracted),
        reason="hash_extraction",
    )


def format_gsm8k_cot_answer(answer_text: str) -> str:
    parsed = parse_gold_answer(answer_text)
    final_answer = sanitize_answer_surface(parsed.extracted_answer)
    reasoning = _GSM8K_ANSWER_RE.sub("", answer_text).strip()
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    if final_answer is None:
        return answer_text.strip()
    if reasoning:
        return f"{reasoning}\nFINAL_ANSWER: {final_answer}"
    return f"FINAL_ANSWER: {final_answer}"


def extract_generated_answer(completion: str) -> ParsedAnswer:
    matches = list(_FINAL_ANSWER_RE.finditer(completion))
    if not matches:
        stripped_lines = [line.strip() for line in completion.splitlines() if line.strip()]
        if len(stripped_lines) == 1:
            extracted = sanitize_answer_surface(stripped_lines[0])
            normalized = normalize_numeric_text(extracted)
            if normalized is not None:
                return ParsedAnswer(
                    extracted_answer=extracted,
                    normalized_answer=normalized,
                    reason="single_line_numeric_fallback",
                )
        hash_matches = _GSM8K_ANSWER_RE.findall(completion)
        if hash_matches:
            extracted = sanitize_answer_surface(hash_matches[-1])
            normalized = normalize_numeric_text(extracted)
            return ParsedAnswer(
                extracted_answer=extracted,
                normalized_answer=normalized,
                reason="hash_answer_fallback",
            )
        return ParsedAnswer(extracted_answer=None, normalized_answer=None, reason="missing_final_answer")
    extracted = sanitize_answer_surface(matches[-1].group(1))
    return ParsedAnswer(
        extracted_answer=extracted,
        normalized_answer=normalize_numeric_text(extracted),
        reason="final_answer",
    )


def extract_reasoning_block(completion: str) -> str | None:
    match = _REASONING_BLOCK_RE.search(completion)
    if not match:
        return None
    return sanitize_answer_surface(match.group(1))
