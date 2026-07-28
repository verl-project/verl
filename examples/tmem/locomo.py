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

"""LoCoMo data preparation and scoring for the TMEM reproduction."""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXTRACTION_SYSTEM_PROMPT = (
    "You are given one problem to solve, previous extracted QA pairs and one conversation session.\n"
    "Your task is to create high-quality supervised fine-tuning (SFT) QA pairs\n"
    "grounded ONLY in this session.\n"
    "Question:\n"
    "<question> {question} </question>\n"
    "Previous extracted QA pairs:\n"
    "<qa_history> {qa_history} </qa_history>\n"
    "Session:\n"
    "<session> {chunk} </session>"
)

MEMORY_WRITING_PROMPT = """Task: Generate grounded SFT QA pairs from the current session.

Given the problem to solve, previous conversation history. Now you should create high-quality supervised fine-tuning
(SFT) QA pairs grounded on the history.

Requirements:
1. Generate QA pairs adaptively based on how much useful information is present in the session.
   - If the session contains rich, concrete facts, generate more QA pairs.
   - If the session has limited useful evidence, generate fewer QA pairs.
   - If there is no usable evidence, return an empty JSON array.
2. You can generate QA pairs that capture the lessons learned from the session to help improve future interactions,
   such as preferences, plans, events, and temporal details, rather than just factual questions.
3. Each question must be answerable using explicit information from the session.
4. Each answer must be concise, factual, and directly supported by the session.
5. Cover diverse types when possible: who/what/when/where, preferences, plans, events, and temporal details.
6. Avoid duplicate or near-duplicate QA pairs, and keep wording natural and clear.

Return ONLY a JSON array. Each item must be:
{
  "instruction": "<question>",
  "output": "<answer>"
}

Output the generated SFT QA pairs in the specified JSON format. Do not include any explanations or additional text."""

EXTRACTION_END_SENTINEL = "</qa_pairs>"

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant whose job is to understand the following conversation and "
    "answer questions based on the conversation.\n"
    "If you don't know the answer to a question, please don't share false information."
)

ANSWER_PROMPT = (
    "Below is a conversation between two people: {speaker_a} and {speaker_b}. The conversation takes place over "
    "multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"
    "{context}\n\n"
    "Based on the above conversations, write a short answer for the following question in a few words. Do not write "
    "complete and lengthy sentences. Answer with exact words from the conversations whenever possible.\n\n"
    "Question: {question}"
)


def load_locomo(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list) or len(data) != 10:
        raise ValueError(f"Expected the official LoCoMo-10 list, got {type(data).__name__} of length {len(data)}")
    return data


def conversation_sessions(sample: dict[str, Any]) -> list[str]:
    conversation = sample["conversation"]
    sessions = []
    index = 1
    while f"session_{index}_date_time" in conversation:
        turns = conversation.get(f"session_{index}")
        if turns:
            lines = [f"DATE: {conversation[f'session_{index}_date_time']}", "CONVERSATION:"]
            for turn in turns:
                text = f'{turn["speaker"]} said, "{turn["text"]}"'
                if turn.get("blip_caption"):
                    text = f"{text}\n and shared {turn['blip_caption']}."
                lines.append(text)
            sessions.append("\n".join(lines))
        index += 1
    return sessions


def pack_context_chunks(sessions: Iterable[str], tokenizer, token_budget: int) -> list[str]:
    """Return trigger contexts followed by the final working context.

    A complete conversation session is first appended to the working context.
    If that makes the context exceed the budget, the accumulated context is
    emitted for memory writing and the working context is cleared.  The final
    element is therefore always the unconsumed working context (possibly
    empty), which is the only raw conversation passed to answer generation.
    """
    trigger_contexts: list[str] = []
    current: list[str] = []
    for session in sessions:
        current.append(session)
        context = "\n\n".join(current)
        if len(tokenizer.encode(context, add_special_tokens=False)) > token_budget:
            trigger_contexts.append(context)
            current = []
    return [*trigger_contexts, "\n\n".join(current)]


@dataclass(frozen=True)
class QAPairParseResult:
    """Structured result for auditing memory-extraction generations."""

    pairs: list[dict[str, str]]
    status: str
    error: str | None = None
    dropped_items: int = 0

    @property
    def valid(self) -> bool:
        return self.status in {"ok", "empty", "partial"}


def _json_array_span(text: str) -> tuple[int, int | None, str]:
    """Locate the first top-level JSON array without treating `]` in strings as its end."""
    start = text.find("[")
    if start < 0:
        return -1, None, "no_array"

    stack: list[str] = []
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if not stack or stack[-1] != expected:
                return start, None, "malformed_brackets"
            stack.pop()
            if not stack:
                return start, index + 1, "complete"
    return start, None, "incomplete_array"


def has_complete_json_array(text: str) -> bool:
    """Return whether text contains a lexically complete top-level JSON array."""
    return _json_array_span(text)[2] == "complete"


def parse_qa_pairs_result(text: str) -> QAPairParseResult:
    """Parse one extraction without silently converting malformed output to no supervision."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "").strip()
    start, end, span_status = _json_array_span(text)
    if span_status != "complete" or end is None:
        return QAPairParseResult([], span_status, f"Could not find a complete top-level JSON array ({span_status})")
    try:
        values = json.loads(text[start:end])
    except json.JSONDecodeError as error:
        return QAPairParseResult([], "invalid_json", str(error))
    if not isinstance(values, list):
        return QAPairParseResult([], "invalid_schema", "Top-level JSON value is not an array")
    pairs = []
    for value in values:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("instruction"), str)
            or not isinstance(value.get("output"), str)
        ):
            continue
        instruction = value["instruction"].strip()
        output = value["output"].strip()
        if (
            not instruction
            or not output
            or "?" not in instruction
            or " ".join(instruction.split()).casefold() == " ".join(output.split()).casefold()
        ):
            continue
        pairs.append({"instruction": instruction, "output": output})
    dropped_items = len(values) - len(pairs)
    if dropped_items and not pairs:
        return QAPairParseResult(
            [],
            "invalid_schema",
            f"All {len(values)} entries lack non-empty string instruction/output",
            dropped_items=dropped_items,
        )
    if dropped_items:
        return QAPairParseResult(
            pairs,
            "partial",
            f"Dropped {dropped_items} of {len(values)} schema-invalid entries",
            dropped_items=dropped_items,
        )
    return QAPairParseResult(pairs, "ok" if pairs else "empty")


def parse_qa_pairs(text: str) -> list[dict[str, str]]:
    """Compatibility parser; extraction training uses :func:`parse_qa_pairs_result`."""
    result = parse_qa_pairs_result(text)
    if result.valid:
        return result.pairs

    # Preserve the public helper's historical tolerance of individually invalid
    # entries. The runner itself deliberately does not use this fallback.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "").strip()
    start, end, span_status = _json_array_span(text)
    if span_status != "complete" or end is None:
        return []
    try:
        values = json.loads(text[start:end])
    except json.JSONDecodeError:
        return []
    if not isinstance(values, list):
        return []
    return [
        {"instruction": value["instruction"].strip(), "output": value["output"].strip()}
        for value in values
        if isinstance(value, dict)
        and isinstance(value.get("instruction"), str)
        and isinstance(value.get("output"), str)
        and value["instruction"].strip()
        and value["output"].strip()
    ]


def deduplicate_qa_pairs(
    pairs: Iterable[dict[str, str]],
    existing_pairs: Iterable[dict[str, str]] = (),
) -> tuple[list[dict[str, str]], int]:
    """Drop exact semantic duplicates without guessing at near-duplicate meaning.

    Case and repeated whitespace are ignored when comparing pairs. The first
    spelling of a pair is preserved so SFT targets are otherwise unchanged.
    """

    def key(pair: dict[str, str]) -> tuple[str, str]:
        return (
            " ".join(pair["instruction"].split()).casefold(),
            " ".join(pair["output"].split()).casefold(),
        )

    seen = {key(pair) for pair in existing_pairs}
    unique = []
    duplicate_count = 0
    for pair in pairs:
        pair_key = key(pair)
        if pair_key in seen:
            duplicate_count += 1
            continue
        seen.add(pair_key)
        unique.append(pair)
    return unique, duplicate_count


def reference_answer(qa: dict[str, Any]) -> str:
    if int(qa["category"]) == 5:
        return "No information available"
    answer = qa.get("answer")
    return "not mentioned" if answer is None else str(answer)


def prepare_question(qa: dict[str, Any], *, no_information_first: bool) -> tuple[str, dict[str, str] | None]:
    """Apply the official LoCoMo temporal and adversarial question protocol."""
    question = qa["question"]
    if qa["category"] == 2:
        return f"{question} Use DATE of CONVERSATION to answer with an approximate date.", None
    if qa["category"] != 5:
        return question, None

    distractor = str(qa["adversarial_answer"])
    unavailable = "No information available"
    if no_information_first:
        options = {"a": unavailable, "b": distractor}
    else:
        options = {"a": distractor, "b": unavailable}
    question = f"{question} (a) {options['a']} (b) {options['b']}. Select the correct answer by writing (a) or (b)."
    return question, options


def postprocess_prediction(text: str, options: dict[str, str] | None) -> str:
    """Normalize the official answer wrapper and resolve category-5 options."""
    text = text.replace('\\"', "'").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    answer = lines[0] if lines else ""
    if options is not None:
        normalized = answer.lower()
        selected = "a" if "(a)" in normalized or normalized in {"a", "a)"} else "b"
        return options[selected]
    answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.IGNORECASE)
    return re.sub(r"^(?:\([ab]\)|[ab]\))\s*", "", answer, flags=re.IGNORECASE).strip()


def normalize_answer(text: str) -> str:
    """Apply the normalization stated in TMEM: case, punctuation, whitespace."""
    punctuation = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(punctuation).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_records(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    records = list(records)
    if not records:
        return {"count": 0, "f1": 0.0, "em": 0.0}
    return {
        "count": len(records),
        "f1": 100 * sum(token_f1(record["prediction"], record["reference"]) for record in records) / len(records),
        "em": 100 * sum(exact_match(record["prediction"], record["reference"]) for record in records) / len(records),
    }


def score_breakdown(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Score all questions, non-adversarial questions, and each category."""
    records = list(records)
    categories = sorted({int(record["category"]) for record in records})
    return {
        **score_records(records),
        "without_category_5": score_records(record for record in records if int(record["category"]) != 5),
        "by_category": {
            str(category): score_records(record for record in records if int(record["category"]) == category)
            for category in categories
        },
    }
