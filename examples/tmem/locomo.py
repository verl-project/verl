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


def parse_qa_pairs(text: str) -> list[dict[str, str]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        values = json.loads(text[start : end + 1])
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
