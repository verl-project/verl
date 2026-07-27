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

EXTRACTION_PROMPT = """You are given one problem to solve, previous extracted QA pairs and one conversation session.
Your task is to create high-quality supervised fine-tuning (SFT) QA pairs
grounded ONLY in this session.
Question:
<question> {question} </question>
Previous extracted QA pairs:
<qa_history> {qa_history} </qa_history>
Session:
<session> {chunk} </session>

Generate more pairs when the session contains rich, concrete facts, fewer when
it contains little useful evidence, and an empty array when it contains none.
Pairs may capture preferences, plans, events, temporal details, and useful
lessons. Every question must be answerable from explicit session information;
answers must be concise, factual, and directly supported. Cover diverse facts
and avoid duplicate or near-duplicate pairs, including pairs already present
in the QA history.

Return only a JSON array. Each item must have this form:
{{"instruction": "<question>", "output": "<answer>"}}"""

ANSWER_PROMPT = """Answer the question using the conversation and facts retained in your model.
If the information is absent, answer "not mentioned". Give only the concise answer.

Conversation:
{context}

Question: {question}
Answer:"""


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
            lines = [f"Date: {conversation[f'session_{index}_date_time']}"]
            for turn in turns:
                text = turn["text"]
                if turn.get("blip_caption"):
                    text = f"{text} [Image: {turn['blip_caption']}]"
                lines.append(f"{turn['speaker']}: {text}")
            sessions.append("\n".join(lines))
        index += 1
    return sessions


def pack_context_chunks(sessions: Iterable[str], tokenizer, token_budget: int) -> list[str]:
    """Pack ordered sessions into chunks no longer than the trigger budget."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for session in sessions:
        session_tokens = tokenizer.encode(session, add_special_tokens=False)
        if len(session_tokens) > token_budget:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            for offset in range(0, len(session_tokens), token_budget):
                chunks.append(tokenizer.decode(session_tokens[offset : offset + token_budget]))
            continue
        if current and current_tokens + len(session_tokens) > token_budget:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(session)
        current_tokens += len(session_tokens)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


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
    answer = qa.get("answer")
    return "not mentioned" if answer is None else str(answer)


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
