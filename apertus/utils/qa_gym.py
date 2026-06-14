"""QA Gym dataset loading helpers."""

from __future__ import annotations

import json
import os
from typing import Any

import datasets


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_qa_gym_rl_pairs_jsonl(path: str) -> datasets.Dataset:
    """Load cleaned QA Gym RL pairs with id, question, prompt, and answer fields."""
    rows = []
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = normalize_text(record.get("prompt"))
            answer = normalize_text(record.get("answer"))
            if not prompt or not answer:
                continue
            rows.append(
                {
                    "id": normalize_text(record.get("id")) or str(row_index),
                    "question": normalize_text(record.get("question")),
                    "prompt": prompt,
                    "answer": answer,
                }
            )
    return datasets.Dataset.from_list(rows)
