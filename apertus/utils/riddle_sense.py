"""Utilities for adapting RiddleSense samples for RL preprocessing."""

from __future__ import annotations

from typing import Any


def normalize_riddle_sense_sample(sample: dict[str, Any]) -> dict[str, Any]:
    choices = sample["choices"]
    labels = [str(label).strip() for label in choices["label"]]
    texts = [str(text).strip() for text in choices["text"]]
    assert len(labels) == len(texts), "RiddleSense labels and choice texts have different lengths"

    answer_key = str(sample["answerKey"]).strip()
    answer_index = None
    for index, label in enumerate(labels):
        if label.upper() == answer_key.upper():
            answer_index = index
            break
    if answer_index is None:
        raise ValueError(f"RiddleSense answer label not found: {answer_key!r}")

    return {
        "question": str(sample["question"]).strip(),
        "choices": texts,
        "answer_index": answer_index,
        "answer_text": texts[answer_index],
        "metadata": {
            "source_labels": labels,
            "source_answer_key": answer_key,
        },
    }
