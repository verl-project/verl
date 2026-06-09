# Copyright 2026 The VERL Team and individual contributors.
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
"""Rule-based reward function for Table-GPT rows."""

from __future__ import annotations

import json
import re
from typing import Any

from sklearn.metrics import f1_score

JSON_PARSING_ERROR = "JSONParsingError"

ANSWER_KEYS = {
    "ColumnFinding": "result",
    "ColumnTypeAnnotation": "chosen_semantic_type",
    "DataImputation": "value",
    "Row2RowTransformation": "output_value",
    "TableQuestion": "answer",
    "EntityMatching": "answer",
    "MissingValueIdentification": ["missing_col", "row_id"],
    "ErrorDetection": "erroneous_cells",
    "SchemaMatching": "column_mappings",
}


def _task_from_inputs(data_source: str, extra_info: dict[str, Any] | None) -> str:
    if isinstance(extra_info, dict) and extra_info.get("task"):
        return str(extra_info["task"])
    prefix = "tablegpt/"
    if isinstance(data_source, str) and data_source.startswith(prefix):
        return data_source[len(prefix) :]
    return ""


def _get_evaluate_fn(task: str) -> callable | None:
    return {
        "EntityMatching": evaluate_em,
        "ErrorDetection": evaluate_ed,
        "ColumnFinding": evaluate_cf,
        "DataImputation": evaluate_di,
        "Row2RowTransformation": evaluate_r2r,
        "TableQuestion": evaluate_tq,
        "SchemaMatching": evaluate_sm,
        "ColumnTypeAnnotation": evaluate_cta,
        "MissingValueIdentification": evaluate_mvi,
    }.get(task)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score a generated completion against a TableGPT row."""
    task = _task_from_inputs(data_source, extra_info)
    answer_key = ANSWER_KEYS.get(task)
    eval_fn = _get_evaluate_fn(task)
    if answer_key is None or eval_fn is None:
        print(f"Warning: Table-GPT task {task!r} is not supported; returning 0 reward.")
        return 0.0

    y_true = [extract_json_answer(ground_truth, answer_key)]
    y_pred = [extract_json_answer(solution_str or "", answer_key)]
    try:
        return float(eval_fn(y_true, y_pred))
    except Exception as exc:
        print(
            f"Warning: Table-GPT scoring failed for task {task!r}; "
            f"returning 0 reward. Error: {exc}"
        )
        return 0.0


def _extract_from_json_obj(result: dict[str, Any], answer_key: str | list[str]) -> Any:
    if isinstance(answer_key, list):
        for key in answer_key:
            if key in result:
                return result[key]
    elif answer_key in result:
        return result[answer_key]
    return JSON_PARSING_ERROR


def extract_json_answer(text: Any, answer_key: str | list[str]) -> Any:
    assert answer_key is not None
    if isinstance(text, dict):
        return _extract_from_json_obj(text, answer_key)
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)

    pattern = r"{[^}]*}"
    matches = re.findall(pattern, text)
    for match in matches:
        try:
            result = json.loads(match)
        except Exception:
            continue
        if isinstance(result, dict):
            extracted = _extract_from_json_obj(result, answer_key)
            if extracted != JSON_PARSING_ERROR:
                return extracted
    return JSON_PARSING_ERROR


def _f1_from_counts(tp_count: int, fp_count: int, fn_count: int) -> float:
    prec = tp_count / (tp_count + fp_count) if tp_count + fp_count else 0.0
    recall = tp_count / (tp_count + fn_count) if tp_count + fn_count else 0.0
    return 2 * prec * recall / (prec + recall) if prec + recall else 0.0


def evaluate_em(y_true: list[Any], y_pred: list[Any]) -> float:
    y_true = [int(str(x).lower() == "yes") for x in y_true]
    y_pred = [int(str(x).lower() == "yes") for x in y_pred]
    return f1_score(y_true, y_pred, zero_division=0)


def evaluate_ed(y_true: list[Any], y_pred: list[Any]) -> float:
    tp_count = 0
    fp_count = 0
    fn_count = 0

    def preprocess(y: Any) -> set[Any]:
        if y == JSON_PARSING_ERROR or y is None:
            y = []
        elif isinstance(y, str):
            if y.lower() == "none":
                y = []
            else:
                y = [y]
        return set(y)

    for y_t, y_p in zip(y_true, y_pred, strict=False):
        y_true_set = preprocess(y_t)
        y_pred_set = preprocess(y_p)
        tp_set = list(y_true_set.intersection(y_pred_set))[:1]

        n_p = min(len(y_true_set), 1)
        n_tp = min(len(tp_set), 1)
        n_fp = 0
        for y_pred_item in y_pred_set:
            if y_pred_item not in y_true_set:
                n_fp += 1
        n_fn = n_p - n_tp

        tp_count += len(tp_set)
        fp_count += n_fp
        fn_count += n_fn

    return _f1_from_counts(tp_count, fp_count, fn_count)


def evaluate_cf(y_true: list[Any], y_pred: list[Any]) -> float:
    corrects = [
        int(str(y_t).lower() == str(y_p).lower())
        for y_t, y_p in zip(y_true, y_pred, strict=False)
    ]
    return sum(corrects) / len(corrects) if corrects else 0.0


def evaluate_di(y_true: list[Any], y_pred: list[Any]) -> float:
    corrects = [
        int(str(y_t).lower() == str(y_p).lower())
        for y_t, y_p in zip(y_true, y_pred, strict=False)
    ]
    return sum(corrects) / len(corrects) if corrects else 0.0


def evaluate_r2r(y_true: list[Any], y_pred: list[Any]) -> float:
    corrects = [int(y_t == y_p) for y_t, y_p in zip(y_true, y_pred, strict=False)]
    return sum(corrects) / len(corrects) if corrects else 0.0


def evaluate_tq(y_true: list[Any], y_pred: list[Any]) -> float:
    corrects = [
        int(str(y_t).lower() == str(y_p).lower())
        for y_t, y_p in zip(y_true, y_pred, strict=False)
    ]
    return sum(corrects) / len(corrects) if corrects else 0.0


def evaluate_sm(y_true: list[Any], y_pred: list[Any]) -> float:
    def parse_matching(label: Any) -> dict[str, str]:
        gt = {}
        if not isinstance(label, list):
            return gt
        for x in label:
            if len(x) != 2:
                continue
            source, target = x
            source = source.strip()
            target = target.strip()
            if len(target) == 0:
                continue
            gt[source] = target
        return gt

    recalls = []
    for y_t, y_p in zip(y_true, y_pred, strict=False):
        y_t = parse_matching(y_t)
        y_p = parse_matching(y_p)
        count = 0
        for key in y_p.keys():
            if key in y_t and y_p[key] == y_t[key]:
                count += 1
        if len(y_t) == 0:
            recall = None
        else:
            recall = count / len(y_t)
        recalls.append(recall)
    valid_recalls = [recall for recall in recalls if recall is not None]
    return sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0


def evaluate_cta(y_true: list[Any], y_pred: list[Any]) -> float:
    tp_count = 0
    fp_count = 0
    fn_count = 0

    for y_t, y_p in zip(y_true, y_pred, strict=False):
        if str(y_t) == "None":
            n_p = 0
        else:
            n_p = 1

        if str(y_p) != "None" and y_p == y_t:
            n_tp = 1
        else:
            n_tp = 0

        if str(y_p) == "None":
            n_pp = 0
        else:
            n_pp = 1

        n_fp = n_pp - n_tp
        n_fn = n_p - n_tp

        tp_count += n_tp
        fp_count += n_fp
        fn_count += n_fn

    return _f1_from_counts(tp_count, fp_count, fn_count)


def evaluate_mvi(y_true: list[Any], y_pred: list[Any]) -> float:
    corrects = [
        int(str(y_t).lower() == str(y_p).lower())
        for y_t, y_p in zip(y_true, y_pred, strict=False)
    ]
    return sum(corrects) / len(corrects) if corrects else 0.0
