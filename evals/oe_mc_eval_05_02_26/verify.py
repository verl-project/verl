#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

# Ensure repo-root + local imports regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import ParseMethod, score_gsm8k_with_verl, write_json
from verl.utils.reward_score import gsm8k as verl_gsm8k


VerificationStatus = str
VALID_STATUSES = {"correct", "incorrect", "unknown", "parsing_error"}
GROUND_TRUTH_FIELDS = ("ground_truth", "canonical_answer", "answer", "gold_answer", "target_answer")


def _derive_out_json(answers_json: str) -> str:
    in_path = Path(answers_json)
    if not in_path.name.endswith("_answer.json"):
        raise ValueError(
            f"Input answers filename must end with '_answer.json'; got '{in_path.name}'."
        )
    return str(in_path.with_name(f"{in_path.name[: -len('_answer.json')]}_evaluation.json"))


def _read_records(path: str) -> List[Dict[str, Any]]:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if in_path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Reading parquet dataset artifacts requires `pyarrow` to be installed.") from exc
        records = pq.read_table(in_path).to_pylist()
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"Parquet file '{path}' must contain rows that convert to objects.")
        return records
    if in_path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with in_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL record at line {line_number} must be an object.")
                records.append(value)
        return records
    if in_path.suffix != ".json":
        raise ValueError(f"Unsupported file type for '{path}'. Expected .json, .jsonl, or .parquet.")
    with in_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"JSON file '{path}' must contain a top-level list.")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"JSON record {index} in '{path}' must be an object.")
    return value


def _normalize_question_id(value: Any) -> str:
    return str(value)


def _load_answers(path: str) -> List[Dict[str, Any]]:
    records = _read_records(path)
    normalized: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        identity = record.get("identity")
        payload = record.get("payload")
        if not isinstance(identity, dict):
            raise ValueError(f"Answers record {index} is missing object 'identity'.")
        if not isinstance(payload, dict):
            raise ValueError(f"Answers record {index} is missing object 'payload'.")
        if "question_id" not in identity:
            raise ValueError(f"Answers record {index} is missing 'identity.question_id'.")
        if "responses" not in payload:
            raise ValueError(f"Answers record {index} is missing 'payload.responses'.")
        responses = payload["responses"]
        if not isinstance(responses, list):
            raise ValueError(f"'payload.responses' for answers record {index} must be a list.")
        if any(not isinstance(response, str) for response in responses):
            raise ValueError(f"All responses for answers record {index} must be strings.")
        normalized.append(
            {
                "question_id": identity["question_id"],
                "responses": responses,
            }
        )
    return normalized


def _extract_ground_truth(example: Dict[str, Any], *, dataset_path: str, record_index: int) -> str:
    for field_name in GROUND_TRUTH_FIELDS:
        raw = example.get(field_name)
        if raw in (None, ""):
            continue
        raw_text = str(raw).strip()
        parsed = verl_gsm8k.extract_solution(raw_text, method="strict")
        return parsed or raw_text
    raise ValueError(
        f"Dataset record {record_index} in '{dataset_path}' is missing a canonical answer field. "
        f"Tried: {', '.join(GROUND_TRUTH_FIELDS)}"
    )


def _load_dataset_lookup(path: str) -> Dict[str, Dict[str, Any]]:
    records = _read_records(path)
    lookup: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(tqdm(records, desc="load dataset", dynamic_ncols=True)):
        if "question_id" not in record:
            raise ValueError(f"Dataset record {index} is missing 'question_id'.")
        normalized_question_id = _normalize_question_id(record["question_id"])
        if normalized_question_id in lookup:
            raise ValueError(f"Duplicate dataset question_id '{record['question_id']}' in '{path}'.")
        raw_answer = None
        for field_name in GROUND_TRUTH_FIELDS:
            raw = record.get(field_name)
            if raw in (None, ""):
                continue
            raw_answer = str(raw)
            break
        lookup[normalized_question_id] = {
            "question_id": record["question_id"],
            "answer": raw_answer,
            "ground_truth": _extract_ground_truth(record, dataset_path=path, record_index=index),
        }
    return lookup


def _verify_response(*, response: str, ground_truth: str, method: ParseMethod) -> Tuple[VerificationStatus, Optional[str]]:
    try:
        score = score_gsm8k_with_verl(completion=response, ground_truth=ground_truth, method=method)
    except Exception:
        return "parsing_error", None
    parsed_solution = score.get("pred")
    if not score.get("format_ok", False):
        return "parsing_error", None
    if score.get("correct") is True:
        return "correct", parsed_solution
    if score.get("correct") is False:
        return "incorrect", parsed_solution
    return "unknown", parsed_solution


def _build_evaluation_records(
    *,
    answers_records: List[Dict[str, Any]],
    dataset_lookup: Dict[str, Dict[str, Any]],
    parse_methods: List[ParseMethod],
) -> List[Dict[str, Any]]:
    evaluation_records: List[Dict[str, Any]] = []
    for record in tqdm(answers_records, desc="verify answers", dynamic_ncols=True):
        question_id = record["question_id"]
        lookup_key = _normalize_question_id(question_id)
        if lookup_key not in dataset_lookup:
            raise ValueError(f"No dataset example found for question_id '{question_id}'.")
        dataset_example = dataset_lookup[lookup_key]
        verify_result: Dict[str, List[List[Optional[str] | VerificationStatus]]] = {}
        for method in parse_methods:
            method_results: List[List[Optional[str] | VerificationStatus]] = []
            for response in record["responses"]:
                status, solution = _verify_response(
                    response=response,
                    ground_truth=dataset_example["ground_truth"],
                    method=method,
                )
                method_results.append([solution, status])
            verify_result[method] = method_results
        evaluation_records.append(
            {
                "question_id": question_id,
                "answer": dataset_example["answer"],
                "verify_result": verify_result,
            }
        )
    return evaluation_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify GSM8K sampled answers against a dataset artifact.")
    parser.add_argument(
        "--answers_json",
        required=True,
        help="Path to *_answer.json containing identity.question_id and payload.responses.",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to local JSON/JSONL/parquet dataset artifact.",
    )
    parser.add_argument(
        "--parse_methods",
        nargs="+",
        choices=["strict", "flexible"],
        required=True,
        help="One or more GSM8K parse methods to apply.",
    )
    parser.add_argument("--out_json", default=None, help="Optional explicit output path.")
    args = parser.parse_args()

    answers_records = _load_answers(args.answers_json)
    dataset_lookup = _load_dataset_lookup(args.dataset_path)
    parse_methods: List[ParseMethod] = [method for method in args.parse_methods]  # type: ignore[assignment]
    evaluation_records = _build_evaluation_records(
        answers_records=answers_records,
        dataset_lookup=dataset_lookup,
        parse_methods=parse_methods,
    )

    out_json = args.out_json or _derive_out_json(args.answers_json)
    write_json(out_json, evaluation_records)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
