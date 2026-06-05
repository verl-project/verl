# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Build a GSM8K multiple-choice dataset from model-sampled open-ended responses.

The builder reads:
- an evaluation file (`*_evaluation.json`) containing per-response correctness
- the sibling answer file (`*_answer.json`) containing the sampled responses

For each GSM8K question it samples:
- one final visible correct option
- N visible wrong options

`--number_of_multiple_choice_correct` controls how many correct source responses
are sampled before selecting the single displayed correct option. This keeps the
final prompt compatible with the current reward/scoring stack, which expects one
ground-truth letter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets

DEFAULT_SAVE_DIR = "~/data/gsm8k_mc_sampled"
DEFAULT_DATA_SOURCE = "gsm8k_mc_sampled"
ALL_OPTION_LABELS = list(string.ascii_uppercase)
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"


def extract_solution(solution_str: str) -> str:
    solution = re.search(r"####\s*(\-?[0-9\.\,]+)", solution_str)
    if solution is None:
        raise ValueError("Unable to extract final solution from GSM8K answer.")
    return solution.group(1).replace(",", "").strip()


def format_multiple_choice_prompt(
    question_raw: str,
    options: dict[str, str],
    include_cot_phrase: bool,
    allow_none: bool,
) -> str:
    option_lines = [f"{label}: {text}" for label, text in options.items()]
    options_block = "\n".join(option_lines)

    suffix = 'output the letter of the final answer choice after "####".'
    if allow_none:
        suffix = 'output the final answer choice after "####". Output NONE if every numeric option is incorrect.'
    if include_cot_phrase:
        suffix = f"Let's think step by step and {suffix}"
    return f"{question_raw}\n\nOptions:\n{options_block}\n\n{suffix}"


def build_question_id(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()


def derive_answer_path(evaluation_path: Path) -> Path:
    name = evaluation_path.name
    if name.endswith("_evaluation.json"):
        return evaluation_path.with_name(name[: -len("_evaluation.json")] + "_answer.json")
    if name == "evaluation.json":
        return evaluation_path.with_name("answer.json")
    raise ValueError(
        f"Cannot derive answer file from {evaluation_path}. Expected a filename ending in '_evaluation.json'."
    )


def config_prefixed_dir(path: str, *, augment: int, num_correct: int, num_wrong: int) -> str:
    expanded = os.path.expanduser(path)
    parent, base = os.path.split(expanded)
    base = base or os.path.basename(os.path.normpath(expanded))
    tags = []
    if augment != 1:
        tags.append(f"aug{augment}x")
    if num_correct != 1 or num_wrong != 3:
        tags.append(f"c{num_correct}w{num_wrong}")
    if not tags:
        return expanded
    return os.path.join(parent, f"{'_'.join(tags)}_{base}")


@dataclass(frozen=True)
class Candidate:
    response_index: int
    response_text: str
    parsed_answer: str
    status: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_eval_map(evaluation_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    eval_map: dict[str, dict[str, Any]] = {}
    for record in evaluation_records:
        question_id = record["question_id"]
        if question_id in eval_map:
            raise ValueError(f"Duplicate question_id in evaluation file: {question_id}")
        eval_map[question_id] = record
    return eval_map


def build_answer_map(answer_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    answer_map: dict[str, dict[str, Any]] = {}
    for record in answer_records:
        identity = record.get("identity") or {}
        if identity.get("dataset_name") != "gsm8k":
            continue
        question_id = identity.get("question_id")
        if question_id is None:
            raise ValueError("Answer record is missing identity.question_id")
        if question_id in answer_map:
            raise ValueError(f"Duplicate question_id in answer file: {question_id}")
        answer_map[question_id] = record
    return answer_map


def group_wrong_candidates(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.parsed_answer, []).append(candidate)
    return grouped


def build_candidate_pools(
    *,
    answer_record: dict[str, Any],
    evaluation_record: dict[str, Any],
    parse_method: str,
) -> tuple[list[Candidate], dict[str, list[Candidate]]]:
    responses = ((answer_record.get("payload") or {}).get("responses")) or []
    verify_results_by_method = evaluation_record.get("verify_result") or {}
    if parse_method not in verify_results_by_method:
        question_id = evaluation_record.get("question_id")
        available = ", ".join(sorted(verify_results_by_method.keys())) or "<none>"
        raise ValueError(
            f"Evaluation record for question_id={question_id} does not contain parse_method={parse_method!r}. "
            f"Available methods: {available}. Re-run verification with this parse method or pass --parse_method."
        )
    verify_result = verify_results_by_method[parse_method] or []

    if len(responses) != len(verify_result):
        question_id = evaluation_record.get("question_id")
        raise ValueError(
            f"Mismatched response/verdict counts for question_id={question_id}: "
            f"{len(responses)} responses vs {len(verify_result)} verdicts."
        )

    correct_candidates: list[Candidate] = []
    wrong_candidates: list[Candidate] = []
    for idx, (response_text, verdict) in enumerate(zip(responses, verify_result, strict=True)):
        if isinstance(verdict, dict):
            parsed_answer = verdict.get("parsed_solution")
            status = verdict.get("label")
        elif isinstance(verdict, list) and len(verdict) >= 2:
            parsed_answer, status = verdict[0], verdict[1]
        else:
            continue
        if status not in {"correct", "incorrect"}:
            continue
        if not isinstance(parsed_answer, str) or not parsed_answer.strip():
            continue
        candidate = Candidate(
            response_index=idx,
            response_text=str(response_text),
            parsed_answer=parsed_answer.strip(),
            status=status,
        )
        if status == "correct":
            correct_candidates.append(candidate)
        else:
            wrong_candidates.append(candidate)

    return correct_candidates, group_wrong_candidates(wrong_candidates)


def sample_variant(
    *,
    question_id: str,
    split: str,
    question_raw: str,
    wrong_candidate_groups: dict[str, list[Candidate]],
    augment_variant: int,
    orig_index: int,
    data_source_name: str,
    rng: random.Random,
    include_cot_phrase: bool,
    number_of_multiple_choice_correct: int,
    number_of_multiple_choice_wrong: int,
    augment_total: int,
    gold_solution: str,
    gold_answer: str,
    parse_method: str,
) -> dict[str, Any]:
    wrong_values = rng.sample(sorted(wrong_candidate_groups.keys()), number_of_multiple_choice_wrong)
    sampled_wrong_candidates = [rng.choice(wrong_candidate_groups[value]) for value in wrong_values]

    options_payload = []
    correct_letter = "NONE"
    correct_choice = None
    correct_option_source = None
    if number_of_multiple_choice_correct == 1:
        options_payload.append(
            {
                "option_text": gold_solution,
                "candidate": {
                    "response_index": None,
                    "response_text": gold_answer,
                    "parsed_answer": gold_solution,
                    "status": "ground_truth",
                    "source": "ground_truth",
                },
                "is_correct": True,
            }
        )
        correct_option_source = "ground_truth"

    for candidate in sampled_wrong_candidates:
        options_payload.append(
            {
                "option_text": candidate.parsed_answer,
                "candidate": {
                    "response_index": candidate.response_index,
                    "response_text": candidate.response_text,
                    "parsed_answer": candidate.parsed_answer,
                    "status": candidate.status,
                    "source": "sampled_wrong",
                },
                "is_correct": False,
            }
        )

    rng.shuffle(options_payload)

    option_labels = ALL_OPTION_LABELS[: len(options_payload)]
    labeled_options: dict[str, str] = {}
    option_metadata = []
    for label, payload in zip(option_labels, options_payload, strict=True):
        labeled_options[label] = payload["option_text"]
        if payload["is_correct"]:
            correct_letter = label
            correct_choice = label
        candidate = payload["candidate"]
        option_metadata.append(
            {
                "label": label,
                "option_text": payload["option_text"],
                "is_correct": payload["is_correct"],
                "response_index": candidate["response_index"],
                "response_text": candidate["response_text"],
                "parsed_answer": candidate["parsed_answer"],
                "status": candidate["status"],
                "source": candidate["source"],
            }
        )

    prompt = format_multiple_choice_prompt(
        question_raw=question_raw,
        options=labeled_options,
        include_cot_phrase=include_cot_phrase,
        allow_none=number_of_multiple_choice_correct == 0,
    )

    return {
        "data_source": data_source_name,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "math_mc",
        "reward_model": {"style": "rule", "ground_truth": correct_letter},
        "extra_info": {
            "split": split,
            "index": orig_index * augment_total + augment_variant,
            "orig_index": orig_index,
            "augment_variant": augment_variant,
            "augment_total": augment_total,
            "question_id": question_id,
            "question_raw": question_raw,
            "gold_solution": gold_solution,
            "correct_choice": correct_choice,
            "correct_option_source": correct_option_source,
            "number_of_multiple_choice_correct": number_of_multiple_choice_correct,
            "number_of_multiple_choice_wrong": number_of_multiple_choice_wrong,
            "sampled_candidate_parse_method": parse_method,
            "options": labeled_options,
            "option_records": option_metadata,
            "sampled_correct_response_indices": [],
            "sampled_correct_response_texts": [],
            "selected_correct_response_index": None,
            "selected_correct_response_text": None,
            "selected_wrong_response_indices": [candidate.response_index for candidate in sampled_wrong_candidates],
            "selected_wrong_response_texts": [candidate.response_text for candidate in sampled_wrong_candidates],
        },
    }


def preprocess_split(
    *,
    dataset_split,
    split_name: str,
    answer_map: dict[str, dict[str, Any]],
    eval_map: dict[str, dict[str, Any]],
    rng: random.Random,
    include_cot_phrase: bool,
    number_of_multiple_choice_correct: int,
    number_of_multiple_choice_wrong: int,
    augment: int,
    data_source_name: str,
    parse_method: str,
) -> tuple[datasets.Dataset, dict[str, int]]:
    records = []
    stats = {
        "seen": 0,
        "missing_eval_or_answer": 0,
        "insufficient_wrong": 0,
        "written": 0,
    }

    for idx, example in enumerate(dataset_split):
        stats["seen"] += 1
        question_raw = example["question"]
        gold_answer = example["answer"]
        gold_solution = extract_solution(example["answer"])
        question_id = build_question_id(question_raw)

        answer_record = answer_map.get(question_id)
        evaluation_record = eval_map.get(question_id)
        if answer_record is None or evaluation_record is None:
            stats["missing_eval_or_answer"] += 1
            continue

        identity = answer_record.get("identity") or {}
        if identity.get("split") != split_name:
            continue

        _, wrong_candidate_groups = build_candidate_pools(
            answer_record=answer_record,
            evaluation_record=evaluation_record,
            parse_method=parse_method,
        )

        if len(wrong_candidate_groups) < number_of_multiple_choice_wrong:
            stats["insufficient_wrong"] += 1
            continue

        for augment_variant in range(augment):
            records.append(
                sample_variant(
                    question_id=question_id,
                    split=split_name,
                    question_raw=question_raw,
                    wrong_candidate_groups=wrong_candidate_groups,
                    augment_variant=augment_variant,
                    orig_index=idx,
                    data_source_name=data_source_name,
                    rng=rng,
                    include_cot_phrase=include_cot_phrase,
                    number_of_multiple_choice_correct=number_of_multiple_choice_correct,
                    number_of_multiple_choice_wrong=number_of_multiple_choice_wrong,
                    augment_total=augment,
                    gold_solution=gold_solution,
                    gold_answer=gold_answer,
                    parse_method=parse_method,
                )
            )
            stats["written"] += 1

    return datasets.Dataset.from_list(records), stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True, help="Path to the *_evaluation.json file.")
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_save_dir", default=DEFAULT_SAVE_DIR, help="Output directory for parquet files.")
    parser.add_argument(
        "--number_of_multiple_choice_wrong",
        type=int,
        default=3,
        help="How many visible incorrect options to show per question.",
    )
    parser.add_argument(
        "--number_of_multiple_choice_correct",
        type=int,
        default=1,
        help="0: create an all-wrong question with target NONE. 1: use the GSM8K ground-truth numeric answer as the single correct option.",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=1,
        help="How many independently sampled variants to create per original question.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for response sampling.")
    parser.add_argument(
        "--parse_method",
        choices=["strict", "flexible"],
        default="flexible",
        help="Verification result bucket to use when selecting sampled wrong answers.",
    )
    parser.add_argument(
        "--no_cot_phrase",
        action="store_true",
        help="Remove the \"Let's think step by step\" phrase from the prompt suffix.",
    )

    args = parser.parse_args()

    if args.number_of_multiple_choice_wrong < 1:
        raise ValueError("--number_of_multiple_choice_wrong must be >= 1.")
    if args.number_of_multiple_choice_correct not in {0, 1}:
        raise ValueError("--number_of_multiple_choice_correct must be 0 or 1 for now.")
    if args.augment < 1:
        raise ValueError("--augment must be >= 1.")
    total_options = args.number_of_multiple_choice_wrong + args.number_of_multiple_choice_correct
    if total_options > len(ALL_OPTION_LABELS):
        raise ValueError(
            f"Total number of visible options ({total_options}) exceeds the supported maximum "
            f"of {len(ALL_OPTION_LABELS)}."
        )

    evaluation_path = Path(os.path.expanduser(args.evaluation)).resolve()
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {evaluation_path}")
    answer_path = derive_answer_path(evaluation_path)
    if not answer_path.exists():
        raise FileNotFoundError(f"Derived answer file not found: {answer_path}")

    legacy_local_dir = args.local_dir
    if legacy_local_dir is not None:
        local_save_dir = legacy_local_dir
    else:
        local_save_dir = args.local_save_dir
        if local_save_dir == DEFAULT_SAVE_DIR:
            local_save_dir = config_prefixed_dir(
                local_save_dir,
                augment=args.augment,
                num_correct=args.number_of_multiple_choice_correct,
                num_wrong=args.number_of_multiple_choice_wrong,
            )

    val_data_source = os.path.basename(os.path.normpath(os.path.expanduser(local_save_dir)))

    rng = random.Random(args.seed)
    evaluation_records = load_json(evaluation_path)
    answer_records = load_json(answer_path)
    if not isinstance(evaluation_records, list):
        raise ValueError("Evaluation JSON must be a list of records.")
    if not isinstance(answer_records, list):
        raise ValueError("Answer JSON must be a list of records.")

    eval_map = build_eval_map(evaluation_records)
    answer_map = build_answer_map(answer_records)
    gsm8k_dataset = datasets.load_dataset(GSM8K_DATASET, GSM8K_CONFIG)

    train_dataset, train_stats = preprocess_split(
        dataset_split=gsm8k_dataset["train"],
        split_name="train",
        answer_map=answer_map,
        eval_map=eval_map,
        rng=rng,
        include_cot_phrase=not args.no_cot_phrase,
        number_of_multiple_choice_correct=args.number_of_multiple_choice_correct,
        number_of_multiple_choice_wrong=args.number_of_multiple_choice_wrong,
        augment=args.augment,
        data_source_name=DEFAULT_DATA_SOURCE,
        parse_method=args.parse_method,
    )
    test_dataset, test_stats = preprocess_split(
        dataset_split=gsm8k_dataset["test"],
        split_name="test",
        answer_map=answer_map,
        eval_map=eval_map,
        rng=rng,
        include_cot_phrase=not args.no_cot_phrase,
        number_of_multiple_choice_correct=args.number_of_multiple_choice_correct,
        number_of_multiple_choice_wrong=args.number_of_multiple_choice_wrong,
        augment=args.augment,
        data_source_name=val_data_source,
        parse_method=args.parse_method,
    )

    if legacy_local_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")

    print(f"Using evaluation file: {evaluation_path}")
    print(f"Using answer file: {answer_path}")
    print(
        "Sampling config: "
        f"correct_source_count={args.number_of_multiple_choice_correct}, "
        f"wrong_visible_count={args.number_of_multiple_choice_wrong}, "
        f"augment={args.augment}, "
        f"parse_method={args.parse_method}"
    )
    print(f"Train stats: {train_stats}")
    print(f"Test stats: {test_stats}")

    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        print(f"Skipping HDFS copy for example. Would copy to {args.hdfs_dir}")
