from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from reliable_gsm8k.backends import create_generation_backend, get_backend_runtime_metadata
from reliable_gsm8k.judge import JudgeVerdict, parse_judge_output
from reliable_gsm8k.parsing import extract_generated_answer, extract_reasoning_block, format_gsm8k_cot_answer, parse_gold_answer, values_equal
from reliable_gsm8k.profiles import get_generator_profile, get_inference_profile, get_judge_profile
from reliable_gsm8k.prompts import (
    build_correct_solution_prompt,
    build_incorrect_solution_prompt,
    build_judge_prompt,
    build_mc_allwrong_prompt,
    build_mc_onecorrect_prompt,
    build_oe_prompt,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _artifact_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _question_id_from_question(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self._handle = None

    def __enter__(self) -> JsonlWriter:
        _ensure_dir(self.path.parent)
        self._handle = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("writer is not open")
        self._handle.write(json.dumps(record, sort_keys=True))
        self._handle.write("\n")
        self._handle.flush()
        self.count += 1


@dataclass
class AcceptedSolution:
    role: str
    completion: str
    reasoning: str | None
    final_answer_text: str
    final_answer_normalized: str
    judge_verdict: str
    judge_reason: str
    generator_model: str
    judge_model: str
    attempt_index: int
    generation_raw_response: dict[str, Any]
    judge_raw_response: dict[str, Any]


@dataclass
class AttemptEvaluation:
    role_requested: str
    attempt_index: int
    completion: str
    extracted_answer: str | None
    normalized_answer: str | None
    label: str
    accepted: bool
    feedback: str
    error_type: str
    parser_message: str | None
    judge_verdict: str
    judge_reason: str
    judge_malformed: bool
    generator_model: str
    judge_model: str
    generation_raw_response: dict[str, Any]
    judge_raw_response: dict[str, Any]
    candidate_source: str
    acceptance_rejection_reason: str | None = None


_INCORRECT_MISTAKE_HINTS = [
    "one arithmetic step is added or subtracted incorrectly",
    "a number from the question is multiplied when it should be divided",
    "a percentage or fraction is applied incorrectly",
    "the result stops one step too early",
    "one quantity is counted twice or omitted once",
    "a ratio part is assigned to the wrong person or group",
    "a unit conversion is skipped",
    "the last subtraction or addition is reversed",
]


def _backend_uses_local_transformers(config: dict[str, Any]) -> bool:
    return str(config.get("backend")) == "transformers_causal_lm"


def _use_compact_generation_prompts(generator_profile_name: str, generator_model_name: str) -> bool:
    profile = generator_profile_name.lower()
    model = generator_model_name.lower()
    return "qwen" in profile or "qwen" in model


def detect_visible_gpu_ids(override_gpu_ids: str | None = None) -> list[str]:
    raw = override_gpu_ids
    if raw is None:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None and raw.strip():
        values = [value.strip() for value in raw.split(",") if value.strip() and value.strip() != "-1"]
        return values
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def should_enable_multi_gpu(
    *,
    generator_profile_name: str,
    judge_profile_name: str | None,
    gpu_ids: list[str],
    use_judge: bool,
) -> bool:
    if len(gpu_ids) <= 1:
        return False
    generator_config = get_generator_profile(generator_profile_name)
    if _backend_uses_local_transformers(generator_config):
        return True
    if use_judge and judge_profile_name is not None:
        judge_config = get_judge_profile(judge_profile_name)
        return _backend_uses_local_transformers(judge_config)
    return False


def _stable_shuffle(seed: int, item_id: str, choices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    shuffled = list(choices)
    rng.shuffle(shuffled)
    return shuffled


def _load_gsm8k_source_records(split: str, max_items: int | None) -> list[tuple[int, dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main")[split]
    if max_items is not None:
        dataset = dataset.select(range(min(max_items, len(dataset))))
    return [(index, example) for index, example in enumerate(dataset)]


def _select_worker_records(
    records: list[tuple[int, dict[str, Any]]],
    *,
    worker_index: int | None,
    num_workers: int,
) -> list[tuple[int, dict[str, Any]]]:
    if worker_index is None or num_workers <= 1:
        return records
    return [(global_index, example) for global_index, example in records if global_index % num_workers == worker_index]


def _judge_candidate(
    *,
    question: str,
    gold_answer: str,
    completion: str,
    judge_backend,
    judge_model_name: str | None,
    judge_max_tokens: int,
    seed: int,
    metadata: dict[str, Any],
) -> tuple[JudgeVerdict, dict[str, Any]]:
    response = judge_backend.generate(
        prompt=build_judge_prompt(question, gold_answer, completion),
        sampling={"temperature": 0.0, "max_tokens": judge_max_tokens, "n": 1, "seed": seed},
        metadata=metadata,
    )[0]
    verdict = parse_judge_output(response.text)
    raw = {
        "text": response.text,
        "raw_response": response.raw_response,
        "judge_model": judge_model_name,
    }
    return verdict, raw


def _mistake_hint_for_attempt(attempt_index: int) -> str:
    return _INCORRECT_MISTAKE_HINTS[attempt_index % len(_INCORRECT_MISTAKE_HINTS)]


def _build_generator_sampling(*, inference_config: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "max_tokens": int(inference_config.get("max_new_tokens", 512)),
        "max_length": int(inference_config.get("max_length", 2048)),
        "do_sample": bool(inference_config.get("do_sample", False)),
        "repetition_penalty": float(inference_config.get("repetition_penalty", 1.0)),
        "temperature": float(inference_config.get("temperature", 1.0)),
        "top_p": float(inference_config.get("top_p", 1.0)),
        "use_chat_template": bool(inference_config.get("use_chat_template", True)),
        "system_prompt": str(inference_config.get("system_prompt", "You are a helpful assistant.") or ""),
        "n": 1,
        "seed": seed,
    }


def _parser_error_type(*, extracted_answer: str | None, normalized_answer: str | None) -> str:
    if extracted_answer is None:
        return "no_final_answer"
    if normalized_answer is None:
        return "invalid_numeric_format"
    return "none"


def _evaluate_attempt(
    *,
    role: str,
    question: str,
    gold_answer: str,
    completion: str,
    generator_model_name: str,
    judge_model_name: str,
    judge_backend,
    judge_max_tokens: int,
    seed: int,
    attempt_index: int,
    metadata: dict[str, Any],
    use_judge: bool,
    seen_incorrect_answers: set[str] | None = None,
) -> tuple[AttemptEvaluation, int]:
    parsed = extract_generated_answer(completion)
    judge_requests_used = 0
    judge_raw: dict[str, Any] = {"mode": "disabled"}
    judge_verdict = "not_invoked"
    judge_reason = "judge_disabled"
    judge_malformed = False

    if parsed.extracted_answer is None or parsed.normalized_answer is None:
        label = "parsing_error"
        error_type = _parser_error_type(
            extracted_answer=parsed.extracted_answer,
            normalized_answer=parsed.normalized_answer,
        )
        feedback = f"parser could not extract a usable numeric final answer ({parsed.reason})"
    else:
        parsed_is_correct = values_equal(parsed.normalized_answer, gold_answer)
        if use_judge:
            judge_requests_used += 1
            verdict, judge_raw = _judge_candidate(
                question=question,
                gold_answer=gold_answer,
                completion=completion,
                judge_backend=judge_backend,
                judge_model_name=str(judge_model_name),
                judge_max_tokens=judge_max_tokens,
                seed=seed,
                metadata=metadata,
            )
        else:
            verdict = JudgeVerdict(
                verdict="correct" if parsed_is_correct else "incorrect",
                reason="parser_only_acceptance",
                malformed=False,
            )
        judge_verdict = verdict.verdict
        judge_reason = verdict.reason
        judge_malformed = verdict.malformed

        if verdict.verdict == "unresolved":
            label = "unknown"
            error_type = "comparison_error"
            feedback = f"judge returned unresolved after parser extracted {parsed.normalized_answer!r}"
        elif verdict.verdict == "correct" and not parsed_is_correct:
            label = "unknown"
            error_type = "comparison_error"
            feedback = "judge said correct but parsed answer does not equal GSM8K gold answer"
        elif verdict.verdict == "incorrect" and parsed_is_correct:
            label = "unknown"
            error_type = "comparison_error"
            feedback = "judge said incorrect but parsed answer equals GSM8K gold answer"
        elif parsed_is_correct:
            label = "correct"
            error_type = "none"
            feedback = "parsed final answer matches the GSM8K gold answer"
        else:
            label = "incorrect"
            error_type = "none"
            feedback = "parsed final answer differs from the GSM8K gold answer"

    accepted = False
    rejection_reason: str | None = None
    if role == "correct":
        accepted = label == "correct"
        if not accepted:
            rejection_reason = f"requested_correct_but_label_was_{label}"
    else:
        accepted = label == "incorrect"
        if accepted and seen_incorrect_answers is not None and parsed.normalized_answer in seen_incorrect_answers:
            accepted = False
            rejection_reason = "duplicate_incorrect_answer"
        elif not accepted:
            rejection_reason = f"requested_incorrect_but_label_was_{label}"

    if rejection_reason is not None:
        feedback = f"{feedback}; not accepted because {rejection_reason}"

    return (
        AttemptEvaluation(
            role_requested=role,
            attempt_index=attempt_index,
            completion=completion.strip(),
            extracted_answer=parsed.extracted_answer,
            normalized_answer=parsed.normalized_answer,
            label=label,
            accepted=accepted,
            feedback=feedback,
            error_type=error_type,
            parser_message=parsed.reason,
            judge_verdict=judge_verdict,
            judge_reason=judge_reason,
            judge_malformed=judge_malformed,
            generator_model=generator_model_name,
            judge_model=str(judge_model_name or "disabled"),
            generation_raw_response=metadata,
            judge_raw_response=judge_raw,
            candidate_source="sampled_from_model",
            acceptance_rejection_reason=rejection_reason,
        ),
        judge_requests_used,
    )


def _accepted_solution_from_attempt(attempt: AttemptEvaluation) -> AcceptedSolution | None:
    if not attempt.accepted or attempt.extracted_answer is None or attempt.normalized_answer is None:
        return None
    return AcceptedSolution(
        role=attempt.role_requested,
        completion=attempt.completion,
        reasoning=extract_reasoning_block(attempt.completion),
        final_answer_text=attempt.extracted_answer,
        final_answer_normalized=attempt.normalized_answer,
        judge_verdict=attempt.judge_verdict,
        judge_reason=attempt.judge_reason,
        generator_model=attempt.generator_model,
        judge_model=attempt.judge_model,
        attempt_index=attempt.attempt_index,
        generation_raw_response=attempt.generation_raw_response,
        judge_raw_response=attempt.judge_raw_response,
    )


def _generate_one(
    *,
    generator_backend,
    prompt: str,
    inference_config: dict[str, Any],
    seed: int,
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    response = generator_backend.generate(
        prompt=prompt,
        sampling=_build_generator_sampling(inference_config=inference_config, seed=seed),
        metadata=metadata,
    )[0]
    return response.text, {"text": response.text, "raw_response": response.raw_response}


def _collect_correct_solution(
    *,
    question: str,
    gold_answer: str,
    generator_backend,
    generator_model_name: str,
    judge_backend,
    judge_model_name: str | None,
    inference_config: dict[str, Any],
    judge_max_tokens: int,
    correct_max_attempts: int,
    seed: int,
    item_id: str,
    compact_prompt: bool,
    use_judge: bool,
) -> tuple[AcceptedSolution | None, int, int, list[AttemptEvaluation]]:
    judge_requests_used = 0
    attempts: list[AttemptEvaluation] = []
    for attempt_index in range(correct_max_attempts):
        prompt = build_correct_solution_prompt(question, compact=compact_prompt)
        completion, generation_raw = _generate_one(
            generator_backend=generator_backend,
            prompt=prompt,
            inference_config=inference_config,
            seed=seed + 1000 + attempt_index,
            metadata={"item_id": item_id, "role": "correct", "attempt_index": attempt_index},
        )
        attempt, judge_requests = _evaluate_attempt(
            role="correct",
            question=question,
            gold_answer=gold_answer,
            completion=completion,
            generator_model_name=generator_model_name,
            judge_model_name=judge_model_name,
            judge_backend=judge_backend,
            judge_max_tokens=judge_max_tokens,
            seed=seed + 2000 + attempt_index,
            attempt_index=attempt_index,
            metadata=generation_raw,
            use_judge=use_judge,
        )
        attempts.append(attempt)
        judge_requests_used += judge_requests
        accepted = _accepted_solution_from_attempt(attempt)
        if accepted is not None:
            return accepted, attempt_index + 1, judge_requests_used, attempts
    return None, correct_max_attempts, judge_requests_used, attempts


def _collect_incorrect_solutions(
    *,
    question: str,
    gold_answer: str,
    generator_backend,
    generator_model_name: str,
    judge_backend,
    judge_model_name: str | None,
    inference_config: dict[str, Any],
    judge_max_tokens: int,
    incorrect_target_count: int,
    incorrect_max_attempts: int,
    seed: int,
    item_id: str,
    compact_prompt: bool,
    use_judge: bool,
) -> tuple[list[AcceptedSolution], int, int, list[AttemptEvaluation]]:
    accepted: list[AcceptedSolution] = []
    attempts: list[AttemptEvaluation] = []
    seen_final_answers: set[str] = set()
    judge_requests_used = 0
    attempts_used = 0
    for attempt_index in range(incorrect_max_attempts):
        if len(accepted) >= incorrect_target_count:
            break
        attempts_used += 1
        prompt = build_incorrect_solution_prompt(
            question,
            gold_answer,
            banned_final_answers=sorted(seen_final_answers),
            mistake_hint=_mistake_hint_for_attempt(attempt_index),
            compact=compact_prompt,
        )
        completion, generation_raw = _generate_one(
            generator_backend=generator_backend,
            prompt=prompt,
            inference_config=inference_config,
            seed=seed + 3000 + attempt_index,
            metadata={"item_id": item_id, "role": "incorrect", "attempt_index": attempt_index},
        )
        attempt, judge_requests = _evaluate_attempt(
            role="incorrect",
            question=question,
            gold_answer=gold_answer,
            completion=completion,
            generator_model_name=generator_model_name,
            judge_model_name=judge_model_name,
            judge_backend=judge_backend,
            judge_max_tokens=judge_max_tokens,
            seed=seed + 4000 + attempt_index,
            attempt_index=attempt_index,
            metadata=generation_raw,
            use_judge=use_judge,
            seen_incorrect_answers=seen_final_answers,
        )
        attempts.append(attempt)
        judge_requests_used += judge_requests
        candidate = _accepted_solution_from_attempt(attempt)
        if candidate is None:
            continue
        accepted.append(candidate)
        seen_final_answers.add(candidate.final_answer_normalized)
    return accepted, attempts_used, judge_requests_used, attempts


def _solution_to_option_text(solution: AcceptedSolution) -> str:
    return solution.final_answer_normalized


def _build_oe_record(
    *,
    item_id: str,
    question_id: str,
    question: str,
    gold_answer: str,
    gold_solution_text: str,
    data_source: str,
) -> dict[str, Any]:
    return {
        "data_source": data_source,
        "question_id": question_id,
        "prompt": [{"role": "user", "content": build_oe_prompt(question)}],
        "target_answer": gold_solution_text,
        "reward_model": {"style": "rule", "ground_truth": gold_answer},
        "extra_info": {
            "item_id": item_id,
            "question_id": question_id,
            "question": question,
            "gold_answer_semantic": gold_answer,
            "candidate_source": "original",
            "solution_source": "gsm8k",
        },
    }


def _build_mc_onecorrect_record(
    *,
    item_id: str,
    question_id: str,
    question: str,
    gold_answer: str,
    correct_solution: AcceptedSolution,
    incorrect_solutions: list[AcceptedSolution],
    data_source: str,
    seed: int,
) -> dict[str, Any]:
    choices = [{"role": "correct", "solution": correct_solution}] + [{"role": "incorrect", "solution": value} for value in incorrect_solutions[:3]]
    shuffled = _stable_shuffle(seed, item_id, choices)
    labels = ["A", "B", "C", "D"]
    options: dict[str, str] = {}
    option_roles: dict[str, str] = {}
    option_final_answers: dict[str, str] = {}
    option_sources: dict[str, str] = {}
    correct_choice = None
    for label, choice in zip(labels, shuffled, strict=True):
        solution = choice["solution"]
        options[label] = _solution_to_option_text(solution)
        option_roles[label] = choice["role"]
        option_final_answers[label] = solution.final_answer_normalized
        option_sources[label] = "sampled_from_model"
        if choice["role"] == "correct":
            correct_choice = label
    return {
        "data_source": data_source,
        "question_id": question_id,
        "prompt": [{"role": "user", "content": build_mc_onecorrect_prompt(question, options)}],
        "target_answer": f"#### {correct_choice}",
        "reward_model": {"style": "rule", "ground_truth": correct_choice},
        "extra_info": {
            "item_id": item_id,
            "question_id": question_id,
            "question": question,
            "gold_answer_semantic": gold_answer,
            "correct_choice": correct_choice,
            "options": options,
            "option_roles": option_roles,
            "option_final_answers": option_final_answers,
            "option_sources": option_sources,
            "candidate_source_mode": "sampled_from_model",
        },
    }


def _build_mc_allwrong_record(
    *,
    item_id: str,
    question_id: str,
    question: str,
    gold_answer: str,
    correct_solution: AcceptedSolution | None,
    incorrect_solutions: list[AcceptedSolution],
    data_source: str,
    seed: int,
) -> dict[str, Any]:
    shuffled = _stable_shuffle(seed, item_id, [{"role": "incorrect", "solution": value} for value in incorrect_solutions[:4]])
    labels = ["A", "B", "C", "D"]
    options: dict[str, str] = {}
    option_final_answers: dict[str, str] = {}
    option_sources: dict[str, str] = {}
    for label, choice in zip(labels, shuffled, strict=True):
        solution = choice["solution"]
        options[label] = _solution_to_option_text(solution)
        option_final_answers[label] = solution.final_answer_normalized
        option_sources[label] = "sampled_from_model"
    return {
        "data_source": data_source,
        "question_id": question_id,
        "prompt": [{"role": "user", "content": build_mc_allwrong_prompt(question, options)}],
        "target_answer": "#### NONE",
        "reward_model": {"style": "rule", "ground_truth": "NONE"},
        "extra_info": {
            "item_id": item_id,
            "question_id": question_id,
            "question": question,
            "gold_answer_semantic": gold_answer,
            "correct_choice": None,
            "options": options,
            "option_final_answers": option_final_answers,
            "option_sources": option_sources,
            "candidate_source_mode": "sampled_from_model",
        },
    }


def _resolve_run_dir(output_root: Path, run_id: str, worker_index: int | None) -> Path:
    if worker_index is None:
        return output_root / run_id
    return output_root / run_id / "shards" / f"shard_{worker_index:02d}"


def _item_id_sort_key(item_id: str | None) -> tuple[str, int]:
    text = str(item_id or "")
    prefix, _, suffix = text.rpartition(":")
    try:
        return prefix, int(suffix)
    except ValueError:
        return text, -1


def _record_sort_key(record: dict[str, Any]) -> tuple[tuple[str, int], str]:
    item_id = record.get("item_id")
    if item_id is None:
        item_id = (record.get("identity") or {}).get("item_id")
    if item_id is None:
        item_id = (record.get("extra_info") or {}).get("item_id")
    secondary = str(record.get("data_source") or "")
    return _item_id_sort_key(item_id), secondary


def _merge_jsonl_files(output_path: Path, input_paths: list[Path]) -> int:
    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        records.extend(_iter_jsonl(input_path))
    records.sort(key=_record_sort_key)
    with JsonlWriter(output_path) as writer:
        for record in records:
            writer.write(record)
        return writer.count


def _write_parquet_from_jsonl(jsonl_path: Path) -> Path | None:
    records = list(_iter_jsonl(jsonl_path))
    if not records:
        return None
    from datasets import Dataset

    parquet_path = jsonl_path.with_suffix(".parquet")
    Dataset.from_list(records).to_parquet(str(parquet_path))
    return parquet_path


def _merge_json_array_files(output_path: Path, input_paths: list[Path]) -> int:
    records: list[dict[str, Any]] = []
    for input_path in input_paths:
        if not input_path.exists():
            continue
        with input_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {input_path}")
        records.extend(value)
    records.sort(key=_record_sort_key)
    _write_json(output_path, records)
    return len(records)


def _attempt_metadata_for_answer(attempt: AttemptEvaluation, response_index: int) -> dict[str, Any]:
    payload = asdict(attempt)
    payload.pop("completion", None)
    payload["response_index"] = response_index
    return payload


def _verification_entry_from_attempt(attempt: AttemptEvaluation, response_index: int) -> dict[str, Any]:
    return {
        "response_index": response_index,
        "label": attempt.label,
        "parsed_solution": attempt.normalized_answer,
        "parsed_answer_raw": attempt.extracted_answer,
        "accepted": attempt.accepted,
        "role_requested": attempt.role_requested,
        "feedback": attempt.feedback,
        "error_type": attempt.error_type,
        "parser_message": attempt.parser_message,
        "candidate_source": attempt.candidate_source,
        "judge_verdict": attempt.judge_verdict,
        "judge_reason": attempt.judge_reason,
        "judge_malformed": attempt.judge_malformed,
        "acceptance_rejection_reason": attempt.acceptance_rejection_reason,
    }


def _build_answer_artifact_record(
    *,
    item_id: str,
    question_id: str,
    split: str,
    question: str,
    gold_answer: str | None,
    attempts: list[AttemptEvaluation],
    generator_profile_name: str,
    inference_profile_name: str,
    judge_profile_name: str | None,
    use_judge: bool,
    artifact_timestamp: str,
    status: str,
) -> dict[str, Any]:
    responses = [attempt.completion for attempt in attempts]
    identity = {
        "question_id": question_id,
        "item_id": item_id,
        "dataset_name": "gsm8k",
        "split": split,
        "inference_id": inference_profile_name,
        "model_id": generator_profile_name,
        "judge_id": judge_profile_name if use_judge else None,
        "timestamp": artifact_timestamp,
    }
    return {
        "question_id": question_id,
        "item_id": item_id,
        "dataset_name": "gsm8k",
        "split": split,
        "question": question,
        "ground_truth": gold_answer,
        "status": status,
        "response_count": len(responses),
        "responses": responses,
        "response_metadata": [
            _attempt_metadata_for_answer(attempt, response_index=index)
            for index, attempt in enumerate(attempts)
        ],
        "identity": identity,
        "payload": {
            "responses": responses,
            "response_metadata": [
                _attempt_metadata_for_answer(attempt, response_index=index)
                for index, attempt in enumerate(attempts)
            ],
        },
    }


def _build_evaluation_artifact_record(
    *,
    item_id: str,
    question_id: str,
    split: str,
    question: str,
    gold_answer: str | None,
    attempts: list[AttemptEvaluation],
    status: str,
) -> dict[str, Any]:
    entries = [
        _verification_entry_from_attempt(attempt, response_index=index)
        for index, attempt in enumerate(attempts)
    ]
    compact_entries = [[entry["parsed_solution"], entry["label"]] for entry in entries]
    return {
        "question_id": question_id,
        "item_id": item_id,
        "dataset_name": "gsm8k",
        "split": split,
        "question": question,
        "answer": gold_answer,
        "ground_truth": gold_answer,
        "status": status,
        "response_count": len(attempts),
        "verification": {
            "reliable_numeric": entries,
        },
        "verify_result": {
            "reliable_numeric": compact_entries,
        },
    }


def _merge_run_stats(shard_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    merged_stats = {
        "source_item_count": 0,
        "gold_parse_failed": 0,
        "items_with_correct_solution": 0,
        "items_with_four_incorrect_solutions": 0,
        "oe_count": 0,
        "mc_onecorrect_count": 0,
        "mc_allwrong_count": 0,
        "generator_requests": 0,
        "judge_requests": 0,
        "skipped_item_ids": [],
    }
    for manifest in shard_manifests:
        stats = manifest.get("stats", {})
        for key in (
            "source_item_count",
            "gold_parse_failed",
            "items_with_correct_solution",
            "items_with_four_incorrect_solutions",
            "oe_count",
            "mc_onecorrect_count",
            "mc_allwrong_count",
            "generator_requests",
            "judge_requests",
        ):
            merged_stats[key] += int(stats.get(key, 0))
        merged_stats["skipped_item_ids"].extend(stats.get("skipped_item_ids", []))
    merged_stats["skipped_item_ids"] = sorted(set(merged_stats["skipped_item_ids"]), key=lambda value: _item_id_sort_key(value))
    return merged_stats


def merge_run_shards(
    *,
    run_id: str,
    split: str,
    output_root: Path,
    generator_profile_name: str,
    inference_profile_name: str,
    judge_profile_name: str | None,
    use_judge: bool,
    num_workers: int,
    gpu_ids: list[str],
    max_items: int | None,
) -> dict[str, Any]:
    run_dir = _ensure_dir(output_root / run_id)
    shard_dirs = [run_dir / "shards" / f"shard_{index:02d}" for index in range(num_workers)]
    missing = [path for path in shard_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard directories: {missing}")

    shard_manifests = []
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
        shard_manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    generator_slug = generator_profile_name
    items_dir = _ensure_dir(run_dir / "items")
    datasets_dir = _ensure_dir(run_dir / "datasets")
    artifacts_dir = _ensure_dir(run_dir / "artifacts")
    oe_dir = _ensure_dir(datasets_dir / f"gsm8k_oe_{generator_slug}")
    mc_onecorrect_dir = _ensure_dir(datasets_dir / f"gsm8k_mc_onecorrect_{generator_slug}")
    mc_allwrong_dir = _ensure_dir(datasets_dir / f"gsm8k_mc_allwrong_{generator_slug}")
    artifact_timestamp = min(str(manifest.get("artifact_timestamp", _artifact_timestamp())) for manifest in shard_manifests)
    answers_path = artifacts_dir / f"answers_{artifact_timestamp}.json"
    evaluation_path = artifacts_dir / f"evaluation_{artifact_timestamp}.json"

    def artifact_input_paths(key: str) -> list[Path]:
        paths: list[Path] = []
        for shard_dir, shard_manifest in zip(shard_dirs, shard_manifests, strict=True):
            raw_path = ((shard_manifest.get("artifacts") or {}).get(key))
            if raw_path is None:
                continue
            path = Path(raw_path)
            paths.append(path if path.is_absolute() else shard_dir / path)
        return paths

    record_counts = {
        "items": _merge_jsonl_files(items_dir / f"{split}.jsonl", [path / "items" / f"{split}.jsonl" for path in shard_dirs]),
        "oe": _merge_jsonl_files(
            oe_dir / f"{split}.jsonl",
            [path / "datasets" / f"gsm8k_oe_{generator_slug}" / f"{split}.jsonl" for path in shard_dirs],
        ),
        "mc_onecorrect": _merge_jsonl_files(
            mc_onecorrect_dir / f"{split}.jsonl",
            [path / "datasets" / f"gsm8k_mc_onecorrect_{generator_slug}" / f"{split}.jsonl" for path in shard_dirs],
        ),
        "mc_allwrong": _merge_jsonl_files(
            mc_allwrong_dir / f"{split}.jsonl",
            [path / "datasets" / f"gsm8k_mc_allwrong_{generator_slug}" / f"{split}.jsonl" for path in shard_dirs],
        ),
        "answers": _merge_json_array_files(answers_path, artifact_input_paths("answers")),
        "evaluation": _merge_json_array_files(evaluation_path, artifact_input_paths("evaluation")),
    }
    parquet_paths = {
        "items": _write_parquet_from_jsonl(items_dir / f"{split}.jsonl"),
        "oe": _write_parquet_from_jsonl(oe_dir / f"{split}.jsonl"),
        "mc_onecorrect": _write_parquet_from_jsonl(mc_onecorrect_dir / f"{split}.jsonl"),
        "mc_allwrong": _write_parquet_from_jsonl(mc_allwrong_dir / f"{split}.jsonl"),
    }

    manifest = {
        "run_id": run_id,
        "artifact_timestamp": artifact_timestamp,
        "started_at": min(str(manifest.get("started_at", "")) for manifest in shard_manifests),
        "completed_at": max(str(manifest.get("completed_at", "")) for manifest in shard_manifests),
        "split": split,
        "max_items": max_items,
        "generator_profile": generator_profile_name,
        "generator_model_path": shard_manifests[0].get("generator_model_path"),
        "inference_profile": inference_profile_name,
        "inference_config": shard_manifests[0].get("inference_config", {}),
        "judge_profile": judge_profile_name if use_judge else None,
        "generator_runtime": shard_manifests[0].get("generator_runtime", {}),
        "judge_runtime": shard_manifests[0].get("judge_runtime", {"enabled": False}),
        "candidate_source_mode": "sampled_from_model_live",
        "supported_candidate_sources": ["original", "sampled_from_model", "programmatic", "path_to_answers"],
        "active_candidate_sources": {
            "gsm8k_oe": ["original"],
            "gsm8k_mc_onecorrect": ["sampled_from_model"],
            "gsm8k_mc_allwrong": ["sampled_from_model"],
        },
        "artifacts": {
            "answers": str(answers_path.resolve()),
            "evaluation": str(evaluation_path.resolve()),
            "parquet": {
                key: str(path.resolve())
                for key, path in parquet_paths.items()
                if path is not None
            },
        },
        "record_counts": record_counts,
        "stats": _merge_run_stats(shard_manifests),
        "multi_gpu": {
            "enabled": True,
            "num_workers": num_workers,
            "gpu_ids": gpu_ids,
            "worker_run_dirs": [str(path.resolve()) for path in shard_dirs],
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest


def build_run(
    *,
    run_id: str,
    split: str,
    max_items: int | None,
    output_root: Path,
    generator_profile_name: str,
    generator_model_path: str | None,
    inference_profile_name: str,
    judge_profile_name: str | None,
    judge_max_tokens: int,
    incorrect_target_count: int,
    incorrect_max_attempts: int,
    correct_max_attempts: int,
    seed: int,
    use_judge: bool,
    worker_index: int | None = None,
    num_workers: int = 1,
    gpu_id: str | None = None,
) -> dict[str, Any]:
    generator_config = get_generator_profile(generator_profile_name)
    if generator_model_path is not None:
        generator_config["model_name"] = generator_model_path
        if "tokenizer_name" in generator_config:
            generator_config["tokenizer_name"] = generator_model_path
    inference_config = get_inference_profile(inference_profile_name)
    generator_backend = create_generation_backend(generator_config)
    generator_model_name = str(generator_config["model_name"])
    judge_config = get_judge_profile(judge_profile_name) if use_judge and judge_profile_name is not None else None
    judge_backend = create_generation_backend(judge_config) if judge_config is not None else None
    judge_model_name = str(judge_config["model_name"]) if judge_config is not None else None
    compact_generation_prompts = _use_compact_generation_prompts(generator_profile_name, generator_model_name)

    run_dir = _ensure_dir(_resolve_run_dir(output_root, run_id, worker_index))
    items_dir = _ensure_dir(run_dir / "items")
    datasets_dir = _ensure_dir(run_dir / "datasets")
    artifacts_dir = _ensure_dir(run_dir / "artifacts")
    artifact_timestamp = _artifact_timestamp()
    answers_path = artifacts_dir / f"answers_{artifact_timestamp}.json"
    evaluation_path = artifacts_dir / f"evaluation_{artifact_timestamp}.json"
    answer_artifact_records: list[dict[str, Any]] = []
    evaluation_artifact_records: list[dict[str, Any]] = []

    generator_slug = generator_profile_name
    oe_dir = _ensure_dir(datasets_dir / f"gsm8k_oe_{generator_slug}")
    mc_onecorrect_dir = _ensure_dir(datasets_dir / f"gsm8k_mc_onecorrect_{generator_slug}")
    mc_allwrong_dir = _ensure_dir(datasets_dir / f"gsm8k_mc_allwrong_{generator_slug}")

    source_records = _load_gsm8k_source_records(split, max_items)
    selected_records = _select_worker_records(source_records, worker_index=worker_index, num_workers=num_workers)
    worker_label = f"worker={worker_index}/{num_workers} gpu={gpu_id}" if worker_index is not None else "worker=single"

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "artifact_timestamp": artifact_timestamp,
        "started_at": _utc_now(),
        "split": split,
        "max_items": max_items,
        "generator_profile": generator_profile_name,
        "generator_model_path": generator_model_path,
        "inference_profile": inference_profile_name,
        "inference_config": inference_config,
        "judge_profile": judge_profile_name if use_judge else None,
        "generator_runtime": get_backend_runtime_metadata(generator_backend, generator_config),
        "judge_runtime": get_backend_runtime_metadata(judge_backend, judge_config) if judge_config is not None else {"enabled": False},
        "candidate_source_mode": "sampled_from_model_live",
        "supported_candidate_sources": ["original", "sampled_from_model", "programmatic", "path_to_answers"],
        "active_candidate_sources": {
            "gsm8k_oe": ["original"],
            "gsm8k_mc_onecorrect": ["sampled_from_model"],
            "gsm8k_mc_allwrong": ["sampled_from_model"],
        },
        "artifacts": {
            "answers": str(answers_path.resolve()),
            "evaluation": str(evaluation_path.resolve()),
        },
        "stats": {
            "source_item_count": len(selected_records),
            "gold_parse_failed": 0,
            "items_with_correct_solution": 0,
            "items_with_four_incorrect_solutions": 0,
            "oe_count": 0,
            "mc_onecorrect_count": 0,
            "mc_allwrong_count": 0,
            "generator_requests": 0,
            "judge_requests": 0,
            "skipped_item_ids": [],
        },
        "multi_gpu": {
            "enabled": num_workers > 1,
            "worker_index": worker_index,
            "num_workers": num_workers,
            "gpu_id": gpu_id,
        },
    }

    with (
        JsonlWriter(items_dir / f"{split}.jsonl") as item_writer,
        JsonlWriter(oe_dir / f"{split}.jsonl") as oe_writer,
        JsonlWriter(mc_onecorrect_dir / f"{split}.jsonl") as mc_writer,
        JsonlWriter(mc_allwrong_dir / f"{split}.jsonl") as allwrong_writer,
    ):
        for local_index, (global_index, example) in enumerate(selected_records):
            item_id = f"gsm8k:{split}:{global_index}"
            question = str(example["question"])
            question_id = _question_id_from_question(question)
            original_answer_text = str(example["answer"])
            gold = parse_gold_answer(original_answer_text)
            if gold.normalized_answer is None or gold.extracted_answer is None:
                manifest["stats"]["gold_parse_failed"] += 1
                manifest["stats"]["skipped_item_ids"].append(item_id)
                answer_artifact_records.append(
                    _build_answer_artifact_record(
                        item_id=item_id,
                        question_id=question_id,
                        split=split,
                        question=question,
                        gold_answer=gold.normalized_answer,
                        attempts=[],
                        generator_profile_name=generator_profile_name,
                        inference_profile_name=inference_profile_name,
                        judge_profile_name=judge_profile_name,
                        use_judge=use_judge,
                        artifact_timestamp=artifact_timestamp,
                        status="skipped_gold_parse_failed",
                    )
                )
                evaluation_artifact_records.append(
                    _build_evaluation_artifact_record(
                        item_id=item_id,
                        question_id=question_id,
                        split=split,
                        question=question,
                        gold_answer=gold.normalized_answer,
                        attempts=[],
                        status="skipped_gold_parse_failed",
                    )
                )
                item_writer.write(
                    {
                        "item_id": item_id,
                        "question_id": question_id,
                        "split": split,
                        "question": question,
                        "original_answer": original_answer_text,
                        "gold_answer_raw": gold.extracted_answer,
                        "gold_answer_semantic": gold.normalized_answer,
                        "status": "skipped_gold_parse_failed",
                    }
                )
                print(
                    f"[reliable-build] {worker_label} item={local_index + 1}/{len(selected_records)} item_id={item_id} "
                    "status=skipped_gold_parse_failed",
                    flush=True,
                )
                continue

            gold_solution_text = format_gsm8k_cot_answer(original_answer_text)
            per_item_seed = seed + global_index * 10000
            correct_solution, correct_attempts_used, correct_judge_requests, correct_attempts = _collect_correct_solution(
                question=question,
                gold_answer=gold.normalized_answer,
                generator_backend=generator_backend,
                generator_model_name=generator_model_name,
                judge_backend=judge_backend,
                judge_model_name=judge_model_name,
                inference_config=inference_config,
                judge_max_tokens=judge_max_tokens,
                correct_max_attempts=correct_max_attempts,
                seed=per_item_seed,
                item_id=item_id,
                compact_prompt=compact_generation_prompts,
                use_judge=use_judge,
            )
            incorrect_solutions, incorrect_attempts_used, incorrect_judge_requests, incorrect_attempts = _collect_incorrect_solutions(
                question=question,
                gold_answer=gold.normalized_answer,
                generator_backend=generator_backend,
                generator_model_name=generator_model_name,
                judge_backend=judge_backend,
                judge_model_name=judge_model_name,
                inference_config=inference_config,
                judge_max_tokens=judge_max_tokens,
                incorrect_target_count=incorrect_target_count,
                incorrect_max_attempts=incorrect_max_attempts,
                seed=per_item_seed,
                item_id=item_id,
                compact_prompt=compact_generation_prompts,
                use_judge=use_judge,
            )
            all_attempts = correct_attempts + incorrect_attempts
            answer_artifact_records.append(
                _build_answer_artifact_record(
                    item_id=item_id,
                    question_id=question_id,
                    split=split,
                    question=question,
                    gold_answer=gold.normalized_answer,
                    attempts=all_attempts,
                    generator_profile_name=generator_profile_name,
                    inference_profile_name=inference_profile_name,
                    judge_profile_name=judge_profile_name,
                    use_judge=use_judge,
                    artifact_timestamp=artifact_timestamp,
                    status="processed",
                )
            )
            evaluation_artifact_records.append(
                _build_evaluation_artifact_record(
                    item_id=item_id,
                    question_id=question_id,
                    split=split,
                    question=question,
                    gold_answer=gold.normalized_answer,
                    attempts=all_attempts,
                    status="processed",
                )
            )

            manifest["stats"]["generator_requests"] += correct_attempts_used + incorrect_attempts_used
            manifest["stats"]["judge_requests"] += correct_judge_requests + incorrect_judge_requests
            if correct_solution is not None:
                manifest["stats"]["items_with_correct_solution"] += 1
            if len(incorrect_solutions) >= incorrect_target_count:
                manifest["stats"]["items_with_four_incorrect_solutions"] += 1

            item_record = {
                "item_id": item_id,
                "question_id": question_id,
                "split": split,
                "question": question,
                "original_answer": original_answer_text,
                "gold_answer_raw": gold.extracted_answer,
                "gold_answer_semantic": gold.normalized_answer,
                "gold_solution_cot": gold_solution_text,
                "correct_solution": asdict(correct_solution) if correct_solution is not None else None,
                "incorrect_solutions": [asdict(value) for value in incorrect_solutions],
                "candidate_source_mode": "sampled_from_model_live",
                "stats": {
                    "correct_attempts_used": correct_attempts_used,
                    "incorrect_attempts_used": incorrect_attempts_used,
                    "accepted_incorrect_count": len(incorrect_solutions),
                },
            }
            item_writer.write(item_record)

            oe_writer.write(
                _build_oe_record(
                    item_id=item_id,
                    question_id=question_id,
                    question=question,
                    gold_answer=gold.normalized_answer,
                    gold_solution_text=gold_solution_text,
                    data_source=f"gsm8k_oe_{generator_slug}",
                )
            )
            manifest["stats"]["oe_count"] += 1

            if correct_solution is not None and len(incorrect_solutions) >= 3:
                mc_writer.write(
                    _build_mc_onecorrect_record(
                        item_id=item_id,
                        question_id=question_id,
                        question=question,
                        gold_answer=gold.normalized_answer,
                        correct_solution=correct_solution,
                        incorrect_solutions=incorrect_solutions,
                        data_source=f"gsm8k_mc_onecorrect_{generator_slug}",
                        seed=seed,
                    )
                )
                manifest["stats"]["mc_onecorrect_count"] += 1

            if len(incorrect_solutions) >= 4:
                allwrong_writer.write(
                    _build_mc_allwrong_record(
                        item_id=item_id,
                        question_id=question_id,
                        question=question,
                        gold_answer=gold.normalized_answer,
                        correct_solution=correct_solution,
                        incorrect_solutions=incorrect_solutions,
                        data_source=f"gsm8k_mc_allwrong_{generator_slug}",
                        seed=seed,
                    )
                )
                manifest["stats"]["mc_allwrong_count"] += 1

            print(
                f"[reliable-build] {worker_label} item={local_index + 1}/{len(selected_records)} item_id={item_id} "
                f"correct={'yes' if correct_solution is not None else 'no'} "
                f"incorrect={len(incorrect_solutions)} "
                f"oe={manifest['stats']['oe_count']} mc1={manifest['stats']['mc_onecorrect_count']} "
                f"mc0={manifest['stats']['mc_allwrong_count']}",
                flush=True,
            )

        manifest["record_counts"] = {
            "items": item_writer.count,
            "oe": oe_writer.count,
            "mc_onecorrect": mc_writer.count,
            "mc_allwrong": allwrong_writer.count,
            "answers": len(answer_artifact_records),
            "evaluation": len(evaluation_artifact_records),
        }

    manifest["completed_at"] = _utc_now()
    _write_json(answers_path, sorted(answer_artifact_records, key=_record_sort_key))
    _write_json(evaluation_path, sorted(evaluation_artifact_records, key=_record_sort_key))
    parquet_paths = {
        "items": _write_parquet_from_jsonl(items_dir / f"{split}.jsonl"),
        "oe": _write_parquet_from_jsonl(oe_dir / f"{split}.jsonl"),
        "mc_onecorrect": _write_parquet_from_jsonl(mc_onecorrect_dir / f"{split}.jsonl"),
        "mc_allwrong": _write_parquet_from_jsonl(mc_allwrong_dir / f"{split}.jsonl"),
    }
    manifest["artifacts"]["parquet"] = {
        key: str(path.resolve())
        for key, path in parquet_paths.items()
        if path is not None
    }
    _write_json(run_dir / "manifest.json", manifest)
    return manifest
