from __future__ import annotations

import hashlib
import random
from typing import Any

import datasets
import numpy as np

from reliable_gsm8k.parsing import extract_generated_answer, parse_gold_answer, values_equal
from reliable_gsm8k.prompts import build_mc_onecorrect_prompt
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.reward_score import gsm8k_mc


STAGE1_SOURCE = "gsm8k_dynamic_mc_stage1"
STAGE2_SOURCE = "gsm8k_dynamic_mc_stage2"
MC_LABELS = ("A", "B", "C", "D")


def _get_int_config(config: Any, key: str, default: int, *, minimum: int) -> int:
    value = int(config.get(key, default))
    if value < minimum:
        raise ValueError(f"data.dynamic_mc.{key} must be >= {minimum}, got {value}.")
    return value


def question_id_from_question(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()


def build_stage1_correct_prompt(question: str) -> str:
    return (
        "Solve the GSM8K math problem step by step.\n"
        "Return exactly this format:\n"
        "REASONING: <step-by-step solution>\n"
        "FINAL_ANSWER: <final numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n\n"
        f"Question:\n{question.strip()}"
    )


def build_stage1_candidate_prompt(question: str) -> str:
    return (
        "Solve the GSM8K math problem step by step.\n"
        "Return exactly this format:\n"
        "REASONING: <step-by-step solution>\n"
        "FINAL_ANSWER: <final numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n\n"
        f"Question:\n{question.strip()}"
    )


def build_stage1_incorrect_prompt(question: str) -> str:
    return (
        "Write a plausible but mathematically incorrect step-by-step solution to the GSM8K problem.\n"
        "The reasoning should look like a realistic student mistake.\n"
        "The FINAL_ANSWER must be numeric and must be wrong.\n"
        "Return exactly this format:\n"
        "REASONING: <step-by-step but incorrect solution>\n"
        "FINAL_ANSWER: <wrong numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n\n"
        f"Question:\n{question.strip()}"
    )


def make_stage1_record(
    *,
    item_id: str,
    question: str,
    gold_answer: str,
    role_requested: str,
    slot: int,
) -> dict[str, Any]:
    question_id = question_id_from_question(question)
    if role_requested == "candidate":
        prompt = build_stage1_candidate_prompt(question)
    elif role_requested == "correct":
        prompt = build_stage1_correct_prompt(question)
    elif role_requested == "incorrect":
        prompt = build_stage1_incorrect_prompt(question)
    else:
        raise ValueError("role_requested must be 'candidate', 'correct', or 'incorrect'.")
    return {
        "data_source": STAGE1_SOURCE,
        "prompt": [{"role": "user", "content": prompt}],
        "question_id": question_id,
        "reward_model": {"style": "rule", "ground_truth": gold_answer},
        "extra_info": {
            "stage": "stage1_candidate",
            "item_id": item_id,
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "role_requested": role_requested,
            "candidate_slot": slot,
            "correct_choice": "",
            "option_roles": {label: "" for label in MC_LABELS},
            "option_final_answers": {label: "" for label in MC_LABELS},
        },
    }


def make_stage1_records_for_question(
    *,
    item_id: str,
    question: str,
    gold_answer: str,
    incorrect_target_count: int = 3,
    prompt_mode: str = "neutral",
) -> list[dict[str, Any]]:
    if prompt_mode == "neutral":
        prompt_count = max(1, incorrect_target_count + 1)
        return [
            make_stage1_record(
                item_id=item_id,
                question=question,
                gold_answer=gold_answer,
                role_requested="candidate",
                slot=slot,
            )
            for slot in range(prompt_count)
        ]
    if prompt_mode != "role":
        raise ValueError("prompt_mode must be 'neutral' or 'role'.")

    records = [
        make_stage1_record(
            item_id=item_id,
            question=question,
            gold_answer=gold_answer,
            role_requested="correct",
            slot=0,
        )
    ]
    for slot in range(incorrect_target_count):
        records.append(
            make_stage1_record(
                item_id=item_id,
                question=question,
                gold_answer=gold_answer,
                role_requested="incorrect",
                slot=slot,
            )
        )
    return records


def make_stage1_records_from_gsm8k_example(
    *,
    split: str,
    index: int,
    example: dict[str, Any],
    incorrect_target_count: int = 3,
    prompt_mode: str = "neutral",
) -> list[dict[str, Any]]:
    question = str(example["question"])
    parsed = parse_gold_answer(str(example["answer"]))
    if parsed.normalized_answer is None:
        return []
    return make_stage1_records_for_question(
        item_id=f"gsm8k:{split}:{index}",
        question=question,
        gold_answer=parsed.normalized_answer,
        incorrect_target_count=incorrect_target_count,
        prompt_mode=prompt_mode,
    )


class VerifiedCandidate:
    __slots__ = ("completion", "final_answer", "role")

    def __init__(self, completion: str, final_answer: str, role: str) -> None:
        self.completion = completion
        self.final_answer = final_answer
        self.role = role


class GSM8KDynamicMCDataset(RLHFDataset):
    """RLHFDataset that turns Stage 1 rollouts into Stage 2 MC prompts inside VERL.

    Initial training rows are Stage 1 candidate-generation prompts. After each VERL
    training batch, `on_batch_end` parses the actor's generated candidate solutions,
    verifies them against the GSM8K gold answer, and appends a Stage 2 MC prompt when
    a question has one correct and three distinct incorrect candidate CoTs.
    """

    supports_trainer_resume = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        dynamic_cfg = self.config.get("dynamic_mc", {})
        self.incorrect_target_count = _get_int_config(dynamic_cfg, "stage2_incorrect_count", 3, minimum=0)
        if self.incorrect_target_count != 3:
            raise ValueError("data.dynamic_mc.stage2_incorrect_count must be 3 for four-option MC prompts.")
        self.max_stage2_per_question = _get_int_config(dynamic_cfg, "max_stage2_per_question", 1, minimum=1)
        self.max_new_stage2_per_batch = _get_int_config(dynamic_cfg, "max_new_stage2_per_batch", 256, minimum=1)
        self.stage2_candidate_max_chars = _get_int_config(dynamic_cfg, "stage2_candidate_max_chars", 2000, minimum=0)
        self.stage2_insert_strategy = str(dynamic_cfg.get("stage2_insert_strategy", "prepend"))
        if self.stage2_insert_strategy not in {"prepend", "append"}:
            raise ValueError("data.dynamic_mc.stage2_insert_strategy must be 'prepend' or 'append'.")
        self.seed = int(dynamic_cfg.get("seed", self.config.get("seed", 7) or 7))
        self._candidate_buffer: dict[str, dict[str, Any]] = {}
        self._stage2_counts: dict[str, int] = {}
        self._hook_calls = 0
        self._accepted_correct_total = 0
        self._accepted_incorrect_total = 0
        self._inserted_stage2_total = 0
        self._pending_stage2_records: list[dict[str, Any]] = []

    def _decode_responses(self, batch: DataProto) -> list[str]:
        prompt_len = batch.batch["prompts"].shape[-1]
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        responses: list[str] = []
        for index in range(len(batch)):
            valid_len = int(valid_response_lengths[index].item())
            responses.append(self.tokenizer.decode(response_ids[index][:valid_len], skip_special_tokens=True))
        return responses

    def _candidate_from_response(self, *, response: str, extra_info: dict[str, Any]) -> VerifiedCandidate | None:
        requested_role = str(extra_info.get("role_requested", ""))
        gold_answer = str(extra_info.get("gold_answer", ""))
        parsed = extract_generated_answer(response)
        if parsed.normalized_answer is None:
            return None
        is_correct = values_equal(parsed.normalized_answer, gold_answer)
        if requested_role == "candidate":
            role = "correct" if is_correct else "incorrect"
        elif requested_role == "correct":
            if not is_correct:
                return None
            role = "correct"
        elif requested_role == "incorrect":
            if is_correct:
                return None
            role = "incorrect"
        else:
            return None
        return VerifiedCandidate(
            completion=response.strip(),
            final_answer=parsed.normalized_answer,
            role=role,
        )

    def _record_candidate(self, *, question_id: str, extra_info: dict[str, Any], candidate: VerifiedCandidate) -> None:
        entry = self._candidate_buffer.setdefault(
            question_id,
            {
                "question": str(extra_info.get("question", "")),
                "gold_answer": str(extra_info.get("gold_answer", "")),
                "item_id": str(extra_info.get("item_id", question_id)),
                "correct": [],
                "incorrect": [],
                "seen_correct_completions": set(),
                "seen_incorrect_answers": set(),
                "correct_cursor": 0,
                "incorrect_cursor": 0,
            },
        )
        if candidate.role == "correct":
            if candidate.completion in entry["seen_correct_completions"]:
                return
            entry["seen_correct_completions"].add(candidate.completion)
            entry["correct"].append(candidate)
            return
        if candidate.final_answer in entry["seen_incorrect_answers"]:
            return
        entry["seen_incorrect_answers"].add(candidate.final_answer)
        entry["incorrect"].append(candidate)

    def _format_stage2_option_completion(self, completion: str) -> str:
        completion = completion.strip()
        if self.stage2_candidate_max_chars <= 0 or len(completion) <= self.stage2_candidate_max_chars:
            return completion

        marker = "FINAL_ANSWER:"
        marker_index = completion.rfind(marker)
        if marker_index == -1:
            return completion[: self.stage2_candidate_max_chars].rstrip() + "\n...[truncated]"

        final_tail = completion[marker_index:].strip()
        head_budget = self.stage2_candidate_max_chars - len(final_tail) - len("\n...\n")
        if head_budget <= 0:
            return final_tail[-self.stage2_candidate_max_chars :]
        return completion[:head_budget].rstrip() + "\n...\n" + final_tail

    def _commit_stage2_candidate_set(self, *, question_id: str, entry: dict[str, Any], incorrect_end: int) -> None:
        self._stage2_counts[question_id] = self._stage2_counts.get(question_id, 0) + 1
        if int(entry.get("correct_cursor", 0)) < len(entry["correct"]):
            entry["correct_cursor"] = int(entry.get("correct_cursor", 0)) + 1
        entry["incorrect_cursor"] = incorrect_end

    def _stage2_record_passes_prompt_filter(self, record: dict[str, Any]) -> bool:
        dataframe = datasets.Dataset.from_list([record])
        return len(self.maybe_filter_out_long_prompts(dataframe)) == 1

    def _build_stage2_record(
        self,
        *,
        question_id: str,
        entry: dict[str, Any],
        commit: bool = True,
    ) -> dict[str, Any] | None:
        if self._stage2_counts.get(question_id, 0) >= self.max_stage2_per_question:
            return None
        if not entry["correct"]:
            return None
        incorrect_start = int(entry.get("incorrect_cursor", 0))
        incorrect_end = incorrect_start + self.incorrect_target_count
        if len(entry["incorrect"]) < incorrect_end:
            return None

        correct_index = int(entry.get("correct_cursor", 0))
        if correct_index >= len(entry["correct"]):
            # Correct solutions are rarer than wrong ones. Reuse the latest verified
            # correct CoT only when new wrong candidates are available.
            correct_index = len(entry["correct"]) - 1
        correct_candidate = entry["correct"][correct_index]
        incorrect_candidates = entry["incorrect"][incorrect_start:incorrect_end]

        choices = [("correct", correct_candidate)] + [("incorrect", value) for value in incorrect_candidates]
        rng = random.Random(f"{self.seed}:{question_id}:{self._stage2_counts.get(question_id, 0)}")
        rng.shuffle(choices)
        options = {
            label: self._format_stage2_option_completion(candidate.completion)
            for label, (_, candidate) in zip(MC_LABELS, choices, strict=True)
        }
        correct_label = next(label for label, (role, _) in zip(MC_LABELS, choices, strict=True) if role == "correct")
        if commit:
            self._commit_stage2_candidate_set(question_id=question_id, entry=entry, incorrect_end=incorrect_end)

        return {
            "data_source": STAGE2_SOURCE,
            "prompt": [{"role": "user", "content": build_mc_onecorrect_prompt(entry["question"], options)}],
            "question_id": question_id,
            "reward_model": {"style": "rule", "ground_truth": correct_label},
            "extra_info": {
                "stage": "stage2_mc",
                "item_id": entry["item_id"],
                "question_id": question_id,
                "question": entry["question"],
                "gold_answer": entry["gold_answer"],
                "role_requested": "",
                "candidate_slot": -1,
                "correct_choice": correct_label,
                "option_roles": {label: role for label, (role, _) in zip(MC_LABELS, choices, strict=True)},
                "option_final_answers": {
                    label: candidate.final_answer for label, (_, candidate) in zip(MC_LABELS, choices, strict=True)
                },
            },
        }

    def _queue_stage2_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self._pending_stage2_records.extend(records)
        print(
            "[GSM8KDynamicMCDataset] "
            f"queued_stage2={len(records)} pending_stage2={len(self._pending_stage2_records)}"
        )

    def has_pending_dynamic_rows(self) -> bool:
        return bool(self._pending_stage2_records)

    def _flush_pending_stage2_records(self) -> int:
        if not self._pending_stage2_records:
            return 0
        records = self._pending_stage2_records
        self._pending_stage2_records = []
        dataframe = datasets.Dataset.from_list(records)
        dataframe = self.maybe_filter_out_long_prompts(dataframe)
        if len(dataframe) == 0:
            return 0
        if self.stage2_insert_strategy == "prepend":
            self.dataframe = datasets.concatenate_datasets([dataframe, self.dataframe])
        else:
            self.dataframe = datasets.concatenate_datasets([self.dataframe, dataframe])
        self._inserted_stage2_total += len(dataframe)
        print(
            "[GSM8KDynamicMCDataset] "
            f"inserted_stage2={len(dataframe)} strategy={self.stage2_insert_strategy} "
            f"inserted_stage2_total={self._inserted_stage2_total} dataset_len={len(self.dataframe)}"
        )
        return len(dataframe)

    def on_epoch_end(self, epoch: int) -> int:
        print(
            "[GSM8KDynamicMCDataset] "
            f"epoch_end={epoch} pending_stage2={len(self._pending_stage2_records)}"
        )
        return self._flush_pending_stage2_records()

    def on_batch_end(self, batch: DataProto) -> None:
        self._hook_calls += 1
        responses = self._decode_responses(batch)
        new_records: list[dict[str, Any]] = []
        stage1_seen = 0
        accepted_correct = 0
        accepted_incorrect = 0
        for index, response in enumerate(responses):
            extra_info = batch[index].non_tensor_batch.get("extra_info", {})
            if not isinstance(extra_info, dict) or extra_info.get("stage") != "stage1_candidate":
                continue
            stage1_seen += 1
            question_id = str(extra_info.get("question_id", ""))
            if not question_id:
                continue
            candidate = self._candidate_from_response(response=response, extra_info=extra_info)
            if candidate is None:
                continue
            if candidate.role == "correct":
                accepted_correct += 1
            else:
                accepted_incorrect += 1
            self._record_candidate(question_id=question_id, extra_info=extra_info, candidate=candidate)
            if len(new_records) < self.max_new_stage2_per_batch:
                entry = self._candidate_buffer[question_id]
                incorrect_end = int(entry.get("incorrect_cursor", 0)) + self.incorrect_target_count
                stage2 = self._build_stage2_record(question_id=question_id, entry=entry, commit=False)
            else:
                stage2 = None
            if stage2 is not None and self._stage2_record_passes_prompt_filter(stage2):
                self._commit_stage2_candidate_set(question_id=question_id, entry=entry, incorrect_end=incorrect_end)
                new_records.append(stage2)
        self._accepted_correct_total += accepted_correct
        self._accepted_incorrect_total += accepted_incorrect
        print(
            "[GSM8KDynamicMCDataset] "
            f"hook={self._hook_calls} stage1_seen={stage1_seen} "
            f"accepted_correct={accepted_correct} accepted_incorrect={accepted_incorrect} "
            f"accepted_correct_total={self._accepted_correct_total} "
            f"accepted_incorrect_total={self._accepted_incorrect_total} "
            f"stage2_ready={len(new_records)}"
        )
        self._queue_stage2_records(new_records)


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    format_score: float = 0.0,
    score: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    if data_source is None and "data_sources" in kwargs:
        return compute_score_batched(
            data_sources=kwargs.pop("data_sources"),
            solution_strs=kwargs.pop("solution_strs"),
            ground_truths=kwargs.pop("ground_truths"),
            extra_infos=kwargs.pop("extra_infos"),
            format_score=format_score,
            score=score,
            **kwargs,
        )
    if data_source is None or solution_str is None or ground_truth is None:
        raise ValueError("compute_score requires data_source, solution_str, and ground_truth for single examples.")

    extra_info = extra_info or {}
    stage = extra_info.get("stage")

    if data_source == STAGE2_SOURCE or stage == "stage2_mc":
        result = gsm8k_mc.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            method=kwargs.get("mc_method", "strict"),
            format_score=format_score,
            score=score,
        )
        return {**result, "stage": "stage2_mc"}

    parsed = extract_generated_answer(solution_str)
    if parsed.normalized_answer is None:
        return {
            "score": 0.0,
            "stage": "stage1_candidate",
            "label": "parsing_error",
            "format_ok": False,
            "pred": "",
            "role_requested": extra_info.get("role_requested"),
        }

    gold_answer = str(extra_info.get("gold_answer") or ground_truth)
    role_requested = str(extra_info.get("role_requested", "correct"))
    is_correct = values_equal(parsed.normalized_answer, gold_answer)
    if role_requested == "incorrect":
        accepted = not is_correct
    else:
        accepted = is_correct
    return {
        "score": score if accepted else format_score,
        "stage": "stage1_candidate",
        "label": "correct" if is_correct else "incorrect",
        "format_ok": True,
        "pred": parsed.normalized_answer,
        "role_requested": role_requested,
        "accepted": accepted,
    }


def compute_score_batched(
    data_sources: list[str] | np.ndarray,
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return [
        compute_score(
            data_source=str(data_source),
            solution_str=solution_str,
            ground_truth=str(ground_truth),
            extra_info=extra_info,
            **kwargs,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        )
    ]
