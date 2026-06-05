from __future__ import annotations

import sys
from pathlib import Path

import datasets
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_SRC = REPO_ROOT / "reliable-gsm8k-builder" / "src"
for path in (REPO_ROOT, BUILDER_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reliable_gsm8k.verl_dynamic_mc import (  # noqa: E402
    GSM8KDynamicMCDataset,
    STAGE1_SOURCE,
    STAGE2_SOURCE,
    VerifiedCandidate,
    _get_int_config,
    compute_score,
    make_stage1_records_for_question,
    question_id_from_question,
)
from verl import DataProto  # noqa: E402


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        values = [int(value) for value in ids.tolist()]
        first = values[0] if values else -1
        return {
            10: "REASONING: 2 + 2 = 4\nFINAL_ANSWER: 4",
            11: "REASONING: A student subtracts one.\nFINAL_ANSWER: 3",
            12: "REASONING: A student adds one.\nFINAL_ANSWER: 5",
            13: "REASONING: A student doubles three.\nFINAL_ANSWER: 6",
            20: "REASONING: 4 + 4 = 8\nFINAL_ANSWER: 8",
            21: "REASONING: A student subtracts one.\nFINAL_ANSWER: 7",
            22: "REASONING: A student adds one.\nFINAL_ANSWER: 9",
            23: "REASONING: A student adds two.\nFINAL_ANSWER: 10",
        }[first]


def _bare_dynamic_dataset() -> GSM8KDynamicMCDataset:
    dataset = GSM8KDynamicMCDataset.__new__(GSM8KDynamicMCDataset)
    dataset.tokenizer = _FakeTokenizer()
    dataset.incorrect_target_count = 3
    dataset.max_stage2_per_question = 1
    dataset.max_new_stage2_per_batch = 256
    dataset.stage2_candidate_max_chars = 2000
    dataset.stage2_insert_strategy = "prepend"
    dataset.seed = 7
    dataset._candidate_buffer = {}
    dataset._stage2_counts = {}
    dataset._hook_calls = 0
    dataset._accepted_correct_total = 0
    dataset._accepted_incorrect_total = 0
    dataset._inserted_stage2_total = 0
    dataset._pending_stage2_records = []
    dataset.dataframe = datasets.Dataset.from_list(
        [
            {
                "data_source": "seed",
                "prompt": [{"role": "user", "content": "seed"}],
                "question_id": "seed",
                "reward_model": {"style": "rule", "ground_truth": "1"},
                "extra_info": {"stage": "stage1_candidate"},
            }
        ]
    )
    dataset.maybe_filter_out_long_prompts = lambda dataframe: dataframe
    return dataset


def test_dynamic_config_int_validation() -> None:
    assert _get_int_config({"max_stage2_per_question": "2"}, "max_stage2_per_question", 1, minimum=1) == 2

    try:
        _get_int_config({"max_stage2_per_question": "0"}, "max_stage2_per_question", 1, minimum=1)
    except ValueError as exc:
        assert "max_stage2_per_question" in str(exc)
    else:
        raise AssertionError("Expected invalid dynamic config value to raise ValueError.")


def test_dynamic_dataset_declares_trainer_resume_unsupported() -> None:
    assert GSM8KDynamicMCDataset.supports_trainer_resume is False


def test_on_batch_end_builds_stage2_mc_row_from_verified_stage1_candidates() -> None:
    question = "What is 2 + 2?"
    question_id = question_id_from_question(question)
    extra_info = [
        {
            "stage": "stage1_candidate",
            "question_id": question_id,
            "question": question,
            "gold_answer": "4",
            "item_id": "gsm8k:train:test",
            "role_requested": "candidate",
        },
        {
            "stage": "stage1_candidate",
            "question_id": question_id,
            "question": question,
            "gold_answer": "4",
            "item_id": "gsm8k:train:test",
            "role_requested": "candidate",
        },
        {
            "stage": "stage1_candidate",
            "question_id": question_id,
            "question": question,
            "gold_answer": "4",
            "item_id": "gsm8k:train:test",
            "role_requested": "candidate",
        },
        {
            "stage": "stage1_candidate",
            "question_id": question_id,
            "question": question,
            "gold_answer": "4",
            "item_id": "gsm8k:train:test",
            "role_requested": "candidate",
        },
    ]
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.ones((4, 1), dtype=torch.long),
            "responses": torch.tensor([[10, 0], [11, 0], [12, 0], [13, 0]], dtype=torch.long),
            "attention_mask": torch.ones((4, 3), dtype=torch.long),
            "extra_info": np.array(extra_info, dtype=object),
        }
    )

    dataset = _bare_dynamic_dataset()
    dataset.on_batch_end(batch)

    assert len(dataset.dataframe) == 1
    assert len(dataset._pending_stage2_records) == 1
    assert dataset.has_pending_dynamic_rows()

    inserted = dataset.on_epoch_end(epoch=0)
    stage2 = dataset.dataframe[0]
    assert inserted == 1
    assert len(dataset.dataframe) == 2
    assert stage2["data_source"] == STAGE2_SOURCE
    assert stage2["extra_info"]["stage"] == "stage2_mc"
    assert stage2["reward_model"]["ground_truth"] in {"A", "B", "C", "D"}
    assert stage2["prompt"][0]["content"].count("FINAL_ANSWER:") == 4
    assert dataset._accepted_correct_total == 1
    assert dataset._accepted_incorrect_total == 3
    assert dataset._inserted_stage2_total == 1
    assert dataset._pending_stage2_records == []
    assert not dataset.has_pending_dynamic_rows()


def test_stage2_prompt_requires_one_correct_and_three_wrong_candidates() -> None:
    dataset = _bare_dynamic_dataset()
    entry = {
        "question": "What is 2 + 2?",
        "gold_answer": "4",
        "item_id": "gsm8k:train:test",
        "correct": [VerifiedCandidate("REASONING: 2 + 2 = 4\nFINAL_ANSWER: 4", "4", "correct")],
        "incorrect": [
            VerifiedCandidate("REASONING: wrong\nFINAL_ANSWER: 3", "3", "incorrect"),
            VerifiedCandidate("REASONING: wrong\nFINAL_ANSWER: 5", "5", "incorrect"),
        ],
    }

    assert dataset._build_stage2_record(question_id="qid", entry=entry) is None


def test_stage2_queue_cap_does_not_drop_later_verified_candidates() -> None:
    first_question = "What is 2 + 2?"
    second_question = "What is 4 + 4?"
    first_qid = question_id_from_question(first_question)
    second_qid = question_id_from_question(second_question)
    rows = [
        (first_qid, first_question, "4", "candidate"),
        (first_qid, first_question, "4", "candidate"),
        (first_qid, first_question, "4", "candidate"),
        (first_qid, first_question, "4", "candidate"),
        (second_qid, second_question, "8", "candidate"),
        (second_qid, second_question, "8", "candidate"),
        (second_qid, second_question, "8", "candidate"),
        (second_qid, second_question, "8", "candidate"),
    ]
    extra_info = [
        {
            "stage": "stage1_candidate",
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "item_id": f"gsm8k:train:test:{question_id}",
            "role_requested": role_requested,
        }
        for question_id, question, gold_answer, role_requested in rows
    ]
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.ones((8, 1), dtype=torch.long),
            "responses": torch.tensor(
                [[10, 0], [11, 0], [12, 0], [13, 0], [20, 0], [21, 0], [22, 0], [23, 0]],
                dtype=torch.long,
            ),
            "attention_mask": torch.ones((8, 3), dtype=torch.long),
            "extra_info": np.array(extra_info, dtype=object),
        }
    )

    dataset = _bare_dynamic_dataset()
    dataset.max_new_stage2_per_batch = 1
    dataset.on_batch_end(batch)

    assert len(dataset._pending_stage2_records) == 1
    assert dataset._accepted_correct_total == 2
    assert dataset._accepted_incorrect_total == 6
    assert len(dataset._candidate_buffer[first_qid]["incorrect"]) == 3
    assert len(dataset._candidate_buffer[second_qid]["incorrect"]) == 3


def test_filtered_stage2_row_does_not_consume_candidate_set() -> None:
    question = "What is 2 + 2?"
    question_id = question_id_from_question(question)
    extra_info = [
        {
            "stage": "stage1_candidate",
            "question_id": question_id,
            "question": question,
            "gold_answer": "4",
            "item_id": "gsm8k:train:test",
            "role_requested": "candidate",
        }
        for _ in range(4)
    ]
    batch = DataProto.from_single_dict(
        {
            "prompts": torch.ones((4, 1), dtype=torch.long),
            "responses": torch.tensor([[10, 0], [11, 0], [12, 0], [13, 0]], dtype=torch.long),
            "attention_mask": torch.ones((4, 3), dtype=torch.long),
            "extra_info": np.array(extra_info, dtype=object),
        }
    )

    dataset = _bare_dynamic_dataset()
    dataset.maybe_filter_out_long_prompts = lambda dataframe: dataframe.select([])
    dataset.on_batch_end(batch)

    entry = dataset._candidate_buffer[question_id]
    assert dataset._pending_stage2_records == []
    assert dataset._stage2_counts.get(question_id, 0) == 0
    assert entry["correct_cursor"] == 0
    assert entry["incorrect_cursor"] == 0


def test_stage2_records_consume_new_candidate_sets() -> None:
    dataset = _bare_dynamic_dataset()
    dataset.max_stage2_per_question = 2
    entry = {
        "question": "What is 2 + 2?",
        "gold_answer": "4",
        "item_id": "gsm8k:train:test",
        "correct": [
            VerifiedCandidate("REASONING: direct\nFINAL_ANSWER: 4", "4", "correct"),
            VerifiedCandidate("REASONING: addition\nFINAL_ANSWER: 4", "4", "correct"),
        ],
        "incorrect": [
            VerifiedCandidate("REASONING: wrong 1\nFINAL_ANSWER: 3", "3", "incorrect"),
            VerifiedCandidate("REASONING: wrong 2\nFINAL_ANSWER: 5", "5", "incorrect"),
            VerifiedCandidate("REASONING: wrong 3\nFINAL_ANSWER: 6", "6", "incorrect"),
            VerifiedCandidate("REASONING: wrong 4\nFINAL_ANSWER: 7", "7", "incorrect"),
            VerifiedCandidate("REASONING: wrong 5\nFINAL_ANSWER: 8", "8", "incorrect"),
            VerifiedCandidate("REASONING: wrong 6\nFINAL_ANSWER: 9", "9", "incorrect"),
        ],
        "correct_cursor": 0,
        "incorrect_cursor": 0,
    }

    first = dataset._build_stage2_record(question_id="qid", entry=entry)
    second = dataset._build_stage2_record(question_id="qid", entry=entry)
    third = dataset._build_stage2_record(question_id="qid", entry=entry)

    assert first is not None
    assert second is not None
    assert third is None
    first_answers = set(first["extra_info"]["option_final_answers"].values())
    second_answers = set(second["extra_info"]["option_final_answers"].values())
    assert {"3", "5", "6"}.issubset(first_answers)
    assert {"7", "8", "9"}.issubset(second_answers)
    assert entry["correct_cursor"] == 2
    assert entry["incorrect_cursor"] == 6


def test_stage2_option_clipping_preserves_final_answer_tail() -> None:
    dataset = _bare_dynamic_dataset()
    dataset.stage2_candidate_max_chars = 80
    completion = "REASONING: " + ("long step " * 30) + "\nFINAL_ANSWER: 123"

    clipped = dataset._format_stage2_option_completion(completion)

    assert len(clipped) <= 80
    assert "FINAL_ANSWER: 123" in clipped


def test_stage1_and_stage2_rows_have_concat_compatible_schema() -> None:
    question = "What is 2 + 2?"
    question_id = question_id_from_question(question)
    dataset = _bare_dynamic_dataset()
    dataset.dataframe = datasets.Dataset.from_list(
        make_stage1_records_for_question(
            item_id="gsm8k:train:test",
            question=question,
            gold_answer="4",
            incorrect_target_count=3,
        )
    )
    entry = {
        "question": question,
        "gold_answer": "4",
        "item_id": "gsm8k:train:test",
        "correct": [VerifiedCandidate("REASONING: 2 + 2 = 4\nFINAL_ANSWER: 4", "4", "correct")],
        "incorrect": [
            VerifiedCandidate("REASONING: wrong\nFINAL_ANSWER: 3", "3", "incorrect"),
            VerifiedCandidate("REASONING: wrong\nFINAL_ANSWER: 5", "5", "incorrect"),
            VerifiedCandidate("REASONING: wrong\nFINAL_ANSWER: 6", "6", "incorrect"),
        ],
        "correct_cursor": 0,
        "incorrect_cursor": 0,
    }
    stage2 = dataset._build_stage2_record(question_id=question_id, entry=entry)
    assert stage2 is not None

    dataset._queue_stage2_records([stage2])
    inserted = dataset.on_epoch_end(epoch=0)

    assert inserted == 1
    assert len(dataset.dataframe) == 5
    assert dataset.dataframe[0]["extra_info"]["stage"] == "stage2_mc"
    assert dataset.dataframe[1]["extra_info"]["stage"] == "stage1_candidate"


def test_stage1_seed_rows_default_to_neutral_candidate_prompts() -> None:
    records = make_stage1_records_for_question(
        item_id="gsm8k:train:test",
        question="What is 2 + 2?",
        gold_answer="4",
        incorrect_target_count=3,
    )

    assert len(records) == 4
    assert {record["extra_info"]["role_requested"] for record in records} == {"candidate"}
    assert all("mathematically incorrect" not in record["prompt"][0]["content"] for record in records)


def test_dynamic_reward_scores_stage1_and_stage2() -> None:
    assert (
        compute_score(
            STAGE1_SOURCE,
            "REASONING: ok\nFINAL_ANSWER: 72",
            "72",
            {"stage": "stage1_candidate", "role_requested": "correct", "gold_answer": "72"},
        )["score"]
        == 1.0
    )
    assert (
        compute_score(
            STAGE1_SOURCE,
            "REASONING: mistake\nFINAL_ANSWER: 71",
            "72",
            {"stage": "stage1_candidate", "role_requested": "candidate", "gold_answer": "72"},
        )["score"]
        == 0.0
    )
    assert (
        compute_score(
            STAGE1_SOURCE,
            "REASONING: legacy candidate mining\nFINAL_ANSWER: 71",
            "72",
            {"stage": "stage1_candidate", "role_requested": "incorrect", "gold_answer": "72"},
        )["score"]
        == 1.0
    )
    assert compute_score(STAGE2_SOURCE, "#### B", "B", {"stage": "stage2_mc"})["score"] == 1.0


def test_dynamic_reward_compute_score_accepts_batch_manager_call_shape() -> None:
    scores = compute_score(
        data_sources=np.array([STAGE1_SOURCE, STAGE2_SOURCE], dtype=object),
        solution_strs=["REASONING: ok\nFINAL_ANSWER: 72", "#### C"],
        ground_truths=["72", "C"],
        extra_infos=[
            {"stage": "stage1_candidate", "role_requested": "correct", "gold_answer": "72"},
            {"stage": "stage2_mc"},
        ],
    )

    assert isinstance(scores, list)
    assert [item["score"] for item in scores] == [1.0, 1.0]
