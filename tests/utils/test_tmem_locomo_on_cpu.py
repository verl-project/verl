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

from types import SimpleNamespace

import torch

from examples.tmem.locomo import (
    exact_match,
    normalize_answer,
    pack_context_chunks,
    parse_qa_pairs,
    postprocess_prediction,
    prepare_question,
    reference_answer,
    score_breakdown,
    score_records,
    token_f1,
)
from examples.tmem.run_locomo import DFlashRollout, JsonArrayEndCriteria


def test_parse_qa_pairs_accepts_fenced_json_and_drops_invalid_entries():
    text = """<think>draft</think>```json
    [{"instruction": " Who? ", "output": " Alice "}, {"instruction": 3, "output": "bad"}]
    ```"""
    assert parse_qa_pairs(text) == [{"instruction": "Who?", "output": "Alice"}]


def test_paper_normalization_and_metrics():
    assert normalize_answer(" Alice,  Bob! ") == "alice bob"
    assert exact_match("Alice!", "alice") == 1
    assert token_f1("red blue", "blue green") == 0.5
    assert score_records([{"prediction": "Alice!", "reference": "alice"}]) == {
        "count": 1,
        "f1": 100.0,
        "em": 100.0,
    }


def test_missing_reference_uses_locomo_unanswerable_label():
    assert reference_answer({"adversarial_answer": "Alice", "category": 5}) == "No information available"
    assert reference_answer({"answer": 2022, "category": 2}) == "2022"


def test_official_question_preprocessing_and_option_resolution():
    temporal, options = prepare_question({"question": "When?", "category": 2}, no_information_first=True)
    assert temporal.endswith("Use DATE of CONVERSATION to answer with an approximate date.")
    assert options is None

    adversarial, options = prepare_question(
        {"question": "Unsupported?", "adversarial_answer": "Alice", "category": 5},
        no_information_first=False,
    )
    assert adversarial.endswith("Select the correct answer by writing (a) or (b).")
    assert options == {"a": "Alice", "b": "No information available"}
    assert postprocess_prediction("Answer: (b)", options) == "No information available"
    assert postprocess_prediction("Answer: Alice", None) == "Alice"
    assert postprocess_prediction("Alice", None) == "Alice"


def test_score_breakdown_reports_non_adversarial_and_categories():
    metrics = score_breakdown(
        [
            {"prediction": "Alice", "reference": "Alice", "category": 1},
            {"prediction": "not mentioned", "reference": "not mentioned", "category": 5},
        ]
    )
    assert metrics["count"] == 2
    assert metrics["without_category_5"] == {"count": 1, "f1": 100.0, "em": 100.0}
    assert metrics["by_category"]["5"] == {"count": 1, "f1": 100.0, "em": 100.0}


def test_context_trigger_includes_session_that_crosses_budget():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return text.split()

    chunks = pack_context_chunks(
        ["one two", "three four", "five", "six seven eight nine"],
        Tokenizer(),
        token_budget=3,
    )

    assert chunks == ["one two\n\nthree four", "five\n\nsix seven eight nine", ""]


def test_json_array_stopping_criteria_is_vectorized_per_sequence():
    class Tokenizer:
        def __len__(self):
            return 3

        def batch_decode(self, token_ids):
            values = ["word", "other", "]\n"]
            return [values[token_id[0]] for token_id in token_ids]

    criteria = JsonArrayEndCriteria(Tokenizer())
    finished = criteria(torch.tensor([[0, 2], [1, 0]]), torch.empty(0))
    torch.testing.assert_close(finished, torch.tensor([True, False]))


def test_dflash_extraction_keeps_json_terminator():
    rollout = object.__new__(DFlashRollout)
    rollout.args = SimpleNamespace(
        max_extraction_tokens=1024,
        max_answer_tokens=50,
        extraction_temperature=0.7,
        extraction_top_p=0.8,
        extraction_top_k=20,
        answer_temperature=0.4,
        answer_top_p=0.9,
        answer_top_k=10,
    )
    rollout.extraction_stop_token_ids = [2, 7]

    params = rollout._sampling_params(extraction=True, sampling_seed=17)

    assert params["stop_token_ids"] == [2, 7]
    assert params["no_stop_trim"] is True
    assert params["max_new_tokens"] == 1024
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.8
    assert params["top_k"] == 20
    assert params["sampling_seed"] == 17


def test_dflash_stats_are_reset_per_seed():
    rollout = object.__new__(DFlashRollout)
    rollout.reset_stats(seed=7)
    rollout.generation_seconds = 2.5
    rollout.completion_tokens = 10
    rollout.spec_verify_count = 4
    rollout.spec_accept_length_sum = 6.0
    rollout.spec_accept_length_count = 2

    assert rollout.stats() == {
        "generation_calls": 0,
        "generation_seconds": 2.5,
        "completion_tokens": 10,
        "spec_verify_count": 4,
        "mean_spec_accept_length": 3.0,
    }
