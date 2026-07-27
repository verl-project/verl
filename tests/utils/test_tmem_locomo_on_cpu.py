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

from examples.tmem.locomo import (
    exact_match,
    normalize_answer,
    parse_qa_pairs,
    reference_answer,
    score_records,
    token_f1,
)


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
    assert reference_answer({"answer": None, "category": 5}) == "not mentioned"
    assert reference_answer({"answer": 2022, "category": 2}) == "2022"
