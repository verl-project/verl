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

import pytest

from examples.tmem.locomo import deduplicate_qa_pairs, has_complete_json_array, parse_qa_pairs_result
from examples.tmem.run_locomo import (
    ExtractionGenerationError,
    _generate_extractions_with_retries,
    extraction_telemetry,
)


def test_top_level_array_detection_ignores_brackets_inside_json_strings():
    text = '[{"instruction":"What does [x] mean?","output":"a ] bracket"}]'

    assert has_complete_json_array(text)
    result = parse_qa_pairs_result(text)
    assert result.status == "ok"
    assert result.pairs == [{"instruction": "What does [x] mean?", "output": "a ] bracket"}]


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ('[{"instruction":"q","output":"a"}', "incomplete_array"),
        ('[{"instruction":"q","output":"bad "quote""}]', "invalid_json"),
        ("not JSON", "no_array"),
        ('[{"instruction":"","output":"a"}]', "invalid_schema"),
    ],
)
def test_strict_parse_reports_failures_instead_of_returning_silent_empty_pairs(text, status):
    result = parse_qa_pairs_result(text)

    assert not result.valid
    assert result.status == status
    assert result.pairs == []
    assert result.error


def test_strict_parse_accepts_an_intentional_empty_array():
    result = parse_qa_pairs_result("[]")

    assert result.valid
    assert result.status == "empty"
    assert result.pairs == []


def test_strict_parse_keeps_valid_entries_and_reports_schema_drops():
    result = parse_qa_pairs_result('[{"instruction":"q?","output":"a"},{"instruction":"","output":"bad"}]')

    assert result.valid
    assert result.status == "partial"
    assert result.pairs == [{"instruction": "q?", "output": "a"}]
    assert result.dropped_items == 1
    assert result.error == "Dropped 1 of 2 schema-invalid entries"


@pytest.mark.parametrize(
    "text",
    [
        '[{"instruction":"27 March, 2023","output":"27 March, 2023"}]',
        '[{"instruction":"Remember this date","output":"27 March, 2023"}]',
    ],
)
def test_strict_parse_rejects_non_question_supervision(text):
    result = parse_qa_pairs_result(text)

    assert not result.valid
    assert result.status == "invalid_schema"
    assert result.pairs == []


def test_deduplicate_qa_pairs_drops_repeats_within_output_and_history():
    existing = [{"instruction": "Where did Alice go?", "output": "New York"}]
    pairs = [
        {"instruction": " where   did ALICE go? ", "output": "new york"},
        {"instruction": "What did Bob buy?", "output": "A bicycle"},
        {"instruction": "What did Bob buy?", "output": "A bicycle"},
    ]

    unique, duplicate_count = deduplicate_qa_pairs(pairs, existing)

    assert unique == [{"instruction": "What did Bob buy?", "output": "A bicycle"}]
    assert duplicate_count == 2


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


class _RetryRollout:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate_batch(self, prompts, *, extraction, adapter_names):
        assert extraction
        self.prompts.extend(prompts)
        return [self.outputs.pop(0) for _ in prompts]


def _args(**overrides):
    values = {
        "extraction_retries": 2,
        "generation_batch_size": 4,
        "max_extraction_tokens": 4,
        "extraction_failure_policy": "empty",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_extraction_retries_only_invalid_requests_and_records_every_attempt():
    rollout = _RetryRollout(
        [
            '[{"instruction":"q1?","output":"a1"}',
            "[]",
            '[{"instruction":"q1?","output":"a1"}]',
        ]
    )
    prompts = [
        [{"role": "user", "content": "extract 1"}],
        [{"role": "user", "content": "extract 2"}],
    ]

    records = _generate_extractions_with_retries(
        rollout,
        _WhitespaceTokenizer(),
        prompts,
        ["episode_1", "episode_2"],
        _args(),
    )

    assert records[0]["pairs"] == [{"instruction": "q1?", "output": "a1"}]
    assert [attempt["status"] for attempt in records[0]["extraction_attempts"]] == ["incomplete_array", "ok"]
    assert [attempt["status"] for attempt in records[1]["extraction_attempts"]] == ["empty"]
    assert len(rollout.prompts) == 3
    assert "previous extraction attempt 1" in rollout.prompts[-1][-1]["content"]


def test_extraction_fails_closed_after_bounded_retries():
    rollout = _RetryRollout(["bad", "still bad"])

    with pytest.raises(ExtractionGenerationError, match="refusing silent empty SFT"):
        _generate_extractions_with_retries(
            rollout,
            _WhitespaceTokenizer(),
            [[{"role": "user", "content": "extract"}]],
            ["episode_1"],
            _args(extraction_retries=1, extraction_failure_policy="error"),
        )


def test_extraction_records_explicit_empty_update_after_bounded_retries():
    rollout = _RetryRollout(["bad", "still bad"])

    record = _generate_extractions_with_retries(
        rollout,
        _WhitespaceTokenizer(),
        [[{"role": "user", "content": "extract"}]],
        ["episode_1"],
        _args(extraction_retries=1),
    )[0]

    assert record["pairs"] == []
    assert record["extraction_failed"] is True
    assert [attempt["status"] for attempt in record["extraction_attempts"]] == ["no_array", "no_array"]


def test_extraction_telemetry_counts_retries_and_legacy_records():
    records = [
        {
            "triggers": [
                {
                    "extraction_failed": True,
                    "duplicate_pairs_dropped": 2,
                    "extraction_attempts": [
                        {"status": "incomplete_array", "at_token_limit": True},
                        {"status": "ok", "at_token_limit": False},
                    ],
                },
                {"raw_extraction": "[]"},
            ]
        }
    ]

    assert extraction_telemetry(records) == {
        "requests": 2,
        "attempts": 3,
        "retries": 1,
        "at_token_limit": 1,
        "failed_requests": 1,
        "duplicate_pairs_dropped": 2,
        "statuses": {"incomplete_array": 1, "legacy_untracked": 1, "ok": 1},
    }
