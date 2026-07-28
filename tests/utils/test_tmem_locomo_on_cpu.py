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
import torch

from examples.tmem.locomo import (
    EXTRACTION_END_SENTINEL,
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
from examples.tmem.merge_shards import merge_rollout_stats
from examples.tmem.run_locomo import (
    OFFICIAL_LOCOMO_ANSWER_HPARAMS,
    PAPER_TMEM_HPARAMS,
    DFlashRollout,
    JsonArrayEndCriteria,
    _decode_sglang_output,
    _render_sglang_prompt,
    parse_args,
    sampling_seed_for_request,
    validate_table1_hparams,
)


def test_parse_qa_pairs_accepts_fenced_json_and_drops_invalid_entries():
    text = """<think>draft</think>```json
    [{"instruction": " Who? ", "output": " Alice "}, {"instruction": 3, "output": "bad"}]
    ```"""
    assert parse_qa_pairs(text) == [{"instruction": "Who?", "output": "Alice"}]


def test_table1_hparams_are_locked_to_paper_values():
    args = SimpleNamespace(**(PAPER_TMEM_HPARAMS | OFFICIAL_LOCOMO_ANSWER_HPARAMS))
    validate_table1_hparams(args)

    args.epochs = 2
    with pytest.raises(ValueError, match=r"epochs.*2.*5"):
        validate_table1_hparams(args)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("max_answer_tokens", 51, 50),
        ("answer_temperature", 0.7, 0.4),
        ("answer_top_p", 0.8, 0.9),
        ("answer_top_k", 20, 10),
    ],
)
def test_table1_hparams_lock_official_locomo_answer_protocol(name, value, expected):
    args = SimpleNamespace(**(PAPER_TMEM_HPARAMS | OFFICIAL_LOCOMO_ANSWER_HPARAMS))
    setattr(args, name, value)

    with pytest.raises(ValueError, match=rf"{name}.*{value}.*{expected}"):
        validate_table1_hparams(args)


def test_cli_defaults_follow_official_locomo_answer_protocol(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_locomo", "--data", "locomo10.json"])

    args = parse_args()

    assert {name: getattr(args, name) for name in OFFICIAL_LOCOMO_ANSWER_HPARAMS} == OFFICIAL_LOCOMO_ANSWER_HPARAMS


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
        def batch_decode(self, token_ids, skip_special_tokens=True):
            values = {0: "[", 1: '{"instruction":"x]y","output":"z"}', 2: "]", 9: "prompt"}
            return ["".join(values[token_id] for token_id in row) for row in token_ids]

    criteria = JsonArrayEndCriteria(Tokenizer(), prompt_length=1)
    finished = criteria(torch.tensor([[9, 0, 1], [9, 0, 2]]), torch.empty(0))
    torch.testing.assert_close(finished, torch.tensor([False, True]))


def test_dflash_extraction_stops_at_unambiguous_sentinel_not_closing_bracket():
    rollout = object.__new__(DFlashRollout)
    rollout.args = SimpleNamespace(
        max_extraction_tokens=4096,
        max_answer_tokens=50,
        extraction_temperature=0.7,
        extraction_top_p=0.8,
        extraction_top_k=20,
        answer_temperature=0.4,
        answer_top_p=0.9,
        answer_top_k=10,
    )

    params = rollout._sampling_params(extraction=True, sampling_seed=17)

    assert "stop_token_ids" not in params
    assert "no_stop_trim" not in params
    assert params["stop"] == EXTRACTION_END_SENTINEL
    assert params["max_new_tokens"] == 4096
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.8
    assert params["top_k"] == 20
    assert params["sampling_seed"] == 17


def test_sglang_extraction_restores_prefilled_json_array_start():
    assert _decode_sglang_output('{"instruction":"q","output":"a"}]', extraction=True) == (
        '[{"instruction":"q","output":"a"}]'
    )
    assert _decode_sglang_output("plain answer", extraction=False) == "plain answer"
    assert (
        _decode_sglang_output(
            f'{{"instruction":"q","output":"a"}}]{EXTRACTION_END_SENTINEL}ignored',
            extraction=True,
        )
        == '[{"instruction":"q","output":"a"}]'
    )


def test_sglang_extraction_prompt_requests_unambiguous_end_sentinel():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tokenize"] is False
            return messages[-1]["content"]

    rendered = _render_sglang_prompt(
        Tokenizer(),
        [{"role": "user", "content": "Return JSON."}],
        extraction=True,
    )

    assert f"append exactly {EXTRACTION_END_SENTINEL}" in rendered
    assert rendered.endswith("[")


def test_dflash_stats_are_reset_per_seed():
    rollout = object.__new__(DFlashRollout)
    rollout.dflash_block_size = 16
    rollout.reset_stats(seed=7)
    rollout.generation_seconds = 2.5
    rollout.completion_tokens = 10
    rollout.spec_verify_count = 4
    rollout.spec_accept_length_sum = 6.0
    rollout.spec_accept_length_count = 2

    assert rollout.stats() == {
        "dflash_block_size": 16,
        "generation_calls": 0,
        "resumed_request_count": 0,
        "generation_seconds": 2.5,
        "completion_tokens": 10,
        "spec_verify_count": 4,
        "spec_accept_length_count": 2,
        "mean_spec_accept_length": 3.0,
    }


def test_dflash_resume_restores_sampling_request_count():
    rollout = object.__new__(DFlashRollout)
    rollout.dflash_block_size = 16
    rollout.reset_stats(seed=3)
    rollout.restore_progress(
        [
            {"trigger_count": 3},
            {"triggers": [{}, {}]},
        ]
    )

    assert rollout.request_count == 7
    assert rollout.resumed_request_count == 7
    assert rollout.stats()["resumed_request_count"] == 7


def test_dflash_sampling_seed_is_batch_and_shard_invariant():
    prompt = "<|im_start|>user\nWhat happened?<|im_end|>"

    assert sampling_seed_for_request(3, "episode_7", prompt) == sampling_seed_for_request(3, "episode_7", prompt)
    assert sampling_seed_for_request(3, "episode_7", prompt) != sampling_seed_for_request(3, "episode_8", prompt)


def test_merge_rollout_stats_weights_acceptance_by_request_count():
    merged = merge_rollout_stats(
        [
            {
                "dflash_block_size": 16,
                "generation_calls": 2,
                "resumed_request_count": 3,
                "generation_seconds": 4.0,
                "completion_tokens": 20,
                "spec_verify_count": 5,
                "spec_accept_length_count": 2,
                "mean_spec_accept_length": 3.0,
            },
            {
                "dflash_block_size": 16,
                "generation_calls": 4,
                "resumed_request_count": 0,
                "generation_seconds": 6.0,
                "completion_tokens": 30,
                "spec_verify_count": 7,
                "spec_accept_length_count": 3,
                "mean_spec_accept_length": 5.0,
            },
        ]
    )

    assert merged == {
        "dflash_block_size": 16,
        "generation_calls": 6,
        "resumed_request_count": 3,
        "generation_gpu_seconds": 10.0,
        "completion_tokens": 50,
        "spec_verify_count": 12,
        "spec_accept_length_count": 5,
        "mean_spec_accept_length": 4.2,
    }
