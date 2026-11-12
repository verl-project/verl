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

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.workers.rollout.utils import (
    extract_response_topk_logprobs_with_coverage,
    validate_topk_logprob_request,
    warn_topk_logprob_budget,
)


@dataclass
class _FlatLogprobs:
    token_ids: list[int]
    logprobs: list[float]
    start_indices: list[int]
    end_indices: list[int]
    ranks: list[int]

    def __len__(self):
        return len(self.start_indices)


def _flat(heads):
    k = len(heads[0])
    return _FlatLogprobs(
        token_ids=[token_id for _ in heads for token_id in [0, *range(k)]],
        logprobs=[value for head in heads for value in [head[0], *head]],
        start_indices=list(range(0, len(heads) * (k + 1), k + 1)),
        end_indices=list(range(k + 1, (len(heads) + 1) * (k + 1), k + 1)),
        ranks=[rank for _ in heads for rank in [1, *range(1, k + 1)]],
    )


def test_ranked_zero_slot_proves_coverage_without_probability_mass_heuristic():
    flat = _flat([[0.0, float("-inf"), float("-inf")], [-1e-8, -20.0, -30.0]])
    _, ids, log_probs, complete = extract_response_topk_logprobs_with_coverage(flat, k=3, vocab_size=11)

    assert complete.dtype == np.bool_ and complete.tolist() == [True, False]
    assert ids.shape == log_probs.shape == (2, 3)
    # The all-finite head rounds to unit mass in fp32 but still lacks proof about omitted tokens.
    assert np.exp(log_probs[1]).sum() == 1


def test_whole_model_vocabulary_proves_coverage_with_all_finite_slots():
    flat = _flat([np.log([0.7, 0.2, 0.1]).tolist()])
    _, _, _, complete = extract_response_topk_logprobs_with_coverage(flat, k=3, vocab_size=3)
    assert complete.tolist() == [True]


def test_legacy_layout_without_provider_evidence_stays_unknown():
    flat = SimpleNamespace(token_ids=[0, 0, 1], logprobs=[0.0, 0.0, float("-inf")])

    class Legacy:
        token_ids, logprobs = flat.token_ids, flat.logprobs

        def __len__(self):
            return 1

    _, _, _, complete = extract_response_topk_logprobs_with_coverage(Legacy(), k=2, vocab_size=2)
    assert complete.tolist() == [False]


def test_short_and_long_flat_rows_cannot_cancel_out_to_invent_zero_slots():
    flat = _flat([[0.0, float("-inf")], [0.0, float("-inf")]])
    flat.end_indices = [2, 6]
    flat.start_indices = [0, 2]
    with pytest.raises(ValueError, match="contiguous flat rows"):
        extract_response_topk_logprobs_with_coverage(flat, k=2)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 1.0])
def test_invalid_provider_logprobs_are_rejected(bad_value):
    flat = _flat([[bad_value, float("-inf")]])
    with pytest.raises(ValueError, match="finite or -inf"):
        extract_response_topk_logprobs_with_coverage(flat, k=2)


def test_duplicate_global_head_ids_are_rejected():
    flat = _flat([[-0.2, -2.0]])
    flat.token_ids = [0, 0, 0]
    with pytest.raises(ValueError, match="unique"):
        extract_response_topk_logprobs_with_coverage(flat, k=2)


def test_individually_valid_logprobs_with_impossible_head_mass_are_rejected():
    flat = _flat([[-0.1, -0.2]])
    with pytest.raises(ValueError, match="mass must not exceed one"):
        extract_response_topk_logprobs_with_coverage(flat, k=2)


def test_sampled_logprob_must_match_a_recorded_head_entry():
    flat = _flat([[-0.1, -2.5]])
    flat.logprobs[0] = -1.0
    with pytest.raises(ValueError, match="match the corresponding"):
        extract_response_topk_logprobs_with_coverage(flat, k=2)


def test_complete_head_cannot_omit_a_sampled_token_with_positive_probability():
    flat = _flat([[0.0, float("-inf")]])
    flat.token_ids[0] = 3
    flat.logprobs[0] = -2.0
    with pytest.raises(ValueError, match="must contain the sampled token"):
        extract_response_topk_logprobs_with_coverage(flat, k=2, vocab_size=4)


def test_complete_evidence_with_nonunit_head_mass_is_corrupt_not_a_fallback():
    flat = _flat([[-0.5, float("-inf")]])
    with pytest.raises(ValueError, match="unit probability mass"):
        extract_response_topk_logprobs_with_coverage(flat, k=2)


def test_sampled_id_must_be_in_known_model_vocabulary_even_without_provider_layout():
    class Legacy:
        token_ids = [3, 0, 1]
        logprobs = [-2.0, -0.1, -2.5]

        def __len__(self):
            return 1

    with pytest.raises(ValueError, match="sampled ids must lie within"):
        extract_response_topk_logprobs_with_coverage(Legacy(), k=2, vocab_size=3)


def test_missing_ranked_slot_cannot_be_used_as_complete_evidence():
    flat = _flat([[0.0, float("-inf")]])
    flat.ranks[-1] = 0
    with pytest.raises(ValueError, match="consecutive provider ranks"):
        extract_response_topk_logprobs_with_coverage(flat, k=2)


def test_out_of_vocabulary_id_cannot_prove_whole_vocab_coverage():
    flat = _flat([[-0.2, -2.0]])
    flat.token_ids[-1] = 2
    with pytest.raises(ValueError, match="model vocabulary"):
        extract_response_topk_logprobs_with_coverage(flat, k=2, vocab_size=2)


def test_custom_token_logprobs_may_pad_missing_records_so_cannot_supply_global_head():
    # Even consecutive ranks and an -inf cannot prove support when ids were explicitly requested.
    params = SimpleNamespace(logprobs=3, logprob_token_ids=[0], temperature=1.0)
    with pytest.raises(ValueError, match="custom logprob_token_ids"):
        validate_topk_logprob_request(params)


def test_greedy_processed_logprobs_are_not_a_stochastic_sampling_head():
    with pytest.raises(ValueError, match="stochastic sampling"):
        validate_topk_logprob_request(SimpleNamespace(logprobs=3, logprob_token_ids=None, temperature=0.0))
    # Baselines and validation still request ordinary sampled-token log-probs without a head.
    validate_topk_logprob_request(SimpleNamespace(logprobs=0, logprob_token_ids=None, temperature=0.0))


def test_stochastic_full_head_request_is_accepted():
    validate_topk_logprob_request(SimpleNamespace(logprobs=3, logprob_token_ids=None, temperature=0.01))


def test_effective_request_budget_warning_is_once_per_profile_even_when_support_fits(caplog):
    params = SimpleNamespace(logprobs=2, top_k=8, top_p=0.2, min_p=0.05, temperature=0.7)
    warned = set()
    warn_topk_logprob_budget(params, warned)
    warn_topk_logprob_budget(params, warned)
    messages = [record.message for record in caplog.records if "effective sampling top_k" in record.message]
    assert len(messages) == 1
    assert "top_k=8" in messages[0] and "temperature=0.7" in messages[0] and "min_p=0.05" in messages[0]
    # Coverage is decided from returned records; the record-budget warning does not force fallback.
    _, _, _, complete = extract_response_topk_logprobs_with_coverage(_flat([[0.0, float("-inf")]]), k=2)
    assert complete.tolist() == [True]


@pytest.mark.parametrize("top_k,record_k", [(-1, 2), (0, 2), (2, 2), (1, 2), (8, 0)])
def test_no_record_budget_warning_when_head_is_not_below_enabled_sampling_top_k(caplog, top_k, record_k):
    params = SimpleNamespace(logprobs=record_k, top_k=top_k, top_p=0.9, min_p=0.0, temperature=1.0)
    warned = set()
    warn_topk_logprob_budget(params, warned)
    assert not warned
    assert not [record for record in caplog.records if "effective sampling top_k" in record.message]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_first", [False, True])
async def test_partial_rollout_resume_keeps_missing_coverage_unknown(monkeypatch, legacy_first):
    from verl.workers.rollout import llm_server
    from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
    from verl.workers.rollout.replica import TokenOutput

    def segment(token_id, complete, reason):
        extra = {
            "response_topk_ids": np.array([[token_id, token_id + 1]], dtype=np.int32),
            "response_topk_log_probs": np.array([[-0.1, -2.0]], dtype=np.float32),
        }
        if complete:
            extra["response_support_complete"] = np.array([True], dtype=np.bool_)
        return TokenOutput(token_ids=[token_id], extra_fields=extra, stop_reason=reason)

    pending = iter([segment(11, not legacy_first, "aborted"), segment(13, legacy_first, "completed")])

    async def fake_generate(self, request_id, *, prompt_ids, **kwargs):
        return next(pending)

    async def no_wait(delay):
        return None

    monkeypatch.setattr(llm_server.LLMServerClient, "generate", fake_generate)
    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)
    client = FullyAsyncLLMServerClient(config=SimpleNamespace(), load_balancer_handle=None)
    output = await client.generate(request_id="r0", prompt_ids=[1], sampling_params={})

    assert output.extra_fields["response_support_complete"].dtype == np.bool_
    assert output.extra_fields["response_support_complete"].tolist() == [not legacy_first, legacy_first]
    assert len(output.extra_fields["response_support_complete"]) == len(output.token_ids)


def test_synthetic_padding_support_evidence_matches_its_complete_dummy_heads():
    from verl.trainer.ppo.padding_utils import construct_minimal_padding_template

    source = {
        "rollout_topk_ids": torch.zeros(6, 3, dtype=torch.int32),
        "rollout_topk_log_probs": torch.zeros(6, 3),
        "rollout_support_complete": torch.zeros(6, dtype=torch.bool),
    }
    padded, _ = construct_minimal_padding_template(source, {}, eos_token_id=1)
    evidence = padded["rollout_support_complete"]
    assert evidence.shape == padded["input_ids"].shape and evidence.dtype == torch.bool
    assert evidence.all()
    assert padded["response_mask"].sum() == 0
