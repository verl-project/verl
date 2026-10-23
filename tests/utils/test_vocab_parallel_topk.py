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
"""The teacher top-k export must equal a full-vocabulary top-k, exactly.

The distillation loss compares the student against the teacher's top-k log-probs,
so a reduction bug (a wrong normaliser, a local instead of global id, a merge that
misses an entry) does not crash — it silently trains against a distribution that
no model produced. These tests pin the export against ``log_softmax(...).topk``,
which is the definition, including the sharded case: the shards are merged with
the same helpers the distributed path uses, so the merge maths is exercised even
though the test runs on one device.
"""

import pytest

torch = pytest.importorskip("torch")

from verl.utils.megatron.vocab_parallel_topk import (  # noqa: E402
    merge_topk_shards,
    shard_logprobs,
    topk_logprobs_from_logits,
)


def _reference(logits, k):
    return torch.log_softmax(logits.float(), dim=-1).topk(k, dim=-1)


@pytest.mark.parametrize("k", [1, 3, 8])
def test_single_rank_matches_log_softmax(k):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 17)
    got_lp, got_ids = topk_logprobs_from_logits(logits, k)
    want_lp, want_ids = _reference(logits, k)
    assert torch.equal(got_ids, want_ids), (got_ids, want_ids)
    torch.testing.assert_close(got_lp, want_lp, rtol=0, atol=1e-6)


def test_ids_are_global_not_local():
    """A two-shard merge must offset the second shard's ids by the shard width."""
    torch.manual_seed(1)
    full = torch.randn(3, 20)
    a, b = full[:, :10], full[:, 10:]

    global_max = full.max(dim=-1, keepdim=True).values
    global_sumexp = (full - global_max).exp().sum(dim=-1, keepdim=True)

    # each shard keeps its own top-4 candidates, ids offset into global space
    la, ia = shard_logprobs(a, global_max, global_sumexp).topk(4, dim=-1)
    lb, ib = shard_logprobs(b, global_max, global_sumexp).topk(4, dim=-1)
    got_lp, got_ids = merge_topk_shards([la, lb], [ia, ib + 10], k=4)

    want_lp, want_ids = _reference(full, 4)
    assert torch.equal(got_ids, want_ids), (got_ids, want_ids)
    torch.testing.assert_close(got_lp, want_lp, rtol=0, atol=1e-6)


def test_merge_uses_candidates_from_every_shard():
    """The global top-k must be reachable when one shard dominates."""
    torch.manual_seed(2)
    full = torch.randn(1, 8)
    full[0, 6] = 50.0  # a single entry in the second shard dominates
    a, b = full[:, :4], full[:, 4:]
    global_max = full.max(dim=-1, keepdim=True).values
    global_sumexp = (full - global_max).exp().sum(dim=-1, keepdim=True)

    la, ia = shard_logprobs(a, global_max, global_sumexp).topk(2, dim=-1)
    lb, ib = shard_logprobs(b, global_max, global_sumexp).topk(2, dim=-1)
    _, got_ids = merge_topk_shards([la, lb], [ia, ib + 4], k=1)
    assert got_ids.tolist() == [[6]]


def test_k_is_clamped_to_the_local_vocabulary():
    logits = torch.randn(2, 5)
    lp, ids = topk_logprobs_from_logits(logits, 9)
    assert ids.shape == (2, 5)
    assert lp.shape == (2, 5)


def test_rejects_non_positive_k():
    with pytest.raises(ValueError, match="positive"):
        topk_logprobs_from_logits(torch.randn(2, 5), 0)


def test_rejects_k_above_candidates():
    with pytest.raises(ValueError, match="cannot take top"):
        merge_topk_shards([torch.randn(1, 2)], [torch.zeros(1, 2, dtype=torch.long)], k=3)


def test_rejects_mismatched_shard_counts():
    with pytest.raises(ValueError, match="log-prob shards"):
        merge_topk_shards([torch.randn(1, 2)], [], k=1)


def test_log_probs_are_normalised_and_sorted():
    """Sanity on the invariant the loss relies on: sorted, and summing to <= 1."""
    torch.manual_seed(3)
    logits = torch.randn(4, 64)
    lp, _ = topk_logprobs_from_logits(logits, 8)
    assert torch.all(lp[..., 1:] <= lp[..., :-1] + 1e-6), "top-k must be sorted descending"
    assert torch.all(lp <= 0), "log-probs must be non-positive"
    assert torch.all(lp.exp().sum(-1) <= 1.0 + 1e-5), "top-k mass must not exceed 1"
