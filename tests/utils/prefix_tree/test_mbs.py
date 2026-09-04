# Copyright 2025 Meituan Ltd. and/or its affiliates
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


"""Unit tests for the reorder-safe micro-batch grouping API."""

from __future__ import annotations

import pytest
import torch

from verl.utils import tensordict_utils as tu
from verl.utils.prefix_tree.dynamic import (
    greedy_build_tries,
    mbs_groups_from_leaf_idx,
    prepare_prefix_tree_micro_batches,
    trie_group_flat_tokens,
)


def _make_samples(n_prompts, rollout_n, prefix_len, resp_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for p in range(n_prompts):
        prefix = torch.randint(0, 151936, (prefix_len,), generator=g)
        for _ in range(rollout_n):
            resp = torch.randint(0, 151936, (resp_len,), generator=g)
            samples.append(torch.cat([prefix, resp]))
    return samples


def _build_trie(samples):
    seq_lists = [s.tolist() if hasattr(s, "tolist") else list(s) for s in samples]
    trie, _ = greedy_build_tries(seq_lists)
    return trie


def _leaf_idx_from_trie(trie, n_samples):
    """Build canonical leaf_idx: sample i -> its leaf's node_idx."""
    leaf_idx = torch.full((n_samples,), -1, dtype=torch.long)
    for node in trie.nodes:
        if not node.children:  # leaf
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = node.node_idx
    assert int(leaf_idx.min().item()) >= 0, "trie has samples with no leaf"
    return leaf_idx


def test_mbs_groups_from_leaf_idx_covers_all_and_respects_budget():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    budget = 500
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=budget)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    for mb in mbs:
        assert trie_group_flat_tokens(mb, trie) <= budget


def test_mbs_groups_from_leaf_idx_reorder_safe():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx0 = _leaf_idx_from_trie(trie, len(samples))
    budget = 500
    perm = torch.randperm(len(samples), generator=torch.Generator().manual_seed(7)).tolist()
    leaf_idx1 = leaf_idx0[perm].clone()
    mbs = mbs_groups_from_leaf_idx(leaf_idx1, trie, max_token_len=budget)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    canon = mbs_groups_from_leaf_idx(leaf_idx0, trie, max_token_len=budget)
    canon_leaves = sorted(sorted({int(leaf_idx0[i]) for i in mb}) for mb in canon)
    perm_leaves = sorted(sorted({int(leaf_idx1[i]) for i in mb}) for mb in mbs)
    assert canon_leaves == perm_leaves


def test_mbs_groups_from_leaf_idx_duplicates_stay_together():
    base = [1, 2, 3, 4, 5]
    seqs = [base, base, [1, 2, 9, 9, 9], [1, 2, 7, 7, 7]]
    trie = _build_trie(seqs)
    leaf_idx = _leaf_idx_from_trie(trie, len(seqs))
    assert int(leaf_idx[0]) == int(leaf_idx[1])  # identical samples share leaf
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=10_000)
    pos0 = next(i for i, mb in enumerate(mbs) if 0 in mb)
    pos1 = next(i for i, mb in enumerate(mbs) if 1 in mb)
    assert pos0 == pos1, f"duplicate samples split: mb0={pos0} mb1={pos1}"


def test_mbs_groups_from_leaf_idx_raises_on_orphan():
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    leaf_idx[1] = -1  # orphan: sample 1 has no leaf
    with pytest.raises(ValueError, match="no leaf assigned"):
        mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)


def test_mbs_groups_from_leaf_idx_allows_non_leaf_ref():
    """leaf_idx pointing to an internal node (strict-prefix sample) groups without error."""
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    internal_node = None
    for node in trie.nodes:
        if node.children:
            internal_node = node
            break
    assert internal_node is not None, "trie needs at least one internal node for this test"
    leaf_idx[0] = internal_node.node_idx
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))


def test_mbs_groups_from_leaf_idx_budget_overrides_uid():
    """Uid atomicity is owned by DP balancing, not mbs grouping: rollouts of one
    prompt may split across micro-batches so the flat-token budget is always
    respected (budget 15 fits one leaf's path but not two rollouts of a prompt)."""
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    mbs = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=15)
    assert sorted(i for mb in mbs for i in mb) == list(range(len(samples)))
    for mb in mbs:
        assert trie_group_flat_tokens(mb, trie) <= 15, f"micro-batch over budget: {trie_group_flat_tokens(mb, trie)}"


def test_mbs_groups_from_leaf_idx_skips_other_rank_leaves():
    samples = _make_samples(4, 2, prefix_len=20, resp_len=10, seed=7)
    trie = _build_trie(samples)
    full_leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    n = len(samples)
    rank0_leaf_idx = full_leaf_idx[: n // 2].clone()
    rank1_leaf_idx = full_leaf_idx[n // 2 :].clone()
    mbs0 = mbs_groups_from_leaf_idx(rank0_leaf_idx, trie, max_token_len=10_000)
    mbs1 = mbs_groups_from_leaf_idx(rank1_leaf_idx, trie, max_token_len=10_000)
    assert sorted(i for mb in mbs0 for i in mb) == list(range(len(rank0_leaf_idx)))
    assert sorted(i for mb in mbs1 for i in mb) == list(range(len(rank1_leaf_idx)))


def test_prepare_prefix_tree_micro_batches_attaches_subtrie():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    n = len(samples)
    seq_len = max(len(s) for s in samples)
    input_ids = torch.zeros((n, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    for i, s in enumerate(samples):
        input_ids[i, : len(s)] = s
        attention_mask[i, : len(s)] = 1
    budget = 500
    td = tu.get_tensordict(
        tensor_dict={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "leaf_idx": leaf_idx,
        },
        non_tensor_dict={
            "prefix_tree": trie,
            "use_dynamic_bsz": True,
            "use_prefix_tree": True,
            "sp_size": 1,
            "force_group_size": 1,
            "max_token_len_per_gpu": budget,
        },
    )
    micro_batches, batch_idx_list = prepare_prefix_tree_micro_batches(td, sp_size=1)
    assert len(micro_batches) == len(batch_idx_list)
    for mb, mb_idx in zip(micro_batches, batch_idx_list, strict=False):
        subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
        assert subtree is not None, "prefix_tree_subtree not attached"
        mb_leaves = sorted(int(x) for x in mb["leaf_idx"].tolist())
        sub_leaves = sorted(subtree.leaf_node_ids)
        assert sub_leaves == mb_leaves, f"{sub_leaves} != {mb_leaves}"
