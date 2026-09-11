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


"""Trie/dynamic-builder unit tests: greedy_build_tries, build_tree_dynamic,"""

from __future__ import annotations

import pickle
import random

import pytest
import torch
from _helpers import build_trie as _build_trie
from _helpers import make_grpo_samples, make_pt_batch

from verl.utils.prefix_tree.dynamic import (
    balance_prefix_tree_blocks,
    build_subtrie_view,
    build_tree_dynamic,
    convert_trie_to_tree_node,
    dfs_leaf_order,
    greedy_build_tries,
)
from verl.utils.prefix_tree.dynamic import (
    mbs_groups_from_leaf_idx as _mbs_groups,
)
from verl.utils.prefix_tree.magi import restore_flat_to_nested
from verl.utils.prefix_tree.trainer import build_global_trie as _build_global_trie
from verl.utils.prefix_tree.tree import PrefixSubTrie
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def test_greedy_build_tries_and_dfs_leaf_order():
    seqs = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    trie, _ = greedy_build_tries(seqs)
    assert trie.is_root
    order = dfs_leaf_order(seqs, trie)
    assert set(order) == {0, 1, 2} and len(order) == 3
    raw = [[1, 2, 10], [5, 6, 20], [1, 2, 11], [5, 6, 21]]
    o2 = dfs_leaf_order(raw, _build_trie(raw))
    a = sorted([o2.index(0), o2.index(2)])
    b = sorted([o2.index(1), o2.index(3)])
    assert a[1] - a[0] == 1 and b[1] - b[0] == 1


def test_build_subtrie_view_all_subset_and_empty():
    trie = _build_trie([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]])
    sub_all = build_subtrie_view(trie, {0, 1, 2})
    assert isinstance(sub_all, PrefixSubTrie)
    assert sorted(sub_all.leaf_to_sample) == [0, 1, 2] and len(sub_all.nodes[0].input_ids) == 2

    sub_one = build_subtrie_view(trie, {0})
    assert sub_one.leaf_to_sample == [0] and len(sub_one.nodes) == 1  # [1,2,3,4] folded into one leaf

    sub_two = build_subtrie_view(trie, {0, 2})
    assert set(sub_two.leaf_to_sample) == {0, 2}
    for lid in sub_two.leaf_node_ids:
        assert 0 <= lid < len(trie.nodes)

    assert build_subtrie_view(trie, set()) is None and build_subtrie_view(trie, {99}) is None


def test_build_tree_dynamic_and_convert_none_cases():
    s2 = [torch.tensor([10, 11, 20, 21]), torch.tensor([10, 11, 30, 31]), torch.tensor([10, 11, 40, 41])]
    r2 = build_tree_dynamic(s2)
    assert r2 is not None and len(r2.nodes[0].input_ids) == 2 and sorted(r2.leaf_to_sample) == [0, 1, 2]
    assert build_tree_dynamic([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]) is None
    assert build_tree_dynamic([]) is None
    # multi-root trie (no shared first token) cannot fold into one tree
    assert convert_trie_to_tree_node(_build_trie([[1, 2], [3, 4]])) is None


def test_layout_token_conservation_and_zero_length_leaf_skipped():
    s2 = [torch.tensor([10, 11, 20, 21]), torch.tensor([10, 11, 30, 31]), torch.tensor([10, 11, 40, 41])]
    p2 = build_layout_from_tree_node(s2, build_tree_dynamic(s2))
    assert p2.tree_packed_tokens.shape[0] >= 8  # at least the raw 8 tokens
    assert p2.prefix_range[0] == 0 and p2.prefix_range[1] >= 1
    assert len(p2.leaf_ranges) == 3
    nested = [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6])]
    pn = build_layout_from_tree_node(nested, build_tree_dynamic(nested))
    assert list(pn.tree_packed_tokens[:2].tolist()) == [1, 2]
    assert set(pn.tree_packed_tokens.tolist()) == {1, 2, 3, 4, 5, 6}


def test_position_ids_are_sample_local():
    """Position IDs reset at branch points — sample-local, not flat 0..N-1."""
    s1 = [torch.tensor([10, 20, 30, 41, 42]), torch.tensor([10, 20, 30, 51])]
    p1 = build_layout_from_tree_node(s1, build_tree_dynamic(s1))
    assert p1.tree_packed_position_ids.tolist() == [0, 1, 2, 3, 4, 3]
    custom_pids = [
        torch.tensor([10, 11, 12, 13, 14]),
        torch.tensor([10, 11, 12, 15]),
    ]
    p2 = build_layout_from_tree_node(s1, build_tree_dynamic(s1), position_ids_by_sample=custom_pids)
    assert p2.tree_packed_position_ids.tolist() == [10, 11, 12, 13, 14, 15]


def test_fuzz_random_tree_round_trip():
    """Fuzz: random tree topologies, verify full restore (includes token collisions)."""
    rng = random.Random(42)
    for _ in range(20):
        n_samples = rng.randint(3, 12)
        prefix_len = rng.randint(2, 10)
        base = [rng.randint(0, 100) for _ in range(prefix_len)]
        samples = [torch.tensor(base)]
        for __ in range(n_samples - 1):
            parent = samples[rng.randint(0, len(samples) - 1)].tolist()
            split = rng.randint(min(len(base), 1), len(parent))
            suffix_len = rng.randint(1, 5)
            suffix = [rng.randint(0, 100) for _ in range(suffix_len)]
            samples.append(torch.tensor(parent[:split] + suffix))
        restored = _build_and_restore(samples)
        assert len(restored) == len(samples)
        for i, (orig, rest) in enumerate(zip(samples, restored, strict=False)):
            assert torch.equal(orig, rest), f"run {_}, sample {i}: {orig.tolist()} != {rest.tolist()}"

    empty_count = 0
    for run in range(8):
        n_trees = 128
        samples = []
        for _t in range(n_trees):
            prefix = [rng.randint(0, 10_000) for _ in range(rng.randint(2, 12))]
            n_leaf = rng.randint(2, 5)
            for _l in range(n_leaf - 1):
                suffix = [rng.randint(0, 10_000) for _ in range(rng.randint(1, 16))]
                samples.append(torch.tensor(prefix + suffix))
            samples.append(torch.tensor(prefix))  # empty leaf: sample == prefix
            empty_count += 1
        subtrie = _make_subtrie([s.tolist() for s in samples], range(len(samples)))
        restored = _build_and_restore(samples, subtrie=subtrie)
        assert len(restored) == len(samples)
        for i, (orig, rest) in enumerate(zip(samples, restored, strict=False)):
            assert torch.equal(orig, rest), f"run {run}, sample {i}: {orig.tolist()} != {rest.tolist()}"

        # Direct iterator-contract checks (regressions surface here, not as restore mismatches).
        in_view = {n.node_idx for n in subtrie.nodes}
        # children_of must yield only in-view children (subtrie view boundary).
        for n in subtrie.nodes:
            assert all(c.node_idx in in_view for c in subtrie.children_of(n)), (
                f"run {run}: children_of leaked out-of-view child"
            )
        # bfs/dfs must visit exactly the subtrie's nodes, no more (no pruned leak).
        assert {n.node_idx for n in subtrie.bfs()} == in_view, "bfs leaked past view"
        assert {n.node_idx for n in subtrie.dfs()} == in_view, "dfs leaked past view"
        # dfs(leaf_only=True) must visit only leaves (nodes with no in-view children).
        bfs_leaves = {n.node_idx for n in subtrie.bfs() if not subtrie.children_of(n)}
        assert {n.node_idx for n in subtrie.dfs(leaf_only=True)} == bfs_leaves, (
            "dfs leaf_only disagrees with children_of-defined leaves"
        )
    assert empty_count == 8 * 128, "empty-leaf case must be exercised for every tree"


def test_fuzz_tree_balance_reduces_imbalance():
    """Fuzz: multiple random trees; tree-level KK balance must never worsen the"""
    rng = random.Random(7)
    improved = 0
    checked = 0
    for _ in range(50):
        n_trees = rng.randint(3, 8)
        seq_lists = []
        block_ids = []
        for _t in range(n_trees):
            base = [rng.randint(0, 10_000) for _ in range(rng.randint(1, 5))]
            n_leaf = rng.randint(1, 5)
            for _l in range(n_leaf):
                suffix = [rng.randint(0, 10_000) for _ in range(rng.randint(0, 30))]
                seq_lists.append(base + suffix)
                block_ids.append(f"t{_t}")
        trie = _build_trie(seq_lists)
        dp = rng.randint(2, min(4, n_trees))
        permutation, partitions, workloads = balance_prefix_tree_blocks(trie, dp, block_ids)
        assert sorted(permutation) == list(range(len(seq_lists)))
        if len(partitions) < dp:
            continue
        balanced = sorted(sum(workloads[i] for i in part) for part in partitions)
        per = len(workloads) // dp
        natural = []
        for i in range(dp):
            lo = i * per
            hi = (i + 1) * per if i < dp - 1 else len(workloads)
            natural.append(sum(workloads[lo:hi]))
        natural = sorted(natural)
        b_imb = balanced[-1] - balanced[0]
        n_imb = natural[-1] - natural[0]
        checked += 1
        assert b_imb <= n_imb, f"run {_}: balanced imbalance {b_imb} > natural {n_imb}"
        if b_imb < n_imb:
            improved += 1
    assert checked >= 40
    assert improved > 0, "tree balance never strictly improved imbalance"


def test_balance_by_prompt_ids_keeps_prompt_blocks_whole():
    """Cross-prompt shared prefix (system prompt) must NOT merge prompt blocks."""
    seq_lists = []
    block_ids = []
    for p in range(6):
        prompt = [10 + p, 20 + p]
        for r in range(3):
            seq_lists.append([1, 2, 3] + prompt + [100 + r])  # system + prompt + response
            block_ids.append(f"uid_{p}_0")
    trie = _build_trie(seq_lists)
    permutation, partitions, workloads = balance_prefix_tree_blocks(trie, 2, block_ids)
    assert sorted(permutation) == list(range(len(seq_lists)))
    assert len(workloads) == 6, f"expected 6 prompt blocks, got {len(workloads)}"
    for p in range(6):
        positions = [permutation.index(i) for i in range(p * 3, p * 3 + 3)]
        assert positions == list(range(min(positions), max(positions) + 1)), f"prompt {p} split"


def test_strict_prefix_zero_length_leaf_boundary_skipped():
    """Strict-prefix sample (zero-length response) should not appear in boundary registry."""
    samples = [torch.tensor([1, 2, 3, 10, 11]), torch.tensor([1, 2, 3])]
    p = build_layout_from_tree_node(samples, build_tree_dynamic(samples))
    registry = getattr(p, "boundary_registry", None)
    if registry:
        for b_pos, leaves_info in registry:
            for sample_idx, _ in leaves_info:
                assert sample_idx != 1, f"sample 1 (zero-length) appeared at boundary {b_pos}"


def _make_subtrie(raw_seqs, keep_ids):
    trie, _ = greedy_build_tries(raw_seqs)
    subtrie = build_subtrie_view(trie, set(keep_ids))
    assert subtrie is not None
    return subtrie


def _samples(raw):
    return [torch.tensor(s, dtype=torch.long) for s in raw]


def test_pickle_round_trip_and_duplicate_leaf_alignment():
    raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    st = _make_subtrie(raw, [0, 1, 2])
    samps = _samples(raw)
    p1 = build_layout_from_tree_node(samps, st)
    st2 = pickle.loads(pickle.dumps(st))
    p2 = build_layout_from_tree_node(samps, st2)
    assert torch.equal(p1.tree_packed_tokens, p2.tree_packed_tokens)
    assert p1.leaf_to_sample == p2.leaf_to_sample
    assert p1.q_ranges == p2.q_ranges
    assert p1.prefix_range == p2.prefix_range
    # children must be reconstructed after unpickling (the layout build walks them)
    valid = {n.node_idx for n in st2.nodes}
    assert any(c.node_idx in valid for c in st2.nodes[0].children.values())
    # duplicate samples: leaf_ranges stays aligned with leaf_to_sample
    dup = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 5, 6]]
    p_d = build_layout_from_tree_node(_samples(dup), _make_subtrie(dup, [0, 1, 2]))
    assert len(p_d.leaf_ranges) == len(p_d.leaf_to_sample) == 3
    assert set(p_d.leaf_to_sample) == {0, 1, 2}


def _build_and_restore(samples: list[torch.Tensor], subtrie=None) -> list[torch.Tensor]:
    """Build flat layout from subtrie, then restore back to per-sample tokens."""
    if subtrie is None:
        subtrie = build_tree_dynamic(samples)
    assert subtrie is not None
    params = build_layout_from_tree_node(samples, subtrie)
    pt_batch = make_pt_batch(params, subtrie)
    restored = restore_flat_to_nested(params.tree_packed_tokens, pt_batch)
    offsets, vals = restored.offsets(), restored.values()
    lengths = offsets.diff().tolist()
    result = []
    pos = 0
    for length in lengths:
        result.append(vals[pos : pos + int(length)])
        pos += int(length)
    return result


@pytest.mark.parametrize(
    "samples",
    [
        [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6])],
        [torch.tensor([10, 11, 20, 21]), torch.tensor([10, 11, 30, 31]), torch.tensor([10, 11, 40, 41])],
        [
            torch.tensor([1, 2, 3, 10]),
            torch.tensor([1, 2, 3, 11]),
            torch.tensor([1, 2, 3, 20, 21]),
            torch.tensor([1, 2, 3, 20, 22]),
        ],
    ],
    ids=["strict-prefix", "nested-prefix", "internal-node-owner"],
)
def test_round_trip(samples):
    """strict-prefix: [1,2] ⊂ [1,2,3,4] ⊂ [1,2,3,4,5,6]; nested-prefix: 3
    samples share [10,11] with different suffixes; internal-node-owner: internal
    nodes shared by non-sample-0 branches must pack correct tokens."""
    restored = _build_and_restore(samples)
    assert len(restored) == len(samples)
    for i, (orig, rest) in enumerate(zip(samples, restored, strict=False)):
        assert torch.equal(orig, rest), f"sample {i}: {orig.tolist()} != {rest.tolist()}"


def test_dp_shard_subtrie_round_trip():
    """First-half DP shard: build subtrie, layout, restore — all samples recovered."""
    prefix = torch.randint(0, 1000, (100,))
    samples = []
    for _ in range(8):
        suffix = torch.randint(0, 1000, (50,))
        samples.append(torch.cat([prefix, suffix]))
    full = build_tree_dynamic(samples)
    assert full is not None

    half = set(range(len(samples) // 2))
    shard_sub = build_subtrie_view(full.source or full, half)
    if shard_sub is None:
        return  # no sharing in this shard, skip
    shard_samples = [samples[i] for i in sorted(half)]
    restored = _build_and_restore(shard_samples, shard_sub)
    assert len(restored) == len(shard_samples)
    for orig, rest in zip(shard_samples, restored, strict=False):
        assert torch.equal(orig, rest)


# ---- worker-side prefix-tree restore contracts ----
# These tests walk the EXACT worker-side flow used in production:
#     build_global_trie (deepest-node leaf_idx)
#     → mbs_groups_from_leaf_idx / create_and_attach_subtrie_views
#       (leaf_to_sample = LOCAL positions within the micro-batch)
#     → build_layout_from_tree_node
#     → restore_flat_to_nested
# Each test locks a regression found in this suite (see per-test docstrings).


def _worker_restore(samples, trie, leaf_idx, order):
    """Build the worker-style subtrie (LOCAL leaf_to_sample) and restore."""
    subtrie = PrefixSubTrie(
        source=trie,
        leaf_node_ids=[int(leaf_idx[i]) for i in order],
        leaf_to_sample=list(range(len(order))),
        batch_size=len(order),
    )
    samples_mb = [samples[i] for i in order]
    params = build_layout_from_tree_node(samples_mb, subtrie)
    pb = make_pt_batch(params, subtrie)
    restored = restore_flat_to_nested(pb.tree_packed_input_ids, pb)
    lengths = restored.offsets().diff().tolist()
    assert lengths == [len(s) for s in samples_mb], (
        f"restored lengths {lengths} != expected {[len(s) for s in samples_mb]}"
    )
    vals = restored.values()
    pos = 0
    for i, s in enumerate(samples_mb):
        assert torch.equal(vals[pos : pos + len(s)], s), f"sample {i} token mismatch"
        pos += len(s)


@pytest.mark.parametrize(
    "order",
    [list(range(16, 32)), list(range(8, 32)) + list(range(8))],
    ids=["late-global-ids", "shuffled-ids"],
)
def test_owner_resolution_order_independent(order):
    """Owner resolution is order-independent (node-id keyed + descendant
    propagation): an mb whose global sample ids are all >= mb size must restore
    exactly instead of raising (previously: raise), and an mb whose order
    differs from global id order must restore the right content (previously:
    wrong content) — ``owner_of`` cross-matched GLOBAL sequence_ids against
    LOCAL leaf_to_sample keys."""
    samples = make_grpo_samples(4, 8, prefix_len=300, resp_len=200, seed=42)
    trie, leaf_idx, _ = _build_global_trie(samples)
    _worker_restore(samples, trie, leaf_idx, order)


def test_duplicate_leaves_restore_exactly():
    """Duplicate leaves (identical sequences sharing one leaf) restore exactly
    even when non-adjacent in the micro-batch (leaf_ranges is emitted per
    leaf_node_ids position, aligned with subtrie.leaf_to_sample)."""
    # adjacent duplicates (trivial ordering)
    samples = make_grpo_samples(3, 4, prefix_len=100, resp_len=50, seed=42, duplicate_pair=(1, 2, 3))
    trie, leaf_idx, _ = _build_global_trie(samples)
    assert int(leaf_idx[6]) == int(leaf_idx[7])
    _worker_restore(samples, trie, leaf_idx, list(range(len(samples))))
    # non-adjacent duplicates (was: content scramble)
    g = torch.Generator().manual_seed(11)
    prefix = torch.randint(0, 100000, (200,), generator=g)
    resp = [torch.randint(0, 100000, (50,), generator=g) for _ in range(4)]
    samples = [
        torch.cat([prefix, resp[0]]),
        torch.cat([prefix, resp[1]]),
        torch.cat([prefix, resp[2]]),
        torch.cat([prefix, resp[3]]),
        torch.cat([prefix, resp[1].clone()]),
    ]
    trie, leaf_idx, _ = _build_global_trie(samples)
    assert int(leaf_idx[1]) == int(leaf_idx[4])
    _worker_restore(samples, trie, leaf_idx, list(range(5)))


def test_strict_prefix_sample_restores_exactly():
    """one response is an exact prefix of another (was: non-leaf ValueError)."""
    g = torch.Generator().manual_seed(3)
    prefix = torch.randint(0, 100000, (100,), generator=g)
    short = torch.randint(0, 100000, (20,), generator=g)
    long_resp = torch.cat([short, torch.randint(0, 100000, (30,), generator=g)])
    samples = [
        torch.cat([prefix, short]),
        torch.cat([prefix, long_resp]),
        torch.cat([prefix, torch.randint(0, 100000, (40,), generator=g)]),
    ]
    trie, leaf_idx, _ = _build_global_trie(samples)
    assert trie is not None
    groups = _mbs_groups(leaf_idx, trie, max_token_len=10**6)
    assert sorted(i for mb in groups for i in mb) == list(range(len(samples)))
    for idx in groups:
        _worker_restore(samples, trie, leaf_idx, idx)


def test_worker_restore_round_trip_full_batch():
    """full worker flow: group by budget then restore every micro-batch exactly."""
    samples = make_grpo_samples(6, 8, prefix_len=300, resp_len=200, seed=42, duplicate_pair=(2, 1, 5))
    trie, leaf_idx, _ = _build_global_trie(samples)
    for budget in (10**9, 5000):
        groups = _mbs_groups(leaf_idx, trie, max_token_len=budget)
        assert sorted(i for mb in groups for i in mb) == list(range(len(samples)))
        for idx in groups:
            _worker_restore(samples, trie, leaf_idx, idx)


def test_unfused_expand_first_no_boundary_patch(monkeypatch):
    """Unfused post-processing expands per-sample BEFORE the logits processor:
    every sample's boundary position gets ITS OWN label's log-prob (no flat
    boundary patch)."""
    import verl.utils.prefix_tree.forward as pt_forward

    # model=None makes the real clear_rope_pids crash via unwrap_model's
    # isinstance against stub classnames; the rope context is unused here.
    monkeypatch.setattr(pt_forward, "clear_rope_pids", lambda model: None)

    tensors = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 3, 6])]
    subtrie = build_tree_dynamic(tensors)
    assert subtrie is not None
    params = build_layout_from_tree_node(tensors, subtrie)
    pb = make_pt_batch(params, subtrie)
    pb.per_sample_labels = [torch.cat([s[1:], torch.zeros(1, dtype=torch.long)]) for s in tensors]

    flat_len = pb.tree_packed_input_ids.shape[0]
    logits = torch.zeros(flat_len, 8)
    for i, lbl in enumerate(pb.tree_packed_labels.tolist()):
        logits[i, lbl] = 100.0
    # Boundary (flat pos 2, shared token 3): give every sample's OWN next token a
    # high, distinct score so per-sample values are distinguishable.
    for k, nxt in enumerate((4, 5, 6)):
        logits[2, nxt] = 100.0 + 10.0 * k

    def processor(logits_, label, temperature=1.0, **kw):
        lp = torch.log_softmax(logits_.squeeze(1), dim=-1)
        log_probs = lp.gather(1, label.long())
        probs = torch.softmax(logits_.squeeze(1), dim=-1)
        entropy = -(probs * lp).sum(-1)
        return {"log_probs": log_probs.squeeze(-1), "entropy": entropy}

    ctx = pt_forward.TreeForwardCtx(pb, None, None, "flex", model=None)
    out = pt_forward.tree_post_processing(ctx, logits.unsqueeze(0), processor, {"temperature": 1.0}, post_process=True)

    lengths = out["log_probs"].offsets().diff().tolist()
    assert lengths == [4, 4, 4]
    vals = out["log_probs"].values()
    pos = 0
    for j, s in enumerate(tensors):
        rolled = torch.cat([s[1:], torch.zeros(1, dtype=torch.long)])
        p_start, p_end = pb.restoration.prefix_range
        s_start, s_end = pb.restoration.segment_ranges[j]
        rows = list(range(p_start, p_end)) + list(range(s_start, s_end))
        lp = torch.log_softmax(logits[rows], dim=-1)
        expected = lp.gather(1, rolled.unsqueeze(1)).squeeze(-1)
        assert torch.allclose(vals[pos : pos + 4], expected), f"sample {j} boundary log-prob mismatch"
        pos += 4
