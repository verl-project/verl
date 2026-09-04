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

import numpy as np
import torch

from verl.utils.prefix_tree.dynamic import (
    balance_prefix_tree_blocks,
    build_subtrie_view,
    build_tree_dynamic,
    convert_trie_to_tree_node,
    dfs_leaf_order,
    greedy_build_tries,
)
from verl.utils.prefix_tree.magi import PackRestorationParam, PrefixTreeMagiBatch, restore_flat_to_nested
from verl.utils.prefix_tree.tree import PrefixSubTrie, PrefixTrie, TrieNode
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


def _build_trie(sequences):
    trie, _ = greedy_build_tries(sequences)
    return trie


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
    assert len(sub_all.leaf_to_sample) == 3 and len(sub_all.nodes[0].input_ids) == 2

    sub_one = build_subtrie_view(trie, {0})
    assert sub_one.leaf_to_sample == [0] and len(sub_one.nodes) == 1  # [1,2,3,4] folded into one leaf

    sub_two = build_subtrie_view(trie, {0, 2})
    assert set(sub_two.leaf_to_sample) == {0, 2}
    for lid in sub_two.leaf_node_ids:
        assert 0 <= lid < len(trie.nodes)

    assert build_subtrie_view(trie, set()) is None and build_subtrie_view(trie, {99}) is None


def test_build_tree_dynamic_and_none_cases():
    s2 = [torch.tensor([10, 11, 20, 21]), torch.tensor([10, 11, 30, 31]), torch.tensor([10, 11, 40, 41])]
    r2 = build_tree_dynamic(s2)
    assert r2 is not None and len(r2.nodes[0].input_ids) == 2 and sorted(r2.leaf_to_sample) == [0, 1, 2]
    assert build_tree_dynamic([torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]) is None
    assert build_tree_dynamic([]) is None


def test_convert_trie_to_tree_node_normal_and_multi_root():
    trie = _build_trie([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]])
    r = convert_trie_to_tree_node(trie)
    assert r is not None and len(r.nodes[0].input_ids) == 2 and len(r.leaf_to_sample) == 3
    multi = _build_trie([[1, 2], [3, 4]])
    assert convert_trie_to_tree_node(multi) is None


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


def test_internal_node_owner_propagation():
    """Internal nodes shared by non-sample-0 branches must pack correct tokens."""
    samples = [
        torch.tensor([1, 2, 3, 10]),
        torch.tensor([1, 2, 3, 11]),
        torch.tensor([1, 2, 3, 20, 21]),
        torch.tensor([1, 2, 3, 20, 22]),
    ]
    restored = _build_and_restore(samples)
    for i, (orig, rest) in enumerate(zip(samples, restored, strict=False)):
        assert torch.equal(orig, rest), f"sample {i}: {orig.tolist()} != {rest.tolist()}"


def test_fuzz_random_tree_round_trip():
    """Fuzz: random tree topologies, verify full restore (includes token collisions)."""
    import random

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
    import random

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
            print(
                f"run {_}: dp={dp} trees={len(workloads)} workload={workloads} "
                f"imbalance {n_imb} -> {b_imb} ({n_imb - b_imb} better)"
            )
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


def _tree_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


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


def _build_params(subtrie, samples):
    return build_layout_from_tree_node(samples, subtrie)


def _samples(raw):
    return [torch.tensor(s, dtype=torch.long) for s in raw]


def test_basic_and_duplicates_round_trip():
    raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    st = _make_subtrie(raw, [0, 1, 2])
    samps = _samples(raw)
    p1 = _build_params(st, samps)
    st2 = pickle.loads(pickle.dumps(st))
    p2 = _build_params(st2, samps)
    assert torch.equal(p1.tree_packed_tokens, p2.tree_packed_tokens)
    assert p1.leaf_to_sample == p2.leaf_to_sample
    assert p1.q_ranges == p2.q_ranges
    assert p1.prefix_range == p2.prefix_range
    dup = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 5, 6]]
    st_dup = _make_subtrie(dup, [0, 1, 2])
    st_dup2 = pickle.loads(pickle.dumps(st_dup))
    p_d1 = _build_params(st_dup, _samples(dup))
    p_d2 = _build_params(st_dup2, _samples(dup))
    assert torch.equal(p_d1.tree_packed_tokens, p_d2.tree_packed_tokens)
    assert set(p_d1.leaf_to_sample) == set(p_d2.leaf_to_sample)


def test_children_reconstructed_after_unpickling():
    raw = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    st2 = pickle.loads(pickle.dumps(_make_subtrie(raw, [0, 1, 2])))
    valid = {n.node_idx for n in st2.nodes}
    children = [c for c in st2.nodes[0].children.values() if c.node_idx in valid]
    assert len(children) > 0


def _build_and_restore(samples: list[torch.Tensor], subtrie=None) -> list[torch.Tensor]:
    """Build flat layout from subtrie, then restore back to per-sample tokens."""
    if subtrie is None:
        subtrie = build_tree_dynamic(samples)
    assert subtrie is not None
    params = build_layout_from_tree_node(samples, subtrie)
    pids = params.tree_packed_position_ids
    assert pids.numel() == params.tree_packed_tokens.numel()
    assert int(pids[0]) == 0, f"position_ids start at {pids[0]}"
    restoration = PackRestorationParam(
        segment_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
        boundary_registry=getattr(params, "boundary_registry", None),
    )
    pt_batch = PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=None,
        flex_key=None,
        restoration=restoration,
        subtrie=subtrie,
    )
    restored = restore_flat_to_nested(params.tree_packed_tokens, pt_batch)
    offsets, vals = restored.offsets(), restored.values()
    lengths = offsets.diff().tolist()
    result = []
    pos = 0
    for length in lengths:
        result.append(vals[pos : pos + int(length)])
        pos += int(length)
    return result


def test_strict_prefix_restore():
    """Sample [1,2] is a strict prefix of [1,2,3,4], which is prefix of [1,2,3,4,5,6]."""
    samples = [torch.tensor([1, 2]), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 4, 5, 6])]
    restored = _build_and_restore(samples)
    assert len(restored) == len(samples)
    for i, (orig, rest) in enumerate(zip(samples, restored, strict=False)):
        assert torch.equal(orig, rest), f"sample {i}: {orig.tolist()} != {rest.tolist()}"


def test_nested_prefix_round_trip():
    """3 samples share [10,11] with different suffixes."""
    samples = [torch.tensor([10, 11, 20, 21]), torch.tensor([10, 11, 30, 31]), torch.tensor([10, 11, 40, 41])]
    restored = _build_and_restore(samples)
    assert len(restored) == 3
    for i, (orig, rest) in enumerate(zip(samples, restored, strict=False)):
        assert torch.equal(orig, rest), f"sample {i}"


def test_dfs_order_leaf_coverage():
    """All samples appear in leaf_to_sample exactly once."""
    seqs = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 6, 7]]
    trie, _ = greedy_build_tries(seqs)
    sub = build_subtrie_view(trie, {0, 1, 2})
    assert sorted(sub.leaf_to_sample) == [0, 1, 2]


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


def test_duplicate_samples_leaf_alignment():
    samples = [
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([1, 2, 3, 4]),  # duplicate of sample 0
        torch.tensor([1, 2, 5, 6]),
    ]
    seq_lists = [s.tolist() for s in samples]
    trie, _ = greedy_build_tries(seq_lists)
    subtrie = build_subtrie_view(trie, set(range(len(samples))))
    params = build_layout_from_tree_node(samples, subtrie)
    assert len(params.leaf_ranges) == len(params.leaf_to_sample) == 3, (
        f"leaf_ranges={params.leaf_ranges} (len={len(params.leaf_ranges)}), "
        f"leaf_to_sample={params.leaf_to_sample} (len={len(params.leaf_to_sample)})"
    )
    assert set(params.leaf_to_sample) == {0, 1, 2}


def _build_global_trie_local(seqs_t):
    """Greedy global trie + leaf_idx, mirroring trainer-side build_global_trie."""
    total_raw = sum(int(s.numel()) for s in seqs_t)
    trie = PrefixTrie(root=TrieNode())
    for seq_id, seq in enumerate(seqs_t):
        trie.insert(np.array(seq if hasattr(seq, "tolist") else [int(x) for x in seq], dtype=np.int64), seq_id)
    trie.finalize()
    if total_raw <= 0:
        return None, None

    leaf_idx = np.full(len(seqs_t), -1, dtype=np.int64)
    for node_idx, node in enumerate(trie.nodes):
        if not node.children:
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = node_idx
    return trie, torch.from_numpy(leaf_idx)


def _make_seqs(n_prompts=2, rollout_n=2, prompt_len=5, resp_len=3):
    seqs = []
    for p in range(n_prompts):
        prompt = list(range(100 + p * 10, 100 + p * 10 + prompt_len))
        for r in range(rollout_n):
            resp = list(range(200 + (p * rollout_n + r) * 10, 200 + (p * rollout_n + r) * 10 + resp_len))
            seqs.append(torch.tensor(prompt + resp, dtype=torch.long))
    return seqs


def test_build_global_trie_leaf_idx_valid():
    seqs = _make_seqs()
    trie, leaf_idx = _build_global_trie_local(seqs)
    assert trie is not None and len(trie.nodes) > 0
    assert isinstance(leaf_idx, torch.Tensor) and leaf_idx.dtype == torch.long
    assert leaf_idx.shape == (4,) and (leaf_idx >= 0).all()
    for i, node_idx in enumerate(leaf_idx.tolist()):
        node = trie.nodes[node_idx]
        assert not node.children and i in node.sequence_ids
