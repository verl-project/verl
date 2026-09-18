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

"""Unit tests for the reorder-safe micro-batch grouping API (grouping, DP
equalization of the micro-batch count, and the fused MAGI+CP boundary grad
flow)."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _helpers import build_layout, free_port
from _helpers import build_trie as _build_trie
from _helpers import make_grpo_samples as _make_samples

from verl.utils import tensordict_utils as tu
from verl.utils.prefix_tree import dynamic as pt_dynamic
from verl.utils.prefix_tree.dynamic import (
    _mbs_groups_dfs,
    mbs_groups_from_leaf_idx,
    prepare_prefix_tree_micro_batches,
    trie_group_flat_tokens,
)
from verl.utils.seqlen_balancing import roundup_divisible


def _leaf_idx_from_trie(trie, n_samples):
    """Build canonical leaf_idx: sample i -> its leaf's node_idx."""
    leaf_idx = torch.full((n_samples,), -1, dtype=torch.long)
    for node in trie.nodes:
        if not node.children:  # leaf
            for seq_id in node.sequence_ids:
                leaf_idx[seq_id] = node.node_idx
    assert int(leaf_idx.min().item()) >= 0, "trie has samples with no leaf"
    return leaf_idx


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


def test_mbs_groups_from_leaf_idx_leaf_ref_validation():
    """An orphan sample (leaf_idx=-1) raises; a ref to an internal node
    (strict-prefix sample) is accepted and groups without error."""
    samples = _make_samples(2, 2, prefix_len=10, resp_len=5, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    leaf_idx[1] = -1  # orphan: sample 1 has no leaf
    with pytest.raises(ValueError, match="no leaf assigned"):
        mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len=500)
    internal_node = next(node for node in trie.nodes if node.children)
    leaf_idx[1] = internal_node.node_idx
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


def _make_td(samples, trie, leaf_idx, budget=None, mbs=None):
    """Tensordict for prepare_prefix_tree_micro_batches (prefix tree): dynamic-bsz
    (token budget) when budget is given, fixed mbs otherwise."""
    n = len(samples)
    seq_len = max(len(s) for s in samples)
    input_ids = torch.zeros((n, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((n, seq_len), dtype=torch.long)
    for i, s in enumerate(samples):
        input_ids[i, : len(s)] = s
        attention_mask[i, : len(s)] = 1
    non_tensor = {"prefix_tree": trie, "use_prefix_tree": True, "force_group_size": 1}
    if budget is not None:
        non_tensor.update(use_dynamic_bsz=True, sp_size=1, max_token_len_per_gpu=budget)
    else:
        non_tensor.update(use_dynamic_bsz=False, micro_batch_size_per_gpu=mbs)
    return tu.get_tensordict(
        tensor_dict={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "leaf_idx": leaf_idx,
        },
        non_tensor_dict=non_tensor,
    )


def test_prepare_prefix_tree_micro_batches_attaches_subtrie():
    samples = _make_samples(4, 4, prefix_len=100, resp_len=20, seed=42)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    td = _make_td(samples, trie, leaf_idx, budget=500)
    micro_batches, batch_idx_list = prepare_prefix_tree_micro_batches(td, sp_size=1)
    assert len(micro_batches) == len(batch_idx_list)
    for mb, mb_idx in zip(micro_batches, batch_idx_list, strict=False):
        subtree = tu.get_non_tensor_data(mb, "prefix_tree_subtree", default=None)
        assert subtree is not None, "prefix_tree_subtree not attached"
        mb_leaves = sorted(int(x) for x in mb["leaf_idx"].tolist())
        sub_leaves = sorted(subtree.leaf_node_ids)
        assert sub_leaves == mb_leaves, f"{sub_leaves} != {mb_leaves}"


def test_refill_lands_exact_count_for_every_target():
    """_refill_to_count must land exactly on target for every target in
    [natural, n_samples]: the binary search finds a budget whose fill count
    equals target when one exists, and the peel fallback tops up any plateau
    gap (while count < target <= n_samples some bucket still has >= 2 samples)."""
    from verl.utils.prefix_tree.dynamic import _leaf_entries_from_leaf_idx, _refill_to_count

    samples = _make_samples(8, 4, prefix_len=64, resp_len=24, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    leaf_entries = _leaf_entries_from_leaf_idx(leaf_idx, trie)
    budget = 1200
    natural = len(_mbs_groups_dfs(leaf_entries, budget))
    assert natural < len(samples), "budget too large to exercise refill"
    for target in range(natural, len(samples) + 1):
        groups = _refill_to_count(leaf_entries, budget, target)
        assert len(groups) == target, f"target {target}: got {len(groups)}"
        assert sorted(p for g in groups for p in g) == list(range(len(samples))), f"target {target}: coverage broken"


def test_refill_exact_landing_has_no_peeled_singletons():
    """When the binary search lands exactly (no plateau gap), no bucket is a
    1-sample peel: singletons only appear via the peel fallback. Guaranteed for
    this data: every leaf holds rollout_n=4 positions, so a non-peel bucket is a
    whole number of leaves (multiples of 4) while a peel bucket has <4 samples
    left over from the largest bucket being split."""
    from verl.utils.prefix_tree.dynamic import _leaf_entries_from_leaf_idx, _refill_to_count

    samples = _make_samples(8, 4, prefix_len=64, resp_len=24, seed=1)
    trie = _build_trie(samples)
    leaf_idx = _leaf_idx_from_trie(trie, len(samples))
    leaf_entries = _leaf_entries_from_leaf_idx(leaf_idx, trie)
    budget = 1200
    natural = len(_mbs_groups_dfs(leaf_entries, budget))
    # one step up from natural: an exact budget landing exists (a budget whose
    # fill splits exactly one more leaf off)
    groups = _refill_to_count(leaf_entries, budget, natural + 1)
    assert len(groups) == natural + 1
    assert all(len(g) > 1 for g in groups), f"peel singletons on exact landing: {[len(g) for g in groups]}"


def test_pp2_divisibility_one_pass():
    """The VPP roundup is baked into the single comm pass: the count is exactly
    roundup_divisible(natural, 2), coverage holds, and buckets stay within the
    hard max_token_len budget."""
    for seed, budget in [(1, 1200), (2, 900), (3, 700), (6, 1200), (7, 2000), (9, 450)]:
        samples = _make_samples(8, 4, prefix_len=64, resp_len=24, seed=seed)
        trie = _build_trie(samples)
        leaf_idx = _leaf_idx_from_trie(trie, len(samples))
        td = _make_td(samples, trie, leaf_idx, budget=budget)
        _, mbs = prepare_prefix_tree_micro_batches(td, sp_size=1, num_batches_divided_by=2)
        natural = len(mbs_groups_from_leaf_idx(leaf_idx, trie, budget))
        expected = roundup_divisible(natural, 2)
        assert len(mbs) == expected, f"seed {seed}: got {len(mbs)} mbs, expected {expected} (natural {natural})"
        assert len(mbs) % 2 == 0, f"seed {seed}: mb count not PP2-divisible"
        assert sorted(p for g in mbs for p in g) == list(range(len(samples))), f"seed {seed}: coverage broken"
        for gi, g in enumerate(mbs):
            cost = trie_group_flat_tokens(g, trie)
            assert cost <= budget, f"seed {seed}: bucket {gi} cost {cost} > budget {budget}"


# ---- 2-rank DP micro-batch count equalization ----
# The tree path must produce the SAME number of micro-batches on every DP rank
# (same_micro_num_in_dp), otherwise the DP gradient collective desyncs. The
# dynbsz path refills at a smaller budget to land on the DP-max count; the
# fixed-mbs path peels single samples (mbs=1 buckets). A rank with fewer
# samples than the target count cannot split that far: it warns and continues
# with unequal counts.


class _CPUDevice:
    def current_device(self):
        return "cpu"


def _equalize_worker(rank, world_size, port, n_local_per_rank, result_queue):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    pt_dynamic.get_torch_device = _CPUDevice
    try:
        n_total = sum(n_local_per_rank)
        all_samples = _make_samples((n_total + 3) // 4, 4, prefix_len=20, resp_len=10, seed=42)
        trie = _build_trie(all_samples)
        # deterministic per-rank slice: rank r takes a contiguous block of samples
        start = sum(n_local_per_rank[:rank])
        local_ids = list(range(start, start + n_local_per_rank[rank]))
        leaf_idx = _leaf_idx_from_trie(trie, len(all_samples))[local_ids]
        data = _make_td([all_samples[i] for i in local_ids], trie, leaf_idx, mbs=4)
        _, groups = prepare_prefix_tree_micro_batches(
            data, sp_size=1, dp_group=dist.group.WORLD, same_micro_num_in_dp=True
        )
        result_queue.put(("ok", rank, len(groups)))
    except ValueError as e:
        result_queue.put(("raise", rank, str(e)))
    except Exception as e:  # pragma: no cover - surfaces unexpected failure
        result_queue.put(("error", rank, repr(e)))
    finally:
        dist.destroy_process_group()


def _run_two_rank(n_local_per_rank):
    port = free_port()
    # spawn-context queue: mp.Queue() defaults to the ambient context, and a
    # fork-context SemLock cannot be shared with spawn workers (Ray sets fork).
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(_equalize_worker, args=(2, port, n_local_per_rank, q), nprocs=2, join=True)
    return [q.get(timeout=30) for _ in range(2)]


def test_equalize_reaches_same_count_when_feasible():
    """Both ranks have 4 samples, mbs=4 -> each makes 1 group. n_mb=1. No padding, equal counts."""
    results = _run_two_rank([4, 4])
    assert sorted(r[0] for r in results) == ["ok", "ok"], f"unexpected results: {results}"
    counts = [r[2] for r in results]
    assert counts[0] == counts[1], f"unequal counts across ranks: {counts}"


def test_fixed_mbs_starved_rank_warns_and_continues():
    """Fixed-mbs branch: rank1 has 1 sample, mbs=4 -> 1 group; rank0 has 8
    samples -> 2 groups. n_mb=2 > rank1's 1 sample: peel padding cannot split a
    singleton, so rank1 warns and keeps 1 group (unequal counts, no raise)."""
    results = _run_two_rank([8, 1])
    assert sorted(r[0] for r in results) == ["ok", "ok"], f"unexpected results: {results}"
    by_rank = {r[1]: r[2] for r in results}  # queue order is nondeterministic
    assert by_rank[0] == 2 and by_rank[1] == 1, f"expected starved rank1 to stay at 1: {by_rank}"


def _equalize_dynbsz_worker(rank, world_size, port, n_local_per_rank, budget, result_queue):
    """Dynbsz variant: each rank fills at the same flat-token budget, natural
    counts differ (different shard sizes); the comm pass agrees on the DP max
    and each below-max rank refills at a smaller budget to land on it."""
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    pt_dynamic.get_torch_device = _CPUDevice
    try:
        n_total = sum(n_local_per_rank)
        # ceil to a whole number of 4-rollout prompts so every rank slice is in bounds
        all_samples = _make_samples((n_total + 3) // 4, 4, prefix_len=64, resp_len=24, seed=42)
        trie = _build_trie(all_samples)
        start = sum(n_local_per_rank[:rank])
        local_ids = list(range(start, start + n_local_per_rank[rank]))
        leaf_idx = _leaf_idx_from_trie(trie, len(all_samples))[local_ids]
        data = _make_td([all_samples[i] for i in local_ids], trie, leaf_idx, budget=budget)
        _, groups = prepare_prefix_tree_micro_batches(
            data, sp_size=1, dp_group=dist.group.WORLD, same_micro_num_in_dp=True
        )
        result_queue.put(("ok", rank, len(groups)))
    except ValueError as e:
        result_queue.put(("raise", rank, str(e)))
    except Exception as e:  # pragma: no cover - surfaces unexpected failure
        result_queue.put(("error", rank, repr(e)))
    finally:
        dist.destroy_process_group()


def _run_two_rank_dynbsz(n_local_per_rank, budget):
    port = free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    mp.spawn(_equalize_dynbsz_worker, args=(2, port, n_local_per_rank, budget, q), nprocs=2, join=True)
    return [q.get(timeout=30) for _ in range(2)]


def test_dynbsz_equalize_refills_to_dp_max():
    """rank0: 32 samples, rank1: 16 samples, same budget -> different natural
    counts; the comm pass agrees on the DP max and rank1 refills to it (no
    peel singletons in the common case: all buckets > 1 sample)."""
    budget = 700  # 16-sample shard fills to fewer buckets than 32-sample shard
    results = _run_two_rank_dynbsz([32, 16], budget)
    assert sorted(r[0] for r in results) == ["ok", "ok"], f"unexpected results: {results}"
    counts = [r[2] for r in results]
    assert counts[0] == counts[1], f"unequal counts across ranks: {counts}"


def test_dynbsz_starved_rank_warns_and_continues():
    """rank1 has 1 sample; DP max > 1 -> rank1 cannot split that far: warns and
    keeps its single group (unequal counts, no raise)."""
    results = _run_two_rank_dynbsz([32, 1], 700)
    assert sorted(r[0] for r in results) == ["ok", "ok"], f"unexpected results: {results}"
    by_rank = {r[1]: r[2] for r in results}  # queue order is nondeterministic
    assert by_rank[0] > by_rank[1], f"expected unequal counts (starved rank): {by_rank}"


# ---- grad flow of the fused MAGI+CP boundary gather (2-proc gloo) ----
# post_processing_packed_lce reassembles boundary log-probs across CP ranks with
# a collective all_gather. Two failure modes:
#   (a) plain all_gather detaches: forward VALUES are right but the actor update
#       backprops NOTHING through boundary tokens;
#   (b) an autograd-aware all_gather whose backward all_reduces: under static CP
#       the restore+loss is REPLICATED on every CP rank, so every rank's
#       consumption reaches the owner and the boundary gradient is counted
#       CP-world-fold.
# Correct semantics: the owner rank patches with its own locally-produced tail
# log-prob (already in its autograd graph — flows exactly once); other ranks
# patch with detached gathered VALUES.

_BOUNDARY_SAMPLES = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3, 5]), torch.tensor([1, 2, 3, 6])]


def _boundary_grad_worker(rank, world_size, port, mode):
    from verl.utils.prefix_tree import forward as fwd  # deferred: pulls codetiming via megatron_utils

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)

    # Point forward's mpu at our real gloo group as the CP group.
    orig = (fwd.mpu.get_context_parallel_world_size, fwd.mpu.get_context_parallel_group)
    fwd.mpu.get_context_parallel_world_size = lambda: world_size
    fwd.mpu.get_context_parallel_group = lambda: dist.group.WORLD
    try:
        pb, _ = build_layout(_BOUNDARY_SAMPLES)
        registry = pb.restoration.boundary_registry
        assert registry, "expected a boundary registry from the 3-way fork"
        boundary_pos, leaves = registry[0]
        assert len(leaves) == 3, f"expected 3 leaves at the fork, got {len(leaves)}"

        # Split the leaves across CP ranks; each rank produces only its own
        # boundary values (grad-carrying), as the LCE pass would.
        if mode == "single_rank":
            owned = list(range(3)) if rank == 0 else []
        else:
            owned = [s for s in range(3) if s % world_size == rank]
        srcs = {s: torch.tensor(100.0 * rank + s, requires_grad=True) for s in owned}
        pb._boundary_local_vals = [(boundary_pos, s, srcs[s]) for s in owned]

        fwd.post_processing_packed_lce(pb, magi_key=object())

        # Coverage + values: every leaf's boundary log-prob arrived with its own value.
        assert set(pb._boundary_logps) == {0, 1, 2}, f"rank{rank}: bad coverage {sorted(pb._boundary_logps)}"
        for s, vals in pb._boundary_logps.items():
            ((pos, v),) = vals
            owner = 0 if mode == "single_rank" else s % world_size
            assert pos == boundary_pos
            assert abs(v.item() - (100.0 * owner + s)) < 1e-4, f"rank{rank} sample{s}: wrong value {v.item()}"
            if owner == rank:
                # The producer's own copy must stay graph-connected — the
                # original all_gather-detach bug cut even this. (requires_grad,
                # not grad_fn: the producer's value may be a leaf tensor.)
                assert v.requires_grad, (
                    f"rank{rank} sample{s}: OWN boundary log-prob is DETACHED — "
                    f"the cross-CP gather cut the autograd graph"
                )

        # Replicated backward: every rank consumes ALL boundary values (static-CP
        # replicas run the same restore+loss). Each producer must receive the
        # gradient EXACTLY ONCE — grad 1.0, not world_size (the 4x overcount).
        # (A rank owning no boundaries has only detached copies: skip backward.)
        all_vals = [v for vals in pb._boundary_logps.values() for _, v in vals]
        if any(v.requires_grad for v in all_vals):
            torch.stack(all_vals).sum().backward()
        for s, t in srcs.items():
            assert t.grad is not None, f"rank{rank}: sample{s} source got no grad at all"
            assert abs(t.grad.item() - 1.0) < 1e-6, (
                f"rank{rank}: sample{s} grad is {t.grad.item()} — expected exactly 1.0; "
                f"a world_size-fold value means every CP replica's consumption was counted"
            )
    finally:
        fwd.mpu.get_context_parallel_world_size, fwd.mpu.get_context_parallel_group = orig
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["interleaved", "single_rank"])
def test_boundary_grad_flow(mode):
    """Both ownership layouts of the 2-proc gloo boundary gather:
    interleaved — rank0 owns leaves {0,2}, rank1 owns {1}: every gathered value
    must keep its graph; single_rank — all boundaries on rank0, rank1 contributes
    none (zero-entry padding path): rank1's copies must still be grad-connected
    so its downstream usage backprops."""
    port = free_port()
    mp.spawn(_boundary_grad_worker, args=(2, port, mode), nprocs=2, join=True)
