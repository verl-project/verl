# Copyright 2025 Meituan Ltd. and/or its affiliates
# Copyright 2025-2026 The AReaL Authors (Ant Group, Tsinghua University, HKUST)
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

"""Dynamic-trie prefix-tree builder: token-by-token trie insertion, micro-batch
    grouping, leaf_idx reorder-safety, DFS balancing.

Algorithm originally derived from AReaL (https://github.com/inclusionAI/AReaL)."""

from __future__ import annotations

import logging as _logging
from typing import Optional

import torch
from torch import Tensor

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_torch_device
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
    roundup_divisible,
)

_log = _logging.getLogger(__name__)

__all__ = [
    "build_tree_dynamic",
    # Lower-level helpers exposed for testing / benchmarking
    "TrieNode",
    "greedy_build_tries",
    "convert_trie_to_tree_node",
    "build_subtrie_view",
    # Load balancing
    "trie_group_flat_tokens",
    "balance_prefix_tree_v0",
]


# TrieNode is canonical in tree.py; import from there (single definition).
# Old code using .ancestors (list) will raise AttributeError immediately.
from verl.utils.prefix_tree.tree import (  # noqa: E402
    PrefixSubTrie,
    PrefixTrie,
    TrieNode,
    _is_prefix_tree_enabled,
    trie_ancestors,
)

# ---------------------------------------------------------------------------
# Module-level collector for post-micro-batch-build micro_batch_shared_ratio.
# Populated by prepare_prefix_tree_micro_batches, consumed by maybe_collect_mbs_metric.
# ---------------------------------------------------------------------------
_mbs_metric_state = {"shared_ratio_sum": 0.0, "count": 0}


def _reset_mbs_metric():
    _mbs_metric_state["shared_ratio_sum"] = 0.0
    _mbs_metric_state["count"] = 0


def _push_mbs_shared_ratio(ratio: Optional[float]) -> None:
    if ratio is None:
        return
    _mbs_metric_state["shared_ratio_sum"] += ratio
    _mbs_metric_state["count"] += 1


def _get_mbs_metric() -> dict:
    s, c = _mbs_metric_state["shared_ratio_sum"], _mbs_metric_state["count"]
    if c == 0:
        return {}
    from verl.utils.metric import AggregationType, Metric

    return {
        "prefix_tree/micro_batch_shared_ratio": Metric(value=s / c, aggregation=AggregationType.MEAN),
    }


def greedy_build_tries(
    sequences: list[list[int]],
) -> tuple[PrefixTrie, int]:
    """Build compressed trie via PrefixTrie.insert (single-pass, no separate compress)."""
    import numpy as np

    trie = PrefixTrie()
    for seq_id, seq in enumerate(sequences):
        trie.insert(np.array(seq, dtype=np.int64), seq_id)
    trie.finalize()
    return trie, len(trie.nodes)


def convert_trie_to_tree_node(
    trie: PrefixTrie,
) -> Optional[PrefixSubTrie]:
    """Convert a compressed trie to a :class:`PrefixSubTrie`.

    Returns ``None`` when there's no real sharing (no children or multi-root).
    Delegates to :func:`build_subtrie_view` with all sequence IDs.
    """
    if not trie.children:
        _log.warning("prefix_tree: convert_trie_to_tree_node: trie has no children; no sharing, returning None")
        return None
    if len(trie.children) > 1:
        _log.warning(
            "prefix_tree: convert_trie_to_tree_node: multiple roots (%d), returning None",
            len(trie.children),
        )
        return None
    all_seq_ids: set[int] = set()
    for child in trie.children.values():
        all_seq_ids.update(_trie_seq_ids(child))
        # Also collect from internal nodes (strict-prefix samples).
        all_seq_ids.update(child.sequence_ids)
    return build_subtrie_view(trie, all_seq_ids)


def build_tree_dynamic(samples: list[Tensor]) -> Optional[PrefixSubTrie]:
    """Token-by-token trie detection. Returns a :class:`PrefixSubTrie` or None.

    Returns ``None`` when there's no shared prefix (empty input, single sample,
    or multi-forest case).
    """
    if not samples:
        return None
    sequences = [t.tolist() for t in samples]
    trie, _ = greedy_build_tries(sequences)
    if not trie.nodes:
        _log.warning(
            "prefix_tree: build_tree_dynamic: empty trie, returning None",
        )
        return None
    return convert_trie_to_tree_node(trie)


def _trie_seq_ids(node: TrieNode) -> list[int]:
    """Collect all sequence IDs from leaf nodes of a compressed-trie subtree."""
    if not node.children:
        return list(node.sequence_ids)
    ids: list[int] = []
    for child in node.children.values():
        ids.extend(_trie_seq_ids(child))
    return ids


def trie_group_flat_tokens(group: list[int], trie: PrefixTrie) -> int:
    """Flat (deduplicated) token count for a subset of sequences within a trie."""
    sub = build_subtrie_view(trie, frozenset(group))
    return sub.flat_tokens if sub else 0


def dfs_leaf_order(
    sequences: list[list[int]],
    trie: PrefixTrie,
) -> list[int]:
    """Return sample indices in DFS pre-order from a pre-built trie.

    Args:
        sequences: per-sample token lists (used only for length check).
        trie: pre-built TrieNode root (from ``build_global_trie``).

    Returns:
        List of sample indices in DFS pre-order (length == len(sequences)).
    """
    if not sequences:
        return []
    if trie is None:
        raise RuntimeError(
            "dfs_leaf_order: trie is None. The driver must call build_global_trie "
            "and pass trie=... Per-call rebuild is disabled."
        )
    return [sid for node in trie.dfs(leaf_only=True) for sid in node.sequence_ids]


def trie_dfs_leaf_order_from_leaf_idx(leaf_idx, trie: PrefixTrie) -> list[int]:
    """Return batch positions in DFS leaf order from leaf_idx (reorder-safe)."""
    leaf_to_positions: dict[int, list[int]] = {}
    for new_pos, leaf_fid in enumerate(leaf_idx.tolist()):
        if leaf_fid < 0:
            raise ValueError(f"leaf_idx[{new_pos}]={leaf_fid}; sample has no leaf assigned.")
        leaf_to_positions.setdefault(int(leaf_fid), []).append(new_pos)
    return [p for node in trie.dfs(leaf_only=True) for p in leaf_to_positions.get(node.node_idx, [])]


def _mbs_groups_dfs(
    leaf_entries: list[tuple[TrieNode, list[int]]],
    max_token_len: int,
) -> list[list[int]]:
    """DFS-budget walk: groups leaf samples into micro-batches by flat (deduplicated) token budget.

    When a leaf holds multiple positions (identical sequences), all stay in the same
    DFS group to avoid singleton groups. Uid atomicity is NOT enforced here - it is
    owned by DP balancing (balance_prefix_tree_blocks); mbs grouping may freely
    split a uid's rollouts across micro-batches to respect the budget."""
    all_groups: list[list[int]] = []
    current_group: list[int] = []
    covered: set[int] = set()
    current_eff = 0  # flat tokens accumulated in current group

    for node, positions in leaf_entries:
        path = trie_ancestors(node) + [node]
        new_nodes = [n for n in path if n.node_idx not in covered]
        inc = sum(len(n.input_ids) for n in new_nodes)
        if current_group and current_eff + inc > max_token_len:
            all_groups.append(current_group[:])
            current_group = []
            covered = set()
            current_eff = 0
            new_nodes = path
            inc = sum(len(n.input_ids) for n in new_nodes)
        current_group.extend(positions)
        covered.update(n.node_idx for n in new_nodes)
        current_eff += inc

    if current_group:
        all_groups.append(current_group[:])
    return all_groups


def mbs_groups_from_leaf_idx(
    leaf_idx,
    trie: PrefixTrie,
    max_token_len: int,
) -> list[list[int]]:
    """Group reordered batch positions into micro-batches from leaf_idx
    (reorder-safe: groups by leaf_idx rather than trie DFS order)."""
    leaf_to_positions: dict[int, list[int]] = {}
    for new_pos, leaf_fid in enumerate(leaf_idx.tolist()):
        if leaf_fid < 0:
            raise ValueError(
                f"leaf_idx[{new_pos}]={leaf_fid}; sample has no leaf assigned. "
                f"_build_global_trie must populate leaf_idx for every sample."
            )
        leaf_to_positions.setdefault(int(leaf_fid), []).append(new_pos)

    leaf_entries: list[tuple[TrieNode, list[int]]] = []
    for node in trie.nodes:
        positions = leaf_to_positions.get(node.node_idx)
        if positions is None:
            continue  # node belongs to a different DP rank - skip
        # True leaves AND internal nodes (strict-prefix samples end at an
        # internal node; their leaf_idx points there).
        leaf_entries.append((node, positions))

    if len(leaf_entries) != len(leaf_to_positions):
        uncovered = set(leaf_to_positions) - {node.node_idx for node in trie.nodes}
        raise ValueError(f"leaf_idx references {len(uncovered)} non-existent node(s): {sorted(uncovered)}")

    return _mbs_groups_dfs(leaf_entries, max_token_len)


def build_subtrie_view(
    trie: PrefixTrie,
    keep_leaf_ids: set[int],
    source: Optional[PrefixTrie] = None,
) -> Optional[PrefixSubTrie]:
    """Extract subtree containing only given leaf sample indices. Returns PrefixSubTrie or None."""
    if not keep_leaf_ids:
        return None
    if source is None:
        source = trie

    def _collect(node: TrieNode) -> None:
        """Walk node, collecting matching leaves (true leaves + strict-prefix internal nodes)."""
        if not node.children:
            kept = [s for s in node.sequence_ids if s in keep_leaf_ids]
            if kept:
                for sid in kept:
                    leaf_to_sample.append(sid)
                    leaf_node_ids.append(node.node_idx)
            return
        # Strict-prefix: sample terminates at this internal node.
        for sid in node.sequence_ids:
            if sid in keep_leaf_ids and not any(sid in c.sequence_ids for c in node.children.values()):
                leaf_to_sample.append(sid)
                leaf_node_ids.append(node.node_idx)
        for child in node.children.values():
            if keep_leaf_ids.isdisjoint(child.sequence_ids):
                continue
            _collect(child)

    leaf_to_sample: list[int] = []
    leaf_node_ids: list[int] = []
    for child in trie.children.values():
        if keep_leaf_ids.isdisjoint(child.sequence_ids):
            continue
        _collect(child)
    if not leaf_to_sample:
        return None
    if set(leaf_to_sample) != keep_leaf_ids:
        _log.warning("prefix_tree: build_subtrie_view: unmatched sequences: FA3 fallback")
        return None
    batch_size = max(leaf_to_sample) + 1 if leaf_to_sample else 0
    subtrie = PrefixSubTrie(
        source=source,
        leaf_node_ids=leaf_node_ids,
        leaf_to_sample=leaf_to_sample,
        batch_size=batch_size,
    )
    return subtrie


_ZERO_PT_METRICS = {
    "prefix_tree/global_shared_ratio": 0.0,
    "prefix_tree/packed_tokens": 0,
    "prefix_tree/raw_tokens": 0,
}


def compute_prefix_tree_metrics(
    input_ids,
    attention_mask=None,
    max_token_len_per_gpu: int | None = None,
    micro_batch_size: int = 0,
    trie: Optional[PrefixTrie] = None,
    leaf_idx=None,
) -> dict:
    """Compute prefix_tree/global_shared_ratio, prefix_tree/packed_tokens, prefix_tree/raw_tokens.

    Uses caller-provided global trie (built once on driver). Per-call rebuild is disabled."""
    if isinstance(input_ids, Tensor) and input_ids.is_nested:
        sequences = [t.tolist() for t in input_ids.unbind()]
    elif isinstance(input_ids, Tensor) and input_ids.dim() == 2:
        seqlens = (
            attention_mask.sum(dim=-1).tolist()
            if attention_mask is not None
            else [input_ids.shape[1]] * input_ids.shape[0]
        )
        sequences = [input_ids[i, : int(seqlens[i])].tolist() for i in range(input_ids.shape[0])]
    elif isinstance(input_ids, list):
        sequences = input_ids
    else:
        return _ZERO_PT_METRICS

    total_raw = sum(len(s) for s in sequences)
    if total_raw == 0:
        return _ZERO_PT_METRICS

    # Reuse the caller-provided global trie (built once on the driver).
    if trie is None:
        raise RuntimeError(
            "compute_prefix_tree_metrics: global trie is None. The driver must call "
            "build_global_trie and pass trie=... Per-call rebuild is disabled."
        )
    flat = sum(len(n.input_ids) for n in trie.nodes)

    return {
        "prefix_tree/global_shared_ratio": 1.0 - flat / total_raw,
        "prefix_tree/packed_tokens": flat,
        "prefix_tree/raw_tokens": total_raw,
    }


def prepare_prefix_tree_micro_batches(
    data,
    sp_size: int,
    dp_group=None,
    same_micro_num_in_dp: bool = True,
    num_batches_divided_by: int | None = None,
    force_group_size: int = 1,
):
    """Prepare micro-batches using prefix-tree grouping (dynamic bsz: flat-token budget; fixed: DFS chunk by seq count).

    Uses driver-built global trie. Per-worker rebuild is disabled."""
    trie = tu.get_non_tensor_data(data, "prefix_tree", default=None)
    leaf_idx = data.get("leaf_idx", None) if hasattr(data, "get") else data["leaf_idx"]
    if trie is None or leaf_idx is None:
        raise RuntimeError(
            "prepare_prefix_tree_micro_batches: global trie (prefix_tree) or leaf_idx is "
            "None. The driver must call build_global_trie to build+attach the global trie "
            "before dispatching to workers. Per-worker rebuild is disabled."
        )

    use_dynamic_bsz_local = tu.get_non_tensor_data(data, "use_dynamic_bsz", default=True)
    if use_dynamic_bsz_local and "max_token_len_per_gpu" in data.keys():
        # Dynamic bsz: group by flat-token budget.
        _log.warning_once(
            "prefix_tree is on: max_token_len_per_gpu is interpreted as "
            "deduplicated (flat trie) token count, not raw sequence length."
        )
        max_token_len = data["max_token_len_per_gpu"] * sp_size
        batch_idx_list = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len)
    else:
        # Fixed mbs: contiguous chunks in batch order (no DFS reorder).
        mbs = data["micro_batch_size_per_gpu"] * force_group_size
        n = len(leaf_idx)
        batch_idx_list = [list(range(i, min(i + mbs, n))) for i in range(0, n, mbs)]

    # Pad to the max micro-batch count across the DP group, then to divisibility.
    target = len(batch_idx_list)
    if torch.distributed.is_initialized() and same_micro_num_in_dp and dp_group is not None:
        n_mb = torch.tensor([len(batch_idx_list)], device=get_torch_device().current_device())
        torch.distributed.all_reduce(n_mb, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        while len(batch_idx_list) < n_mb.item():
            idx = max(range(len(batch_idx_list)), key=lambda i: len(batch_idx_list[i]))
            if len(batch_idx_list[idx]) <= 1:
                break
            g = batch_idx_list[idx]
            batch_idx_list[idx] = g[:-1]
            batch_idx_list.append([g[-1]])

    if num_batches_divided_by is not None:
        target = roundup_divisible(len(batch_idx_list), num_batches_divided_by)
        while len(batch_idx_list) < target:
            idx = max(range(len(batch_idx_list)), key=lambda i: len(batch_idx_list[i]))
            if len(batch_idx_list[idx]) <= 1:
                break
            g = batch_idx_list[idx]
            batch_idx_list[idx] = g[:-1]
            batch_idx_list.append([g[-1]])

    # Build subtries ONCE with LOCAL leaf_to_sample (required by downstream
    # restore in build_layout_from_tree_node). The cached flat_tokens
    # property provides the flat token count without a separate rebuild.
    _leaf_idx_list = leaf_idx.tolist()
    subtries = []
    for g in batch_idx_list:
        leaf_node_ids = [_leaf_idx_list[j] for j in g]
        subtries.append(
            PrefixSubTrie(
                source=trie,
                leaf_node_ids=leaf_node_ids,
                leaf_to_sample=list(range(len(g))),
                batch_size=len(g),
            )
        )
    tokens_per_group = [s.flat_tokens for s in subtries]

    # Reorder micro-batches inc-then-dec by flat-token count to reduce PP bubble.
    if use_dynamic_bsz_local and len(batch_idx_list) > 1:
        indices = list(range(len(batch_idx_list)))
        sorted_groups = sorted(zip(tokens_per_group, indices, batch_idx_list, subtries, strict=False))
        ordered_tokens = [t for t, _, _, _ in sorted_groups]
        ordered_groups = [g for _, _, g, _ in sorted_groups]
        ordered_subtries = [s for _, _, _, s in sorted_groups]
        batch_idx_list = ordered_groups[::2] + ordered_groups[1::2][::-1]
        tokens_per_group = ordered_tokens[::2] + ordered_tokens[1::2][::-1]
        subtries = ordered_subtries[::2] + ordered_subtries[1::2][::-1]

    micro_batches = [tu.index_select_tensor_dict(data, idx) for idx in batch_idx_list]

    # Compute accurate per-mb sharing ratio from actual engine grouping and push to module collector.
    _input_ids = data["input_ids"]
    _is_nested = isinstance(_input_ids, Tensor) and _input_ids.is_nested
    if _is_nested:
        _seq_lens = _input_ids.offsets().diff().tolist()
    else:
        _attn = data.get("attention_mask")
        if _attn is not None:
            _seq_lens = _attn.sum(dim=-1).tolist()
        else:
            _seq_lens = [_input_ids.shape[1]] * len(_input_ids)
    for group, flat in zip(batch_idx_list, tokens_per_group, strict=False):
        group_raw = sum(_seq_lens[i] for i in group)
        if group_raw == 0:
            continue
        _push_mbs_shared_ratio(1.0 - flat / group_raw)

    # Attach subtries (built once above, no rebuild).
    for mb, sub in zip(micro_batches, subtries, strict=False):
        tu.assign_non_tensor(mb, prefix_tree_subtree=sub)
    return micro_batches, batch_idx_list


def _blocks_by_ids(trie, block_ids, active_samples=None) -> tuple[list[tuple[int, list[int]]], list[int]]:
    """Group samples into blocks by an external identity (e.g. prompt session key).

    Returns (blocks, flat_list): blocks = [(flat_tokens, sorted_sample_ids)] in
    first-seen order of block id; flat_list parallel to blocks. Flat tokens count
    only trie nodes whose sequence_ids lie ENTIRELY within the block — ancestors
    shared across blocks (system prompt) are excluded, and they cancel out in the
    per-rank workload comparison anyway.

    When ``active_samples`` is given (a set of global sample indices), only those
    samples form blocks, and a trie node is counted only if ALL its sequence_ids
    are in the active set (per-minibatch balancing: the global trie is reused, but
    block formation + token counting are restricted to the active minibatch).
    """
    if active_samples is None:
        active = None
    else:
        active = set(active_samples)

    blocks_by_id: dict = {}
    for i, bid in enumerate(block_ids):
        if active is not None and i not in active:
            continue
        blocks_by_id.setdefault(bid, []).append(i)

    flat_by_id = {bid: 0 for bid in list(blocks_by_id)}
    for node in trie.nodes:
        sids = node.sequence_ids
        if not sids:
            continue
        if active is not None and not all(s in active for s in sids):
            continue
        bid = block_ids[sids[0]]
        if all(block_ids[s] == bid for s in sids):
            iids = node.input_ids
            flat_by_id[bid] += len(iids) if iids is not None else 0

    blocks = [(flat_by_id[bid], sorted(blocks_by_id[bid])) for bid in list(blocks_by_id)]
    return blocks, [b[0] for b in blocks]


def balance_prefix_tree_blocks(
    trie,
    dp_size: int,
    block_ids,
    active_samples=None,
) -> tuple[list[int], list[list[int]], list[int]]:
    """Balance whole trees across DP ranks by flat-token workload.

    Blocks are atomic units defined by ``block_ids`` (per-sample prompt/session
    identity): a block's samples are never split across ranks, so intra-rank
    prefix dedup (and same-prompt GRPO advantage grouping) is preserved.

    When ``active_samples`` is given, only those samples (global indices) are
    balanced — used for per-minibatch balancing (keep_minibatch), where the
    global trie is reused but block formation is restricted to the minibatch.

    Returns:
        permutation: new sample order, rank-major (rank 0's blocks first, ...).
            ``permutation[new_pos]`` = original sample index.
        partitions: ``partitions[r]`` = list of block indices assigned to rank r.
        workloads: workload per block (same indexing as block order).
    """
    blocks, flat_list = _blocks_by_ids(trie, block_ids, active_samples=active_samples)

    # Treat each block as one sortable unit: apply the standard transformer
    # workload formula (24576*n + n²) to its flat (deduplicated) token count.
    workloads = calculate_workload(torch.tensor(flat_list, dtype=torch.float32)).tolist()

    if dp_size <= 1:
        permutation = [s for _, samples in blocks for s in samples]
        return permutation, [list(range(len(blocks)))], workloads

    if len(blocks) < dp_size:
        partitions = [[i] for i in range(len(blocks))]
    else:
        partitions = get_seqlen_balanced_partitions(workloads, dp_size, equal_size=False)

    permutation = []
    for part in partitions:
        for block_idx in part:
            permutation.extend(blocks[block_idx][1])
    return permutation, partitions, workloads


def balance_prefix_tree_v0(
    data,
    config_or_data: dict,
    dp_size: int,
    *,
    attention_mask=None,
    metrics: dict | None = None,
    logging_prefix: str = "global_seqlen",
    keep_minibatch: bool = False,
    minibatch_size: int | None = None,
) -> bool:
    """Reorder the batch so each DP rank receives whole trees balanced by per-tree deduped workload."""
    if not _is_prefix_tree_enabled(config_or_data):
        return False

    trie = None
    if hasattr(data, "meta_info"):
        trie = data.meta_info.get("prefix_tree", None)
    else:
        trie = tu.get_non_tensor_data(data, "prefix_tree", default=None)
    if trie is None:
        return False

    if hasattr(data, "batch"):
        n_samples = data.batch["input_ids"].shape[0]
    else:
        n_samples = len(data["input_ids"])

    # Block by prompt identity: uid is the GRPO advantage group (same prompt's
    # rollouts must stay together). Required — advantage computation also
    # indexes non_tensor_batch["uid"], so it is always present in GRPO.
    non_tensor = getattr(data, "non_tensor_batch", None)
    if non_tensor is None or "uid" not in non_tensor:
        raise ValueError(
            "balance_prefix_tree_v0: non_tensor_batch['uid'] is required to block samples by prompt identity."
        )
    uids = non_tensor["uid"]
    if len(uids) != n_samples:
        raise ValueError(f"uid count {len(uids)} != n_samples {n_samples}")
    block_ids = [str(u) for u in uids]

    if keep_minibatch:
        # Per-minibatch tree-block balancing: balance each minibatch's samples
        # across DP ranks independently (sort NOT cross-minibatch), reusing the
        # global trie via the active_samples filter. Mirrors _balance_batch's
        # non-tree keep_minibatch branch (ray_trainer.py:1210).
        if minibatch_size is None:
            raise ValueError("balance_prefix_tree_v0: keep_minibatch=True requires minibatch_size.")
        minibatch_num = n_samples // minibatch_size
        permutation: list[int] = []
        all_partitions: list[list[int]] = [[] for _ in range(dp_size)]
        all_workloads: list[int] = []
        for i in range(minibatch_num):
            active = set(range(i * minibatch_size, (i + 1) * minibatch_size))
            mb_perm, mb_parts, mb_workloads = balance_prefix_tree_blocks(
                trie, dp_size, block_ids, active_samples=active
            )
            # mb_perm is already global indices (active_samples are global); concatenate.
            permutation.extend(mb_perm)
            for j, part in enumerate(mb_parts):
                all_partitions[j].extend(part)
            all_workloads.extend(mb_workloads)
        if len(permutation) != n_samples:
            raise RuntimeError(f"keep_minibatch permutation covered {len(permutation)}/{n_samples} samples.")
        if hasattr(data, "reorder"):
            data.reorder(torch.tensor(permutation))
        else:
            data = tu.index_select_tensor_dict(data, torch.tensor(permutation))
        if metrics is not None:
            stats = log_seqlen_unbalance(
                seqlen_list=all_workloads,
                partitions=all_partitions,
                prefix=logging_prefix,
            )
            metrics.update(stats)
        return True

    permutation, partitions, workloads = balance_prefix_tree_blocks(trie, dp_size, block_ids)
    if len(permutation) != n_samples:
        raise RuntimeError(
            f"balance_prefix_tree_blocks covered {len(permutation)}/{n_samples} samples: "
            "trie does not cover every sample (build_global_trie bug?)."
        )

    if hasattr(data, "reorder"):
        data.reorder(torch.tensor(permutation))
    else:
        data = tu.index_select_tensor_dict(data, torch.tensor(permutation))

    if metrics is not None:
        stats = log_seqlen_unbalance(
            seqlen_list=workloads,
            partitions=partitions,
            prefix=logging_prefix,
        )
        metrics.update(stats)
    return True


def balance_prefix_tree_v1(batch, trie, metrics: dict, dp_size: int, logging_prefix: str):
    """v1 (KVBatchMeta) adapter over :func:`balance_prefix_tree_blocks`: reorder the
    batch so each DP rank receives whole uid blocks balanced by flat-token workload.

    Synthetic padding samples (``is_padding`` tag) are excluded from the balance and
    re-appended after the permutation, mirroring v0's pure-batch balancing."""
    # Key format is {uid}_{session_id}_{index}: uid is the prompt group whose n
    # trajectories share the full prompt — that shared prompt is the main prefix-tree
    # win, so blocks are per-uid (v0 parity). rsplit("_", 2) tolerates "_" inside uid.
    block_ids = [k.rsplit("_", 2)[0] for k in batch.keys]
    real_positions = [i for i, tag in enumerate(batch.tags) if not tag.get("is_padding", False)]
    pad_positions = [i for i in range(len(batch.keys)) if batch.tags[i].get("is_padding", False)]
    permutation, partitions, workloads = balance_prefix_tree_blocks(
        trie, dp_size, block_ids, active_samples=real_positions
    )
    if len(permutation) != len(real_positions):
        raise RuntimeError(
            f"balance_prefix_tree_v1 covered {len(permutation)}/{len(real_positions)} real samples: "
            "trie does not cover every sample."
        )
    batch.reorder(permutation + pad_positions)
    stats = log_seqlen_unbalance(seqlen_list=workloads, partitions=partitions, prefix=logging_prefix)
    metrics.update(stats)
    return batch
