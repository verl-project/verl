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

"""Dynamic-trie prefix-tree builder.

Token-by-token trie insertion supporting arbitrary tree depth. Detects the
shared-prefix tree directly from input tokens; no rollout-side metadata
required. Invoked by
:func:`verl.utils.prefix_tree.magi.build_prefix_tree_micro_batch` when
when ``prefix_segments_batch`` is not provided.

Algorithm originally derived from AReaL
(https://github.com/inclusionAI/AReaL).
"""

from __future__ import annotations

import logging as _logging
from typing import Any, Optional

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

__all__ = [
    "build_tree_dynamic",
    "dfs_leaf_order",
    "dfs_micro_batch_groups",
    # Lower-level helpers exposed for testing / benchmarking
    "TrieNode",
    "greedy_build_tries",
    "mbs_groups_from_trie",
    "convert_trie_to_tree_node",
    "subtrie_view",
    # Load balancing
    "trie_group_flat_tokens",
    "get_dfs_balanced_partitions",
    "reorder_and_balance_for_prefix_tree",
    "create_and_attach_subtrie_views",
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
# Module-level collector for the post-micro-batch-build micro_batch_shared_ratio.
# Populated by ``prepare_prefix_tree_micro_batches`` (the ACTUAL grouping the
# engine dispatches) and pulled into engine output by
# ``maybe_collect_mbs_metric`` in ``prefix_tree_patch_impl``.
# ---------------------------------------------------------------------------
_mbs_metric_state = {"shared_ratio_sum": 0.0, "count": 0}


def _reset_mbs_metric():
    _mbs_metric_state["shared_ratio_sum"] = 0.0
    _mbs_metric_state["count"] = 0


def _push_mbs_shared_ratio(ratio: float) -> None:
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


class _BuildNode:
    """Internal: temporary uncompressed node used during insertion."""

    __slots__ = ("tree_id", "token_id", "node_id", "children", "is_end", "sequence_ids")

    def __init__(self, tree_id: int, token_id: int, node_id: int):
        self.tree_id = tree_id
        self.token_id = token_id
        self.node_id = node_id
        self.children: dict[int, _BuildNode] = {}
        self.is_end = False
        self.sequence_ids: list[int] = []


def _count_additional_nodes(root: _BuildNode, sequence: list[int]) -> int:
    current = root
    for idx, token in enumerate(sequence):
        child = current.children.get(token)
        if child is None:
            return len(sequence) - idx
        current = child
    return 0


def _insert_sequence(
    root: _BuildNode,
    all_nodes: list[_BuildNode],
    sequence: list[int],
    tree_id: int,
    sequence_id: int,
) -> int:
    """Insert sequence into tree. Returns number of NEW nodes created."""
    current = root
    new_nodes = 0
    for token in sequence:
        if token not in current.children:
            node_id = len(all_nodes)
            current.children[token] = _BuildNode(tree_id, token, node_id)
            all_nodes.append(current.children[token])
            new_nodes += 1
        current.children[token].sequence_ids.append(sequence_id)
        current = current.children[token]
    current.is_end = True
    return new_nodes


def _compress_trie(root: _BuildNode) -> TrieNode:
    trie_root = TrieNode(tree_id=root.tree_id)

    def _compress_chain(node: _BuildNode, parent: Optional[TrieNode]) -> TrieNode:
        tokens: list[int] = []
        current = node
        start_id = node.node_id
        while True:
            tokens.append(current.token_id)
            if len(current.children) != 1 or current.is_end:
                break
            next_child = next(iter(current.children.values()))
            if current.sequence_ids != next_child.sequence_ids:
                raise ValueError("Sequence IDs mismatch along chain")
            if next_child.node_id != current.node_id + 1:
                raise ValueError("Node IDs not consecutive along chain")
            current = next_child

        flat_idx = len(trie_root.nodes)
        trie_node = TrieNode(
            tree_id=root.tree_id,
            start_idx=start_id,
            end_idx=current.node_id,
            input_ids=tokens,
            sequence_ids=current.sequence_ids.copy(),
            ancestor=parent,
            flat_idx=flat_idx,
        )
        trie_root.nodes.append(trie_node)
        if current.children:
            for token, child in sorted(current.children.items()):
                trie_node.children[token] = _compress_chain(child, trie_node)
        return trie_node

    if root.children:
        for token, child in sorted(root.children.items()):
            trie_root.children[token] = _compress_chain(child, None)
    return trie_root


def greedy_build_tries(
    sequences: list[list[int]],
    max_tokens_per_tree: int,
) -> tuple[list[TrieNode], list[int]]:
    """Token-by-token greedy trie packing across samples.

    Args:
        sequences: per-sample token lists.
        max_tokens_per_tree: upper bound on uncompressed nodes per tree (set to
            a huge value when you want a single forest).

    Returns:
        (tries, num_tokens_list): list of compressed TrieNode roots + total
        uncompressed nodes per tree.
    """
    # Fast path: when max_tokens_per_tree is huge (e.g., sum*10), build single tree
    total_tokens = sum(len(s) for s in sequences)
    if max_tokens_per_tree >= total_tokens and sequences:
        root = _BuildNode(0, -1, -1)
        all_nodes: list[_BuildNode] = []
        for seq_id, seq in enumerate(sequences):
            _insert_sequence(root, all_nodes, seq, 0, seq_id)
        forests = [{"root": root, "all_nodes": all_nodes, "nodes": len(all_nodes)}]
    else:
        # Greedy packing: try to fit sequences into existing trees
        forests: list[dict[str, Any]] = []
        for seq_id, seq in enumerate(sequences):
            inserted = False
            for tree_id, tree in enumerate(forests):
                additional = _count_additional_nodes(tree["root"], seq)
                if tree["nodes"] + additional <= max_tokens_per_tree:
                    actual_new = _insert_sequence(tree["root"], tree["all_nodes"], seq, tree_id, seq_id)
                    tree["nodes"] += actual_new
                    inserted = True
                    break
            if inserted:
                continue
            if len(seq) > max_tokens_per_tree:
                raise ValueError(f"Sequence length {len(seq)} exceeds max_tokens_per_tree {max_tokens_per_tree}")
            new_tree_id = len(forests)
            new_root = _BuildNode(new_tree_id, -1, -1)
            all_nodes: list[_BuildNode] = []
            _insert_sequence(new_root, all_nodes, seq, new_tree_id, seq_id)
            forests.append({"root": new_root, "all_nodes": all_nodes, "nodes": len(seq)})

    tries = [_compress_trie(f["root"]) for f in forests]
    num_tokens_list = [f["nodes"] for f in forests]
    return tries, num_tokens_list


def convert_trie_to_tree_node(
    trie: TrieNode,
) -> Optional[PrefixSubTrie]:
    """Convert a compressed trie to a :class:`PrefixSubTrie`.

    Returns ``None`` when there's no real sharing (no children or multi-root).
    Delegates to :func:`subtrie_view` with all sequence IDs.
    """
    if not trie.children:
        _logging.getLogger(__name__).warning(
            "prefix_tree: convert_trie_to_tree_node: trie has no children; no sharing, returning None"
        )
        return None
    if len(trie.children) > 1:
        _logging.getLogger(__name__).warning(
            "prefix_tree: convert_trie_to_tree_node: multiple roots (%d), returning None",
            len(trie.children),
        )
        return None
    all_seq_ids = {s for child in trie.children.values() for s in _trie_seq_ids(child)}
    return subtrie_view(trie, all_seq_ids)


def build_tree_dynamic(samples: list[Tensor]) -> Optional[PrefixSubTrie]:
    """Token-by-token trie detection. Returns a :class:`PrefixSubTrie` or None.

    Returns ``None`` when there's no shared prefix (empty input, single sample,
    or multi-forest case).
    """
    if not samples:
        return None
    sequences = [t.tolist() for t in samples]
    max_tokens_per_tree = sum(len(s) for s in sequences) * 10  # one forest
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens_per_tree)
    if not tries or len(tries) > 1:
        _logging.getLogger(__name__).warning(
            "prefix_tree: build_tree_dynamic: multi-forest or empty tries (len=%d), returning None",
            len(tries),
        )
        return None
    return convert_trie_to_tree_node(tries[0])


def _trie_seq_ids(node: TrieNode) -> list[int]:
    """Collect all sequence IDs from leaf nodes of a compressed-trie subtree."""
    if not node.children:
        return list(node.sequence_ids)
    ids: list[int] = []
    for child in node.children.values():
        ids.extend(_trie_seq_ids(child))
    return ids


def trie_group_flat_tokens(group: list[int], trie: TrieNode) -> int:
    """Flat (deduplicated) token count for a subset of sequences within a trie.

    Counts tokens on the minimal sub-trie spanning exactly the sequences in
    ``group``, i.e. the effective forward-pass token budget when those
    sequences are processed together with prefix sharing.

    Args:
        group: Sequence indices as stored in ``TrieNode.sequence_ids``.
        trie: Root of the compressed trie (``trie.is_root == True``).

    Returns:
        Total number of unique tokens required to process this group.
    """
    keep = frozenset(group)

    def _count(node: TrieNode) -> int:
        if not node.children:
            return len(node.input_ids) if any(s in keep for s in node.sequence_ids) else 0
        has_relevant = False
        relevant_total = 0
        for child in node.children.values():
            if any(s in keep for s in child.sequence_ids):
                has_relevant = True
                relevant_total += _count(child)
        return relevant_total + len(node.input_ids) if has_relevant else 0

    return sum(_count(c) for c in trie.children.values())


def dfs_leaf_order(
    sequences: list[list[int]],
    trie: Optional[TrieNode] = None,
) -> list[int]:
    """Return sample indices in DFS pre-order.

    If ``trie`` is provided (pre-built), walks it directly, no rebuild.
    Otherwise builds one via greedy token-by-token insertion.

    Args:
        sequences: per-sample token lists.
        trie: optional pre-built TrieNode root.

    Returns:
        List of sample indices in DFS pre-order (length == len(sequences)).
    """
    if not sequences:
        return []

    if trie is not None:
        return trie_dfs_leaf_order(trie)

    max_tokens = sum(len(s) for s in sequences) * 10
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens)
    ordered: list[int] = []
    for trie_root in tries:
        ordered.extend(trie_dfs_leaf_order(trie_root))
    return ordered


def dfs_micro_batch_groups(
    sequences: list[list[int]],
    max_token_len: int,
) -> list[list[int]]:
    """Group sequences into micro-batches in DFS trie order, budgeted by flat trie tokens.

    Builds ONE trie over all sequences (a forest in the rare greedy-split case),
    then delegates per-trie grouping to :func:`mbs_groups_from_trie`.  The
    budget is flat (deduplicated) trie tokens: prefix counted once + unique
    branch tokens; not raw sequence lengths.  This means a micro-batch of k
    sequences that share a long common prefix uses far fewer budget tokens than
    k × seq_len, allowing more sequences per batch.

    Args:
        sequences: per-sample token lists (the full mini-batch).
        max_token_len: flat-token budget per micro-batch.

    Returns:
        List of micro-batch groups; each group is a list of sample indices in
        DFS pre-order.
    """
    if not sequences:
        return []

    max_tokens = sum(len(s) for s in sequences) * 10  # one big forest
    tries, _ = greedy_build_tries(sequences, max_tokens_per_tree=max_tokens)

    all_groups: list[list[int]] = []
    for trie_root in tries:
        all_groups.extend(mbs_groups_from_trie(trie_root, max_token_len))
    return all_groups


def _trie_dfs_leaf_order(trie: TrieNode, leaf_positions_fn) -> list[int]:
    """Shared DFS pre-order walk over trie leaves.

    Args:
        trie: root node (``is_root`` handled).
        leaf_positions_fn: called with each leaf ``TrieNode``; returns the
            list of positions to emit for that leaf.
    """
    ordered: list[int] = []

    def _walk(node: TrieNode) -> None:
        if not node.children:
            ordered.extend(leaf_positions_fn(node))
        else:
            for child in node.children.values():
                _walk(child)

    if trie.is_root:
        for child in trie.children.values():
            _walk(child)
    else:
        _walk(trie)
    return ordered


def trie_dfs_leaf_order(trie: TrieNode) -> list[int]:
    """Return sample indices in DFS pre-order from an existing trie."""
    return _trie_dfs_leaf_order(trie, lambda n: list(n.sequence_ids))


def trie_dfs_leaf_order_from_leaf_idx(leaf_idx, trie: TrieNode) -> list[int]:
    """Return batch positions in DFS leaf order, reading from ``leaf_idx``.

    Counterpart of :func:`trie_dfs_leaf_order` that reads from ``leaf_idx``
    (reorder-aware) instead of the trie's ``sequence_ids`` (stale after
    reorder).

    ``leaf_idx`` may be a numpy array or a ``torch.long`` tensor; both support
    ``.tolist()``.
    """
    leaf_to_positions: dict[int, list[int]] = {}
    for new_pos, leaf_fid in enumerate(leaf_idx.tolist()):
        if leaf_fid < 0:
            raise ValueError(f"leaf_idx[{new_pos}]={leaf_fid}; sample has no leaf assigned.")
        leaf_to_positions.setdefault(int(leaf_fid), []).append(new_pos)

    return _trie_dfs_leaf_order(trie, lambda n: leaf_to_positions.get(n.flat_idx, []))


def _mbs_groups_dfs(
    leaf_entries: list[tuple[TrieNode, list[int]]],
    max_token_len: int,
) -> list[list[int]]:
    """Shared DFS-budget walk for micro-batch grouping.

    Args:
        leaf_entries: ``(leaf_node, positions)`` pairs in DFS pre-order.
            ``positions`` are the sample indices (or batch positions) to emit
            for that leaf.
        max_token_len: flat-token budget per micro-batch.

    Budget is flat (deduplicated) tokens.  When a trie leaf holds multiple
    positions (identical sequences), all stay in the same DFS group.
    Duplicates are handled correctly: build_layout_from_tree_node includes the
    representative's leaf range in the duplicate's ancestor_segment_ranges so
    restore_flat_to_nested reconstructs the full sequence for each.
    Keeping duplicates in-group avoids an extra singleton group that would
    cause same_micro_num_in_dp to pad the other DP rank.
    """
    all_groups: list[list[int]] = []
    current_group: list[int] = []
    covered: set[int] = set()
    current_eff = 0  # flat tokens accumulated in current group

    for node, positions in leaf_entries:
        path = trie_ancestors(node) + [node]
        new_nodes = [n for n in path if n.flat_idx not in covered]
        inc = sum(len(n.input_ids) for n in new_nodes)
        if current_group and current_eff + inc > max_token_len:
            all_groups.append(current_group[:])
            current_group.clear()
            covered.clear()
            current_eff = 0
            new_nodes = path
            inc = sum(len(n.input_ids) for n in new_nodes)
        current_group.extend(positions)
        covered.update(n.flat_idx for n in new_nodes)
        current_eff += inc

    if current_group:
        all_groups.append(current_group[:])
    return all_groups


def mbs_groups_from_trie(
    trie: TrieNode,
    max_token_len: int,
) -> list[list[int]]:
    """Group sequences into micro-batches from an existing trie.

    Budget is flat (deduplicated) tokens.  When a trie leaf holds multiple
    sequence_ids (identical sequences), all IDs stay in the same DFS group.
    Duplicates are handled correctly: build_layout_from_tree_node includes the
    representative's leaf range in the duplicate's ancestor_segment_ranges so
    restore_flat_to_nested reconstructs the full sequence for each.
    Keeping duplicates in-group avoids an extra singleton group that would
    cause same_micro_num_in_dp to pad the other DP rank.
    """
    leaf_entries: list[tuple[TrieNode, list[int]]] = []

    def _collect(node: TrieNode) -> None:
        if not node.children:
            leaf_entries.append((node, list(node.sequence_ids)))
        else:
            for child in node.children.values():
                _collect(child)

    if trie.is_root:
        for child in trie.children.values():
            _collect(child)
    else:
        _collect(trie)

    return _mbs_groups_dfs(leaf_entries, max_token_len)


def mbs_groups_from_leaf_idx(
    leaf_idx,
    trie: TrieNode,
    max_token_len: int,
) -> list[list[int]]:
    """Group reordered-batch positions into micro-batches using ``leaf_idx``.

    Counterpart of :func:`mbs_groups_from_trie` that reads from ``leaf_idx``
    (a ``torch.long`` tensor in ``batch``, automatically reordered by
    ``DataProto.reorder``) instead of the trie's ``sequence_ids`` (which go
    stale after reorder).
    """
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
        if node.children:
            continue
        if node.flat_idx not in leaf_to_positions:
            raise ValueError(
                f"trie leaf flat_idx={node.flat_idx} has no sample in leaf_idx: trie and leaf_idx are out of sync."
            )
        leaf_entries.append((node, leaf_to_positions[node.flat_idx]))

    return _mbs_groups_dfs(leaf_entries, max_token_len)


def subtrie_view(
    trie: TrieNode,
    keep_leaf_ids: set[int],
    source: Optional[PrefixTrie] = None,
) -> Optional[PrefixSubTrie]:
    """Extract a subtree containing only the given leaf sample indices.

    Returns a :class:`PrefixSubTrie` ready for downstream use, or ``None``
    if no leaves match.  When ``source`` is provided it is attached as the
    global-trie back-reference on the returned ``PrefixSubTrie``; otherwise
    one is constructed automatically from ``trie``.
    """
    if not keep_leaf_ids:
        return None
    if source is None:
        source = PrefixTrie(root=trie)

    def _collect(node: TrieNode) -> bool:
        """Walk node, collecting matching leaves."""
        if not node.children:
            kept = [s for s in node.sequence_ids if s in keep_leaf_ids]
            if not kept:
                return True
            # All samples (including duplicates) map to the same flat_idx.
            # build_layout_from_tree_node handles duplicates via ancestor_segment_ranges.
            for sid in kept:
                leaf_to_sample.append(sid)
                leaf_node_ids.append(node.flat_idx)
            return True
        for child in node.children.values():
            if keep_leaf_ids.isdisjoint(child.sequence_ids):
                continue
            if not _collect(child):
                return False
        return True

    leaf_to_sample: list[int] = []
    leaf_node_ids: list[int] = []
    for child in trie.children.values():
        if keep_leaf_ids.isdisjoint(child.sequence_ids):
            continue
        if not _collect(child):
            return None
    if not leaf_to_sample:
        return None
    if set(leaf_to_sample) != keep_leaf_ids:
        _logging.getLogger(__name__).warning("prefix_tree: subtrie_view: unmatched sequences: FA3 fallback")
        return None
    batch_size = max(leaf_to_sample) + 1 if leaf_to_sample else 0
    subtrie = PrefixSubTrie(
        source=source,
        leaf_node_ids=leaf_node_ids,
        leaf_to_sample=leaf_to_sample,
        batch_size=batch_size,
    )
    return subtrie


def compute_prefix_tree_metrics(
    input_ids,
    attention_mask=None,
    max_token_len_per_gpu: int | None = None,
    micro_batch_size: int = 0,
    trie: Optional[TrieNode] = None,
    leaf_idx=None,
) -> dict:
    """Compute prefix-tree metrics as a ``prefix_tree/`` namespace dict.

    Returns a dict with keys:
        ``prefix_tree/global_shared_ratio``      : fraction of tokens saved by deduplication
        ``prefix_tree/packed_tokens``           : deduplicated packed trie token count
        ``prefix_tree/raw_tokens``              : total raw token count across all sequences

    Note: ``micro_batch_shared_ratio`` is no longer computed here. The trainer
    previously computed it pre-forward on the full batch (before DP dispatch
    and reorder), so it did not match what the engine actually dispatches. The
    accurate post-micro-batch-build version is now computed inside
    ``prepare_prefix_tree_micro_batches`` and surfaced by
    ``maybe_collect_mbs_metric`` from the engine output.

    Args:
        input_ids: NestedTensor, padded 2-D Tensor, or list[list[int]].
        attention_mask: Optional mask for padded 2-D case.
        max_token_len_per_gpu: kept for backward-compat; no longer used here.
        micro_batch_size: kept for backward-compat; no longer used here.
        trie: Pre-built compressed TrieNode root. When provided, skips
            ``greedy_build_tries`` entirely (the caller already built it).
            ``input_ids`` is still needed for sequence lengths.
        leaf_idx: kept for backward-compat; no longer used here.

    Returns:
        dict of float metrics, all zero if no sequences.
    """
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
        return {
            "prefix_tree/global_shared_ratio": 0.0,
            "prefix_tree/packed_tokens": 0,
            "prefix_tree/raw_tokens": 0,
        }

    total_raw = sum(len(s) for s in sequences)
    if total_raw == 0:
        return {
            "prefix_tree/global_shared_ratio": 0.0,
            "prefix_tree/packed_tokens": 0,
            "prefix_tree/raw_tokens": 0,
        }

    # Build the trie once (or reuse the caller-provided one).
    if trie is None:
        tries, num_tokens = greedy_build_tries(sequences, max_tokens_per_tree=total_raw * 10)
        flat = sum(num_tokens)
    else:
        flat = sum(len(n.input_ids) for n in trie.nodes) if trie else 0

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
    """Prepare micro-batches using prefix-tree grouping.

    Works with both dynamic and fixed micro-batch sizes:
    - ``use_dynamic_bsz=True``: reads ``max_token_len_per_gpu`` and groups by flat token budget.
    - ``use_dynamic_bsz=False``: reads ``micro_batch_size_per_gpu`` and chunks by sequence count
      using DFS trie order so same-prefix sequences stay together.

    Expects a pre-built trie stored via ``tu.assign_non_tensor(data, prefix_tree=trie)``.
    If not present, falls back to token-by-token trie construction (dynbsz) or plain range
    order (fixed mbs).
    """
    trie = tu.get_non_tensor_data(data, "prefix_tree", default=None)
    leaf_idx = data.get("leaf_idx", None) if hasattr(data, "get") else data["leaf_idx"]
    if trie is not None and leaf_idx is None:
        raise ValueError(
            "prepare_prefix_tree_micro_batches: trie is attached but leaf_idx is "
            "missing from batch.  _build_global_trie must attach both."
        )

    use_dynamic_bsz_local = tu.get_non_tensor_data(data, "use_dynamic_bsz", default=True)
    if use_dynamic_bsz_local and "max_token_len_per_gpu" in data.keys():
        # Dynamic bsz: group by flat-token budget.
        _logging.getLogger(__name__).warning_once(
            "prefix_tree is on: max_token_len_per_gpu is interpreted as "
            "deduplicated (flat trie) token count, not raw sequence length."
        )
        max_token_len = data["max_token_len_per_gpu"] * sp_size
        if trie is not None:
            batch_idx_list = mbs_groups_from_leaf_idx(leaf_idx, trie, max_token_len)
        else:
            input_ids = data["input_ids"]
            seqs = [t.tolist() for t in input_ids.unbind()]
            batch_idx_list = dfs_micro_batch_groups(seqs, max_token_len)
    else:
        # Fixed mbs: chunk by sequence count in DFS trie order so same-prefix
        # sequences land in the same micro-batch.
        mbs = data["micro_batch_size_per_gpu"] * force_group_size
        n = len(data)
        if trie is not None:
            dfs_order = trie_dfs_leaf_order_from_leaf_idx(leaf_idx, trie)
        else:
            dfs_order = list(range(n))
        batch_idx_list = [dfs_order[i : i + mbs] for i in range(0, n, mbs)]

    if torch.distributed.is_initialized() and same_micro_num_in_dp and dp_group is not None:
        n_mb = torch.tensor([len(batch_idx_list)], device=get_torch_device().current_device())
        torch.distributed.all_reduce(n_mb, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        while len(batch_idx_list) < n_mb.item():
            batch_idx_list.append(batch_idx_list[-1])

    if num_batches_divided_by is not None:
        target = roundup_divisible(len(batch_idx_list), num_batches_divided_by)
        while len(batch_idx_list) < target:
            batch_idx_list.append(batch_idx_list[-1])

    micro_batches = [tu.index_select_tensor_dict(data, idx) for idx in batch_idx_list]
    # Compute deduplicated (flat) token count per micro-batch, used for PP
    # sort and for imbalance diagnostics.
    if trie is not None:
        tokens_per_group = [trie_group_flat_tokens(g, trie) for g in batch_idx_list]

        # Reorder micro-batches in inc-then-dec flat-token pattern to reduce PP bubble.
        # Preserves prefix locality: samples within a group share prefixes and stay together.
        if use_dynamic_bsz_local and len(batch_idx_list) > 1:
            sorted_groups = sorted(zip(tokens_per_group, batch_idx_list, range(len(batch_idx_list)), strict=False))
            ordered_tokens = [t for t, _, _ in sorted_groups]
            ordered_groups = [g for _, g, _ in sorted_groups]
            batch_idx_list = ordered_groups[::2] + ordered_groups[1::2][::-1]
            tokens_per_group = ordered_tokens[::2] + ordered_tokens[1::2][::-1]
            micro_batches = [tu.index_select_tensor_dict(data, idx) for idx in batch_idx_list]

        # Compute the accurate per-micro-batch sharing ratio from the ACTUAL
        # grouping the engine will dispatch (post-reorder, post-pad) and push
        # it into the module-level collector.  ``maybe_collect_mbs_metric``
        # pulls it into the engine output and resets the state.  Per-group
        # ratio is 1 - flat_tokens / raw_tokens.
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

    create_and_attach_subtrie_views(micro_batches, batch_idx_list, trie)
    return micro_batches, batch_idx_list


def create_and_attach_subtrie_views(micro_batches, batch_idx_list, trie) -> None:
    """Create a subtrie view per micro-batch and attach it (shared by dynbsz and fixed-mbs).

    Reads ``leaf_idx`` from each microbatch's ``batch`` (which survives
    ``DataProto.reorder`` and ``chunk`` automatically via torch indexing)
    to determine which leaf each local sample belongs to, no dependence
    on the trie's ``sequence_ids`` (which go stale after reorder).
    """
    if trie is None or batch_idx_list is None:
        return
    pt_global = PrefixTrie(root=trie)
    for idx, mb in zip(batch_idx_list, micro_batches, strict=False):
        mb_leaf_idx = mb.get("leaf_idx", None) if hasattr(mb, "get") else mb["leaf_idx"]
        if mb_leaf_idx is None:
            raise ValueError(
                "create_and_attach_subtrie_views: microbatch has no leaf_idx in "
                "batch.  _build_global_trie must attach leaf_idx and it "
                "must survive reorder/chunk: this is a bug."
            )
        leaf_to_local: dict[int, int] = {}
        leaf_node_ids: list[int] = []
        leaf_to_sample: list[int] = []
        for local_pos, leaf_fid in enumerate(mb_leaf_idx.tolist()):
            if leaf_fid < 0:
                raise ValueError(
                    f"microbatch leaf_idx[{local_pos}]={leaf_fid}; no leaf assigned (bug in _build_global_trie)."
                )
            if leaf_fid not in leaf_to_local:
                leaf_to_local[leaf_fid] = len(leaf_node_ids)
            leaf_node_ids.append(leaf_fid)
            leaf_to_sample.append(local_pos)
        local_subtree = PrefixSubTrie(
            source=pt_global,
            leaf_node_ids=leaf_node_ids,
            leaf_to_sample=leaf_to_sample,
            batch_size=len(idx),
        )
        tu.assign_non_tensor(mb, prefix_tree_subtree=local_subtree)


def get_dfs_balanced_partitions(
    data,
    config_or_data: dict,
    dp_size: int,
    *,
    attention_mask=None,
    contiguous_partitions: bool = False,
):
    """Re-order batch in DFS trie order and return balanced partitions."""
    if not _is_prefix_tree_enabled(config_or_data):
        return None

    batch_size = data.batch["input_ids"].shape[0] if hasattr(data, "batch") else len(data["input_ids"])
    _ids = data.batch["input_ids"] if hasattr(data, "batch") else data["input_ids"]
    _mask = (
        attention_mask
        if attention_mask is not None
        else (data.batch.get("attention_mask", None) if hasattr(data, "batch") else None)
    )

    if _mask is not None:
        seqs = [_ids[i][_mask[i].bool()].tolist() or [0] for i in range(batch_size)]
    else:
        seqs = [_ids[i].tolist() for i in range(batch_size)]

    # Reuse globally-built trie if attached (built once in ray_trainer._build_global_trie).
    # Falls back to building a throwaway trie inside dfs_leaf_order when absent.
    if hasattr(data, "batch"):
        attached_trie = data.meta_info.get("prefix_tree", None)
    else:
        attached_trie = tu.get_non_tensor_data(data, "prefix_tree", default=None)
    dfs_order = dfs_leaf_order(seqs, trie=attached_trie)
    if len(dfs_order) < batch_size:
        missing = [i for i in range(batch_size) if i not in set(dfs_order)]
        dfs_order = dfs_order + missing

    if hasattr(data, "reorder"):
        data.reorder(torch.tensor(dfs_order))
    else:
        data = tu.index_select_tensor_dict(data, torch.tensor(dfs_order))

    if hasattr(data, "batch") and "attention_mask" in data.batch:
        global_seqlen_lst = data.batch["attention_mask"].view(batch_size, -1).sum(-1)
    else:
        global_seqlen_lst = torch.Tensor([item.size()[0] for item in data["input_ids"]])

    if contiguous_partitions:
        per_rank = batch_size // dp_size
        partition_lst = [list(range(i * per_rank, (i + 1) * per_rank)) for i in range(dp_size)]
    else:
        partition_lst = get_seqlen_balanced_partitions(
            calculate_workload(global_seqlen_lst), k_partitions=dp_size, equal_size=True
        )

    return partition_lst, global_seqlen_lst, data


def reorder_and_balance_for_prefix_tree(
    data,
    config_or_data: dict,
    dp_size: int,
    *,
    attention_mask=None,
    metrics: dict | None = None,
    logging_prefix: str = "global_seqlen",
) -> bool:
    """DFS-reorder batch and compute contiguous partitions for prefix-tree."""
    if not _is_prefix_tree_enabled(config_or_data):
        return False

    result = get_dfs_balanced_partitions(
        data,
        config_or_data,
        dp_size,
        attention_mask=attention_mask,
        contiguous_partitions=True,
    )
    if result is None:
        return False

    global_partition_lst, global_seqlen_lst, _ = result
    global_idx = torch.arange(global_seqlen_lst.shape[0])
    data.reorder(global_idx)
    if metrics is not None:
        stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(),
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(stats)
    return True
