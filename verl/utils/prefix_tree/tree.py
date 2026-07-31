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
"""Prefix-tree data structures.

Class hierarchy:

    TrieNode:       compressed trie node; root holds flat DFS-ordered ``nodes``
                      list; ``nodes[i].flat_idx == i`` for O(1) lookup and future
                      KV-cache indexing.

    PrefixTrie:     common interface for navigating a prefix trie.
                      Both the global trie and per-micro-batch view expose this
                      interface so callers need not distinguish between them.
                      Future: ``kv_cache: list[Tensor]`` indexed by ``flat_idx``.

    PrefixSubTrie:  per-micro-batch view into a PrefixTrie.  Inherits the
                      PrefixTrie interface.  Serialisable: ``leaf_node_ids`` and
                      ``leaf_to_sample`` are plain int lists; ``source`` is a
                      local-only back-reference, never transmitted cross-node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# TrieNode: compressed trie node (immutable after construction)
# ---------------------------------------------------------------------------


@dataclass
class TrieNode:
    """Compressed-trie node produced by the ``_compress`` pass.

    Each non-root node represents a contiguous run of tokens shared by the same
    set of sequences.  The root has no tokens and no parent (``ancestor=None``).

    ``nodes`` (root-only): flat DFS-ordered list of all non-root nodes.
    ``nodes[i].flat_idx == i`` (assigned once during construction and never
    mutated, making the trie effectively immutable after build).
    """

    tree_id: int
    start_idx: int = -1
    end_idx: int = -1
    input_ids: list[int] = field(default_factory=list)
    children: dict[int, TrieNode] = field(default_factory=dict)
    # Sequence IDs that pass through this node (from trie construction).
    sequence_ids: list[int] = field(default_factory=list)
    # Direct parent reference: single hop upward; None on root.
    ancestor: Optional[TrieNode] = None
    # Root-only: flat DFS list; nodes[i].flat_idx == i.
    nodes: list[TrieNode] = field(default_factory=list)
    # Root-only: leaves[sample_idx] = leaf TrieNode for that sample.
    # Direct O(1) index → leaf map; populated during tree construction.
    leaves: list[Optional[TrieNode]] = field(default_factory=list)
    # DFS index in root.nodes: set once by _compress_trie; -1 on root.
    flat_idx: int = -1

    @property
    def is_root(self) -> bool:
        return self.flat_idx == -1


# ---------------------------------------------------------------------------
# PrefixTrie: common interface (global trie + subtrie view)
# ---------------------------------------------------------------------------


class PrefixTrie:
    """Common interface for navigating a prefix trie.

    Concrete subclasses:
      - PrefixTrie itself:  wraps the global TrieNode root for a full batch
      - PrefixSubTrie:      filtered view for one micro-batch

    Both expose the same interface so downstream code (layout builder, MAGI
    key construction, KV-cache lookup) works identically on either.
    """

    nodes: list[TrieNode]  # flat DFS list; nodes[i].flat_idx == i
    root: TrieNode  # virtual root (is_root=True)

    def __init__(self, root: TrieNode) -> None:
        self.root = root
        self.nodes = root.nodes  # shared reference, no copy

    # ── navigation ────────────────────────────────────────────────────────

    def __getitem__(self, flat_idx: int) -> TrieNode:
        """O(1) node lookup by flat_idx."""
        return self.nodes[flat_idx]

    # ── metrics ───────────────────────────────────────────────────────────

    # ── future: KV cache ──────────────────────────────────────────────────
    # kv_cache: list[Optional[Tensor]]  # indexed by flat_idx, same as nodes


# ---------------------------------------------------------------------------
# PrefixSubTrie: per-micro-batch view
# ---------------------------------------------------------------------------


class PrefixSubTrie(PrefixTrie):
    """Per-micro-batch view into a PrefixTrie.

    Exposes the same PrefixTrie interface so callers treat it identically to
    the global trie.  ``nodes`` contains only the TrieNodes relevant to this
    micro-batch (leaves and their ancestor path).

    Serialisation
    -------------
    ``leaf_node_ids`` and ``leaf_to_sample`` are plain int lists (tiny cross-
    node footprint).  ``source`` is a local-only reference (not serialised);
    re-attach after deserialisation to enable KV-cache lookup.
    """

    leaf_to_sample: list[int]  # leaf i → global sample index
    leaf_node_ids: list[int]  # leaf i → flat_idx of its TrieNode in source
    source: Optional[PrefixTrie]  # back-ref to global trie; not serialised
    # leaf_ids[local_pos] = flat_idx of that sample's leaf; -1 if absent.
    # Indexed by shard-local position (0..len(global_sample_ids)-1).
    # Use global_sample_ids[local_pos] to recover the global sample index.
    leaf_ids: np.ndarray  # shape (len(global_sample_ids),), dtype int64
    # global_sample_ids[i] = global sample index for shard-local position i.
    global_sample_ids: list[int]

    # MAGI key cache for OLP→actor_update reuse. Populated on first forward,
    # reused if tree structure and CP group haven't changed.
    _cached_magi_key: Optional[object] = None

    def __init__(
        self,
        source: PrefixTrie,
        leaf_node_ids: list[int],
        leaf_to_sample: list[int],
        batch_size: int,
    ) -> None:
        import numpy as np

        self.source = source
        self.leaf_node_ids = leaf_node_ids
        self.leaf_to_sample = leaf_to_sample
        self.root = source.root
        self.nodes = self._collect_nodes(source, leaf_node_ids)
        # Build shard-local leaf_ids: indexed by local position (0..shard_size-1).
        # global_sample_ids[i] is the global sample index for local position i.
        self.global_sample_ids = sorted(set(leaf_to_sample))
        _global_to_local: dict[int, int] = {g: i for i, g in enumerate(self.global_sample_ids)}
        local_batch_size = len(self.global_sample_ids)
        self.leaf_ids = np.full(local_batch_size, -1, dtype=np.int64)
        for i, sample_idx in enumerate(leaf_to_sample):
            self.leaf_ids[_global_to_local[sample_idx]] = leaf_node_ids[i]

    @staticmethod
    def _collect_nodes(source: PrefixTrie, leaf_node_ids: list[int]) -> list[TrieNode]:
        """Collect nodes reachable from the given leaves (leaves + all ancestors)."""
        seen: set[int] = set()
        result: list[TrieNode] = []
        for idx in leaf_node_ids:
            node = source[idx]
            path: list[TrieNode] = []
            cur: Optional[TrieNode] = node
            while cur is not None and cur.flat_idx not in seen:
                path.append(cur)
                cur = cur.ancestor
            for n in reversed(path):
                if n.flat_idx not in seen:
                    seen.add(n.flat_idx)
                    result.append(n)
        return result

    def __getstate__(self) -> dict:
        """Pickle without the full trie back-references.

        TrieNode.children dicts point to siblings outside the subtrie, causing
        pickle to serialise the entire global trie.  Store compact per-node data
        and reconstruct detached nodes on __setstate__.
        """
        nodes_data = [
            (n.flat_idx, n.input_ids, n.ancestor.flat_idx if n.ancestor else -1, n.sequence_ids) for n in self.nodes
        ]
        return {
            "leaf_node_ids": self.leaf_node_ids,
            "leaf_to_sample": self.leaf_to_sample,
            "leaf_ids": self.leaf_ids,
            "global_sample_ids": self.global_sample_ids,
            "nodes_data": nodes_data,
        }

    def __setstate__(self, state: dict) -> None:
        self.source = None
        self.root = TrieNode(tree_id=-1)  # dummy root (not used after unpickling)
        self.leaf_node_ids = state["leaf_node_ids"]
        self.leaf_to_sample = state["leaf_to_sample"]
        self.leaf_ids = state["leaf_ids"]
        self.global_sample_ids = state.get("global_sample_ids", sorted(set(state["leaf_to_sample"])))
        # Reconstruct detached TrieNode objects (subtrie-only children links).
        by_flat_idx: dict[int, TrieNode] = {}
        for flat_idx, input_ids, _anc, sequence_ids in state["nodes_data"]:
            node = TrieNode(tree_id=-1, input_ids=list(input_ids), sequence_ids=list(sequence_ids), flat_idx=flat_idx)
            by_flat_idx[flat_idx] = node

        for flat_idx, input_ids, ancestor_flat_idx, _seq in state["nodes_data"]:
            node = by_flat_idx[flat_idx]
            if ancestor_flat_idx != -1 and ancestor_flat_idx in by_flat_idx:
                node.ancestor = by_flat_idx[ancestor_flat_idx]
                # Key by flat_idx (always unique) not first_token: keying by first
                # token causes silent collision when two siblings start with the same
                # token (e.g. two rollout responses both begin with the same word).
                by_flat_idx[ancestor_flat_idx].children[flat_idx] = node

        self.nodes = [by_flat_idx[fid] for fid, _, _, _ in state["nodes_data"]]
        self._cached_magi_key = None


# ---------------------------------------------------------------------------
# Segment-based global trie construction (O(N), no token comparison)
# ---------------------------------------------------------------------------


def build_global_tree_from_segments(
    samples: list,
    segment_hashes,
    segment_lengths,
) -> Optional[TrieNode]:
    """Build a global TrieNode from segment metadata, supporting multilevel trees.

    O(N) construction using known prefix structure, no token-by-token
    comparison.  Returns a TrieNode root compatible with ``mbs_groups_from_trie``
    and ``subtrie_view``.

    Recurses through every segment level: samples sharing levels 0..k get an
    ancestor node at each shared level.  A sample bottoms out into a leaf when
    its segments are exhausted, when its subgroup becomes a singleton, or when
    no further sharing is found.  Leaf ``input_ids`` = all remaining tokens
    after the last shared segment.

    Children of the root are keyed by level-0 hash (deterministic DFS order).
    Children of intermediate nodes are keyed by ``flat_idx`` (unique).

    Args:
        samples: List of 1-D token tensors or lists (one per sequence).
        segment_hashes: (N,) object-dtype numpy array of hash lists.
        segment_lengths: (N,) object-dtype numpy array of length lists.

    Returns:
        TrieNode root, or None if fewer than 2 samples.
    """
    if not samples or len(samples) < 2:
        return None

    from verl.utils.prefix_tree.segment_grouper import group_by_segment_hash

    # Normalize samples to list[list[int]]: production caller passes lists,
    # tests may pass tensors.
    samples = [s.tolist() if hasattr(s, "tolist") else list(s) for s in samples]

    groups = group_by_segment_hash(segment_hashes, segment_lengths, level=0)

    trie_root = TrieNode(tree_id=0)
    trie_root.nodes = []
    trie_root.leaves = [None] * len(samples)

    for uid_hash in sorted(groups.keys()):
        group = groups[uid_hash]
        all_seq_ids = [sid for sid, _ in group]

        if len(group) >= 2:
            first_idx, prefix_len = group[0]
            prefix_tokens = samples[first_idx][:prefix_len]

            prefix_node = TrieNode(
                tree_id=0,
                input_ids=prefix_tokens,
                sequence_ids=list(all_seq_ids),
                ancestor=None,
                flat_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(prefix_node)
            trie_root.children[uid_hash] = prefix_node

            _build_segment_subtree(
                samples,
                segment_hashes,
                segment_lengths,
                group_sids=all_seq_ids,
                level=1,
                accumulated_len=prefix_len,
                parent_node=prefix_node,
                trie_root=trie_root,
            )
        else:
            seq_idx = group[0][0]
            all_tokens = samples[seq_idx]
            leaf = TrieNode(
                tree_id=0,
                input_ids=all_tokens,
                sequence_ids=[seq_idx],
                ancestor=None,
                flat_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(leaf)
            trie_root.children[uid_hash] = leaf
            trie_root.leaves[seq_idx] = leaf

    return trie_root if trie_root.nodes else None


def _build_segment_subtree(
    samples,
    segment_hashes,
    segment_lengths,
    group_sids: list[int],
    level: int,
    accumulated_len: int,
    parent_node: TrieNode,
    trie_root: TrieNode,
) -> None:
    """Recursively build ancestor nodes for shared segments at ``level`` >= 1.

    Samples in *group_sids* already share levels 0..level-1.  Subgroup by hash
    at *level*: 2+ sharing → ancestor node + recurse on the suffix; singleton
    subgroup or segment-exhausted sample → leaf with all remaining tokens.
    """
    # Samples whose segment list ended before this level → leaves with remaining tokens.
    for sid in group_sids:
        if level >= len(segment_hashes[sid]):
            remaining = samples[sid][accumulated_len:]
            leaf = TrieNode(
                tree_id=0,
                input_ids=remaining,
                sequence_ids=[sid],
                ancestor=parent_node,
                flat_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(leaf)
            parent_node.children[leaf.flat_idx] = leaf
            trie_root.leaves[sid] = leaf

    # Subgroup remaining samples by hash at this level.
    subgroups: dict[int, list[int]] = {}
    for sid in group_sids:
        if level < len(segment_hashes[sid]):
            h = int(segment_hashes[sid][level])
            subgroups.setdefault(h, []).append(sid)

    for hash_val in sorted(subgroups.keys()):
        subgroup = subgroups[hash_val]
        if len(subgroup) >= 2:
            seg_len = int(segment_lengths[subgroup[0]][level])
            seg_tokens = samples[subgroup[0]][accumulated_len : accumulated_len + seg_len]
            node = TrieNode(
                tree_id=0,
                input_ids=seg_tokens,
                sequence_ids=list(subgroup),
                ancestor=parent_node,
                flat_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(node)
            parent_node.children[node.flat_idx] = node
            _build_segment_subtree(
                samples,
                segment_hashes,
                segment_lengths,
                group_sids=subgroup,
                level=level + 1,
                accumulated_len=accumulated_len + seg_len,
                parent_node=node,
                trie_root=trie_root,
            )
        else:
            sid = subgroup[0]
            remaining = samples[sid][accumulated_len:]
            leaf = TrieNode(
                tree_id=0,
                input_ids=remaining,
                sequence_ids=[sid],
                ancestor=parent_node,
                flat_idx=len(trie_root.nodes),
            )
            trie_root.nodes.append(leaf)
            parent_node.children[leaf.flat_idx] = leaf
            trie_root.leaves[sid] = leaf


def trie_ancestors(node: TrieNode) -> list[TrieNode]:
    """Return ancestor chain from root-child down to node's parent (exclusive of node)."""
    chain: list[TrieNode] = []
    cur = node.ancestor
    while cur is not None:
        chain.append(cur)
        cur = cur.ancestor
    chain.reverse()
    return chain


def _is_prefix_tree_enabled(config_or_data) -> bool:
    if isinstance(config_or_data, dict):
        return config_or_data.get("use_prefix_tree", False)
    return bool(getattr(config_or_data, "use_prefix_tree", False))
