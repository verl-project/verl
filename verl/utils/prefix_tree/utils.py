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

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from verl.utils.prefix_tree.tree import PrefixSubTrie, TrieNode

RangeSpec = tuple[int, int]

# Boundary registry: (boundary_flat_position, [(sample_idx, next_token), ...]).
BoundaryRegistry = list[tuple[int, list[tuple[int, int]]]]


@dataclass
class PrefixTreeParams:
    """Metadata for a flattened PrefixTree batch."""

    prefix_range: RangeSpec
    leaf_ranges: list[RangeSpec]
    leaf_to_sample: list[int]
    sample_to_leaf_range: dict[int, RangeSpec]
    q_ranges: list[RangeSpec]
    k_ranges: list[RangeSpec]
    mask_types: list[str]
    total_seqlen_q: int
    total_seqlen_k: int
    tree_packed_tokens: Optional[Tensor] = None
    tree_packed_labels: Optional[Tensor] = None
    tree_packed_loss_mask: Optional[Tensor] = None
    tree_packed_position_ids: Optional[Tensor] = None

    # Boundary registry for the LCE boundary-patch fix (see prepare_packed_label).
    # Plain Python (list of tuples of ints) — serialises fine over RPC.
    # None when no boundaries exist (single-leaf subtries or no branching).
    boundary_registry: Optional[BoundaryRegistry] = None

    def __post_init__(self) -> None:
        if len(self.leaf_ranges) != len(self.leaf_to_sample):
            raise ValueError("leaf_ranges and leaf_to_sample must have the same length")
        if len(self.q_ranges) != len(self.k_ranges) or len(self.q_ranges) != len(self.mask_types):
            raise ValueError("q_ranges, k_ranges, and mask_types must have the same length")
        if set(self.leaf_to_sample) != set(self.sample_to_leaf_range):
            raise ValueError("sample_to_leaf_range must cover exactly the samples in leaf_to_sample")

        prefix_start, prefix_end = self.prefix_range
        if prefix_start != 0:
            raise ValueError("prefix_range must start at 0 in flattened PrefixTree layout")
        if prefix_end < prefix_start:
            raise ValueError("prefix_range must be non-decreasing")

        if any(e < s for s, e in self.leaf_ranges):
            raise ValueError("leaf ranges must be non-decreasing")

        if self.total_seqlen_q != self.total_seqlen_k:
            raise ValueError("PrefixTree expects matching q/k sequence lengths")
        if self.leaf_ranges and max(end for _, end in self.leaf_ranges) != self.total_seqlen_q:
            raise ValueError("max leaf range end must equal total sequence length")
        if not self.leaf_ranges and self.prefix_range[1] != self.total_seqlen_q:
            raise ValueError("prefix-only PrefixTree must end at total sequence length")

        for sample_idx, leaf_range in zip(self.leaf_to_sample, self.leaf_ranges, strict=False):
            if self.sample_to_leaf_range[sample_idx] != leaf_range:
                raise ValueError("sample_to_leaf_range does not match leaf_to_sample ordering")

        for name, tensor in {
            "tree_packed_tokens": self.tree_packed_tokens,
            "tree_packed_labels": self.tree_packed_labels,
            "tree_packed_position_ids": self.tree_packed_position_ids,
        }.items():
            if tensor is not None and tensor.numel() != self.total_seqlen_q:
                raise ValueError(f"{name} must have total_seqlen_q elements")

    @property
    def prefix_len(self) -> int:
        return self.prefix_range[1] - self.prefix_range[0]

    @property
    def num_samples(self) -> int:
        return len(self.leaf_to_sample)


__all__ = [
    "RangeSpec",
    "PrefixTreeParams",
    "BoundaryRegistry",
    "build_layout_from_tree_node",
    "prepare_packed_label",
]


def prepare_packed_label(
    samples: Sequence[Tensor],
    subtrie: PrefixSubTrie,
    leaf_node_id_to_samples: dict[int, list[int]],
    flat_end: dict[int, int],
    owner_offset: dict[int, int],
) -> BoundaryRegistry:
    """Build boundary registry for LCE boundary-patch: maps flat positions with ≥2 branching children to per-leaf
    (sample_idx, next_token) pairs, so restore_flat_to_nested can patch non-owner leaf boundary log-probs after LCE."""

    def _collect_leaf_descendants(node: TrieNode) -> list[TrieNode]:
        """All leaf nodes (no in-view children) in the subtree rooted at *node*."""
        return list(subtrie.dfs(roots=subtrie.children_of(node), leaf_only=True))

    registry: BoundaryRegistry = []

    # BFS walk to find branching nodes (≥2 children → boundary).
    for node in subtrie.bfs():
        children = subtrie.children_of(node)
        # Boundary condition: ≥2 children AND node has ≥1 token to emit.
        # A node with 0 tokens has no flat position (no predictor), and a
        # node with <2 children doesn't cause divergence here.
        if len(children) < 2 or len(node.input_ids) < 1:
            continue

        # The boundary is the LAST token of this shared ancestor.
        b_pos = flat_end[node.node_idx] - 1
        next_token_pos = owner_offset[node.node_idx] + len(node.input_ids)

        leaves_info: list[tuple[int, int]] = []
        for leaf in _collect_leaf_descendants(node):
            # Expand to ALL samples sharing this leaf node (duplicates).
            for sample_idx in leaf_node_id_to_samples.get(leaf.node_idx, []):
                # Zero-length leaf: next_token is past sample end; skip to
                # avoid IndexError (no token to predict at the boundary).
                sample_len = samples[sample_idx].shape[0]
                if next_token_pos >= sample_len:
                    continue
                next_token = int(samples[sample_idx][next_token_pos].item())
                leaves_info.append((sample_idx, next_token))

        if len(leaves_info) >= 2:
            registry.append((b_pos, leaves_info))

    return registry


def build_layout_from_tree_node(
    samples: Sequence[Tensor],
    subtrie: PrefixSubTrie,
    loss_masks_by_sample: Optional[Sequence[Tensor]] = None,
    position_ids_by_sample: Optional[Sequence[Tensor]] = None,
) -> PrefixTreeParams:
    """generate metadata (PrefixTreeParams) from tree structure"""
    # Map node_idx → ordered list of sample_ids (first = representative, rest = duplicates)
    leaf_node_id_to_samples: dict[int, list[int]] = {}
    for nid, sid in zip(subtrie.leaf_node_ids, subtrie.leaf_to_sample, strict=False):
        leaf_node_id_to_samples.setdefault(nid, []).append(sid)

    # LOCAL keyed: leaf_to_sample is local mb positions; node.sequence_ids are global — never cross-match.
    leaf_node_id_to_sample: dict[int, int] = {nid: sids[0] for nid, sids in leaf_node_id_to_samples.items()}

    device = samples[0].device
    rolled_samples = [torch.cat([s[1:], torch.zeros(1, dtype=s.dtype, device=s.device)]) for s in samples]

    # Local layout dicts — no TrieNode mutation.
    flat_start: dict[int, int] = {}
    flat_end: dict[int, int] = {}
    owner_offset: dict[int, int] = {}

    q_ranges: list[RangeSpec] = []
    k_ranges: list[RangeSpec] = []
    mask_types: list[str] = []
    flat_pieces: list[Tensor] = []
    flat_label_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []

    bfs_order = list(subtrie.bfs())

    # Pass 1: assign flat positions (forward BFS). owner_offset (sample-space
    # offset) is precomputed as node.sample_start during subtrie compression.
    pos = 0
    for node in bfs_order:
        nid = node.node_idx
        flat_start[nid] = pos
        flat_end[nid] = pos + len(node.input_ids)
        owner_offset[nid] = node.sample_start
        pos = flat_end[nid]

    # Owners: leaf entries first (node_idx keyed — order-independent), then
    # internal nodes inherit from the first in-view descendant with a known owner.
    owner_sample: dict[int, Optional[int]] = {}
    for node in bfs_order:
        owner_sample[node.node_idx] = leaf_node_id_to_sample.get(node.node_idx)
    for node in reversed(bfs_order):
        nid = node.node_idx
        if owner_sample[nid] is not None:
            continue
        stack = list(node.children.values())
        while stack:
            child = stack.pop()
            child_owner = owner_sample.get(child.node_idx)
            if child_owner is not None:
                owner_sample[nid] = child_owner
                break
            stack.extend(child.children.values())

    # Pass 2: pack tokens/labels/masks/position_ids + attention rectangles.
    for node in bfs_order:
        if len(node.input_ids) == 0:
            continue
        nid = node.node_idx
        s, e = owner_offset[nid], owner_offset[nid] + len(node.input_ids)
        if s >= e:
            continue
        src = owner_sample.get(nid)
        if src is None:
            continue  # pruned node, not owned by any sample in this shard
        flat_pieces.append(samples[src][s:e])
        flat_label_pieces.append(rolled_samples[src][s:e])
        if flat_lm_pieces is not None:
            flat_lm_pieces.append(loss_masks_by_sample[src][s:e])

        fs, fe = flat_start[nid], flat_end[nid]
        if flat_pid_pieces is not None:
            flat_pid_pieces.append(position_ids_by_sample[src][s:e])
        else:
            default_pid_pieces.append(torch.arange(s, e, device=device, dtype=torch.long))

        node_range: RangeSpec = (fs, fe)
        q_ranges.append(node_range)
        k_ranges.append(node_range)
        mask_types.append("causal")
        for desc in subtrie.bfs(roots=subtrie.children_of(node)):
            if len(desc.input_ids) > 0:
                did = desc.node_idx
                q_ranges.append((flat_start[did], flat_end[did]))
                k_ranges.append(node_range)
                mask_types.append("full")

    # Assemble packed tensors.
    tree_packed_tokens = _cat_or_empty(flat_pieces, samples[0].dtype, device)
    tree_packed_loss_mask = torch.cat(flat_lm_pieces) if flat_lm_pieces is not None else None
    tree_packed_labels_tensor = (
        torch.cat(flat_label_pieces) if flat_label_pieces else torch.zeros_like(tree_packed_tokens)
    )
    if flat_pid_pieces is not None:
        tree_packed_position_ids = torch.cat(flat_pid_pieces)
    else:
        tree_packed_position_ids = _cat_or_empty(default_pid_pieces, torch.long, device)

    # One entry per leaf_node_ids position so restore stays aligned with subtrie.leaf_to_sample.
    leaf_ranges: list[RangeSpec] = []
    leaf_to_sample_list: list[int] = []
    leaf_ancestor_ranges: list[list[RangeSpec]] = []

    first_rep_range: dict[int, RangeSpec] = {}
    for i, nid in enumerate(subtrie.leaf_node_ids):
        if nid not in flat_start:
            continue
        sids = leaf_node_id_to_samples.get(nid, [])
        if not sids:
            continue
        sample_idx = subtrie.leaf_to_sample[i]

        chain: list[RangeSpec] = []
        cur = next((n for n in subtrie.nodes if n.node_idx == nid), None)
        if cur is not None:
            anc = cur.ancestor
            while anc is not None and anc in subtrie.nodes:
                cid = anc.node_idx
                if cid in flat_start:
                    chain.append((flat_start[cid], flat_end[cid]))
                anc = anc.ancestor
        chain.reverse()

        if nid not in first_rep_range:
            rep_range: RangeSpec = (flat_start[nid], flat_end[nid])
            first_rep_range[nid] = rep_range
            leaf_ranges.append(rep_range)
            leaf_ancestor_ranges.append(chain)
        else:
            rep_range = first_rep_range[nid]
            leaf_ranges.append((rep_range[1], rep_range[1]))
            leaf_ancestor_ranges.append(chain + [rep_range])
        leaf_to_sample_list.append(sample_idx)

    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample_list, leaf_ranges, strict=False)}
    prefix_range = (flat_start[subtrie.roots[0].node_idx], flat_end[subtrie.roots[0].node_idx])

    params = PrefixTreeParams(
        prefix_range=prefix_range,
        leaf_ranges=leaf_ranges,
        leaf_to_sample=leaf_to_sample_list,
        sample_to_leaf_range=sample_to_leaf_range,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        mask_types=mask_types,
        total_seqlen_q=tree_packed_tokens.numel(),
        total_seqlen_k=tree_packed_tokens.numel(),
        tree_packed_tokens=tree_packed_tokens,
        tree_packed_labels=tree_packed_labels_tensor,
        tree_packed_loss_mask=tree_packed_loss_mask,
        tree_packed_position_ids=tree_packed_position_ids,
        boundary_registry=prepare_packed_label(samples, subtrie, leaf_node_id_to_samples, flat_end, owner_offset),
    )
    params._leaf_ancestor_ranges = leaf_ancestor_ranges

    return params


def _cat_or_empty(pieces: list[Tensor], dtype: torch.dtype, device: torch.device) -> Tensor:
    """torch.cat(pieces) or an empty tensor of dtype/device when pieces is empty."""
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=dtype, device=device)
