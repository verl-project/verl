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

        for leaf_range in self.leaf_ranges:
            leaf_start, leaf_end = leaf_range
            if leaf_end < leaf_start:
                raise ValueError("leaf ranges must be non-decreasing")

        if self.total_seqlen_q != self.total_seqlen_k:
            raise ValueError("PrefixTree expects matching q/k sequence lengths")
        if self.leaf_ranges and self.leaf_ranges[-1][1] != self.total_seqlen_q:
            raise ValueError("last leaf range must end at total sequence length")
        if not self.leaf_ranges and self.prefix_range[1] != self.total_seqlen_q:
            raise ValueError("prefix-only PrefixTree must end at total sequence length")

        for sample_idx, leaf_range in zip(self.leaf_to_sample, self.leaf_ranges, strict=False):
            if self.sample_to_leaf_range[sample_idx] != leaf_range:
                raise ValueError("sample_to_leaf_range does not match leaf_to_sample ordering")

        for name, tensor in {
            "tree_packed_tokens": self.tree_packed_tokens,
            "tree_packed_labels": self.tree_packed_labels,
            "tree_packed_loss_mask": self.tree_packed_loss_mask,
            "tree_packed_position_ids": self.tree_packed_position_ids,
        }.items():
            if tensor is not None and tensor.numel() != self.total_seqlen_q:
                raise ValueError(f"{name} must have total_seqlen_q elements")

    @property
    def prefix_len(self) -> int:
        return self.prefix_range[1] - self.prefix_range[0]

    @property
    def branch_lengths(self) -> list[int]:
        return [end - start for start, end in self.leaf_ranges]

    @property
    def num_samples(self) -> int:
        return len(self.leaf_to_sample)

    def get_leaf_range(self, sample_idx: int) -> RangeSpec:
        return self.sample_to_leaf_range[sample_idx]


__all__ = [
    "RangeSpec",
    "PrefixTreeParams",
    "build_layout_from_tree_node",
]


def build_layout_from_tree_node(
    samples: Sequence[Tensor],
    subtrie: PrefixSubTrie,
    loss_masks_by_sample: Optional[Sequence[Tensor]] = None,
    position_ids_by_sample: Optional[Sequence[Tensor]] = None,
) -> PrefixTreeParams:
    """Build flat layout (PrefixTreeParams) from a PrefixSubTrie.

    Walks only the nodes in ``subtrie``, emitting tokens in DFS pre-order.
    Leaf ordering matches ``subtrie.leaf_to_sample``.

    Labels are derived from the flat token layout after packing: the donation
    mechanism ensures ``labels[k] = tokens[k+1]`` is correct everywhere except
    the last position of each leaf segment, which is zeroed (EOS).
    """

    valid_ids: set[int] = {n.flat_idx for n in subtrie.nodes}
    # Map flat_idx → ordered list of sample_ids (first = representative, rest = zero-len duplicates)
    leaf_node_id_to_samples: dict[int, list[int]] = {}
    for nid, sid in zip(subtrie.leaf_node_ids, subtrie.leaf_to_sample, strict=False):
        leaf_node_id_to_samples.setdefault(nid, []).append(sid)
    leaf_node_id_to_sample: dict[int, int] = {nid: sids[0] for nid, sids in leaf_node_id_to_samples.items()}

    def _subtrie_children(node: TrieNode) -> list[TrieNode]:
        return [c for c in node.children.values() if c.flat_idx in valid_ids]

    root_nodes = [n for n in subtrie.nodes if n.ancestor is None or n.ancestor.flat_idx not in valid_ids]

    # Assign flat positions, build attention spec rectangles.
    q_ranges: list[RangeSpec] = []
    k_ranges: list[RangeSpec] = []
    mask_types: list[str] = []

    def _assign_offsets(node: TrieNode, start: int, donated_in: bool = False) -> int:
        """Assign flat positions with boundary-shift: each non-leaf donates its last
        token to every child so the child independently packs it from its own sample,
        giving the correct per-sample label at the prefix/leaf boundary.

        donated_in: parent donated its last token → this node includes 1 extra token.
        donated_out: this node donates its last token → children each include 1 extra.
        """
        children = _subtrie_children(node)
        node._flat_start = start
        extra_in = 1 if donated_in else 0
        if children and len(node.input_ids) >= 1:
            donated_out = True
            node._flat_end = start + extra_in + len(node.input_ids) - 1
        else:
            donated_out = False
            node._flat_end = start + extra_in + len(node.input_ids)
        node._donated_in = donated_in
        node._donated_out = donated_out
        cur = node._flat_end
        for child in children:
            cur = _assign_offsets(child, cur, donated_out)
        return cur

    pos = 0
    for root in root_nodes:
        pos = _assign_offsets(root, pos)

    def _collect_descendants(node: TrieNode) -> list[TrieNode]:
        result: list[TrieNode] = []
        for child in _subtrie_children(node):
            result.append(child)
            result.extend(_collect_descendants(child))
        return result

    def _emit_attn(node: TrieNode) -> None:
        if not node.input_ids or node._flat_start >= node._flat_end:
            # No tokens or empty range after donation (e.g. single-token node that
            # donated its only token to children): skip rects, still recurse.
            for child in _subtrie_children(node):
                _emit_attn(child)
            return
        node_range: RangeSpec = (node._flat_start, node._flat_end)
        q_ranges.append(node_range)
        k_ranges.append(node_range)
        mask_types.append("causal")
        children = _subtrie_children(node)
        if not children:
            return
        for desc in _collect_descendants(node):
            if not desc.input_ids:
                continue
            q_ranges.append((desc._flat_start, desc._flat_end))
            k_ranges.append(node_range)
            mask_types.append("full")
        for child in children:
            _emit_attn(child)

    for root in root_nodes:
        _emit_attn(root)

    # Build flat token layout.
    leaves_in_dfs: list[TrieNode] = []
    parent_of: dict[int, TrieNode] = {}

    def _annotate(node: TrieNode, owner_offset: int) -> int:
        node._owner_offset = owner_offset
        children = _subtrie_children(node)
        if not children:
            sample_idx = leaf_node_id_to_sample[node.flat_idx]
            node._owner_sample = sample_idx
            leaves_in_dfs.append(node)
            return sample_idx
        child_offset = owner_offset + len(node.input_ids)
        first_owner: Optional[int] = None
        for i, child in enumerate(children):
            parent_of[child.flat_idx] = node
            owner = _annotate(child, child_offset)
            if i == 0:
                first_owner = owner
        node._owner_sample = first_owner
        return first_owner

    for root in root_nodes:
        _annotate(root, 0)

    device = samples[0].device
    flat_pieces: list[Tensor] = []
    flat_lm_pieces: Optional[list[Tensor]] = [] if loss_masks_by_sample is not None else None
    flat_pid_pieces: Optional[list[Tensor]] = [] if position_ids_by_sample is not None else None
    default_pid_pieces: list[Tensor] = []

    def _emit(node: TrieNode) -> None:
        donated_in = getattr(node, "_donated_in", False)
        if node.input_ids or donated_in:
            children = _subtrie_children(node)
            s = node._owner_offset
            e = s + len(node.input_ids)
            donated_out = getattr(node, "_donated_out", False)
            # Slice into sample sequence: include donated boundary from parent (-1 on s),
            # exclude token donated to children (-1 on e).
            s_emit = s - 1 if donated_in else s
            e_emit = e - 1 if donated_out else e
            if not children:
                # Leaf: use its own sample so the donated boundary token comes from
                # this leaf's sequence (making tokens[k+1] correct for label derivation).
                leaf_sample = leaf_node_id_to_sample.get(node.flat_idx)
                src = leaf_sample if leaf_sample is not None else node._owner_sample
            else:
                # Non-leaf: use owner's tokens (shared prefix/ancestor tokens).
                src = node._owner_sample
            if s_emit < e_emit:
                flat_pieces.append(samples[src][s_emit:e_emit])
                if flat_lm_pieces is not None:
                    flat_lm_pieces.append(loss_masks_by_sample[src][s_emit:e_emit])
                if flat_pid_pieces is not None:
                    flat_pid_pieces.append(position_ids_by_sample[src][s_emit:e_emit])
                else:
                    default_pid_pieces.append(torch.arange(s_emit, e_emit, device=device, dtype=torch.long))
        for child in _subtrie_children(node):
            _emit(child)

    for root in root_nodes:
        _emit(root)

    tree_packed_tokens = (
        torch.cat(flat_pieces) if flat_pieces else torch.empty(0, dtype=samples[0].dtype, device=device)
    )
    tree_packed_loss_mask = torch.cat(flat_lm_pieces) if flat_lm_pieces is not None else None
    # Labels: tokens[k+1] at every position (donation ensures correctness at prefix/leaf
    # boundaries); zero out the last position of each leaf segment (EOS crossing guard).
    tree_packed_labels_tensor = torch.zeros_like(tree_packed_tokens)
    if tree_packed_tokens.numel() > 1:
        tree_packed_labels_tensor[:-1] = tree_packed_tokens[1:]
    for leaf in leaves_in_dfs:
        end = leaf._flat_end
        if end > leaf._flat_start and end <= tree_packed_tokens.numel():
            tree_packed_labels_tensor[end - 1] = 0
    if flat_pid_pieces is not None:
        tree_packed_position_ids = torch.cat(flat_pid_pieces)
    else:
        tree_packed_position_ids = (
            torch.cat(default_pid_pieces) if default_pid_pieces else torch.empty(0, dtype=torch.long, device=device)
        )

    # Build leaf ranges and ancestor chains in DFS order, interleaving zero-length
    # duplicate entries immediately after their representative so the last entry
    # (the last real leaf) still ends at total_seqlen_q (satisfying PrefixTreeParams).
    leaf_ranges: list[RangeSpec] = []
    leaf_to_sample_list: list[int] = []
    leaf_ancestor_ranges: list[list[RangeSpec]] = []

    for leaf in leaves_in_dfs:
        chain: list[RangeSpec] = []
        cur = parent_of.get(leaf.flat_idx)
        while cur is not None:
            chain.append((cur._flat_start, cur._flat_end))
            cur = parent_of.get(cur.flat_idx)
        chain.reverse()  # root first

        rep_range: RangeSpec = (leaf._flat_start, leaf._flat_end)
        sids = leaf_node_id_to_samples[leaf.flat_idx]

        # Representative entry
        leaf_ranges.append(rep_range)
        leaf_to_sample_list.append(sids[0])
        leaf_ancestor_ranges.append(chain)

        # Zero-length entries for duplicates: ancestor chain extended with the rep's
        # leaf range so restore_flat_to_nested reconstructs the full sequence correctly.
        zero_range: RangeSpec = (rep_range[1], rep_range[1])
        for dup_sid in sids[1:]:
            leaf_ranges.append(zero_range)
            leaf_to_sample_list.append(dup_sid)
            leaf_ancestor_ranges.append(chain + [rep_range])

    sample_to_leaf_range = {s: r for s, r in zip(leaf_to_sample_list, leaf_ranges, strict=False)}

    # prefix_range: the shared root segment (first root_node for single-prefix trees)
    prefix_range = (root_nodes[0]._flat_start, root_nodes[0]._flat_end)

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
    )
    params._leaf_ancestor_ranges = leaf_ancestor_ranges
    return params
