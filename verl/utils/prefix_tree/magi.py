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
"""Prefix-tree + MAGI utilities: flat layout packing, MAGI/flex keys,
rope/decoder overrides, and flat→nested restore."""

from __future__ import annotations

import contextlib
import functools
import logging as _log
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.prefix_tree.tree import PrefixSubTrie
from verl.utils.prefix_tree.utils import build_layout_from_tree_node

try:
    from verl.utils.megatron_utils import unwrap_model
except ImportError:  # local dev without megatron; tests monkeypatch this symbol

    def unwrap_model(m):
        return m


@dataclass
class PackRestorationParam:
    """Per-micro-batch layout info for restoring flat tensors to per-sample."""

    segment_ranges: list[tuple[int, int]]
    prefix_range: tuple[int, int]
    ancestor_segment_ranges: Optional[list[list[tuple[int, int]]]] = None
    boundary_registry: Optional[object] = None


@dataclass
class PrefixTreeMagiBatch:
    """Holds the tree-packed layout and MAGI key for one prefix-tree micro-batch."""

    # tree-packed input tensors ready to pass to model(...)
    tree_packed_input_ids: Tensor  # (total_tokens,)
    tree_packed_position_ids: Tensor  # (total_tokens,)

    # Attention keys: one will be None depending on prefix_tree_attention setting
    magi_key: object  # MAGI key (None when using flex)
    flex_key: object  # flex_attention block_mask (None when using magi)

    # Per-token labels derived from tree_packed_tokens via within-segment shift
    tree_packed_labels: Optional[Tensor] = None  # (total_tokens,)

    # Number of real (non-padding) tokens; may be < tree_packed_input_ids.shape[0]
    # when _finalize pads to TP/CP divisibility. Use this (not shape[0]) to strip
    # padding before restore/undispatch, else padding log-probs leak into samples.
    real_tokens: int = 0

    # Per-sample rolled labels (each sample's tokens shifted left, 0-padded at the
    # end) used by the unfused path to compute per-sample log-probs without the
    # flat boundary patch.
    per_sample_labels: Optional[list[Tensor]] = None

    # Restoration params for unpacking model output to per-sample tensors
    restoration: Optional[PackRestorationParam] = None

    # Live subtrie reference for restoration lookups (leaf_to_sample, etc.)
    subtrie: Optional[object] = None


def build_prefix_tree_micro_batch(
    model,
    input_ids: NestedTensor,
    loss_mask: Optional[NestedTensor] = None,
    position_ids: Optional[NestedTensor] = None,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
    subtrie: Optional[PrefixSubTrie] = None,
) -> Optional[PrefixTreeMagiBatch]:
    """Build a PrefixTreeMagiBatch from a per-mb subtrie (built once per step in prepare_prefix_tree_micro_batches).

    The subtrie is reused across OLP + actor update forwards. Returns None when subtrie absent."""
    # Lazy import to avoid cycle with forward.py.
    from verl.utils.prefix_tree.forward import _finalize_prefix_tree_batch, _unpack_nested_to_list

    samples = _unpack_nested_to_list(input_ids, mask=loss_mask)
    if not samples:
        _log.getLogger(__name__).warning("prefix_tree: build_prefix_tree_micro_batch got empty samples; returning None")
        return None
    loss_masks_by_sample = _unpack_nested_to_list(loss_mask)
    position_ids_by_sample = _unpack_nested_to_list(position_ids, mask=loss_mask)

    if subtrie is None:
        raise RuntimeError(
            "build_prefix_tree_micro_batch: prefix_tree_subtree is None. The global "
            "trie was not built/transmitted to this worker (build_global_trie not called "
            "on the driver, or prefix_tree_subtree did not survive dispatch). Per-microbatch "
            "rebuild is disabled — fix the driver to attach the global trie."
        )

    params = build_layout_from_tree_node(
        samples,
        subtrie,
        loss_masks_by_sample=loss_masks_by_sample,
        position_ids_by_sample=position_ids_by_sample,
    )
    pb = _finalize_prefix_tree_batch(
        params,
        model=model,
        num_samples=len(samples),
        attention_type=attention_type,
        tp_size=tp_size,
        cp_size=cp_size,
        subtrie=subtrie,
    )
    pb.per_sample_labels = [torch.cat([s[1:], torch.zeros(1, dtype=s.dtype, device=s.device)]) for s in samples]
    return pb


def _build_per_sample_tensor(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
    boundary_logps: Optional[dict[int, list[tuple[int, Tensor]]]] = None,
) -> list:
    """Build per-sample tensor list from flat tensor, applying per-leaf boundary log-prob patches via split-and-cat.

    boundary_logps maps sample_idx → [(boundary_flat_pos, per_leaf_logp), ...] to fix the
    dedup→one-slot→copy-to-all-leaves bug where non-owner leaves inherit wrong boundary log-probs."""
    # One entry per sample (duplicates sharing a leaf included); output slot per sample, None until filled.
    n = (max(pt_batch.subtrie.leaf_to_sample) + 1) if pt_batch.subtrie.leaf_to_sample else 0
    sample_tensors: list[Optional[Tensor]] = [None] * n
    # Packed-sequence ranges of the shared trie nodes on each leaf's root->parent path.
    ancestor_ranges = pt_batch.restoration.ancestor_segment_ranges
    if ancestor_ranges is None:
        ancestor_ranges = [[pt_batch.restoration.prefix_range] for _ in range(n)]
    # iterate each sample, from leaf
    for leaf_idx, sample_idx in enumerate(pt_batch.subtrie.leaf_to_sample):
        leaf_start, leaf_end = pt_batch.restoration.segment_ranges[leaf_idx]
        leaf_slice = flat_tensor[leaf_start:leaf_end]
        pieces: list[Tensor] = []
        boundary_tokens = boundary_logps.get(sample_idx, []) if boundary_logps is not None else []
        # iterate the segment to build the entire sequence
        for start, end in ancestor_ranges[leaf_idx]:
            patched = False
            for pos, leaf_val in boundary_tokens:
                # find the correct boundary token apply here
                if start <= pos < end:
                    pieces.extend([flat_tensor[start:pos], leaf_val.reshape(1)])
                    patched = True
                    break
            if not patched:
                pieces.append(flat_tensor[start:end])
        pieces.append(leaf_slice)
        # torch cat the entire pieces togather
        sample_tensors[sample_idx] = torch.cat(pieces, dim=0)
    return sample_tensors


def restore_flat_to_nested(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
    apply_boundary_patch: bool = False,
) -> NestedTensor:
    """Restore flat tensor to per-sample NestedTensor. Set apply_boundary_patch=True for log_probs."""
    boundary_logps = None
    if apply_boundary_patch:
        boundary_logps = getattr(pt_batch, "_boundary_logps", None)
    sample_tensors = _build_per_sample_tensor(flat_tensor, pt_batch, boundary_logps=boundary_logps)
    if not all(t is not None for t in sample_tensors):
        raise RuntimeError("restore_flat_to_nested: some sample indices were not covered by segment_to_sample")
    # as_nested_tensor (not nested_tensor) preserves grad_fn through the cat ops.
    return torch.nested.as_nested_tensor(sample_tensors, layout=torch.jagged)


def set_rope_pids(model, position_ids: Optional[Tensor]) -> None:
    """Set per-token position_ids on model.rotary_pos_emb for global RoPE patch to read."""
    rope_mod = getattr(unwrap_model(model), "rotary_pos_emb", None)
    if rope_mod is not None and position_ids is not None:
        rope_mod._pids = position_ids.reshape(-1)


def clear_rope_pids(model) -> None:
    rope_mod = getattr(unwrap_model(model), "rotary_pos_emb", None)
    if rope_mod is not None:
        rope_mod._pids = None


@contextlib.contextmanager
def prefix_tree_decoder_key_context(model, magi_attention_key=None, flex_attention_key=None):
    """Override model.decoder.forward to inject magi/flex attention key into kwargs for one call."""
    if magi_attention_key is None and flex_attention_key is None:
        yield
        return
    _real_decoder_forward = model.decoder.forward

    @functools.wraps(_real_decoder_forward)
    def _decoder_forward_with_key(*args, **kw):
        return _real_decoder_forward(
            *args,
            magi_attention_key=magi_attention_key,
            flex_attention_key=flex_attention_key,
            **kw,
        )

    model.decoder.forward = _decoder_forward_with_key
    try:
        yield
    finally:
        model.decoder.forward = _real_decoder_forward


# model-forward helpers: consumed by verl/models/mcore/model_forward.py


_PREFIX_TREE_KEYS = frozenset(
    {
        "loss_mask",
        "use_prefix_tree",
        "prefix_tree_attention",
        "prefix_tree_subtree",
    }
)


def strip_prefix_tree_args(logits_processor_args: dict | None) -> None:
    """Remove prefix-tree specific keys"""
    if logits_processor_args is None:
        return
    for k in _PREFIX_TREE_KEYS:
        logits_processor_args.pop(k, None)


def get_prefix_tree_logits_args(batch, tu) -> dict:
    """Build the prefix-tree fragment for logits_processor_args from a batch.

    The per-mb subtrie (built once in prepare_prefix_tree_micro_batches as
    a pruned view of the global trie) is the only thing needed here.
    """
    use_prefix_tree = tu.get_non_tensor_data(batch, key="use_prefix_tree", default=False)
    if not use_prefix_tree:
        return {}
    return {
        "use_prefix_tree": True,
        "prefix_tree_attention": tu.get_non_tensor_data(batch, key="prefix_tree_attention", default="flex"),
        "prefix_tree_subtree": tu.get_non_tensor_data(batch, "prefix_tree_subtree", default=None),
    }
