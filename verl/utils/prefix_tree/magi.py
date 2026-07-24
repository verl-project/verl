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
"""Prefix-tree + MAGI utilities for verl SFT training.

Dispatches a micro-batch through either the hash-based
(:mod:`verl.utils.prefix_tree.segment_grouper`) or dynamic-trie
(:mod:`verl.utils.prefix_tree.dynamic`) detection path, materialises a flat
layout via :func:`verl.utils.prefix_tree.utils.build_layout_from_tree_node`,
and builds a MAGI / flex attention key for the result.

Usage (inside gptmodel_forward_model_engine):

    pt_batch = build_prefix_tree_micro_batch(model, input_ids, loss_mask, position_ids)
    if pt_batch is not None:
        output = model(
            input_ids=pt_batch.tree_packed_input_ids,
            attention_mask=None,
            position_ids=pt_batch.tree_packed_position_ids,
            packed_seq_params=None,
            magi_attention_key=pt_batch.magi_key,
        )
        output = restore_flat_to_nested(output, pt_batch)
"""

from __future__ import annotations

import contextlib
import functools
import logging as _log
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.prefix_tree.dynamic import build_tree_dynamic
from verl.utils.prefix_tree.tree import PrefixSubTrie
from verl.utils.prefix_tree.utils import build_layout_from_tree_node


@dataclass
class PrefixTreeMagiBatch:
    """Holds the tree-packed layout and MAGI key for one prefix-tree micro-batch."""

    # tree-packed input tensors ready to pass to model(...)
    tree_packed_input_ids: Tensor  # (total_tokens,)
    tree_packed_position_ids: Tensor  # (total_tokens,)

    # Attention keys: one will be None depending on prefix_tree_attention setting
    magi_key: object  # MAGI key (None when using flex)
    flex_key: object  # flex_attention block_mask (None when using magi)

    # mapping needed for output restoration
    # segment_to_sample[i] = original sample index for leaf i
    segment_to_sample: list[int]
    # segment_ranges[i] = (start, end) token offset in flat layout for leaf i
    segment_ranges: list[tuple[int, int]]
    prefix_range: tuple[int, int]

    # original batch size (= number of leaves for single-level tree)
    original_batch_size: int

    # per-token labels derived from tree_packed_tokens via within-segment shift
    tree_packed_labels: Optional[Tensor] = None  # (total_tokens,)

    # number of real (non-padding) tokens; may be < tree_packed_input_ids.shape[0]
    # when tp_size > 1 padding was added for sequence-parallel divisibility
    real_tokens: int = 0

    # ancestor_segment_ranges[i] = list of (start,end) flat ranges that precede leaf i
    # For single-level: None (use prefix_range directly)
    # For multilevel: [(0, root_end), (turn2_start, turn2_end)] etc.
    ancestor_segment_ranges: Optional[list[list[tuple[int, int]]]] = None

    # CP-local tensors: after magi dispatch, each CP rank only processes its assigned tokens.
    # When CP=1, these equal tree_packed_input_ids/tree_packed_position_ids.
    # Shape: (local_tokens, ...) where local_tokens = total_tokens / cp_effective
    local_tree_packed_input_ids: Optional[Tensor] = None
    local_tree_packed_position_ids: Optional[Tensor] = None

    def __post_init__(self):
        if self.real_tokens == 0:
            self.real_tokens = int(self.tree_packed_input_ids.shape[0])
        # Default local to full when not set (CP=1 or flex path)
        if self.local_tree_packed_input_ids is None:
            self.local_tree_packed_input_ids = self.tree_packed_input_ids
        if self.local_tree_packed_position_ids is None:
            self.local_tree_packed_position_ids = self.tree_packed_position_ids


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
    """Build a PrefixTreeMagiBatch from a micro-batch using a per-mb subtrie.

    The subtrie is produced once per training step in
    ``prepare_prefix_tree_micro_batches``:
      1. ``greedy_build_tries`` builds a global trie from ALL batch samples.
      2. ``subtrie_view`` prunes it to this mb's sample subset → the subtrie.

    The subtrie is then reused across all forward passes (OLP + actor update)
    for this mb without rebuilding.

    Returns None when the subtrie is not available (prefix sharing not
    detected or dynamic bsz disabled), signalling the caller to fall back
    to standard attention.

    Args:
        model: Megatron model (used to read num_heads / head_dim from config).
        input_ids: NestedTensor of shape (batch_size, variable_seqlen).
        loss_mask: Optional NestedTensor matching input_ids shape.
        position_ids: Optional NestedTensor matching input_ids shape.
        attention_type: ``"flex"`` or ``"magi"``.
        tp_size / cp_size: Tensor / context parallel world sizes.
        subtrie: Per-mb subtrie from ``prepare_prefix_tree_micro_batches``.

    Returns:
        PrefixTreeMagiBatch or None.
    """
    # Lazy import: ``_unpack_nested_to_list`` and ``_finalize_prefix_tree_batch``
    # live in :mod:`verl.utils.prefix_tree.forward`, which imports this module at
    # load time; a top-level import here would create a cycle.
    from verl.utils.prefix_tree.forward import (
        _finalize_prefix_tree_batch,
        _unpack_nested_to_list,
    )

    samples = _unpack_nested_to_list(input_ids, mask=loss_mask)
    if not samples:
        _log.getLogger(__name__).warning("prefix_tree: build_prefix_tree_micro_batch got empty samples; returning None")
        return None
    loss_masks_by_sample = _unpack_nested_to_list(loss_mask)
    position_ids_by_sample = _unpack_nested_to_list(position_ids, mask=loss_mask)

    if subtrie is None:
        # No pre-built subtrie (e.g. use_dynamic_bsz=False): build locally.
        # build_tree_dynamic does token-by-token trie detection on this mb's
        # samples; slower than the global path but correct.
        subtrie = build_tree_dynamic(samples)

    if subtrie is None:
        _log.getLogger(__name__).error(
            "build_prefix_tree_micro_batch: no prefix sharing found (n=%d); falling back to standard attention",
            len(samples),
        )
        return None

    try:
        params = build_layout_from_tree_node(
            samples,
            subtrie,
            loss_masks_by_sample=loss_masks_by_sample,
            position_ids_by_sample=position_ids_by_sample,
        )
        return _finalize_prefix_tree_batch(
            params,
            model=model,
            num_samples=len(samples),
            attention_type=attention_type,
            tp_size=tp_size,
            cp_size=cp_size,
            subtrie=subtrie,
        )
    except (ValueError, KeyError, IndexError) as _e:
        _log.getLogger(__name__).exception(
            "build_prefix_tree_micro_batch: falling back to standard attention (%s: %s) "
            "subtrie_nodes=%d subtrie_leaves=%d",
            type(_e).__name__,
            _e,
            len(subtrie.nodes) if subtrie is not None else -1,
            len(subtrie.leaf_node_ids) if subtrie is not None else -1,
        )
        return None


def _build_sample_tensors(flat_tensor: Tensor, pt_batch: PrefixTreeMagiBatch) -> list:
    """Build a per-sample list of tensors from a flat deduplicated tensor.

    Returns sample_tensors[sample_idx] = cat(ancestor_slices..., leaf_slice).
    """
    prefix_start, prefix_end = pt_batch.prefix_range
    prefix_slice = flat_tensor[prefix_start:prefix_end]
    n = pt_batch.original_batch_size
    sample_tensors: list[Optional[Tensor]] = [None] * n
    for leaf_idx, sample_idx in enumerate(pt_batch.segment_to_sample):
        s, e = pt_batch.segment_ranges[leaf_idx]
        leaf_slice = flat_tensor[s:e]
        if pt_batch.ancestor_segment_ranges is not None:
            parts = [flat_tensor[a:b] for a, b in pt_batch.ancestor_segment_ranges[leaf_idx]]
            parts.append(leaf_slice)
            sample_tensors[sample_idx] = torch.cat(parts, dim=0)
        else:
            sample_tensors[sample_idx] = torch.cat([prefix_slice, leaf_slice], dim=0)
    return sample_tensors


def restore_flat_to_nested(
    flat_tensor: Tensor,
    pt_batch: PrefixTreeMagiBatch,
) -> NestedTensor:
    """Restore a flat (total_tokens, ...) tensor to a per-sample NestedTensor.

    Each sample's view is ``[prefix_tokens || ancestor_tokens... || leaf_tokens]``
    concatenated, matching the original per-sample sequence length.

    Args:
        flat_tensor: Tensor with first dimension == total_tokens.
        pt_batch: PrefixTreeMagiBatch from build_prefix_tree_micro_batch.

    Returns:
        NestedTensor of shape (batch_size, variable_seqlen, ...).
    """
    sample_tensors = _build_sample_tensors(flat_tensor, pt_batch)
    assert all(t is not None for t in sample_tensors), (
        "restore_flat_to_nested: some sample indices were not covered by segment_to_sample"
    )
    # as_nested_tensor (not nested_tensor) preserves grad_fn through the cat ops.
    return torch.nested.as_nested_tensor(sample_tensors, layout=torch.jagged)


@contextlib.contextmanager
def prefix_tree_rope_context(model, position_ids: Optional[Tensor]):
    """Override ``rotary_pos_emb.forward`` to use per-token *position_ids*.

    Shared by both fused and unfused prefix-tree paths.  Megatron's default
    RoPE slicing assumes each CP rank holds sequential positions
    ``[r·T/CP .. (r+1)·T/CP]``; after MAGI dispatch each rank holds
    non-sequential tokens whose ``position_ids`` are arbitrary.

    Two RoPE families are handled:

    - **Standard ``RotaryEmbedding``** (``forward(max_seq_len, ...)``): builds
      a sequential table ``[0..max_seq_len-1]``.  The override builds the full
      table (``cp_group=None``) and indexes it by ``position_ids``.
    - **M-RoPE** (``Qwen3VLMultimodalRotaryEmbedding`` /
      ``MultimodalRotaryEmbedding``: ``forward(position_ids, mrope_section)``):
      builds freqs directly from ``position_ids``, so each token's RoPE is
      whatever ``position_ids`` says.  The override broadcasts the 1D per-token
      positions to 3D ``[3, 1, T]`` (text-only: all three dims identical) and
      passes them to the original forward with ``cp_group=None`` (MAGI already
      dispatched; no internal CP slicing).  No table indexing needed.

    No-op when ``model`` has no ``rotary_pos_emb`` or ``position_ids`` is None.
    """
    rope_mod = getattr(model, "rotary_pos_emb", None)
    if rope_mod is None or position_ids is None:
        yield
        return

    pids = position_ids.reshape(-1)
    _real_rope_fwd = rope_mod.forward

    # M-RoPE modules don't have get_emb; their forward takes position_ids (3D)
    # instead of max_seq_len (int).  Qwen3.5 text-only uses this path.
    _is_mrope = not hasattr(rope_mod, "get_emb")

    if _is_mrope:
        _mrope_section = getattr(model, "mrope_section", None)

        def _rope_fwd_with_pids(*args, **kwargs):
            # Broadcast 1D per-token positions to [3, 1, T]; text-only: all
            # three M-RoPE dims (temporal/height/width) are identical.
            pids_3d = pids.view(1, 1, -1).expand(3, 1, -1).contiguous()
            return _real_rope_fwd(pids_3d, _mrope_section, cp_group=None)
    else:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        _orig_rope_fn = RotaryEmbedding.forward.__wrapped__  # bypass lru_cache

        def _rope_fwd_with_pids(max_seq_len, offset=0, packed_seq=False, cp_group=None):
            actual_seq_len = int(pids.max().item()) + 1
            emb = _orig_rope_fn(rope_mod, actual_seq_len, offset=0, packed_seq=True, cp_group=None)
            # All PP stages use seq-first Q=(seq,1,H,D); freqs=(seq,1,1,dim)
            # broadcasts correctly: Q×freqs → (seq,1,H,D).
            indexed = emb[pids.to(emb.device)]
            return indexed

    rope_mod.forward = _rope_fwd_with_pids
    try:
        yield
    finally:
        rope_mod.forward = _real_rope_fwd


@contextlib.contextmanager
def prefix_tree_decoder_key_context(model, magi_attention_key=None, flex_attention_key=None):
    """Override ``model.decoder.forward`` to inject the attention key.

    Shared by both fused and unfused prefix-tree paths.  The decoder's forward
    signature doesn't accept ``magi_attention_key`` / ``flex_attention_key``;
    the patched TEDotProductAttention reads them from its module's forward
    kwargs.  This context wraps ``decoder.forward`` to inject the keys for the
    duration of one call.
    """
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
        "response_attention_mask",
    }
)


def strip_prefix_tree_args(logits_processor_args: dict | None) -> None:
    """Remove prefix-tree keys from *logits_processor_args* (mutates dict).

    Called after the prefix-tree path has consumed them so they don't
    leak into the downstream logits processor.
    """
    if logits_processor_args is None:
        return
    for k in _PREFIX_TREE_KEYS:
        logits_processor_args.pop(k, None)


def read_prefix_tree_batch_config(batch, tu, use_remove_padding: bool = True) -> tuple[bool, str]:
    """Read and validate prefix-tree flags from a batch non-tensor dict.

    Returns (use_prefix_tree, prefix_tree_attention).
    """
    use_prefix_tree = tu.get_non_tensor_data(batch, key="use_prefix_tree", default=False)
    prefix_tree_attention = tu.get_non_tensor_data(batch, key="prefix_tree_attention", default="flex")
    if use_prefix_tree:
        assert use_remove_padding, (
            "use_prefix_tree=True requires use_remove_padding=True (THD format). "
            "Set model.use_remove_padding=True in your config."
        )
        assert prefix_tree_attention in ("flex", "magi"), (
            f"prefix_tree_attention must be 'flex' or 'magi', got {prefix_tree_attention!r}"
        )
    return use_prefix_tree, prefix_tree_attention


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
