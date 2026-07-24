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
"""Prefix-tree forward-path implementations.

Split out of :mod:`verl.utils.prefix_tree.magi` to keep the data-structure /
config helpers separate from the actual forward-pass code.  Everything in this
module is consumed only by the prefix-tree forward path
(``verl.models.mcore.model_forward`` and ``model_forward_fused``) or by other
functions in this module.

Public entry points:

- :func:`unfuse_try_forward_prefix_tree`: unfused-path driver.
- :func:`fuse_try_forward_prefix_tree`: fused-path driver.
- :func:`fuse_forward_body`: fused-path body invoked by the patched
  ``_fused_GPTModel_forward``.
- :func:`dispatch_magi` (renamed from ``dispatch_pt_batch``): slices
  per-CP-rank local tensors via magi dispatch.
"""

from __future__ import annotations

import logging as _log
from typing import Optional

import torch
import torch.distributed as _dist
from magi_attention.api import (
    DistAttnConfig,
    get_position_ids,
    magi_attn_flex_key,
    undispatch,
)
from magi_attention.common import AttnRanges
from magi_attention.common.enum import AttnMaskType
from magi_attention.meta.solver.dispatch_solver import DispatchConfig
from megatron.core import parallel_state as mpu
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask

from verl.utils.megatron_utils import unwrap_model
from verl.utils.prefix_tree.magi import (
    PrefixTreeMagiBatch,
    build_prefix_tree_micro_batch,
    prefix_tree_decoder_key_context,
    prefix_tree_rope_context,
    restore_flat_to_nested,
    strip_prefix_tree_args,
)

# ---------------------------------------------------------------------------
# Shared helpers (extracted from the forward functions below)
# ---------------------------------------------------------------------------


def _prepare_attn_inputs(
    pb: PrefixTreeMagiBatch,
    prefix_tree_attention: str,
) -> tuple[Tensor, Tensor, dict]:
    """Build local input ids / position ids + attention kwargs for one forward.

    Shared by :func:`unfuse_forward_prefix_tree` and
    :func:`fuse_try_forward_prefix_tree`.  For the ``magi`` branch the returned
    tensors are CP-local slices obtained via :func:`dispatch_magi`; for the
    ``flex`` branch they are the full tree-packed tensors with a leading
    batch dim.  The caller is responsible for wrapping the magi branch in
    :func:`prefix_tree_rope_context` if needed.
    """
    if prefix_tree_attention == "magi":
        local_input_ids, local_position_ids = dispatch_magi(pb)
        attn_kwargs = {"magi_attention_key": pb.magi_key}
    else:
        local_input_ids = pb.local_tree_packed_input_ids.unsqueeze(0)
        local_position_ids = pb.local_tree_packed_position_ids.unsqueeze(0)
        attn_kwargs = {"flex_attention_key": pb.flex_key}
    return local_input_ids, local_position_ids, attn_kwargs


def _restore_to_nested_per_sample(
    flat_tensor: Tensor,
    pb: PrefixTreeMagiBatch,
) -> Tensor:
    """Restore a flat dedup tensor to per-sample nested (jagged) format.

    Returns a NestedTensor matching non-tree model output: per-sample
    constituents are prefix + ancestors + leaf concatenated, with DP-padding
    tokens excluded. ``postprocess_batch_func`` and ``no_padding_2_padding``
    handle this identically to origin's nested output.
    """
    return restore_flat_to_nested(flat_tensor, pb)


def _expand_temperature(t, pt_batch: PrefixTreeMagiBatch, total_flat: int, device) -> Tensor:
    """Expand a temperature spec to a ``(total_flat, 1)`` per-token tensor.

    Handles three cases:
      * NestedTensor (per-sample): fill every token (prefix, each leaf, and
        each leaf's ancestor chain) with the sample's temperature.
      * Scalar ``Tensor``: broadcast via ``torch.full``.
      * Plain scalar (``float`` / ``int``): same broadcast.

    When ``t`` is None a ones tensor is returned (prior default).
    """
    if t is None:
        return torch.ones(total_flat, 1, dtype=torch.float32, device=device)
    if isinstance(t, torch.Tensor) and t.is_nested:
        # Per-sample temperature: expand to match tree-packed structure.
        # The flat layout contains prefix root + internal ancestor nodes +
        # leaf nodes, so we must fill every token: covering prefix, each
        # leaf, and each leaf's ancestor chain (ancestor_segment_ranges).
        # Missing the internal ancestor tokens shrinks the cat below total_flat.
        temp_by_sample = t.values()  # (batch_size,)
        tree_packed_t = torch.ones(total_flat, 1, dtype=torch.float32, device=device)
        for leaf_idx, sample_idx in enumerate(pt_batch.segment_to_sample):
            t_val = temp_by_sample[sample_idx].item()
            if pt_batch.ancestor_segment_ranges is not None:
                for a, b in pt_batch.ancestor_segment_ranges[leaf_idx]:
                    if b > a:
                        tree_packed_t[a:b] = t_val
            s, e = pt_batch.segment_ranges[leaf_idx]
            if e > s:
                tree_packed_t[s:e] = t_val
        # Shared prefix keeps sample[0]'s temp (prior convention); refill
        # last so ancestor writes from other leaves don't override it.
        prefix_start, prefix_end = pt_batch.prefix_range
        if prefix_end > prefix_start:
            tree_packed_t[prefix_start:prefix_end] = temp_by_sample[0].item()
        return tree_packed_t
    if isinstance(t, torch.Tensor):
        scalar_t = t.flatten()[0].item()
        return torch.full((total_flat, 1), scalar_t, dtype=torch.float32, device=device)
    scalar_t = float(t)
    return torch.full((total_flat, 1), scalar_t, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Low-level builders
# ---------------------------------------------------------------------------


def _unpack_nested_to_list(x, pad_token_id=None, mask: Optional[Tensor] = None) -> Optional[list[Tensor]]:
    """Unpack a NestedTensor or padded 2-D Tensor into a list of 1-D tensors.

    - NestedTensor (jagged): uses ``.offsets()``
    - Padded 2-D Tensor ``(B, T)``:
      * If ``mask`` is provided: uses ``mask.sum(dim=-1).tolist()`` as
        sequence lengths
      * If ``mask`` is None: returns None (cannot safely unpack)
    - ``None``: returns ``None``
    """
    if x is None:
        return None
    if hasattr(x, "is_nested") and x.is_nested:
        offsets = x.offsets()
        lengths = offsets.diff().tolist()
        vals = x.values()
        out: list[Tensor] = []
        pos = 0
        for length in lengths:
            out.append(vals[pos : pos + int(length)])
            pos += int(length)
        return out
    if x.dim() == 2:
        if mask is not None:
            seqlens = mask.sum(dim=-1).tolist()
            return [x[i, : int(seqlens[i])] for i in range(x.shape[0])]
        return None
    return None


def _build_flex_key(params, device):
    """Build a torch flex_attention block_mask from PrefixTreeParams.

    The mask encodes the prefix-tree attention pattern:
    - Prefix tokens: causal self-attention
    - Leaf tokens: full attention to prefix + causal self-attention within same leaf
    - Cross-leaf attention: blocked (leaf_i cannot see leaf_j)

    Returns a compiled block_mask usable with torch.nn.attention.flex_attention.
    """
    total = params.total_seqlen_q
    prefix_end = params.prefix_range[1]  # == prefix_len

    leaf_id = torch.full((total,), -1, dtype=torch.int32)
    for i, (s, e) in enumerate(params.segment_ranges):
        leaf_id[s:e] = i
    leaf_id = leaf_id.to(device)

    def prefix_tree_mask(b, h, q_idx, kv_idx):
        q_leaf = leaf_id[q_idx]
        k_leaf = leaf_id[kv_idx]
        in_prefix_k = kv_idx < prefix_end
        same_leaf = (q_leaf == k_leaf) & (q_leaf >= 0)
        causal = kv_idx <= q_idx
        return (in_prefix_k & causal) | (same_leaf & causal) | (in_prefix_k & (q_leaf >= 0))

    # _compile=False: avoid Triton JIT which takes minutes for new shapes.
    # Memory is handled at the call site via torch.utils.checkpoint.
    block_mask = create_block_mask(
        prefix_tree_mask, B=None, H=None, Q_LEN=total, KV_LEN=total, device=device, _compile=False
    )
    block_mask._leaf_id = leaf_id  # keep closure alive
    return block_mask


def _build_magi_key(model, params):
    """Construct a magi_attn_flex_key from PrefixTreeParams and model config."""

    cfg = unwrap_model(model).config
    tp_size = mpu.get_tensor_model_parallel_world_size()
    # Per-rank head counts: ColumnParallelLinear (linear_qkv) shards heads
    # across TP ranks, so each rank's Q/KV tensors hold heads/tp heads.
    # The kernel reads head counts from q.size(1)/k.size(1), but the key's
    # num_heads_q must match for the flatten_head_groups path (enabled via
    # MAGI_ATTENTION_FLATTEN_HEAD_GROUPS=1) which asserts equality.
    num_heads_q = cfg.num_attention_heads // tp_size
    # GQA: num_query_groups may be set; fall back to num_attention_heads (full) if not
    num_heads_kv = (getattr(cfg, "num_query_groups", None) or cfg.num_attention_heads) // tp_size
    head_dim = cfg.kv_channels  # hidden_size // num_attention_heads

    try:
        cp_group = mpu.get_context_parallel_group()
    except Exception:
        cp_group = _dist.group.WORLD

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges(params.q_ranges),
        k_ranges=AttnRanges.from_ranges(params.k_ranges),
        attn_mask_type=[AttnMaskType(m) for m in params.mask_types],
        total_seqlen_q=params.total_seqlen_q,
        total_seqlen_k=params.total_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        dist_attn_config=DistAttnConfig(dispatch_config=DispatchConfig(uneven_shard=True)),
    )


def _finalize_prefix_tree_batch(
    params,
    model,
    num_samples: int,
    attention_type: str = "flex",
    tp_size: int = 1,
    cp_size: int = 1,
    subtrie=None,
) -> PrefixTreeMagiBatch:
    """Common downstream step for both detection paths.

    Pads to TP/CP divisibility, builds the requested attention key, and wraps
    the result into a :class:`PrefixTreeMagiBatch`. Padding tokens are not
    added to the attention rectangles; they are stripped before loss, and
    MAGI assigns zero attention weight to out-of-range positions.
    """
    real_tokens = params.tree_packed_tokens.shape[0]
    if tp_size > 1:
        align_size = (tp_size * cp_size * 2) if cp_size > 1 else tp_size
        pad_len = (align_size - real_tokens % align_size) % align_size
        if pad_len > 0:
            params.tree_packed_tokens = torch.cat(
                [params.tree_packed_tokens, params.tree_packed_tokens.new_zeros(pad_len)]
            )
            params.tree_packed_position_ids = torch.cat(
                [params.tree_packed_position_ids, params.tree_packed_position_ids.new_zeros(pad_len)]
            )
            params.total_seqlen_q += pad_len
            params.total_seqlen_k += pad_len

    if attention_type == "magi":
        # Cache the MAGI key on the subtrie: OLP and actor_update process the same
        # micro-batch (same sequences, same seqlen) so the key is valid for both passes.
        # TODO(dynamic-cp): if dynamic_context_parallel is enabled, dump this cache.
        if subtrie is not None and getattr(subtrie, "_cached_magi_key", None) is not None:
            magi_key = subtrie._cached_magi_key
        else:
            magi_key = _build_magi_key(model, params)
            if subtrie is not None:
                subtrie._cached_magi_key = magi_key
        flex_key = None
    else:
        flex_key = _build_flex_key(params, params.tree_packed_tokens.device)
        magi_key = None

    return PrefixTreeMagiBatch(
        tree_packed_input_ids=params.tree_packed_tokens,
        tree_packed_position_ids=params.tree_packed_position_ids,
        tree_packed_labels=params.tree_packed_labels,
        magi_key=magi_key,
        flex_key=flex_key,
        segment_to_sample=params.leaf_to_sample,
        segment_ranges=params.leaf_ranges,
        prefix_range=params.prefix_range,
        original_batch_size=num_samples,
        real_tokens=real_tokens,
        ancestor_segment_ranges=getattr(params, "_leaf_ancestor_ranges", None),
        local_tree_packed_input_ids=params.tree_packed_tokens,
        local_tree_packed_position_ids=params.tree_packed_position_ids,
    )


def dispatch_magi(pt_batch: PrefixTreeMagiBatch) -> tuple[Tensor, Tensor]:
    """Slice local_input_ids / local_position_ids from tree-packed tensors via magi dispatch.

    Shared by both fused and unfused paths.  Each CP rank processes only its
    assigned token slice through embedding / FFN / layer norms; cross-rank
    attention is handled by ``calc_attn`` inside the patched attention layer.
    When CP=1, ``local_indices`` covers all tokens.

    Args:
        pt_batch: PrefixTreeMagiBatch with a non-None ``magi_key``.

    Returns:
        (local_input_ids (1, local_tokens), local_position_ids (1, local_tokens)).
    """
    local_indices = get_position_ids(pt_batch.magi_key)
    local_input_ids = pt_batch.tree_packed_input_ids[local_indices].unsqueeze(0)
    local_position_ids = pt_batch.tree_packed_position_ids[local_indices].unsqueeze(0)
    return local_input_ids, local_position_ids


# ---------------------------------------------------------------------------
# Forward-path drivers
# ---------------------------------------------------------------------------


def build_prefix_tree_batch(model, input_ids, logits_processor_args, vision_model, mtp_enable_train):
    """Build prefix-tree micro-batch from *logits_processor_args*.

    Returns :class:`PrefixTreeMagiBatch` or ``None`` when the per-mb subtrie
    is not available.  Caller must gate on use_prefix_tree and skip conditions.
    """
    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")
    loss_mask_nested = (logits_processor_args or {}).get("loss_mask", None)
    position_ids_nested = (logits_processor_args or {}).get("position_ids", None)
    # Per-mb subtrie built once in prepare_prefix_tree_micro_batches (global trie
    # pruned to this mb's samples) and attached to the mb's non-tensor data.
    subtrie = (logits_processor_args or {}).get("prefix_tree_subtree")

    return build_prefix_tree_micro_batch(
        model,
        input_ids,
        loss_mask_nested,
        position_ids=position_ids_nested,
        attention_type=prefix_tree_attention,
        tp_size=mpu.get_tensor_model_parallel_world_size(),
        cp_size=mpu.get_context_parallel_world_size(),
        subtrie=subtrie,
    )


def unfuse_forward_prefix_tree(
    model, pt_batch, prefix_tree_attention, logits_processor, logits_processor_args, post_process, model_kwargs
):
    """Unfused-path: forward pass for prefix-tree batches using magi or flex attention."""
    tree_packed_input_ids = pt_batch.local_tree_packed_input_ids.unsqueeze(0)
    # Use the layout builder's per-sample position IDs (resets within each sample,
    # stays within max_position_embeddings).  torch.arange(flat_tokens) would produce
    # monotonic IDs up to 172437+ which OOB the RoPE embedding table on large batches.
    tree_packed_position_ids = pt_batch.local_tree_packed_position_ids.unsqueeze(0)

    strip_prefix_tree_args(logits_processor_args)

    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pt_batch, prefix_tree_attention)
    if prefix_tree_attention == "magi":
        with prefix_tree_rope_context(model, local_position_ids):
            output_orig = model(
                input_ids=local_input_ids,
                attention_mask=None,
                position_ids=local_position_ids,
                packed_seq_params=None,
                **attn_kwargs,
                **model_kwargs,
            )
    else:
        output_orig = model(
            input_ids=tree_packed_input_ids,
            attention_mask=None,
            position_ids=tree_packed_position_ids,
            packed_seq_params=None,
            **attn_kwargs,
            **model_kwargs,
        )

    real_tokens = pt_batch.real_tokens
    if output_orig.shape[0] == 1:
        output_orig = output_orig[:, :real_tokens]
    else:
        output_orig = output_orig[:real_tokens].permute(1, 0, 2)

    if post_process and logits_processor is not None:
        logits_flat = output_orig.squeeze(0)  # (flat_tokens, vocab)
        tree_packed_ids = pt_batch.tree_packed_input_ids[:real_tokens]  # (flat_tokens,)

        # Labels are derived from tree_packed_tokens via within-segment shift; leaf-end positions are 0.
        tree_packed_label = pt_batch.tree_packed_labels[:real_tokens].unsqueeze(1)

        orig_args = logits_processor_args or {}
        total_flat = tree_packed_ids.shape[0]
        tree_packed_t = _expand_temperature(
            orig_args.get("temperature"), pt_batch, total_flat, tree_packed_label.device
        )
        flat_args = {
            k: v for k, v in orig_args.items() if k not in ("label", "temperature", "loss_mask", "use_prefix_tree")
        }

        # For MAGI: logits are CP-local (local_tokens, vocab). Slice label/temp to match.
        # For flex: logits are full flat (real_tokens, vocab). Use as-is.
        if prefix_tree_attention == "magi":
            local_indices = get_position_ids(pt_batch.magi_key)  # (local_tokens,)
            flat_padded = pt_batch.tree_packed_input_ids.shape[0]
            pad = flat_padded - real_tokens

            def _pad_to_full(x):
                return torch.cat([x, x.new_zeros((pad,) + x.shape[1:])]) if pad > 0 else x

            flat_args["label"] = _pad_to_full(tree_packed_label)[local_indices]
            flat_args["temperature"] = _pad_to_full(tree_packed_t)[local_indices]
            n_logits = local_indices.shape[0]
        else:
            flat_args["label"] = tree_packed_label
            flat_args["temperature"] = tree_packed_t
            n_logits = total_flat

        output_dict = logits_processor(logits_flat.clone().unsqueeze(1), **flat_args)

        if isinstance(output_dict, dict):
            for key, val in output_dict.items():
                if isinstance(val, torch.Tensor):
                    val_1d = val.reshape(-1)
                    if val_1d.shape[0] == n_logits:
                        if prefix_tree_attention == "magi":
                            val_1d = undispatch(val_1d, pt_batch.magi_key)[:real_tokens]
                        output_dict[key] = _restore_to_nested_per_sample(val_1d, pt_batch)
        return output_dict
    else:
        # Intermediate PP stage (post_process=False) or no logits_processor.
        # output_orig is (1, flat_tokens, hidden_dim) after normalization above.
        # Stage 0 transposes BSH→SBHD internally (embedding → seq-first).
        # We must send the same seq-first format (seq, 1, hidden) so all downstream
        # stages also get seq-first Q; no per-stage conditional needed, PP=N safe.
        return output_orig.permute(1, 0, 2)  # (1,seq,hid) → (seq,1,hid)


def unfuse_try_forward_prefix_tree(
    model,
    input_ids,
    logits_processor_args,
    prefix_tree_attention,
    logits_processor,
    post_process,
    model_kwargs,
    vision_model=False,
    mtp_enable_train=False,
):
    """Unfused-path: try to build + forward a prefix-tree batch; returns output dict or None.

    Consolidates build/forward/strip into one call.  Returns None when no
    prefix sharing is detected, in which case prefix-tree keys are stripped
    from *logits_processor_args* so the caller can fall through to the
    standard THD path.
    """
    # Unfused path blocks VLM unconditionally (3D M-RoPE not wired here).
    # Fused path (fuse_try_forward_prefix_tree) only blocks VLM-with-images
    # so text-only VLM batches proceed; see vision_model+has_vision_data guard there.
    if vision_model or mtp_enable_train:
        _log.getLogger(__name__).warning(
            "prefix_tree: skipping prefix-tree path (vision_model=%s, mtp_enable_train=%s); "
            "falling back to standard THD",
            vision_model,
            mtp_enable_train,
        )
        strip_prefix_tree_args(logits_processor_args)
        return None

    pb = build_prefix_tree_batch(
        model,
        input_ids,
        logits_processor_args,
        vision_model,
        mtp_enable_train,
    )
    if pb is not None:
        return unfuse_forward_prefix_tree(
            model,
            pb,
            prefix_tree_attention,
            logits_processor,
            logits_processor_args,
            post_process,
            model_kwargs,
        )

    _log.getLogger(__name__).warning(
        "prefix_tree: build_prefix_tree_batch returned None; falling back to standard THD path "
        "(post_process=%s). If this appears for one PP stage but not the other, the hidden-state "
        "format will mismatch between stages.",
        post_process,
    )
    strip_prefix_tree_args(logits_processor_args)
    return None


# ---------------------------------------------------------------------------
# Fused-path
# ---------------------------------------------------------------------------


def _run_lce(
    hidden_states: Tensor,
    output_weight: Tensor,
    labels: Tensor,
    temperature: float,
    model,
    magi_key=None,
    pt_batch: Optional[PrefixTreeMagiBatch] = None,
) -> tuple[Tensor, Tensor]:
    """Fused LCE for both MAGI and flex branches.

    Shared part: gather from sequence-parallel region (if enabled), then call
    :func:`linear_cross_entropy`.  When ``magi_key`` is given (MAGI branch),
    labels are padded to flat-padded length and sliced by CP-local indices
    before the call, and the 1D outputs are undispatched back to full flat
    order and trimmed to ``real_tokens``.  Without ``magi_key`` (flex branch),
    labels pass through directly and no undispatch is needed.
    """
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy as _lce

    if model.config.sequence_parallel:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    if magi_key is not None:
        local_indices = get_position_ids(magi_key)
        flat_padded = pt_batch.tree_packed_input_ids.shape[0]
        pad = flat_padded - labels.shape[0]
        labels_full = torch.cat([labels, labels.new_zeros(pad)]) if pad > 0 else labels
        lce_labels = labels_full[local_indices]
    else:
        lce_labels = labels

    logprobs, entropy = _lce(
        hidden_states,
        output_weight,
        lce_labels,
        temperature,
        "none",
        mpu.get_tensor_model_parallel_group(),
    )

    if magi_key is not None:
        logprobs = undispatch(logprobs.reshape(-1), magi_key)[: pt_batch.real_tokens]
        entropy = undispatch(entropy.reshape(-1), magi_key)[: pt_batch.real_tokens]
    return logprobs, entropy


def fused_prefix_tree_forward(
    model,
    *,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    labels: Optional[Tensor],
    temperature: float,
    pt_batch,
    decoder_input: Optional[Tensor],
    packed_seq_params,
    extra_block_kwargs: Optional[dict],
    inference_context,
    kwargs: dict,
):
    """Fused-path prefix-tree forward used by the patched ``_fused_GPTModel_forward``.

    Pops ``magi_attention_key`` / ``flex_attention_key`` from ``kwargs`` and installs
    rope + decoder-key contexts before delegating to :func:`fuse_forward_body`.
    Returns ``None`` when both attention keys are absent so the caller falls back
    to the standard fused path.
    """
    _magi_key = kwargs.pop("magi_attention_key", None)
    _flex_key = kwargs.pop("flex_attention_key", None)
    if _magi_key is None and _flex_key is None:
        return None

    with (
        prefix_tree_rope_context(model, position_ids),
        prefix_tree_decoder_key_context(model, _magi_key, _flex_key),
    ):
        return fuse_forward_body(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            temperature=temperature,
            pt_batch=pt_batch,
            magi_key=_magi_key,
            flex_key=_flex_key,
            decoder_input=decoder_input,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )


def fuse_forward_body(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Optional[Tensor],
    labels: Tensor,
    temperature: float,
    pt_batch: PrefixTreeMagiBatch,
    magi_key=None,
    flex_key=None,
    **kwargs,
):
    """Fused-path forward body for prefix-tree: preprocess → decoder → LCE.

    Shared entry point invoked by the unified ``_gpt_forward`` patch when the
    fused prefix-tree path is selected (``pt_batch`` present + attention key).
    Mirrors ``_fused_GPTModel_forward`` but assumes rope override and decoder
    key injection are already active (installed by the caller via
    :func:`prefix_tree_rope_context` and :func:`prefix_tree_decoder_key_context`).

    Vocab projection stays fused via :func:`linear_cross_entropy`: no
    ``(flat_tokens, vocab)`` logits tensor is materialised.
    """
    from collections import OrderedDict as _OrderedDict

    from megatron.core.config_logger import has_config_logger_enabled as _has_cfg_log
    from megatron.core.config_logger import log_config_to_disk as _log_cfg
    from megatron.core.utils import deprecate_inference_params as _dep_inf

    from verl.utils.model import CausalLMOutputForPPO as _CLMOutput

    inference_context = kwargs.pop("inference_context", None)
    inference_params = kwargs.pop("inference_params", None)
    inference_context = _dep_inf(inference_context, inference_params)
    decoder_input = kwargs.pop("decoder_input", None)
    packed_seq_params = kwargs.pop("packed_seq_params", None)
    extra_block_kwargs = kwargs.pop("extra_block_kwargs", None)

    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
    )
    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    hidden_states = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **(extra_block_kwargs or {}),
        **kwargs,
    )

    if not model.post_process:
        return hidden_states

    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    else:
        output_weight = model.embedding.word_embeddings.weight

    if magi_key is not None:
        logprobs, entropy = _run_lce(
            hidden_states, output_weight, labels, temperature, model, magi_key=magi_key, pt_batch=pt_batch
        )
    else:
        logprobs, entropy = _run_lce(hidden_states, output_weight, labels, temperature, model)

    if _has_cfg_log(model.config):
        payload = _OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        _log_cfg(model.config, payload, prefix="input_and_logits")

    output = _CLMOutput(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )
    output.entropy = entropy
    output.log_probs = logprobs
    return output


def fuse_try_forward_prefix_tree(
    model,
    input_ids,
    labels,
    temperature: float,
    logits_processor_args: dict,
    calculate_entropy: bool,
    *,
    vision_model: bool = False,
    has_vision_data: bool = False,
):
    """Fused-path: try to build + forward a prefix-tree batch with fused vocab projection.

    Counterpart of :func:`unfuse_try_forward_prefix_tree` for the
    ``use_fused_kernels=True`` path.  The vocab projection + log-prob
    computation stays fused inside ``_fused_GPTModel_forward`` via
    :func:`linear_cross_entropy`: the unfused path materialises
    ``(flat_tokens, vocab)`` logits and runs ``logits_processor`` outside the
    model, but the fused path never materialises the full vocab tensor.

    Limitations vs unfused path:
      - **Scalar temperature only.**  ``linear_cross_entropy`` asserts
        ``isinstance(temperature, float)``.  Per-sample temperature must use
        the unfused path.
      - **PP support**: on non-last stages (``not post_process``), returns the
        raw hidden-state tensor (pipeline schedule sends it to the next stage).
        Last stage (``post_process=True``) returns the log_probs/entropy dict.

    Args:
        model: Megatron GPTModel (forward patched to ``_fused_GPTModel_forward``).
        input_ids: NestedTensor of shape (batch_size, variable_seqlen).
        labels: NestedTensor, used for per-sample offsets only; actual labels
            come from ``pt_batch.tree_packed_labels`` (pre-shifted per sample).
        temperature: scalar float.
        logits_processor_args: dict containing ``use_prefix_tree``,
            ``prefix_tree_attention``, ``segment_hashes``, ``segment_lengths``,
            ``prefix_tree_subtree``.  Prefix-tree keys are stripped on return.
        calculate_entropy: whether to return ``entropy`` alongside ``log_probs``.
        vision_model: whether the model is a VLM-config model (has vision_config).
        has_vision_data: whether ``pixel_values`` is present in multi_modal_inputs.

    Returns:
        ``{"log_probs": NestedTensor, "entropy": NestedTensor}`` (entropy only
        when ``calculate_entropy=True``), or ``None`` when no prefix sharing is
        detected; caller falls through to the standard fused path.
    """

    # VLM-with-images: 3D M-RoPE position handling not yet wired
    # (prefix_tree_rope_context assumes 1D). Text-only on ViT-config models
    # passes through to the standard fused path below.
    if vision_model and has_vision_data:
        strip_prefix_tree_args(logits_processor_args)
        return None

    prefix_tree_attention = (logits_processor_args or {}).get("prefix_tree_attention", "flex")

    pb = build_prefix_tree_batch(
        model,
        input_ids,
        logits_processor_args,
        vision_model=False,
        mtp_enable_train=False,
    )
    if pb is None:
        _log.getLogger(__name__).warning(
            "prefix_tree: build_prefix_tree_batch returned None; falling back to standard fused path"
        )
        strip_prefix_tree_args(logits_processor_args)
        return None

    local_input_ids, local_position_ids, attn_kwargs = _prepare_attn_inputs(pb, prefix_tree_attention)

    strip_prefix_tree_args(logits_processor_args)

    post_process = unwrap_model(model).post_process

    # Only the last PP stage (post_process=True) needs labels for LCE.
    # Non-last stages pass labels=None; fuse_forward_body returns before LCE.
    real_tokens = pb.real_tokens
    if post_process:
        if pb.tree_packed_labels is None:
            _log.getLogger(__name__).warning(
                "prefix_tree[fused]: tree_packed_labels is None; falling back to standard fused path"
            )
            return None
        # Pass flat (deduped) labels; LCE runs on real_tokens, not total_expanded.
        labels_arg = pb.tree_packed_labels[:real_tokens]
    else:
        labels_arg = None

    output_orig = model(
        input_ids=local_input_ids,
        attention_mask=None,
        position_ids=local_position_ids,
        packed_seq_params=None,
        labels=labels_arg,
        temperature=temperature,
        pt_batch=pb,
        **attn_kwargs,
    )

    if not post_process:
        return output_orig

    # output_orig.log_probs / .entropy are (real_tokens,) flat; restore to per-sample nested.
    output = {"log_probs": _restore_to_nested_per_sample(output_orig.log_probs.reshape(-1), pb)}
    if calculate_entropy:
        output["entropy"] = _restore_to_nested_per_sample(output_orig.entropy.reshape(-1), pb)
    return output
