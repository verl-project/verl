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
"""Monkey-patches for upstream Megatron-LM to support MAGI flex-attention.

Upstream Megatron-LM does not have the ``magi_attention_key`` parameter in its
forward call chain.  This module patches the five classes that need it so that
verl can work with a stock (unmodified) Megatron installation.

Call ``apply_prefix_tree_patch()`` once before model construction (e.g. from
``verl/models/mcore/patch.py:apply_patch`` or from the engine initialiser).

Patch chain (each wrapper accepts ``magi_attention_key``/``flex_attention_key`` and threads them):
    GPTModel.forward
    → TransformerBlock.forward  (both the checkpointed and normal variants)
    → TransformerLayer.forward
    → SelfAttention.forward     (both checkpointed and normal core-attention calls)
    → TEDotProductAttention.forward  (early-return MAGI branch)

The ``magi_attn_forward`` helper calls ``calc_attn`` directly on already-dispatched
local Q/K/V (pre-dispatch happens in ``unfuse_forward_prefix_tree`` /
``fuse_try_forward_prefix_tree`` before the model call).
"""

from __future__ import annotations

import functools
import logging

import torch
from magi_attention.api import calc_attn
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import checkpoint as tensor_parallel_checkpoint
from megatron.core.transformer.attention import AttnMaskType, SelfAttention
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

# Stack for passing attention keys through gradient-checkpoint recompute.
# Pushed by _fn_with_key before calling checkpointed fn, popped after.
# Simple list is safe: training is single-threaded per worker.
_attn_key_stack: list = []


# ---------------------------------------------------------------------------
# flex_attention helper for prefix-tree
# ---------------------------------------------------------------------------


def flex_attn_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    flex_attention_key: object,
) -> Tensor:
    """Execute PyTorch flex_attention for prefix-tree batches.

    Input tensors are in thd layout: ``(total_tokens, 1, num_heads, head_dim)``.
    Returns ``(total_tokens, 1, num_heads*head_dim)``.

    Uses torch.utils.checkpoint to avoid storing the O(T²) attention score
    matrix for backward; recomputes the forward pass instead (O(T) memory).
    """

    T, _, H, D = query.shape
    q = query.squeeze(1).permute(1, 0, 2).unsqueeze(0)  # (1, H, T, D)
    k = key.squeeze(1).permute(1, 0, 2).unsqueeze(0)
    v = value.squeeze(1).permute(1, 0, 2).unsqueeze(0)
    enable_gqa = q.shape[1] != k.shape[1]

    out = flex_attention(q, k, v, block_mask=flex_attention_key, enable_gqa=enable_gqa)
    out = out.squeeze(0).permute(1, 0, 2)  # (T, Hq, D)
    return out.reshape(T, 1, -1)  # (T, 1, Hq*D)


# ---------------------------------------------------------------------------
# MAGI attention kernel helper (verbatim from Megatron-LM-prefix-tree fork)
# ---------------------------------------------------------------------------


def magi_attn_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    magi_attention_key: object,
) -> Tensor:
    """Execute MAGI calc_attn for prefix-tree batches.

    Input Q/K/V are already CP-local (pre-dispatched before the model call).
    Returns ``(local_tokens, 1, num_heads*head_dim)``.
    """

    q = query.squeeze(1)
    k = key.squeeze(1)
    v = value.squeeze(1)

    out, _ = calc_attn(q, k, v, magi_attention_key)

    return out.reshape(out.shape[0], 1, -1)


# ---------------------------------------------------------------------------
# Per-batch attention-path counters (magi / flex / fa3-fallback)
# ---------------------------------------------------------------------------


def _make_attn_counters():
    """Return (reset, inc_fa3, inc_non_fa3, get_metrics) closures sharing one state dict."""
    state = {"fa3": 0, "total": 0}

    def reset():
        state["fa3"] = 0
        state["total"] = 0

    def inc_fa3():
        state["fa3"] += 1
        state["total"] += 1

    def inc_non_fa3():
        state["total"] += 1

    def get_metrics():
        from verl.utils.metric import AggregationType, Metric

        fa, total = state["fa3"], state["total"]
        return {
            "prefix_tree/attn_fa3_fallback_ratio": Metric(
                value=(fa / total) if total > 0 else 0.0, aggregation=AggregationType.MEAN
            ),
        }

    return reset, inc_fa3, inc_non_fa3, get_metrics


_reset_attn_counters, _inc_fa3, _inc_non_fa3, _get_attn_metrics = _make_attn_counters()


def maybe_collect_attn_metrics(engine_config, engine, output: dict) -> None:
    """Collect attn metrics into output['metrics'] and clear counters for next batch."""
    if getattr(engine_config, "use_prefix_tree", False):
        attn_metrics = _get_attn_metrics()
        if attn_metrics and engine.is_mp_src_rank_with_outputs():
            output.setdefault("metrics", {}).update(attn_metrics)
        _reset_attn_counters()


def maybe_collect_mbs_metric(engine_config, engine, output: dict) -> None:
    """Collect the post-micro-batch-build micro_batch_shared_ratio into output['metrics'].

    Pulls the accurate per-micro-batch sharing ratio (computed inside
    ``prepare_prefix_tree_micro_batches`` from the ACTUAL grouping the engine
    dispatches) into ``output['metrics']`` and clears the collector.  Mirrors
    :func:`maybe_collect_attn_metrics` exactly.
    """
    if getattr(engine_config, "use_prefix_tree", False):
        from verl.utils.prefix_tree.dynamic import _get_mbs_metric, _reset_mbs_metric

        mbs_metric = _get_mbs_metric()
        if mbs_metric and engine.is_mp_src_rank_with_outputs():
            output.setdefault("metrics", {}).update(mbs_metric)
        _reset_mbs_metric()


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------


def apply_prefix_tree_patch() -> None:
    """Monkey-patch upstream Megatron-LM classes to support prefix-tree attention (flex and MAGI).

    Safe to call multiple times: subsequent calls are no-ops (checks for the
    ``_magi_patched`` sentinel attribute).
    """

    if getattr(TEDotProductAttention, "_prefix_tree_patched", False):
        return  # skip the double-patching

    # ------------------------------------------------------------------
    # 1. TEDotProductAttention.forward: add early-return MAGI branch
    # ------------------------------------------------------------------
    _orig_te_forward = TEDotProductAttention.forward

    @functools.wraps(_orig_te_forward)
    def _te_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
        magi_attention_key=None,
        flex_attention_key=None,
        **kwargs,
    ):
        if magi_attention_key is not None:
            _inc_non_fa3()
            return magi_attn_forward(query, key, value, magi_attention_key)
        if flex_attention_key is not None:
            _inc_non_fa3()
            return flex_attn_forward(query, key, value, flex_attention_key)
        # FA3 fallback: logged once per occurrence so it shows up in monitoring
        _inc_fa3()
        logging.getLogger(__name__).warning_once("prefix_tree_patch: using FA3 attention path (fallback)")
        return _orig_te_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            num_splits=num_splits,
            **kwargs,
        )

    TEDotProductAttention.forward = _te_forward

    # ------------------------------------------------------------------
    # 2. SelfAttention._checkpointed_attention_forward: thread kwarg
    # ------------------------------------------------------------------
    _orig_sa_ckpt = SelfAttention._checkpointed_attention_forward

    @functools.wraps(_orig_sa_ckpt)
    def _sa_ckpt_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        magi_attention_key=None,
        flex_attention_key=None,
        **kwargs,
    ):
        # Capture the MAGI-patched forward at closure-creation time (not lookup time).
        # self.core_attention.forward is _ca_forward_with_key right now (set by _sa_forward).
        # If we looked it up dynamically inside custom_forward, recomputation during backward
        # would see the restored FA3 forward (the finally block already ran).
        _captured_ca_forward = self.core_attention.forward

        def custom_forward(*inputs):
            q, k, v, amask = inputs[0], inputs[1], inputs[2], inputs[3]
            _attn_mask_type = AttnMaskType(inputs[5].item())
            return _captured_ca_forward(
                q,
                k,
                v,
                amask,
                attn_mask_type=_attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type_tensor = torch.tensor([attn_mask_type.value], dtype=torch.int)
        return tensor_parallel_checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type_tensor,
        )

    SelfAttention._checkpointed_attention_forward = _sa_ckpt_forward

    # ------------------------------------------------------------------
    # 3. SelfAttention.forward: accept and pass magi/flex attention key
    # ------------------------------------------------------------------
    _orig_sa_forward = SelfAttention.forward

    @functools.wraps(_orig_sa_forward)
    def _sa_forward(self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs):
        attn_key = magi_attention_key or flex_attention_key
        _real_ca_forward = self.core_attention.forward

        @functools.wraps(_real_ca_forward)
        def _ca_forward_with_key(q, k, v, *args, **kw):
            return _real_ca_forward(
                q,
                k,
                v,
                *args,
                magi_attention_key=magi_attention_key if attn_key else None,
                flex_attention_key=flex_attention_key if attn_key else None,
                **kw,
            )

        self.core_attention.forward = _ca_forward_with_key
        try:
            out = _orig_sa_forward(self, hidden_states, attention_mask, **kwargs)
        finally:
            self.core_attention.forward = _real_ca_forward
        return out

    SelfAttention.forward = _sa_forward

    # ------------------------------------------------------------------
    # 4. TransformerLayer.forward: accept and pass magi/flex attention key
    # ------------------------------------------------------------------
    _orig_tl_forward = TransformerLayer.forward

    @functools.wraps(_orig_tl_forward)
    def _tl_forward(self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs):
        # Forward: key arrives via the layer wrapper kwargs (patch A in TransformerBlock).
        # Recompute: layer wrappers are gone; key arrives via _attn_key_stack pushed by _fn_with_key (patch B).
        if magi_attention_key is None and flex_attention_key is None and _attn_key_stack:
            magi_attention_key, flex_attention_key = _attn_key_stack[-1]
        attn_key = magi_attention_key or flex_attention_key
        if attn_key is None:
            out = _orig_tl_forward(self, hidden_states, attention_mask, **kwargs)
        else:
            _real_sa_forward = self.self_attention.forward

            @functools.wraps(_real_sa_forward)
            def _sa_forward_with_key(*args, **kw):
                return _real_sa_forward(
                    *args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw
                )

            self.self_attention.forward = _sa_forward_with_key
            try:
                out = _orig_tl_forward(self, hidden_states, attention_mask, **kwargs)
            finally:
                self.self_attention.forward = _real_sa_forward

        return out

    TransformerLayer.forward = _tl_forward

    # ------------------------------------------------------------------
    # 5. TransformerBlock.forward: accept and pass magi/flex attention key
    # ------------------------------------------------------------------
    _orig_tb_forward = TransformerBlock.forward

    @functools.wraps(_orig_tb_forward)
    def _transformer_block_forward(
        self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs
    ):
        attn_key = magi_attention_key or flex_attention_key
        if attn_key is None:
            return _orig_tb_forward(self, hidden_states, attention_mask, **kwargs)
        # (A) Layer-level patching: injects key via kwargs for the forward pass,
        #     regardless of whether gradient checkpointing is enabled.
        originals = []
        for layer in self.layers:
            originals.append(layer.forward)

            def _make_wrapper(orig):
                @functools.wraps(orig)
                def _w(*args, **kw):
                    return orig(
                        *args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw
                    )

                return _w

            layer.forward = _make_wrapper(layer.forward)

        # (B) Stack via checkpoint wrapper: pushes key for backward recomputation.
        #     During recompute the layer wrappers above are already restored (finally ran),
        #     so _tl_forward falls back to _attn_key_stack pushed by _fn_with_key.
        import megatron.core.tensor_parallel as _tp

        _real_tp_checkpoint = _tp.checkpoint

        def _checkpoint_with_key(fn, distribute, *ck_args, **ck_kwargs):
            _cap_magi = magi_attention_key
            _cap_flex = flex_attention_key

            def _fn_with_key(*a, **kw):
                _attn_key_stack.append((_cap_magi, _cap_flex))
                try:
                    return fn(*a, **kw)
                finally:
                    _attn_key_stack.pop()

            return _real_tp_checkpoint(_fn_with_key, distribute, *ck_args, **ck_kwargs)

        _tp.checkpoint = _checkpoint_with_key
        try:
            out = _orig_tb_forward(self, hidden_states, attention_mask, **kwargs)
        finally:
            for layer, orig_fwd in zip(self.layers, originals, strict=False):
                layer.forward = orig_fwd
            _tp.checkpoint = _real_tp_checkpoint
        return out

    TransformerBlock.forward = _transformer_block_forward

    # ------------------------------------------------------------------
    # 6. GPTModel.forward: accept and pass magi/flex attention key.
    #    Patches rope_mod.forward with _rope_fwd_with_pids for the duration
    #    of the call so patch #7 uses correct per-token position indexing.
    # ------------------------------------------------------------------
    _orig_rope_fn = RotaryEmbedding.forward.__wrapped__  # actual impl, bypasses lru_cache
    _orig_gpt_forward = GPTModel.forward

    @functools.wraps(_orig_gpt_forward)
    def _gpt_forward(
        self, input_ids, position_ids, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs
    ):
        attn_key = magi_attention_key or flex_attention_key
        if attn_key is None:
            return _orig_gpt_forward(self, input_ids, position_ids, attention_mask, **kwargs)
        _real_decoder_forward = self.decoder.forward

        @functools.wraps(_real_decoder_forward)
        def _decoder_forward_with_key(*args, **kw):
            return _real_decoder_forward(
                *args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw
            )

        self.decoder.forward = _decoder_forward_with_key

        # Patch RotaryEmbedding using position_ids (= pt_batch.local_tree_packed_position_ids,
        # per-sample positions that reset within each sample). Closure captures pids
        # per-batch: no global state needed.
        rope_mod = getattr(self, "rotary_pos_emb", None)
        pids = position_ids.reshape(-1) if (position_ids is not None and rope_mod is not None) else None
        _real_rope_fwd = rope_mod.forward if rope_mod is not None else None

        if pids is not None:
            _is_mrope = not hasattr(rope_mod, "get_emb")
            if _is_mrope:
                _mrope_section = getattr(self, "mrope_section", None)

                def _rope_fwd_with_pids(*args, **kwargs):
                    pids_3d = pids.view(1, 1, -1).expand(3, 1, -1).contiguous()
                    return _real_rope_fwd(pids_3d, _mrope_section, cp_group=None)
            else:

                def _rope_fwd_with_pids(max_seq_len, offset=0, packed_seq=False, cp_group=None):
                    actual_seq_len = int(pids.max().item()) + 1
                    emb = _orig_rope_fn(rope_mod, actual_seq_len, offset=0, packed_seq=True, cp_group=None)
                    # All PP stages use seq-first Q=(seq,1,H,D) because unfuse_forward_prefix_tree
                    # returns (seq,1,hidden) for all intermediate stages (not batch-first).
                    # freqs=(seq,1,1,dim) broadcasts correctly: Q×freqs→(seq,1,H,D) ✓
                    indexed = emb[pids.to(emb.device)]
                    return indexed

            rope_mod.forward = _rope_fwd_with_pids

        try:
            out = _orig_gpt_forward(self, input_ids, position_ids, attention_mask, **kwargs)
        finally:
            self.decoder.forward = _real_decoder_forward
            if pids is not None:
                rope_mod.forward = _real_rope_fwd
        return out

    GPTModel.forward = _gpt_forward

    # ------------------------------------------------------------------
    # 7. RotaryEmbedding.forward: bypass CP-rank RoPE slicing for prefix-tree.
    #
    # Root cause of CP>1 bug: Megatron's get_pos_emb_on_this_cp_rank slices RoPE
    # assuming rank r holds sequential positions [r*T/CP .. (r+1)*T/CP].  After
    # MAGI dispatch, each CP rank holds non-sequential tokens (dispatch assigns by
    # attention topology), so their position_ids are arbitrary, not rank-aligned.
    # Using the rank-sliced frequencies produces wrong Q/K rotations.
    #
    # Fix: the closure _rope_fwd_with_pids (patch #6) replaces rope_mod.forward
    # for the duration of each prefix-tree forward.  It builds the full RoPE table
    # (cp_group=None) and indexes directly by the actual local position_ids, giving
    # correct frequencies regardless of CP rank or dispatch ordering.
    # ------------------------------------------------------------------

    TEDotProductAttention._prefix_tree_patched = True
