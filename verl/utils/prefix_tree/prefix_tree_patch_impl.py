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
"""Monkey-patches for Megatron-LM: MAGI/flex key threaded to TEDotProductAttention.forward via
TransformerLayer (core_attention wrap) + checkpointed-attention capture, TransformerBlock, GPTModel,
+ RoPE CP slicing override."""

from __future__ import annotations

import functools
import logging

import megatron.core.tensor_parallel as tp
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

# Eager flex_attention materializes the full scores matrix and burns ~100ms+/layer of
# Python/autograd dispatch; the compiled form emits a fused kernel instead.
_compiled_flex_attention = torch.compile(flex_attention)

# Stack for passing attention keys through gradient-checkpoint recompute.
# Pushed by _fn_with_key before calling checkpointed fn, popped after.
# Simple list is safe: training is single-threaded per worker.
_attn_key_stack: list = []


# flex_attention helper


def flex_attn_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    flex_attention_key: object,
) -> Tensor:
    """Execute PyTorch flex_attention for prefix-tree batches: THD→(1,H,T,D), flex_attention, back→THD."""

    T, _, H, D = query.shape
    q = query.squeeze(1).permute(1, 0, 2).unsqueeze(0)  # (1, H, T, D)
    k = key.squeeze(1).permute(1, 0, 2).unsqueeze(0)
    v = value.squeeze(1).permute(1, 0, 2).unsqueeze(0)
    enable_gqa = q.shape[1] != k.shape[1]

    out = _compiled_flex_attention(q, k, v, block_mask=flex_attention_key, enable_gqa=enable_gqa)
    out = out.squeeze(0).permute(1, 0, 2)  # (T, Hq, D)
    return out.reshape(T, 1, -1)  # (T, 1, Hq*D)


# MAGI attention kernel helper


def magi_attn_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    magi_attention_key: object,
) -> Tensor:
    """Execute MAGI calc_attn for prefix-tree: squeeze pre-dispatched CP-local Q/K/V, call calc_attn, reshape."""

    q = query.squeeze(1).contiguous()
    k = key.squeeze(1).contiguous()
    v = value.squeeze(1).contiguous()

    out, _ = calc_attn(q, k, v, magi_attention_key)

    return out.reshape(out.shape[0], 1, -1)


# Per-batch attention-path counters (magi/flex/fa3)
def _make_attn_counters():
    """Return (reset, inc_fa3, inc_non_fa3, get_metrics) closures tracking FA3 fallback ratio."""
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


def maybe_collect_prefix_tree_metrics(engine_config, engine, output: dict) -> None:
    """Collect prefix-tree metrics (FA3 fallback ratio, mb shared ratio) into output['metrics']."""
    if not getattr(engine_config, "use_prefix_tree", False):
        return
    from verl.utils.prefix_tree.dynamic import _get_mbs_metric, _reset_mbs_metric

    metrics = dict(_get_attn_metrics())
    mbs_metric = _get_mbs_metric()
    if mbs_metric:
        metrics.update(mbs_metric)
    if metrics and engine.is_mp_src_rank_with_outputs():
        output.setdefault("metrics", {}).update(metrics)
    _reset_attn_counters()
    _reset_mbs_metric()


# Patch
def apply_prefix_tree_patch() -> None:
    """Monkey-patch Megatron classes for prefix-tree attention (flex and MAGI). Safe to call multiple times."""

    if getattr(TEDotProductAttention, "_prefix_tree_patched", False):
        return  # skip the double-patching

    # 0. RoPE: patch globally once — set ._pids before forward, patch reads it.
    _orig_rope_fn = RotaryEmbedding.forward.__wrapped__  # actual impl, bypasses lru_cache
    _real_rope_forward = RotaryEmbedding.forward

    @functools.wraps(_real_rope_forward)
    def _rope_forward(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        pids = getattr(self, "_pids", None)
        if pids is None:
            return _real_rope_forward(self, max_seq_len, offset, packed_seq, cp_group)
        if not hasattr(self, "get_emb"):  # M-RoPE
            pids_3d = pids.reshape(-1).view(1, 1, -1).expand(3, 1, -1).contiguous()
            return _real_rope_forward(self, pids_3d, getattr(self, "mrope_section", None), cp_group=None)
        actual_seq_len = int(pids.reshape(-1).max().item()) + 1
        emb = _orig_rope_fn(self, actual_seq_len, offset=0, packed_seq=True, cp_group=None)
        return emb[pids.reshape(-1).to(emb.device)]

    RotaryEmbedding.forward = _rope_forward

    # 1. TEDotProductAttention.forward: add early-return MAGI/flex branches.
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

    # 2. SelfAttention._checkpointed_attention_forward: capture patched core_attention.forward at closure-creation time.
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
        # Capture magi-patched core_attention.forward at closure time (not lookup time) for recompute safety.
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

    # 3. TransformerLayer.forward: accept and pass magi/flex attention key, with recompute stack fallback.
    _orig_tl_forward = TransformerLayer.forward

    @functools.wraps(_orig_tl_forward)
    def _tl_forward(self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs):
        # Forward: key arrives via the layer wrapper kwargs (patch A in TransformerBlock).
        # Recompute: layer wrappers are gone; key arrives via _attn_key_stack pushed by _fn_with_key (patch B).
        if magi_attention_key is None and flex_attention_key is None and _attn_key_stack:
            magi_attention_key, flex_attention_key = _attn_key_stack[-1]
        attn_key = magi_attention_key or flex_attention_key
        # Inject the key at core_attention.forward — the one seam shared by stock
        # SelfAttention and subclasses (e.g. Qwen3.5 Qwen3_5VLSelfAttention), all of
        # which call self.core_attention(...) internally. Never wrap self_attention.forward:
        # subclasses may not accept the key kwargs. Layers without core_attention (GDN/SSM)
        # route through their own ssm_gdn_patch and skip the key entirely.
        _core_attn = getattr(self.self_attention, "core_attention", None) if attn_key is not None else None
        if _core_attn is None:
            out = _orig_tl_forward(self, hidden_states, attention_mask, **kwargs)
        else:
            _real_ca_forward = _core_attn.forward

            @functools.wraps(_real_ca_forward)
            def _ca_forward_with_key(*args, **kw):
                return _real_ca_forward(
                    *args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw
                )

            _core_attn.forward = _ca_forward_with_key
            try:
                out = _orig_tl_forward(self, hidden_states, attention_mask, **kwargs)
            finally:
                _core_attn.forward = _real_ca_forward

        return out

    TransformerLayer.forward = _tl_forward

    # 4. TransformerBlock.forward: accept and pass magi/flex attention key with checkpoint wrapper for recompute.
    _orig_tb_forward = TransformerBlock.forward

    @functools.wraps(_orig_tb_forward)
    def _transformer_block_forward(
        self, hidden_states, attention_mask, magi_attention_key=None, flex_attention_key=None, **kwargs
    ):
        attn_key = magi_attention_key or flex_attention_key
        if attn_key is None:
            return _orig_tb_forward(self, hidden_states, attention_mask, **kwargs)
        # (A) Layer-level patching: inject key via kwargs for forward pass.
        originals = [layer.forward for layer in self.layers]

        def _make_wrapper(orig):
            @functools.wraps(orig)
            def _w(*args, **kw):
                return orig(*args, magi_attention_key=magi_attention_key, flex_attention_key=flex_attention_key, **kw)

            return _w

        for layer in self.layers:
            layer.forward = _make_wrapper(layer.forward)

        # (B) Checkpoint wrapper: push key onto _attn_key_stack for backward recompute.
        _real_tp_checkpoint = tp.checkpoint

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

        tp.checkpoint = _checkpoint_with_key
        try:
            out = _orig_tb_forward(self, hidden_states, attention_mask, **kwargs)
        finally:
            for layer, orig_fwd in zip(self.layers, originals, strict=False):
                layer.forward = orig_fwd
            tp.checkpoint = _real_tp_checkpoint
        return out

    TransformerBlock.forward = _transformer_block_forward

    # 5. GPTModel.forward: accept and pass magi/flex attention key.
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

        # Set position_ids for global RoPE patch.
        rope_mod = getattr(self, "rotary_pos_emb", None)
        if rope_mod is not None and position_ids is not None:
            rope_mod._pids = position_ids.reshape(-1)

        try:
            out = _orig_gpt_forward(self, input_ids, position_ids, attention_mask, **kwargs)
        finally:
            if rope_mod is not None:
                rope_mod._pids = None
            self.decoder.forward = _real_decoder_forward
        return out

    GPTModel.forward = _gpt_forward

    TEDotProductAttention._prefix_tree_patched = True
