# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Make ``attn_implementation="flash_attention_2"`` usable on Gemma4.

Gemma4 alternates two attention flavours (e.g. 5 sliding : 1 full over 30 layers in
``gemma-4-26B-A4B``):

  * ``sliding_attention``  ``head_dim=256``, sliding window  -> flash-legal
  * ``full_attention``     ``head_dim=512``, full causal     -> flash-illegal

FlashAttention caps ``head_dim`` at 256, so loading Gemma4 with a flash implementation raises
``FlashAttention forward only supports head dimension at most 256`` on the first *global* layer.
The sliding layers -- the large majority -- are ``head_dim=256`` and run under flash fine:
``transformers`` already plumbs ``self.sliding_window`` per layer into the kernel and turns it
into ``window_size=(sliding_window - 1, sliding_window - 1)`` alongside ``causal=True``.

So the whole incompatibility is those few global layers, and the fix is to run *them* on SDPA
while everything else takes the stock path. This matters because SDPA is a poor fit for the
sliding layers: given an additive mask it computes the full ``S x S`` score matrix and then
masks it, discarding the sliding-window sparsity. Measured on an H200 at ``S=8192`` (fwd+bwd,
single sliding layer): ~72 ms under SDPA vs ~2.25 ms under flash.

Behaviour by ``attn_implementation``
------------------------------------
* ``sdpa`` / ``eager`` -- every layer takes the stock dispatch. This patch is a no-op.
* ``flash_attention_2`` (or any flash flavour) -- sliding layers take the stock flash path,
  global layers fall back to SDPA.

Why patch the kernel dispatch rather than write a custom attention module
------------------------------------------------------------------------
Gemma4 uses **two** RoPEs -- sliding layers use ``rope_theta=1e4`` with full rotary, global
layers use ``rope_theta=1e6`` with ``partial_rotary_factor=0.25`` and
``rope_type="proportional"``. A hand-written attention module has to reproduce both, and
applying one scheme to both layer types silently corrupts the global layers.

That is avoidable: in ``Gemma4TextAttention.forward``, QK-norm, both RoPEs and the
``attention_k_eq_v`` global aliasing are all applied *before* the attention kernel is called,
which then dispatches through ``ALL_ATTENTION_FUNCTIONS``. So this module keeps that forward
verbatim and changes only which kernel the global layers reach -- RoPE correctness is
structural, and the sliding layers are byte-identical to upstream.

``transformers`` resolves ``config._attn_implementation`` to a single string per text tower
(both the mask builders in ``masking_utils`` and the attention forward read it), so there is no
native per-layer-type dispatch; hence the module-level patch.

Two masks, not one
------------------
The mask format is decided by ``config._attn_implementation``, so a flash config would hand the
global layers a mask SDPA cannot use: ``(batch, kv_len)`` padding-only under flash (causality
lives in the kernel's ``causal`` flag) versus the ``(batch, 1, q_len, kv_len)`` tensor carrying
causality *and* padding that SDPA needs.

Gemma4 builds that mapping per layer type, and every path routes the ``full_attention`` entry
through a builder resolved in the ``modeling_gemma4`` namespace::

    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

So this module rebinds those builders *in that namespace* to force the ``full_attention`` entry
into SDPA format, leaving the sliding entry alone. The stock ``transformers`` builders still do
the work -- no hand-rolled masking here -- and ``masking_utils`` itself is untouched, so no other
model's masks change.

Doing it at the builder rather than in a model forward matters, because there are two callers.
``Gemma4TextModel.forward`` builds the mapping when it receives a plain tensor, but the composite
``Gemma4Model.forward`` (``Gemma4ForConditionalGeneration``, which is what ``AutoModel`` resolves
the released 26B checkpoint to) builds the mapping *itself* and passes it down as a dict. Patching
only the text tower's forward would leave the composite path building flash-format masks and
feeding them to SDPA -- silently, since a packed batch then attends across sequence boundaries in
the global layers instead of crashing.

This is what makes ``use_remove_padding`` work. The rmpad path packs a batch into one
``(1, total_nnz)`` row with ``attention_mask=None``, leaving sequence boundaries encoded only in
``position_ids``. The flash path recovers them as ``cu_seqlens`` (``_is_packed_sequence``); the
SDPA builder independently turns them into a **block-diagonal** mask. Both respect the
boundaries, so packed sequences do not attend across each other.

Ulysses sequence parallel is still excluded: it shards the sequence across ranks and relies on
veRL's patched flash-attention forward, which the global layers do not go through.
"""

import copy

__all__ = ["apply_gemma4_flash_attention"]

# Set by apply_gemma4_flash_attention() when it rebinds the builder below.
_stock_create_causal_mask = None


def _as_sdpa(config):
    """Shallow copy of ``config`` requesting sdpa, or ``config`` itself if it already is."""
    from transformers.utils.generic import is_flash_attention_requested

    if not is_flash_attention_requested(config):
        return config
    sdpa_config = copy.copy(config)
    sdpa_config._attn_implementation = "sdpa"
    return sdpa_config


def _global_mask_in_sdpa_format(config, *args, **kwargs):
    """``create_causal_mask`` with the config forced to sdpa.

    This builder feeds *only* the ``full_attention`` entry -- the sliding entry comes from
    ``create_sliding_window_causal_mask``, which is left alone so those layers keep the flash
    mask their (stock, flash) kernel path expects.
    """
    return _stock_create_causal_mask(_as_sdpa(config), *args, **kwargs)


def _global_attention_sdpa(module, query, key, value, attention_mask, dropout, **kwargs):
    """SDPA for the head_dim-512 global layers.

    The mask arrives in SDPA format because the model forward above built it that way, so this is
    the stock SDPA interface with nothing reconstructed.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=module.scaling,
        **kwargs,
    )


def _patched_gemma4_text_attention_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    shared_kv_states,
    past_key_values=None,
    **kwargs,
):
    """Mirror of ``Gemma4TextAttention.forward`` with a per-layer-type kernel dispatch.

    Adapted from transformers 5.8.0. Everything outside the marked dispatch block is verbatim
    upstream, so that QK-norm, both RoPEs, the K==V global aliasing and the KV cache behave
    unchanged -- re-check that body against upstream when bumping transformers, since a change
    there would be silently dropped rather than conflicting.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, eager_attention_forward
    from transformers.utils.generic import is_flash_attention_requested

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
    # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
    # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
    if self.is_kv_shared_layer:
        key_states, value_states = shared_kv_states[self.layer_type]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

    if past_key_values is not None and not self.is_kv_shared_layer:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    if self.store_full_length_kv:
        shared_kv_states[self.layer_type] = key_states, value_states

    # --- kernel dispatch: the only divergence from upstream ------------------------------------
    # Global layers are head_dim 512, past flash's 256 cap, so under a flash config they take
    # SDPA. Sliding layers (and every layer under a non-flash config) take the stock path.
    if self.is_sliding or not is_flash_attention_requested(self.config):
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
    else:
        attn_output, attn_weights = _global_attention_sdpa(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            **kwargs,
        )
    # -------------------------------------------------------------------------------------------

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def apply_gemma4_flash_attention(model=None, ulysses_sp_size: int = 1) -> bool:
    """Let Gemma4 run under a flash ``attn_implementation`` by keeping its global layers on SDPA.

    Patches ``Gemma4TextAttention.forward`` (kernel dispatch) at class level, and rebinds the two
    mask builders in the ``modeling_gemma4`` namespace so the ``full_attention`` entry is always
    built in SDPA format regardless of which model class is instantiated. Idempotent, and a
    structural no-op unless ``attn_implementation`` is a flash flavour.

    Raises if flash is requested together with Ulysses sequence parallel (see the module
    docstring). Returns True if the patch was installed, False if Gemma4 is unavailable.
    """
    global _stock_create_causal_mask
    try:
        from transformers.models.gemma4 import modeling_gemma4
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
        from transformers.utils.generic import is_flash_attention_requested
    except Exception as e:  # pragma: no cover - transformers without gemma4
        print(f"[gemma4] transformers has no Gemma4TextAttention ({e}); patch skipped")
        return False

    flash = model is not None and is_flash_attention_requested(model.config)
    if flash and getattr(model.config.get_text_config(), "use_bidirectional_attention", None) != "vision":
        raise ValueError(
            "Gemma4 variants with use_bidirectional_attention="
            f"{getattr(model.config.get_text_config(), 'use_bidirectional_attention', None)!r} are not "
            "supported with a flash attn_implementation. Their forward builds the per-layer-type mask "
            "mapping through `create_masks_for_generate`, which resolves each layer type via "
            "masking_utils' pattern mapping and so never reaches the `create_causal_mask` rebinding "
            "this patch relies on -- the head_dim-512 global layers would run SDPA against a "
            "flash-format mask and, under remove-padding, attend across packed-sequence boundaries "
            "without raising. Use `attn_implementation='sdpa'`."
        )
    if flash and ulysses_sp_size > 1:
        raise ValueError(
            "Gemma4 does not support Ulysses sequence parallel with a flash attn_implementation "
            f"({model.config._attn_implementation!r}, ulysses_sequence_parallel_size="
            f"{ulysses_sp_size}). Its head_dim-512 global layers exceed flash's 256 cap and run on "
            "SDPA, which does not go through the sequence-parallel flash-attention forward that "
            "Ulysses patches in, so the sharded sequence would be attended incorrectly. Set "
            "`ulysses_sequence_parallel_size=1`, or use `attn_implementation='sdpa'`."
        )

    if not getattr(Gemma4TextAttention.forward, "_gemma4_attention_dispatch", False):
        _patched_gemma4_text_attention_forward._gemma4_attention_dispatch = True
        Gemma4TextAttention.forward = _patched_gemma4_text_attention_forward

    # Rebind in the `modeling_gemma4` namespace only -- `masking_utils` and every other model file
    # keep the stock builders. Rebinding `masking_utils.create_causal_mask` instead would change
    # the mask format for Gemma3, Mistral and everything else that imports it.
    if not getattr(modeling_gemma4.create_causal_mask, "_gemma4_sdpa_global_mask", False):
        _stock_create_causal_mask = modeling_gemma4.create_causal_mask
        _global_mask_in_sdpa_format._gemma4_sdpa_global_mask = True
        modeling_gemma4.create_causal_mask = _global_mask_in_sdpa_format

    if flash:
        print(
            "[gemma4] flash attention enabled: sliding (head_dim 256) layers use "
            f"{model.config._attn_implementation}, global (head_dim 512) layers fall back to SDPA"
        )
    return True
