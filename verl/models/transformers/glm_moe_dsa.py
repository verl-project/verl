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

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)


def _build_causal_mask_from_position_ids(position_ids: torch.Tensor, seq_len: int, dtype: torch.dtype):
    """Build a 4D causal mask [B, 1, S, S] from position_ids that respects sequence boundaries.

    In packed sequences, position_ids resets at sequence boundaries:
    e.g., [0, 1, 2, 0, 1, 2, 3, 0, 1] for 3 packed sequences.

    The mask ensures:
    - Causal: position i can only attend to positions j where j <= i
    - Cross-sequence isolation: position i can only attend to positions in the same sequence
    """
    batch_size = position_ids.shape[0]
    device = position_ids.device

    boundaries = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
    if seq_len > 1:
        boundaries[:, 1:] = (position_ids[:, 1:] <= position_ids[:, :-1]).long()
    segment_ids = boundaries.cumsum(dim=1)  # [B, S]

    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)  # [B, S, S]

    indices = torch.arange(seq_len, device=device)
    causal = indices.unsqueeze(0) <= indices.unsqueeze(1)  # [S, S], causal[i, j] = (j <= i)

    mask = same_segment & causal.unsqueeze(0)  # [B, S, S]

    float_mask = torch.where(mask, torch.zeros(1, device=device, dtype=dtype), torch.finfo(dtype).min)
    return float_mask.unsqueeze(1)  # [B, 1, S, S]


def glm_moe_dsa_attn_forward_with_dsa(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    prev_topk_indices: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Drop-in replacement for GlmMoeDsaAttention.forward with use_remove_padding + Ulysses SP.

    When attention_mask is None (use_remove_padding mode), generates a causal mask
    from position_ids that respects packed sequence boundaries.
    DSA indexer is fully preserved.

    When Ulysses SP > 1, uses all-to-all to gather full sequence / scatter heads
    for attention, and all-gathers hidden_states for the indexer.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    batch_size, seq_length = hidden_states.shape[:-1]
    cos, sin = position_embeddings
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    # Always rebuild a float additive mask from position_ids.
    # The model's create_causal_mask may return a bool mask (incompatible with the DSA indexer
    # which expects float 0/-inf) or a local-sized mask (wrong when ulysses_sp_size > 1).
    if ulysses_sp_size > 1 and position_ids is not None:
        from verl.utils.ulysses import get_ulysses_sequence_parallel_group

        sp_group = get_ulysses_sequence_parallel_group()
        position_ids_full = _all_gather_seq(position_ids, sp_group, ulysses_sp_size)
        full_seq_length = seq_length * ulysses_sp_size
        attention_mask = _build_causal_mask_from_position_ids(position_ids_full, full_seq_length, hidden_states.dtype)
    elif position_ids is not None:
        attention_mask = _build_causal_mask_from_position_ids(position_ids, seq_length, hidden_states.dtype)

    # ===== Query path (MLA) =====
    if self.q_lora_rank is None:
        query_states = self.q_proj(hidden_states)
        q_resid = None
    else:
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid)
    query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = _apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

    # ===== KV path (MLA compressed) =====
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_compressed = self.kv_a_layernorm(k_compressed)

    kv_expanded = self.kv_b_proj(k_compressed)
    kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_nope = k_nope.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    k_pe = _apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

    # Assemble full Q and K
    query_states = torch.cat([q_nope, q_pe], dim=-1)
    key_states = torch.cat([k_nope, k_pe], dim=-1)

    # Cache update
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    # ===== Ulysses SP: all-to-all to gather full seq, scatter heads =====
    if ulysses_sp_size > 1:
        # [B, H, S_local, D] -> [B, H/sp, S_full, D]
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        full_seq_length = query_states.shape[2]
    else:
        full_seq_length = seq_length

    # ===== Indexer (DSA sparse mask) =====
    if not self.skip_topk or prev_topk_indices is None:
        if ulysses_sp_size > 1:
            # Indexer needs full-sequence hidden_states and q_resid
            from verl.utils.ulysses import get_ulysses_sequence_parallel_group

            sp_group = get_ulysses_sequence_parallel_group()
            # All-gather hidden_states along seq dim for indexer
            hidden_states_full = _all_gather_seq(hidden_states, sp_group, ulysses_sp_size)
            q_resid_full = _all_gather_seq(q_resid, sp_group, ulysses_sp_size) if q_resid is not None else None
            # Position embeddings also need full seq
            cos_full = _all_gather_seq(cos, sp_group, ulysses_sp_size)
            sin_full = _all_gather_seq(sin, sp_group, ulysses_sp_size)
            position_embeddings_full = (cos_full, sin_full)
        else:
            hidden_states_full = hidden_states
            q_resid_full = q_resid
            position_embeddings_full = position_embeddings

        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1)
            if attention_mask is not None
            else None
        )
        topk_indices = self.indexer(
            hidden_states_full,
            q_resid_full,
            position_embeddings_full,
            indexer_mask,
            use_cache=past_key_values is not None,
        )
    else:
        topk_indices = prev_topk_indices

    # Build combined DSA + causal mask
    total_len = key_states.shape[2]
    # topk_indices is [B, S_full, topk]
    index_mask = torch.full(
        (batch_size, full_seq_length, total_len),
        float("-inf"),
        device=hidden_states.device,
        dtype=query_states.dtype,
    )
    index_mask.scatter_(-1, topk_indices, 0.0)
    index_mask = index_mask.unsqueeze(1)  # [B, 1, S_full, T]
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask[..., :total_len]
        combined_mask = index_mask + causal_mask
    else:
        combined_mask = (
            attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
            if attention_mask is not None
            else index_mask
        )

    # DSA produces a dense 4D float mask incompatible with flash attention.
    attn_impl = "sdpa"

    # SDPA requires Q/K/V to have same head_dim; pad V if needed
    if self.qk_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(attn_impl, _eager_attention_forward)

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        combined_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    if self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    # ===== Ulysses SP: all-to-all back to scatter seq, gather heads =====
    # attention_interface returns [B, S_full, H/sp, v_head_dim] (already transposed internally)
    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    else:
        attn_output = attn_output.reshape(batch_size, full_seq_length, -1).contiguous()

    attn_output = self.o_proj(attn_output)

    topk_for_next = topk_indices if self.next_skip_topk else None
    return attn_output, attn_weights, topk_for_next


def _all_gather_seq(tensor: torch.Tensor, group, sp_size: int) -> torch.Tensor:
    """All-gather tensor along sequence dimension (dim=1)."""
    if sp_size <= 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(sp_size)]
    torch.distributed.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _is_flash_attention_requested(config):
    try:
        from transformers.utils.generic import is_flash_attention_requested

        return is_flash_attention_requested(config)
    except ImportError:
        return getattr(config, "_attn_implementation", None) in ("flash_attention_2", "flash_attention_3")


def _eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    """Fallback eager attention (imported at runtime to avoid circular deps)."""
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import eager_attention_forward

    return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
