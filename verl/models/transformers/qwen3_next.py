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

import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

from verl.utils.ulysses import (
    all_gather_tensor,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.cp import build_cp_context
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError as err:
    raise ImportError("Please install flash-linear-attention for Qwen3-Next") from err

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group):
        ctx.group = group
        ctx.part_size = local_tensor.size(0)
        return all_gather_tensor(local_tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_local = torch.empty(
            (ctx.part_size, *grad_output.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(grad_local, grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_local, None


# Adapted from https://github.com/huggingface/transformers/blob/c9ea365a7b56326418769a4ba4682864d407ed63/src/transformers/models/qwen3_next/modular_qwen3_next.py#L428
class PatchQwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ShortConvolution(
            hidden_size=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
        )

        # projection of the input hidden states
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
            device=None,
            dtype=config.dtype if config.dtype is not None else torch.get_default_dtype(),
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        cp_context=None,
    ):
        if cp_context is not None:
            use_cp_mode = True
            cu_seqlens = cp_context.cu_seqlens
        elif cu_seqlens is not None:
            # pre-process: [bsz, seq, h] -> [seq, bsz, h] -> [seq * sp, bsz, h] -> [bsz, seq * sp, h]
            use_cp_mode = False
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            hidden_states = _AllGather.apply(hidden_states, get_ulysses_sequence_parallel_group())
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            raise ValueError("cu_seqlens or cp_context is required")

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)

        mixed_qkv, _ = self.conv1d(
            x=mixed_qkv,
            cu_seqlens=cu_seqlens,
            cp_context=cp_context if use_cp_mode else None,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_cp_mode:
            core_attn_out, _ = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cp_context=cp_context,
            )
        else:
            core_attn_out, _ = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        if not use_cp_mode:
            # post-process: [bsz, seq * sp, h] -> [bsz, seq, h]
            output = slice_input_tensor(output, dim=1, padding=False)
        return output


# Adapted from https://github.com/huggingface/transformers/blob/c9ea365a7b56326418769a4ba4682864d407ed63/src/transformers/models/qwen3_next/modular_qwen3_next.py#L680
def patch_qwen3_next_decoder_layer_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    cache_position=None,
    gdn_use_cp: bool = True,
    **kwargs,
):
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Token Mixer
    if self.layer_type == "linear_attention":
        # 1. Get the global position ids
        ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
        ulysses_sp_group = get_ulysses_sequence_parallel_group()
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=ulysses_sp_group)
        position_ids = torch.concat(position_ids_list, dim=-1)
        # 2. Get the cu_seqlens by position_ids
        (cu_seqlens_q, _), _ = prepare_fa_kwargs_from_position_ids(position_ids)
        if gdn_use_cp:
            cp_context = build_cp_context(
                cu_seqlens_q,
                group=ulysses_sp_group,
                conv1d_kernel_size=self.linear_attn.conv_kernel_size,
            )
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cp_context=cp_context,
            )
        else:
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens_q,
            )
    elif self.layer_type == "full_attention":
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # For the MoE layers, we need to unpack
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states
