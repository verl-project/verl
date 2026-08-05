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
from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
)

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _call_accepts_kwarg(fn, name: str) -> bool:
    params = signature(fn).parameters
    return name in params or any(param.kind == param.VAR_KEYWORD for param in params.values())


def _prepare_packed_seq_idx(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: Optional[torch.LongTensor]):
    try:
        from fla.ops.utils import prepare_sequence_ids

        return prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu).to(torch.int32).unsqueeze(0)
    except Exception:
        offsets = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.detach().cpu()).tolist()
        seq_idx = [
            torch.full((end - start,), i, device=cu_seqlens.device, dtype=torch.int32)
            for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
        ]
        return torch.cat(seq_idx, dim=0).unsqueeze(0)


def _split_packed_args(
    cu_seqlens: torch.LongTensor,
    tensors: tuple[torch.Tensor, ...],
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    dim: int = 1,
):
    offsets = (cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.detach().cpu()).tolist()
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        chunks = []
        for tensor in tensors:
            split_dim = dim if dim >= 0 else tensor.ndim + dim
            slices = [slice(None)] * tensor.ndim
            slices[split_dim] = slice(start, end)
            chunks.append(tensor[tuple(slices)])
        yield tuple(chunks)


class _ConvPrefixExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tails: torch.Tensor, group: dist.ProcessGroup):
        from fla.ops.cp.comm import conv_cp_send_recv_fwd

        ctx.group = group
        return conv_cp_send_recv_fwd(tails.contiguous(), group)

    @staticmethod
    def backward(ctx, grad_prefix: torch.Tensor):
        from fla.ops.cp.comm import conv_cp_send_recv_bwd

        return conv_cp_send_recv_bwd(grad_prefix.contiguous(), ctx.group), None


def _build_fla_cp_context(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: Optional[torch.LongTensor],
    conv_kernel_size: int,
):
    group = get_ulysses_sequence_parallel_group()
    if group is None:
        return None
    from fla.ops.cp.context import build_cp_context

    return build_cp_context(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        group=group,
        conv1d_kernel_size=conv_kernel_size,
    )


def _prepend_cp_conv_prefix(mixed_qkv: torch.Tensor, cp_context) -> tuple[torch.Tensor, int]:
    prefix_len = max(int(cp_context.conv1d_kernel_size or 0) - 1, 0)
    if prefix_len == 0:
        return mixed_qkv, 0

    if mixed_qkv.shape[-1] >= prefix_len:
        tails = mixed_qkv[..., -prefix_len:]
    else:
        tails = F.pad(mixed_qkv, (prefix_len - mixed_qkv.shape[-1], 0))

    prefix = _ConvPrefixExchange.apply(tails, cp_context.group)
    valid_prefix_len = min(prefix_len, int(cp_context.pre_num_conv_tokens or 0))
    if valid_prefix_len < prefix_len:
        prefix = prefix.clone()
        prefix[..., : prefix_len - valid_prefix_len] = 0
    return torch.cat((prefix, mixed_qkv), dim=-1), prefix_len


def _as_channel_last_conv1d_input(x: torch.Tensor) -> torch.Tensor:
    if x.stride(1) == 1:
        return x
    return x.transpose(1, 2).contiguous().transpose(1, 2)


def _prepare_cp_conv_seq_idx(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: Optional[torch.LongTensor],
    prefix_len: int,
):
    seq_idx = _prepare_packed_seq_idx(cu_seqlens, cu_seqlens_cpu)
    if prefix_len == 0:
        return seq_idx
    prefix_idx = seq_idx[:, :1].expand(seq_idx.shape[0], prefix_len)
    return torch.cat((prefix_idx, seq_idx), dim=1)


def _packed_causal_conv1d_fallback(
    self,
    mixed_qkv: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
):
    outputs = []
    for (segment,) in _split_packed_args(cu_seqlens, (mixed_qkv,), cu_seqlens_cpu=cu_seqlens_cpu, dim=2):
        outputs.append(F.silu(self.conv1d(segment)[:, :, : segment.shape[-1]]))
    return torch.cat(outputs, dim=-1)


def _packed_chunk_gated_delta_rule(self, query, key, value, g, beta, cu_seqlens, cu_seqlens_cpu, cp_context=None):
    kwargs = {
        "g": g,
        "beta": beta,
        "initial_state": None,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": True,
    }
    if cu_seqlens is None:
        return self.chunk_gated_delta_rule(query, key, value, **kwargs)

    if cp_context is not None:
        if not _call_accepts_kwarg(self.chunk_gated_delta_rule, "cp_context"):
            raise NotImplementedError("Qwen3.5 Ulysses SP requires FLA chunk_gated_delta_rule cp_context support.")
        kwargs["cp_context"] = cp_context
        return self.chunk_gated_delta_rule(query, key, value, **kwargs)

    if _call_accepts_kwarg(self.chunk_gated_delta_rule, "cu_seqlens"):
        kwargs["cu_seqlens"] = cu_seqlens
        if _call_accepts_kwarg(self.chunk_gated_delta_rule, "cu_seqlens_cpu"):
            kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
        return self.chunk_gated_delta_rule(query, key, value, **kwargs)

    outputs = []
    for q_i, k_i, v_i, g_i, beta_i in _split_packed_args(
        cu_seqlens, (query, key, value, g, beta), cu_seqlens_cpu=cu_seqlens_cpu
    ):
        split_kwargs = dict(kwargs)
        split_kwargs["g"] = g_i
        split_kwargs["beta"] = beta_i
        out_i, _ = self.chunk_gated_delta_rule(q_i, k_i, v_i, **split_kwargs)
        outputs.append(out_i)
    return torch.cat(outputs, dim=1), None


def qwen3_5_gated_delta_net_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
):
    if cu_seqlens is not None and cache_params is not None:
        raise NotImplementedError("Packed Qwen3.5 linear attention does not support cached forward.")

    module = import_module(self.__class__.__module__)
    hidden_states = module.apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape
    cp_context = None
    model_cu_seqlens = cu_seqlens
    model_cu_seqlens_cpu = cu_seqlens_cpu
    if cu_seqlens is not None:
        if batch_size != 1:
            raise ValueError("Packed Qwen3.5 linear attention expects batch size 1.")
        total_seq_len = int(cu_seqlens[-1].item())
        ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
        if total_seq_len != seq_len:
            if (
                ulysses_sp_size <= 1
                or total_seq_len % ulysses_sp_size != 0
                or seq_len != total_seq_len // ulysses_sp_size
            ):
                raise ValueError(f"Packed cu_seqlens end {total_seq_len} does not match seq_len {seq_len}.")
            cp_context = _build_fla_cp_context(cu_seqlens, cu_seqlens_cpu, self.conv_kernel_size)
            model_cu_seqlens = cp_context.cu_seqlens
            model_cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1
    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            conv_prefix_len = 0
            conv_input = mixed_qkv
            if cp_context is not None:
                conv_input, conv_prefix_len = _prepend_cp_conv_prefix(mixed_qkv, cp_context)
            if model_cu_seqlens is not None:
                conv_input = _as_channel_last_conv1d_input(conv_input)
            seq_idx = (
                _prepare_cp_conv_seq_idx(model_cu_seqlens, model_cu_seqlens_cpu, conv_prefix_len)
                if model_cu_seqlens is not None
                else None
            )
            conv_output = self.causal_conv1d_fn(
                x=conv_input,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
            mixed_qkv = conv_output[..., conv_prefix_len:] if conv_prefix_len else conv_output
        elif model_cu_seqlens is not None:
            if cp_context is not None:
                mixed_qkv, conv_prefix_len = _prepend_cp_conv_prefix(mixed_qkv, cp_context)
                model_cu_seqlens = model_cu_seqlens + conv_prefix_len
                model_cu_seqlens[0] = 0
                if model_cu_seqlens_cpu is not None:
                    model_cu_seqlens_cpu = model_cu_seqlens_cpu + conv_prefix_len
                    model_cu_seqlens_cpu[0] = 0
                mixed_qkv = _packed_causal_conv1d_fallback(self, mixed_qkv, model_cu_seqlens, model_cu_seqlens_cpu)[
                    ..., conv_prefix_len:
                ]
            else:
                mixed_qkv = _packed_causal_conv1d_fallback(self, mixed_qkv, model_cu_seqlens, model_cu_seqlens_cpu)
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            self.key_dim,
            self.key_dim,
            self.value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = _packed_chunk_gated_delta_rule(
            self, query, key, value, g, beta, model_cu_seqlens, model_cu_seqlens_cpu, cp_context
        )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


def qwen3_5_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    **kwargs,
):
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    cu_seqlens_cpu = kwargs.pop("cu_seqlens_cpu", None)
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states

    return hidden_states


def fast_pos_embed_interpolate(self, grid_thw):
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    # Modification: # Get device from grid_thw to avoid self.pos_embed being on CPU when FSDP2 enables cpu_offload
    device = grid_thw.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def _get_input_embeds(
    model: "Qwen3_5CausalLMOutputWithPast",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def qwen3_5_base_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    kwargs.update(input_kwargs)
    return self.language_model(
        input_ids=None,
        **kwargs,
    )


@dataclass
class Qwen3_5CausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_with_normal_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_cpu is not None:
        kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    return Qwen3_5CausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_cpu is not None:
        kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # See `dense_common.forward_with_torch_backend` for the `shift_labels`
    # rationale (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    vocab_weights = self.lm_head.weight
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()

    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
    if shift_labels is None and ulysses_sequence_parallel_size > 1:
        rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
            rolled_labels, position_ids_rmpad=None, sp_size=ulysses_sequence_parallel_size
        )
    hidden_states = hidden_states.to(vocab_weights.dtype)  # bf16 to float
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=vocab_weights,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_cpu: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    if cu_seqlens_cpu is not None:
        kwargs["cu_seqlens_cpu"] = cu_seqlens_cpu
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # See `dense_common.forward_with_torch_backend` for the `shift_labels`
    # rationale (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")
    ulysses_sequence_parallel_size = get_ulysses_sequence_parallel_world_size()
    if shift_labels is None and ulysses_sequence_parallel_size > 1:
        rolled_labels, _, _ = ulysses_pad_and_slice_inputs(
            rolled_labels, position_ids_rmpad=None, sp_size=ulysses_sequence_parallel_size
        )

    vocab_weights = self.lm_head.weight
    hidden_states = hidden_states.to(vocab_weights.dtype)
    if isinstance(vocab_weights, DTensor):
        vocab_weights = vocab_weights.full_tensor()

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        vocab_weights,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
