"""BagelForTraining – FSDP-compatible BAGEL MoT module for flow-matching training.

Ported from vllm-omni/BAGEL with the following correctness-critical details:
  * MoT (Mixture-of-Thought): dual pathways for text vs generation tokens
  * start_of_image / end_of_image boundary tokens are required
  * All latent tokens share ONE RoPE position (spatial via 2-D sincos embed)
  * QK-norm + RoPE in float32; cast to bfloat16 only for SDPA
  * Attention mask: text-context is causal & cannot see image region

Dependencies: torch, numpy, safetensors, einops, transformers (AutoTokenizer)
NO dependency on vllm or vllm-omni.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# ===================================================================
#  Config
# ===================================================================

@dataclass
class BagelTrainingConfig:
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    # Bagel-specific
    latent_patch_size: int = 2
    max_latent_size: int = 32
    latent_channel: int = 16
    vae_downsample: int = 8
    start_of_image_id: int = 151652   # <|vision_start|>
    end_of_image_id: int = 151653     # <|vision_end|>

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size ** 2 * self.latent_channel

    @classmethod
    def from_model_path(cls, model_path: str) -> "BagelTrainingConfig":
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            root_cfg = json.load(f)
        llm = root_cfg.get("llm_config", {})
        vae = root_cfg.get("vae_config", {})
        return cls(
            hidden_size=llm.get("hidden_size", 3584),
            intermediate_size=llm.get("intermediate_size", 18944),
            num_hidden_layers=llm.get("num_hidden_layers", 28),
            num_attention_heads=llm.get("num_attention_heads", 28),
            num_key_value_heads=llm.get("num_key_value_heads", 4),
            vocab_size=llm.get("vocab_size", 152064),
            rms_norm_eps=llm.get("rms_norm_eps", 1e-6),
            rope_theta=llm.get("rope_theta", 1_000_000.0),
            max_position_embeddings=llm.get("max_position_embeddings", 32768),
            latent_patch_size=root_cfg.get("latent_patch_size", 2),
            max_latent_size=root_cfg.get("max_latent_size", 32),
            latent_channel=vae.get("z_channels", 16),
            vae_downsample=vae.get("downsample", 8),
        )


# ===================================================================
#  VAE AutoEncoder (from FLUX / BAGEL, Apache-2.0)
# ===================================================================

@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    downsample: int = 8
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] | tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 16
    scale_factor: float = 0.3611
    shift_factor: float = 0.1159


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        return x + self.proj_out(h_)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        h = _swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = _swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class _Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, resolution: int, in_channels: int, ch: int,
                 ch_mult: list[int], num_res_blocks: int, z_channels: int):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = _Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = _swish(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, ch: int, out_ch: int, ch_mult: list[int],
                 num_res_blocks: int, in_channels: int, resolution: int, z_channels: int):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = _swish(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        return mean


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(
            resolution=params.resolution, in_channels=params.in_channels,
            ch=params.ch, ch_mult=list(params.ch_mult),
            num_res_blocks=params.num_res_blocks, z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution, in_channels=params.in_channels,
            ch=params.ch, out_ch=params.out_ch, ch_mult=list(params.ch_mult),
            num_res_blocks=params.num_res_blocks, z_channels=params.z_channels,
        )
        self.reg = DiagonalGaussian()
        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    def encode(self, x: Tensor) -> Tensor:
        z = self.reg(self.encoder(x))
        return self.scale_factor * (z - self.shift_factor)

    def decode(self, z: Tensor) -> Tensor:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def load_ae(path: str) -> tuple[AutoEncoder, AutoEncoderParams]:
    """Load VAE autoencoder from a safetensors checkpoint."""
    params = AutoEncoderParams()
    ae = AutoEncoder(params)
    if path is not None:
        from safetensors.torch import load_file
        sd = load_file(path)
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        if missing:
            print(f"VAE load: {len(missing)} missing keys")
        if unexpected:
            print(f"VAE load: {len(unexpected)} unexpected keys")
    return ae, params


# ===================================================================
#  Tokenizer & data utilities (replaces BAGEL/data/data_utils.py)
# ===================================================================

def load_tokenizer(model_path: str):
    """Load tokenizer with special tokens for BAGEL using transformers.AutoTokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    all_special = set()
    for v in tokenizer.special_tokens_map.values():
        if isinstance(v, str):
            all_special.add(v)
        elif isinstance(v, list):
            all_special.update(v)

    new_tokens = []
    for t in ['<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|vision_end|>']:
        if t not in all_special and t not in tokenizer.get_vocab():
            new_tokens.append(t)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)

    new_token_ids = {
        'bos_token_id': tokenizer.convert_tokens_to_ids('<|im_start|>'),
        'eos_token_id': tokenizer.convert_tokens_to_ids('<|im_end|>'),
        'start_of_image': tokenizer.convert_tokens_to_ids('<|vision_start|>'),
        'end_of_image': tokenizer.convert_tokens_to_ids('<|vision_end|>'),
    }
    return tokenizer, new_token_ids


def get_flattened_position_ids(img_h: int, img_w: int,
                               patch_size: int, max_num_patches_per_side: int) -> torch.Tensor:
    """Compute flattened 2-D position IDs for latent patches (extrapolate mode)."""
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


# ===================================================================
#  Transformer building blocks
# ===================================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class BagelMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ===================================================================
#  RoPE helpers
# ===================================================================

def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(q, k, cos, sin):
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int = 32768,
                 theta: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor):
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq.to(position_ids.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


# ===================================================================
#  MoT Attention & Layer
# ===================================================================

class BagelMoTAttention(nn.Module):
    """MoT attention with separate standard and generation projections."""

    def __init__(self, config: BagelTrainingConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
    ) -> Tensor:
        B, L, _ = hidden_states.shape
        text_idx = text_mask.nonzero(as_tuple=True)
        latent_idx = latent_mask.nonzero(as_tuple=True)

        q = hidden_states.new_zeros(B, L, self.num_heads * self.head_dim)
        k = hidden_states.new_zeros(B, L, self.num_kv_heads * self.head_dim)
        v = hidden_states.new_zeros(B, L, self.num_kv_heads * self.head_dim)

        text_hs = hidden_states[text_idx]
        q[text_idx] = self.q_proj(text_hs)
        k[text_idx] = self.k_proj(text_hs)
        v[text_idx] = self.v_proj(text_hs)

        latent_hs = hidden_states[latent_idx]
        q[latent_idx] = self.q_proj_moe_gen(latent_hs)
        k[latent_idx] = self.k_proj_moe_gen(latent_hs)
        v[latent_idx] = self.v_proj_moe_gen(latent_hs)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim)

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        q_normed = q.new_zeros(q.shape)
        k_normed = k.new_zeros(k.shape)
        q_normed[text_idx] = self.q_norm(q[text_idx])
        k_normed[text_idx] = self.k_norm(k[text_idx])
        q_normed[latent_idx] = self.q_norm_moe_gen(q[latent_idx])
        k_normed[latent_idx] = self.k_norm_moe_gen(k[latent_idx])

        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_normed, k_normed = _apply_rotary_emb(q_normed, k_normed, cos, sin)

        q_normed = q_normed.to(torch.bfloat16)
        k_normed = k_normed.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k_normed = k_normed.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, L, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, L, self.num_heads, self.head_dim)

        # Split attention: no boolean mask → SDPA uses flash backend
        # Original vllm-omni uses flash_attn_varlen_func; matching that
        # requires avoiding the "math" fallback that boolean masks trigger.
        #   Text tokens: causal self-attention (only see prior text)
        #   Image tokens (soi/latent/eoi): full attention to everything
        q_normed = q_normed.transpose(1, 2)   # (B, H, L, D)
        k_normed = k_normed.transpose(1, 2)
        v = v.transpose(1, 2)

        if L_ctx > 0:
            # Text self-attention (causal, flash backend)
            text_out = F.scaled_dot_product_attention(
                q_normed[:, :, :L_ctx],
                k_normed[:, :, :L_ctx],
                v[:, :, :L_ctx],
                is_causal=True,
            )
            # Image attention to full sequence (no mask, flash backend)
            img_out = F.scaled_dot_product_attention(
                q_normed[:, :, L_ctx:],
                k_normed,
                v,
                is_causal=False,
            )
            attn_out = torch.cat([text_out, img_out], dim=2)
        else:
            attn_out = F.scaled_dot_product_attention(
                q_normed, k_normed, v, is_causal=False,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)

        out = hidden_states.new_zeros(B, L, self.hidden_size)
        out[text_idx] = self.o_proj(attn_out[text_idx])
        out[latent_idx] = self.o_proj_moe_gen(attn_out[latent_idx])
        return out


class BagelMoTLayer(nn.Module):
    def __init__(self, config: BagelTrainingConfig):
        super().__init__()
        self.self_attn = BagelMoTAttention(config)
        self.mlp = BagelMLP(config.hidden_size, config.intermediate_size)
        self.mlp_moe_gen = BagelMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
    ) -> Tensor:
        text_idx = text_mask.nonzero(as_tuple=True)
        latent_idx = latent_mask.nonzero(as_tuple=True)

        normed = hidden_states.new_zeros(hidden_states.shape)
        normed[text_idx] = self.input_layernorm(hidden_states[text_idx])
        normed[latent_idx] = self.input_layernorm_moe_gen(hidden_states[latent_idx])

        attn_out = self.self_attn(normed, cos, sin, text_mask, latent_mask, L_ctx)
        hidden_states = hidden_states + attn_out

        residual = hidden_states
        mlp_out = hidden_states.new_zeros(hidden_states.shape)
        mlp_out[text_idx] = self.mlp(self.post_attention_layernorm(hidden_states[text_idx]))
        mlp_out[latent_idx] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(hidden_states[latent_idx])
        )
        hidden_states = residual + mlp_out
        return hidden_states


# ===================================================================
#  Position embedding helpers
# ===================================================================

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = freq_dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        pos_embed = _get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.pos_embed = nn.Parameter(
            torch.from_numpy(pos_embed).float(), requires_grad=False
        )

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.pos_embed[position_ids]


# ===================================================================
#  Main module: BagelForTraining
# ===================================================================

class BagelForTraining(nn.Module):
    """Standalone Bagel MoT module for FlowGRPO FSDP training.

    Forward signature:
        hidden_states:  (B, L_latent, patch_latent_dim) — noisy latent patches
        timestep:       (B,) — diffusion timestep scalars
        text_token_ids: (B, L_text) — tokenized prompt IDs (with bos/eos)
        latent_pos_ids: (B, L_latent) — 2-D position indices for latent patches
    """

    def __init__(self, config: BagelTrainingConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BagelMoTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.vae2llm = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(config.max_latent_size, config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        text_token_ids: Optional[Tensor],
        latent_pos_ids: Tensor,
        **kwargs,
    ) -> tuple[Tensor]:
        """Forward pass.

        When text_token_ids is None the sequence is [soi, latent, eoi] only
        (no text context). This is used for the CFG unconditional pass.
        """
        B = hidden_states.shape[0]
        L_latent = hidden_states.shape[1]
        dev = hidden_states.device

        # 1. Embed text context
        if text_token_ids is not None:
            text_embeds = self.embed_tokens(text_token_ids)
            L_ctx = text_embeds.shape[1]
        else:
            L_ctx = 0

        # 2. SOI / EOI boundary tokens
        soi_ids = torch.full((B, 1), self.config.start_of_image_id, dtype=torch.long, device=dev)
        eoi_ids = torch.full((B, 1), self.config.end_of_image_id, dtype=torch.long, device=dev)
        soi_emb = self.embed_tokens(soi_ids)
        eoi_emb = self.embed_tokens(eoi_ids)

        # 3. Latent projection
        t_emb = self.time_embedder(timestep)
        pos_emb = self.latent_pos_embed(latent_pos_ids)
        latent_embeds = self.vae2llm(hidden_states) + t_emb.unsqueeze(1) + pos_emb
        latent_embeds = latent_embeds.to(soi_emb.dtype)

        # 4. Sequence: [text?, soi, latent_0..N, eoi]
        L_total = L_ctx + 1 + L_latent + 1
        if L_ctx > 0:
            sequence = torch.cat([text_embeds, soi_emb, latent_embeds, eoi_emb], dim=1)
        else:
            sequence = torch.cat([soi_emb, latent_embeds, eoi_emb], dim=1)

        # 5. MoT routing masks
        #    text pathway: text_ctx + soi + eoi
        #    gen pathway:  latent tokens only
        text_mask = torch.zeros(B, L_total, dtype=torch.bool, device=dev)
        text_mask[:, : L_ctx + 1] = True   # text + soi
        text_mask[:, -1] = True             # eoi
        latent_mask = ~text_mask

        # 6. RoPE positions
        if L_ctx > 0:
            ctx_pos = torch.arange(L_ctx, device=dev)
            img_pos = ctx_pos.new_full((1 + L_latent + 1,), L_ctx)
            position_ids = torch.cat([ctx_pos, img_pos]).unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.zeros(1, L_total, dtype=torch.long, device=dev).expand(B, -1)
        cos, sin = self.rotary_emb(position_ids)

        # 7. Transformer layers (split attention: text causal + image full)
        for layer in self.layers:
            sequence = layer(sequence, cos, sin, text_mask, latent_mask, L_ctx)

        # 8. Final norm with MoT routing
        normed = sequence.new_zeros(sequence.shape)
        t_idx = text_mask.nonzero(as_tuple=True)
        l_idx = latent_mask.nonzero(as_tuple=True)
        normed[t_idx] = self.norm(sequence[t_idx])
        normed[l_idx] = self.norm_moe_gen(sequence[l_idx])

        # 9. Extract latent output
        latent_output = normed[:, L_ctx + 1 : L_ctx + 1 + L_latent, :]
        velocity = self.llm2vae(latent_output)

        return (velocity,)

    # ------------------------------------------------------------------
    #  Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16) -> "BagelForTraining":
        config = BagelTrainingConfig.from_model_path(model_path)
        ckpt_path = os.path.join(model_path, "ema.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)

        if "latent_pos_embed.pos_embed" in state_dict:
            actual_len = state_dict["latent_pos_embed.pos_embed"].shape[0]
            grid = int(actual_len ** 0.5)
            if grid * grid == actual_len and grid != config.max_latent_size:
                config.max_latent_size = grid

        model = cls(config)
        mapped = _map_checkpoint_to_training(state_dict, config)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            import logging
            logging.getLogger(__name__).warning(
                f"Missing keys when loading BagelForTraining: {len(missing)} keys"
            )

        model = model.to(torch_dtype)
        return model


def _map_checkpoint_to_training(
    state_dict: dict[str, Tensor], config: BagelTrainingConfig
) -> dict:
    """Map ema.safetensors keys to BagelForTraining parameter names."""
    mapped: dict[str, Tensor] = {}
    for src_key, tensor in state_dict.items():
        dst_key: str | None = None
        if src_key.startswith("language_model.model."):
            dst_key = src_key[len("language_model.model."):]
        elif src_key.startswith("language_model."):
            continue
        elif src_key.startswith(("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")):
            dst_key = src_key
        if dst_key is not None:
            mapped[dst_key] = tensor
    return mapped
