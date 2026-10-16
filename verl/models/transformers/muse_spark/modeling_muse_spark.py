# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Muse Spark text model for FSDP2 LoRA RL; parameter names follow the HF checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_muse_spark import MuseSparkTextConfig


def rms_norm(x, eps, weight=None, *, weight_first=False, zero_centered=False, out_dtype=None):
    value = x.float()
    gain = None
    if weight is not None:
        gain = weight.float() + (1.0 if zero_centered else 0.0)
    if weight_first and gain is not None:
        value = value * gain
    value = value * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + eps)
    if not weight_first and gain is not None:
        value = value * gain
    return value.to(out_dtype or x.dtype)


class MuseSparkRMSNorm(nn.Module):
    def __init__(self, size, eps, *, weight_first=False, zero_centered=True):
        super().__init__()
        self.eps = eps
        self.weight_first = weight_first
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x, out_dtype=None):
        return rms_norm(
            x,
            self.eps,
            self.weight,
            weight_first=self.weight_first,
            zero_centered=self.zero_centered,
            out_dtype=out_dtype,
        )


class MuseSparkResidualGate(nn.Module):
    """stream * alpha + branch * beta with beta = sigmoid(g / T), alpha = sqrt(sigmoid(-g / T) * (1 + beta))."""

    def __init__(self, size, temperature):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(size))
        self.temperature = temperature

    def forward(self, stream, branch):
        w = self.gate.float() / self.temperature
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
        return alpha * stream.float() + beta * branch.float()


def apply_rope(x, position_ids, theta):
    """Half-split RoPE: q/k rows are in checkpoint order (vLLM reorders them for interleaved RoPE)."""
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) * 2 / x.shape[-1]))
    angles = position_ids.float().unsqueeze(-1) * inv_freq
    cos, sin = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
    x1, x2 = x.float()[..., :half], x.float()[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)


class MuseSparkAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5 * config.qk_scale_factor
        self.rope_theta = float(config.layer_rope_theta[layer_idx])
        self.sliding_window = (
            int(config.sliding_window) if config.layer_types[layer_idx] == "sliding_attention" else None
        )
        self.eps = config.rms_norm_eps
        q, kv, h = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, config.hidden_size
        self.q_proj = nn.Linear(h, q, bias=False)
        self.k_proj = nn.Linear(h, kv, bias=False)
        self.v_proj = nn.Linear(h, kv, bias=False)
        self.gate_proj = nn.Linear(h, q, bias=False)  # output gate
        self.o_proj = nn.Linear(q, h, bias=False)

    def _mask(self, attention_mask, seq_len, device):
        idx = torch.arange(seq_len, device=device)
        allowed = idx[None, :] <= idx[:, None]
        if self.sliding_window is not None:
            allowed &= idx[None, :] > idx[:, None] - self.sliding_window
        allowed = allowed[None, None]
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool()[:, None, None, :]
            allowed = allowed | torch.eye(seq_len, dtype=torch.bool, device=device)[None, None]  # no empty pad rows
        return allowed

    def forward(self, x, position_ids, attention_mask):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        gate = self.gate_proj(x).view(b, s, self.num_heads, self.head_dim)
        q, k = rms_norm(q, self.eps), rms_norm(k, self.eps)
        if self.rope_theta > 0:
            q, k = apply_rope(q, position_ids, self.rope_theta), apply_rope(k, position_ids, self.rope_theta)
        rep = self.num_heads // self.num_kv_heads
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.repeat_interleave(rep, 2).transpose(1, 2),
            v.repeat_interleave(rep, 2).transpose(1, 2),
            attn_mask=self._mask(attention_mask, s, x.device),
            scale=self.scaling,
        ).transpose(1, 2)
        out = (rms_norm(out, self.eps, out_dtype=torch.float32) * torch.sigmoid(gate.float())).to(x.dtype)
        return self.o_proj(out.flatten(-2))


class MuseSparkRouter(nn.Module):
    """Sigmoid scores, top-k on scores + bias, renormalized and scaled weights of the selected experts."""

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.num_local_experts, config.hidden_size, dtype=torch.float32))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_local_experts, dtype=torch.float32))
        self.top_k = config.num_experts_per_tok
        self.output_multiplier = config.output_multiplier
        self.route_scale = config.routed_scaling_factor

    def forward(self, x, routed_index=None):
        """``routed_index``: R3 top-k expert ids; all-zero rows keep the native selection."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = F.linear(x.float(), self.weight.float() * self.output_multiplier)
            scores = torch.sigmoid(logits)
            index = torch.topk(scores + self.e_score_correction_bias.float(), self.top_k, dim=-1).indices
            if routed_index is not None:
                replay = routed_index.to(index.device, torch.long)
                index = torch.where((replay != 0).any(-1, keepdim=True), replay, index)
            weights = scores.gather(-1, index)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-15) * self.route_scale
        return weights, index


try:  # as a PEFT tuner layer, PEFT trains, saves and disables it like its own LoRA
    from peft.tuners.tuners_utils import BaseTunerLayer as _TunerLayer
except ImportError:
    _TunerLayer = object


class MuseSparkExpertsLoRA(nn.Module, _TunerLayer):
    """Shared-outer expert LoRA: gate_up A and down B are shared by all experts, the inner factors are per expert."""

    adapter_layer_names = ("lora_gate_up_A", "lora_gate_up_B", "lora_down_A", "lora_down_B")

    def __init__(self, e, h, i, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self._active_adapter = "default"
        self._disable_adapters = False
        self.merged_adapters = []
        self.lora_gate_up_A = nn.ParameterDict({"default": nn.Parameter(torch.zeros(rank, h))})
        self.lora_gate_up_B = nn.ParameterDict({"default": nn.Parameter(torch.zeros(e, 2 * i, rank))})
        self.lora_down_A = nn.ParameterDict({"default": nn.Parameter(torch.zeros(e, rank, i))})
        self.lora_down_B = nn.ParameterDict({"default": nn.Parameter(torch.zeros(h, rank))})

    def reset_lora_parameters(self):
        """PEFT's default init: kaiming-uniform A, zero B."""
        for a in (self.lora_gate_up_A["default"], self.lora_down_A["default"]):
            nn.init.kaiming_uniform_(a.data.view(-1, a.shape[-1]), a=math.sqrt(5))
        nn.init.zeros_(self.lora_gate_up_B["default"])
        nn.init.zeros_(self.lora_down_B["default"])

    @property
    def active(self):
        return not self._disable_adapters


class MuseSparkExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        e, h, i = config.num_local_experts, config.moe_hidden_size, config.moe_intermediate_size
        self.num_experts = e
        self.gate_up_proj = nn.Parameter(torch.zeros(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.zeros(e, h, i))
        self.post_expert_norm = MuseSparkRMSNorm(h, config.post_norm_eps, weight_first=True, zero_centered=True)
        r = getattr(config, "expert_lora_rank", 0)
        if r:
            self.lora = MuseSparkExpertsLoRA(e, h, i, r, config.expert_lora_alpha)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        shared = hasattr(self, "lora") and self.lora.active
        if shared:
            lo, dt = self.lora, hidden_states.dtype  # PEFT may keep the factors in fp32
            s = lo.scaling
            xa = F.linear(hidden_states, lo.lora_gate_up_A["default"].to(dt))
            gu_b, d_a = lo.lora_gate_up_B["default"].to(dt).unbind(0), lo.lora_down_A["default"].to(dt).unbind(0)
            d_b = lo.lora_down_B["default"].to(dt)
        out = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
        for expert in torch.unique(top_k_index).tolist():
            token, slot = torch.where(top_k_index == expert)
            gate_up = F.linear(hidden_states[token], self.gate_up_proj[expert])
            if shared:
                gate_up = gate_up + s * F.linear(xa[token], gu_b[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
            y = F.linear(act, self.down_proj[expert])
            if shared:
                y = y + s * F.linear(F.linear(act, d_a[expert]), d_b)
            y = self.post_expert_norm(y, out_dtype=torch.float32)
            out = out.index_add(0, token, y * top_k_weights[token, slot, None].float())
        return out.to(hidden_states.dtype)


class MuseSparkMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = MuseSparkRouter(config)
        self.pre_expert_proj = nn.Linear(config.hidden_size, config.moe_hidden_size, bias=False)
        self.pre_expert_norm = MuseSparkRMSNorm(config.moe_hidden_size, config.rms_norm_eps, zero_centered=True)
        self.experts = MuseSparkExperts(config)
        self.post_expert_proj = nn.Linear(config.moe_hidden_size, config.hidden_size, bias=False)

    def forward(self, x, routed_index=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if routed_index is not None:
            routed_index = routed_index.reshape(-1, routed_index.shape[-1])
        weights, index = self.gate(flat, routed_index)
        pooled = self.pre_expert_norm(self.pre_expert_proj(flat))
        return self.post_expert_proj(self.experts(pooled, index, weights)).view(*shape[:-1], -1)


class MuseSparkDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        t = config.residual_gate_temperature
        self.input_layernorm = MuseSparkRMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)
        self.self_attn = MuseSparkAttention(config, layer_idx)
        self.post_attention_residual_gate = MuseSparkResidualGate(config.hidden_size, t)
        self.pre_feedforward_layernorm = MuseSparkRMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=True)
        self.mlp = MuseSparkMoE(config)
        self.post_feedforward_layernorm = MuseSparkRMSNorm(
            config.hidden_size, config.post_norm_eps, weight_first=True, zero_centered=True
        )
        self.post_feedforward_residual_gate = MuseSparkResidualGate(config.hidden_size, t)
        self.post_norm_eps = config.post_norm_eps

    def forward(self, stream, position_ids, attention_mask, routed_index=None):
        dtype = self.self_attn.q_proj.weight.dtype
        attn = self.self_attn(self.input_layernorm(stream, out_dtype=dtype), position_ids, attention_mask)
        stream = self.post_attention_residual_gate(stream, rms_norm(attn, self.post_norm_eps, out_dtype=torch.float32))
        ffn = self.mlp(self.pre_feedforward_layernorm(stream, out_dtype=dtype), routed_index)
        return self.post_feedforward_residual_gate(
            stream, self.post_feedforward_layernorm(ffn, out_dtype=torch.float32)
        )


class MuseSparkTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(MuseSparkDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = MuseSparkRMSNorm(config.hidden_size, config.rms_norm_eps, zero_centered=False)
        self.gradient_checkpointing = False

    def forward(self, input_ids, attention_mask, position_ids, routed_experts=None):
        """``routed_experts``: [B, S, num_layers, top_k] expert ids for R3."""
        stream = rms_norm(self.embed_tokens(input_ids), self.config.rms_norm_eps, out_dtype=torch.float32)
        for i, layer in enumerate(self.layers):
            routed = None if routed_experts is None else routed_experts[:, :, i]
            if self.gradient_checkpointing and self.training:
                stream = checkpoint(layer, stream, position_ids, attention_mask, routed, use_reentrant=False)
            else:
                stream = layer(stream, position_ids, attention_mask, routed)
        return self.norm(stream, out_dtype=self.embed_tokens.weight.dtype)


class MuseSparkModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = MuseSparkTextModel(config)


class MuseSparkPreTrainedModel(PreTrainedModel):
    config_class = MuseSparkTextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MuseSparkDecoderLayer"]
    _keep_in_fp32_modules = ["mlp.gate"]
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear | nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, MuseSparkExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, MuseSparkExpertsLoRA):
            module.reset_lora_parameters()
        elif isinstance(module, MuseSparkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.data.zero_()


class MuseSparkForCausalLM(MuseSparkPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MuseSparkModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if getattr(config, "expert_lora_rank", 0):
            # freeze the base: PEFT keeps existing trainability once the model holds a tuner layer
            for name, param in self.named_parameters():
                param.requires_grad_(".experts.lora." in name)

    def export_serving_lora(self, params, peft_scaling):
        """Expand the shared-outer expert LoRA to the per-expert layout vLLM loads (A rows expert-major, B columns
        ``rank * E + expert``); the expert scaling relative to ``peft_scaling`` is folded into B."""
        c = self.config.expert_lora_alpha / self.config.expert_lora_rank / peft_scaling
        out = {}
        for name, t in params.items():
            prefix, sep, factor = name.rpartition(".experts.lora.lora_")
            if not sep:
                out[name] = t
                continue
            e = self.config.num_local_experts
            stem = f"{prefix}.experts"
            if factor == "gate_up_A":  # (r, H) shared -> (E*r, H)
                out[f"{stem}.base_layer.lora_A.weight"] = t[None].expand(e, *t.shape).reshape(-1, t.shape[-1])
            elif factor == "gate_up_B":  # (E, 2I, r) -> (2I, r*E)
                out[f"{stem}.base_layer.lora_B.weight"] = (c * t).permute(1, 2, 0).reshape(t.shape[1], -1)
            elif factor == "down_A":  # (E, r, I) -> (E*r, I)
                out[f"{stem}.lora_A.weight"] = t.reshape(-1, t.shape[-1])
            elif factor == "down_B":  # (H, r) shared -> (H, r*E)
                out[f"{stem}.lora_B.weight"] = (c * t)[:, :, None].expand(*t.shape, e).reshape(t.shape[0], -1)
            else:
                raise ValueError(name)
        return out

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.language_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.language_model.gradient_checkpointing = False

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):  # read by PEFT
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, labels=None, routed_experts=None, **kwargs
    ):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        hidden = self.model.language_model(input_ids, attention_mask, position_ids, routed_experts)
        logits = self.lm_head(hidden).float() * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]), labels[..., 1:].reshape(-1), ignore_index=-100
            )
        return CausalLMOutput(loss=loss, logits=logits)


def register_muse_spark_training():
    AutoConfig.register(MuseSparkTextConfig.model_type, MuseSparkTextConfig, exist_ok=True)
    AutoModelForCausalLM.register(MuseSparkTextConfig, MuseSparkForCausalLM, exist_ok=True)


__all__ = ["MuseSparkForCausalLM", "MuseSparkPreTrainedModel", "register_muse_spark_training"]
