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
"""MiniMax M3 text model for FSDP2 training.

Written from the vLLM implementation (``vllm/models/minimax_m3/nvidia/model.py``)
and the released checkpoint. Only the text backbone is modeled; the vision tower,
projector and MTP heads are not part of the training graph.

Precision choices (each item follows vLLM's computation for that op):

* Gemma RMSNorm statistics in FP32, output in the activation dtype.
* Per-head QK norm followed by partial NeoX RoPE, both in FP32, one rounding to
  the activation dtype. cos/sin are computed in FP32 and rounded to BF16, as in
  vLLM's ``cos_sin_cache``.
* Router logits, sigmoid, bias-corrected selection and the routing weights in
  FP32 (vLLM's ``GateLinear`` keeps weight and output in FP32).
* Indexer block scores in FP32.

The trainer runs the forward under a BF16 autocast region, which downcasts
matmul-like ops even when their inputs are FP32. Every FP32 section above
therefore disables autocast locally.

Known differences from vLLM serving (not emulated):

* Adapter add order: an adapter delta is added to the FP32 base product and the
  sum is rounded once; vLLM adds it to the BF16-rounded product.
* Routed-expert FC2 finalize: each expert's ``base + delta`` is rounded, then
  weighted and summed in FP32; vLLM accumulates the routing-weighted base outputs
  and the weighted deltas separately before its final store.
* Residual add and RMSNorm: the residual sum is rounded to BF16 before the norm;
  vLLM's fused add + RMSNorm normalizes the FP32 sum.
* Tensor parallelism: row-parallel projections and the FC2 shards are summed in
  FP32 here; vLLM rounds rank-local outputs before the TP reduction.
* The NVFP4 KV cache and indexer cache of the rollout are not emulated.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import MISSING, dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .act_quant import NVFP4ActQuantLinear

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # transformers < 4.52
    GradientCheckpointingLayer = nn.Module

__all__ = ["MiniMaxM3Args", "MiniMaxM3ForCausalLM"]

# Flash for the dense causal case, memory-efficient for block masks (and FP32).
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


def _fp32_region(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


@dataclass
class MiniMaxM3Args:
    """Text backbone hyper-parameters, read from the checkpoint's ``text_config``.

    Defaults equal vLLM's ``MiniMaxM3TextConfig`` for keys the checkpoint omits
    (only ``swiglu_beta`` for the released checkpoint).
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    dense_intermediate_size: int
    shared_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    swiglu_alpha: float
    swiglu_limit: float
    num_local_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    routed_scaling_factor: float
    moe_layer_freq: list[int]
    sparse_attention_config: dict[str, Any]
    swiglu_beta: float = 1.0
    tie_word_embeddings: bool = False
    # Training-only switches (not checkpoint semantics), set through
    # ``text_config`` overrides (see ``_training_overrides``).
    # Routed-expert NVFP4 activation QAT emulating the rollout's quantizer (act_quant.py).
    expert_activation_qat: bool = False
    # FC2-input scales are per tensor-parallel shard of the intermediate dim in the
    # rollout (TRT-LLM MoE quantizes each rank's I/TP columns): set to rollout TP.
    expert_qat_tp_shards: int = 4
    # "kernel": FlashInfer's per-token NVFP4 quantizer, as the rollout (bitwise equal to it);
    # "torch": the IEEE PyTorch mirror (~1e-4 of codes differ by one step).
    expert_qat_backend: str = "kernel"
    # MXFP8 activation QAT of the attention / shared-expert / dense-MLP projections:
    # the rollout runs them as MXFP8 W8A8 GEMMs that quantize their input.
    linear_activation_qat: bool = False

    _REQUIRED_EQUAL = {
        "use_gemma_norm": True,
        "use_qk_norm": True,
        "qk_norm_type": "per_head",
        "hidden_act": "swigluoai",
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "attention_output_gate": False,
    }

    @classmethod
    def from_text_config(cls, text: dict[str, Any], overrides: Optional[dict[str, Any]] = None) -> MiniMaxM3Args:
        for key, expected in cls._REQUIRED_EQUAL.items():
            if key in text and text[key] != expected:
                raise ValueError(f"MiniMax M3 training model requires {key}={expected!r}, got {text[key]!r}")
        if text.get("partial_rotary_factor") is not None:
            rotary_from_factor = int(text["head_dim"] * text["partial_rotary_factor"])
            if text.get("rotary_dim", rotary_from_factor) != rotary_from_factor:
                raise ValueError("rotary_dim and partial_rotary_factor disagree")
        sparse = dict(text["sparse_attention_config"])
        if not sparse.get("use_sparse_attention", False):
            raise ValueError("Only the all-sparse-attention MiniMax M3 layout is implemented")
        if sparse.get("sparse_score_type", "max") != "max":
            raise ValueError(f"Unsupported sparse_score_type {sparse.get('sparse_score_type')!r}")
        for flag in ("sparse_enable_kv_share", "sparse_enable_vo_rope", "sparse_enable_estimate_lse"):
            if sparse.get(flag, False):
                raise ValueError(f"Unsupported sparse attention option {flag}")
        if any(f == 0 for f in sparse.get("sparse_disable_index_value", [])):
            raise ValueError("index value/output projections are not implemented")
        if sparse["sparse_num_index_heads"] != text["num_key_value_heads"]:
            raise ValueError("vLLM requires sparse_num_index_heads == num_key_value_heads")
        names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {k: v for k, v in text.items() if k in names}
        values["sparse_attention_config"] = sparse
        values.update(overrides or {})
        args = cls(**values)
        n = args.num_hidden_layers
        freq = sparse.get("sparse_attention_freq", [1] * n)
        if len(freq) < n or any(f == 0 for f in freq[:n]):
            raise ValueError("Every layer must use sparse attention")
        if len(args.moe_layer_freq) < n:
            raise ValueError("moe_layer_freq shorter than num_hidden_layers")
        return args

    def is_moe_layer(self, layer_id: int) -> bool:
        return self.moe_layer_freq[layer_id] != 0


def _read_text_config(config) -> dict[str, Any]:
    """The checkpoint's raw ``text_config`` dict.

    The shipped config class coerces ``text_config`` through an unrelated
    Transformers config class, which may rename or drop keys. Reading
    ``config.json`` keeps the checkpoint as the only source of truth.
    """
    path = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
    if not path or not os.path.isfile(os.path.join(path, "config.json")):
        raise FileNotFoundError(
            f"MiniMax M3 training model needs the checkpoint's config.json in a local directory, got {path!r}"
        )
    with open(os.path.join(path, "config.json")) as f:
        raw = json.load(f)
    raw_text = dict(raw.get("text_config", raw))
    _reject_config_overrides(config, raw_text)
    return raw_text


def _reject_config_overrides(config, raw_text: dict[str, Any]) -> None:
    """Fail if the in-memory config disagrees with ``config.json`` on a checkpoint key.

    The model is built from ``config.json``; silently ignoring an in-memory
    override (for example ``override_config.text_config.rope_theta``) would train
    a different model than the config object reports.
    """
    text = getattr(config, "text_config", None)
    if text is None:
        return
    live = text if isinstance(text, dict) else (text.to_dict() if hasattr(text, "to_dict") else vars(text))
    # Only the keys this model consumes: a config class may normalize unrelated ones.
    consumed = set(MiniMaxM3Args.__dataclass_fields__) | set(MiniMaxM3Args._REQUIRED_EQUAL)  # type: ignore[attr-defined]
    consumed |= {"partial_rotary_factor"}
    fields = MiniMaxM3Args.__dataclass_fields__  # type: ignore[attr-defined]
    changed = {}
    for key in sorted(consumed - set(_TRAINING_SWITCHES)):
        if key not in live or live[key] is None:
            continue
        if key in raw_text:
            source = raw_text[key]
        elif key in fields and fields[key].default is not MISSING:
            source = fields[key].default  # config.json omits it: the model uses the default
        else:
            continue
        # JSON round trip: a config class may hold tuples where the file has lists.
        if json.loads(json.dumps(live[key])) != source:
            changed[key] = (source, live[key])
    if changed:
        raise ValueError(
            "In-memory text_config overrides of checkpoint keys are not supported by the MiniMax M3 "
            f"training model (config.json value, in-memory value): {changed}"
        )


_TRAINING_SWITCHES = {
    "expert_activation_qat": bool,
    "expert_qat_tp_shards": int,
    "expert_qat_backend": str,
    "linear_activation_qat": bool,
}


def _training_overrides(config) -> dict[str, Any]:
    text = getattr(config, "text_config", None)
    out = {}
    if text is None:
        return out
    for key, cast in _TRAINING_SWITCHES.items():
        value = text.get(key) if isinstance(text, dict) else getattr(text, key, None)
        if value is not None:
            out[key] = cast(value)
    return out


# --------------------------------------------------------------------------- #
# Norms, RoPE
# --------------------------------------------------------------------------- #
class MiniMaxM3RMSNorm(nn.Module):
    """Gemma RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + w)`` with FP32 statistics."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _fp32_region(x.device):
            x32 = x.float()
            y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
            y = y * (1.0 + self.weight.float())
        return y.to(x.dtype)


def rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 NeoX cos/sin rounded to BF16, as stored in vLLM's ``cos_sin_cache``.

    Returns two ``[T, rotary_dim // 2]`` FP32 tensors holding BF16 values.
    """
    with _fp32_region(positions.device):
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=positions.device) / rotary_dim)
        )
        # Broadcast product, not einsum: einsum is lowered by CUDA autocast.
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
        cos = freqs.cos().to(torch.bfloat16).float()
        sin = freqs.sin().to(torch.bfloat16).float()
    return cos, sin


def qk_norm_rope(
    x: torch.Tensor, weight: torch.Tensor, eps: float, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Per-head Gemma norm then partial NeoX RoPE, in FP32, one final rounding.

    x: [T, H, D]; cos/sin: [T, rotary_dim // 2].
    """
    with _fp32_region(x.device):
        x32 = x.float()
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        y = y * (1.0 + weight.float())
        half = rotary_dim // 2
        x1 = y[..., :half]
        x2 = y[..., half:rotary_dim]
        c = cos[:, None, :]
        s = sin[:, None, :]
        rotated = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s, y[..., rotary_dim:]), dim=-1)
    return rotated.to(x.dtype)


# --------------------------------------------------------------------------- #
# Linear layers
# --------------------------------------------------------------------------- #
class MiniMaxM3ExpertLinear(nn.Linear):
    """Routed-expert projection holding NVFP4 weights in factored form.

    ``weight`` stores ``e2m1 * block_scale`` (exact in BF16) and the per-tensor
    ``weight_global_scale`` stays FP32, so the true weight is
    ``weight / weight_global_scale`` (compressed-tensors convention). The global
    scale is applied to the GEMM output, as vLLM's NVFP4 GEMM applies it to the
    accumulator.

    ``act_quant`` = ``(mode, shards, backend)`` enables the deployed NVFP4
    activation quantization of this projection's input (``act_quant.py``).
    ``out_fp32`` returns the scaled FP32 accumulator without rounding: FC1 feeds
    the activation from it, and FC2's adapter delta is added to it before the
    expert rounds the sum once. ``input_global_scale`` is loaded for
    checkpoint completeness but unused: the rollout's per-token dynamic scaling
    replaces it with 1.
    """

    def __init__(self, in_features: int, out_features: int, out_fp32: bool = False):
        super().__init__(in_features, out_features, bias=False, dtype=torch.bfloat16)
        self.register_buffer("weight_global_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("input_global_scale", torch.ones((), dtype=torch.float32))
        self.act_quant: Optional[tuple[str, int, str]] = None
        self.out_fp32 = out_fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.requires_grad and torch.is_grad_enabled():
            # The custom backward returns no weight gradient: only adapters train.
            raise RuntimeError("MiniMax M3 routed experts require a frozen base weight")
        mode, shards, backend = self.act_quant or ("none", 1, "torch")
        return NVFP4ActQuantLinear.apply(x, self.weight, self.weight_global_scale, mode, shards, backend, self.out_fp32)


MXFP8_BLOCK = 32


def mxfp8_qdq(x: torch.Tensor) -> torch.Tensor:
    """MXFP8 activation quantize + dequantize of FP32 rows ``[rows, in]``.

    Blocks of 32 along the input dim; the block scale is the power of two that
    brings the block maximum under the largest E4M3 value (448); values are
    rounded to E4M3. This is what the rollout's MXFP8 GEMM does to its input.
    """
    rows, cols = x.shape
    blocks = x.view(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0)).unsqueeze(-1)
    return ((blocks / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale).view(rows, cols)


class _FrozenLinearFP32(torch.autograd.Function):
    """``x @ W.T`` with an FP32 product for a frozen BF16 weight.

    Only the weight is kept for the backward (the FP32 copy is a temporary), and
    the input gradient is a BF16 product, as under the trainer's autocast. With
    ``act_quant`` the input is MXFP8-quantized first (straight-through backward).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, act_quant: bool) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.x_dtype = x.dtype
        with _fp32_region(x.device):
            x32 = x.float()
            if act_quant:
                x32 = mxfp8_qdq(x32.reshape(-1, x32.shape[-1])).view(x32.shape)
            return F.linear(x32, weight.float())

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (weight,) = ctx.saved_tensors
        with _fp32_region(grad.device):
            return (grad.to(weight.dtype) @ weight).to(ctx.x_dtype), None, None


class MiniMaxM3Linear(nn.Linear):
    """BF16 projection that returns its FP32 product without rounding.

    An adapter wrapped around it adds its delta to the FP32 product, and the
    caller rounds the sum once (``_rounded``). Adding the delta to a product
    that is already rounded to BF16 drops the part of it below half an ulp of
    each element. The rollout adds the delta to its rounded BF16 product (per TP
    rank on row-parallel layers), so neither order reproduces it exactly; here
    only the final rounding can drop part of the delta.

    ``act_quant`` MXFP8-quantizes the base product's input, as the rollout's
    MXFP8 GEMM does; an adapter around this layer still reads the original input.
    """

    act_quant = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.requires_grad and torch.is_grad_enabled():
            # A trainable base weight needs its own gradient: plain autograd.
            with _fp32_region(x.device):
                x32 = x.float()
                if self.act_quant:
                    quantized = mxfp8_qdq(x32.detach().reshape(-1, x32.shape[-1])).view(x32.shape)
                    x32 = x32 + (quantized - x32.detach())
                return F.linear(x32, self.weight.float())
        return _FrozenLinearFP32.apply(x, self.weight, self.act_quant)


def _linear(in_features: int, out_features: int, act_quant: bool = False) -> nn.Linear:
    layer = MiniMaxM3Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
    if act_quant:
        if in_features % MXFP8_BLOCK:
            raise ValueError(f"MXFP8 activation QAT needs in_features divisible by {MXFP8_BLOCK}, got {in_features}")
        layer.act_quant = True
    return layer


def _rounded(proj: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``proj(x)`` (base product plus any adapter delta, FP32) rounded once to ``x``'s dtype."""
    return proj(x).to(x.dtype)


def swigluoai(
    gate: torch.Tensor,
    up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """vLLM ``SiluAndMulWithClamp`` / ``swigluoai_uninterleave`` (computed in FP32)."""
    with _fp32_region(gate.device):
        g = gate.float().clamp(max=limit)
        u = up.float().clamp(min=-limit, max=limit)
        out = g * torch.sigmoid(alpha * g) * (u + beta)
    return out.to(out_dtype or gate.dtype)


class MiniMaxM3MLP(nn.Module):
    """Dense / shared-expert SwiGLU-OAI MLP (``gate_proj``/``up_proj``/``down_proj``)."""

    def __init__(self, args: MiniMaxM3Args, intermediate_size: int):
        super().__init__()
        qat = args.linear_activation_qat
        self.gate_proj = _linear(args.hidden_size, intermediate_size, qat)
        self.up_proj = _linear(args.hidden_size, intermediate_size, qat)
        self.down_proj = _linear(intermediate_size, args.hidden_size, qat)
        self.act = (args.swiglu_alpha, args.swiglu_beta, args.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = swigluoai(_rounded(self.gate_proj, x), _rounded(self.up_proj, x), *self.act)
        return _rounded(self.down_proj, hidden)


class MiniMaxM3Expert(nn.Module):
    """One routed expert; checkpoint names w1 = gate, w3 = up, w2 = down."""

    def __init__(self, args: MiniMaxM3Args):
        super().__init__()
        # FC1 (w1 gate, w3 up) hands its FP32 accumulator to the activation; the
        # activation output is rounded to BF16 and is the FC2 (w2) input. FC2 is
        # rounded once, after its adapter delta (``_rounded``).
        self.w1 = MiniMaxM3ExpertLinear(args.hidden_size, args.intermediate_size, out_fp32=True)
        self.w2 = MiniMaxM3ExpertLinear(args.intermediate_size, args.hidden_size, out_fp32=True)
        self.w3 = MiniMaxM3ExpertLinear(args.hidden_size, args.intermediate_size, out_fp32=True)
        self.act = (args.swiglu_alpha, args.swiglu_beta, args.swiglu_limit)
        if args.expert_activation_qat:
            if args.intermediate_size % (16 * args.expert_qat_tp_shards):
                raise ValueError("intermediate_size must split into 16-column blocks per TP shard")
            self.w1.act_quant = ("fc1", 1, args.expert_qat_backend)
            self.w3.act_quant = ("fc1", 1, args.expert_qat_backend)
            self.w2.act_quant = ("fc2", args.expert_qat_tp_shards, args.expert_qat_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rounded(self.w2, swigluoai(self.w1(x), self.w3(x), *self.act, out_dtype=x.dtype))


class MiniMaxM3Router(nn.Module):
    """FP32 router weight. A buffer, so FSDP mixed precision never casts it."""

    def __init__(self, args: MiniMaxM3Args):
        super().__init__()
        self.register_buffer("weight", torch.empty(args.num_local_experts, args.hidden_size, dtype=torch.float32))


class MiniMaxM3MoE(nn.Module):
    """Sigmoid-routed MoE with bias-corrected top-k and one shared expert."""

    def __init__(self, args: MiniMaxM3Args):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.gate = MiniMaxM3Router(args)
        self.register_buffer("e_score_correction_bias", torch.empty(args.num_local_experts, dtype=torch.float32))
        self.experts = nn.ModuleList(MiniMaxM3Expert(args) for _ in range(args.num_local_experts))
        self.shared_experts = (
            MiniMaxM3MLP(args, args.intermediate_size * args.n_shared_experts) if args.n_shared_experts else None
        )

    def route(self, x: torch.Tensor, replay_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """vLLM ``topk_sigmoid``: select on ``sigmoid + bias``, weight by the unbiased
        sigmoid renormalized over the selection, times ``routed_scaling_factor``.

        ``replay_ids`` (``[tokens, top_k]``, R3) replaces the selection with the experts the
        rollout used; the weights still come from this model's own scores. A row with a
        repeated id carries no recorded choice (all-zero padding) and keeps the native one.
        """
        with _fp32_region(x.device):
            logits = F.linear(x.float(), self.gate.weight)
            scores = torch.sigmoid(logits)
            ids = torch.topk(scores.detach() + self.e_score_correction_bias, self.top_k, dim=-1).indices
            if replay_ids is not None:
                lo, hi, num_experts = int(replay_ids.min()), int(replay_ids.max()), scores.shape[-1]
                if replay_ids.shape != ids.shape or lo < 0 or hi >= num_experts:
                    raise ValueError(
                        f"routed_experts {tuple(replay_ids.shape)} (ids {lo}..{hi}) does not match "
                        f"routing {tuple(ids.shape)} over {num_experts} experts"
                    )
                ordered = replay_ids.sort(-1).values
                recorded = (ordered[:, 1:] != ordered[:, :-1]).all(-1)
                ids = torch.where(recorded[:, None], replay_ids, ids)
            weights = scores.gather(-1, ids)
            denom = weights.sum(-1, keepdim=True)
            denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            weights = weights / denom * self.routed_scaling_factor
        return weights, ids

    def forward(self, x: torch.Tensor, replay_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        weights, ids = self.route(x, replay_ids)
        flat_ids = ids.flatten()
        order = torch.argsort(flat_ids, stable=True)
        counts = torch.bincount(flat_ids, minlength=len(self.experts)).tolist()
        token_of = order // self.top_k
        routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        # With gradients enabled every expert runs, unrouted ones on zero rows:
        # FSDP2 reduce-scatters the gradients of the parameters that have one,
        # so the set must be identical on all ranks (a missing gradient on one
        # rank desynchronizes the collective). Zero rows give exact zero grads.
        run_empty = torch.is_grad_enabled()
        start = 0
        for expert_id, count in enumerate(counts):
            if count == 0 and not run_empty:
                continue
            sel = order[start : start + count]
            tok = token_of[start : start + count]
            start += count
            y = self.experts[expert_id](x[tok])
            with _fp32_region(y.device):
                contrib = y.float() * weights.flatten()[sel, None]
            routed = routed.index_add(0, tok, contrib)
        # Deployed finalize: the weighted sum is accumulated in FP32 and written as
        # BF16; the shared expert's BF16 output is then added in BF16.
        out = routed.to(x.dtype)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


# --------------------------------------------------------------------------- #
# Sparse attention
# --------------------------------------------------------------------------- #
@torch.no_grad()
def select_blocks(
    index_q: torch.Tensor, index_k: torch.Tensor, block_size: int, topk: int, local_blocks: int, init_blocks: int
) -> torch.Tensor:
    """Per (index head, query token) block selection of one sequence.

    vLLM ``minimax_m3_index_score`` + ``minimax_m3_index_topk``: the score of block
    ``b`` for token ``t`` is ``max_{s in b, s <= t} q_t . k_s`` in FP32; blocks
    ``b > t // block_size`` are invalid; the last ``local_blocks`` valid blocks and
    the first ``init_blocks`` are forced in; top-``topk`` of the rest.

    index_q: [L, H, D]; index_k: [L, D]. Returns a bool mask [H, L, num_blocks].
    """
    seq_len, num_heads, _ = index_q.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    device = index_q.device
    pos = torch.arange(seq_len, device=device)
    query_block = pos // block_size
    block_ids = torch.arange(num_blocks, device=device)
    valid = block_ids[None, :] <= query_block[:, None]  # [L, NB]
    if num_blocks <= topk:
        return valid[None].expand(num_heads, -1, -1).clone()

    padded = num_blocks * block_size
    k = index_k.float()
    if padded != seq_len:
        k = torch.cat([k, k.new_zeros(padded - seq_len, k.shape[-1])])
    scores = torch.empty(num_heads, seq_len, num_blocks, dtype=torch.float32, device=device)
    chunk = 512
    with _fp32_region(device):
        for t0 in range(0, seq_len, chunk):
            t1 = min(seq_len, t0 + chunk)
            q = index_q[t0:t1].float()  # [c, H, D]
            qk = torch.matmul(q.transpose(0, 1), k.t())  # [H, c, padded]
            causal = pos[None, t0:t1, None] >= torch.arange(padded, device=device)[None, None, :]
            qk = qk.masked_fill(~causal, float("-inf"))
            scores[:, t0:t1] = qk.view(num_heads, t1 - t0, num_blocks, block_size).amax(-1)
    scores = torch.nan_to_num(scores, nan=-1e30)
    scores = scores.masked_fill(~valid[None], float("-inf"))
    if init_blocks:
        scores = torch.where((block_ids < init_blocks)[None, None, :] & valid[None], 1e30, scores)
    if local_blocks:
        local = (block_ids[None, :] >= (query_block[:, None] + 1 - local_blocks)) & valid
        scores = torch.where(local[None], 1e29, scores)
    top = torch.topk(scores, topk, dim=-1).indices
    selected = torch.zeros(num_heads, seq_len, num_blocks, dtype=torch.bool, device=device)
    selected.scatter_(-1, top, True)
    return selected & valid[None]


class MiniMaxM3Attention(nn.Module):
    def __init__(self, args: MiniMaxM3Args):
        super().__init__()
        sparse = args.sparse_attention_config
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.rotary_dim = args.rotary_dim
        self.eps = args.rms_norm_eps
        self.num_index_heads = sparse["sparse_num_index_heads"]
        self.index_head_dim = sparse["sparse_index_dim"]
        self.block_size = sparse["sparse_block_size"]
        self.topk_blocks = sparse["sparse_topk_blocks"]
        self.local_blocks = sparse.get("sparse_local_block", 0)
        self.init_blocks = sparse.get("sparse_init_block", 0)
        hidden = args.hidden_size
        qat = args.linear_activation_qat
        self.q_proj = _linear(hidden, self.num_heads * self.head_dim, qat)
        self.k_proj = _linear(hidden, self.num_kv_heads * self.head_dim, qat)
        self.v_proj = _linear(hidden, self.num_kv_heads * self.head_dim, qat)
        self.o_proj = _linear(self.num_heads * self.head_dim, hidden, qat)
        self.index_q_proj = _linear(hidden, self.num_index_heads * self.index_head_dim, qat)
        self.index_k_proj = _linear(hidden, self.index_head_dim, qat)
        self.q_norm = MiniMaxM3RMSNorm(self.head_dim, self.eps)
        self.k_norm = MiniMaxM3RMSNorm(self.head_dim, self.eps)
        self.index_q_norm = MiniMaxM3RMSNorm(self.index_head_dim, self.eps)
        self.index_k_norm = MiniMaxM3RMSNorm(self.index_head_dim, self.eps)

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, selected: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Causal attention restricted to the selected blocks, one sequence.

        q: [L, Hq, D]; k, v: [L, Hkv, D]; selected: [Hkv, L, NB] or None (dense).
        """
        seq_len = q.shape[0]
        group = self.num_heads // self.num_kv_heads
        pos = torch.arange(seq_len, device=q.device)
        out = []
        # Pin the SDPA backends: the default choice depends on process-global
        # switches (importing vLLM disables cuDNN SDPA), which silently changed
        # the trainer's attention numerics between processes.
        with sdpa_kernel(_SDPA_BACKENDS):
            for h in range(self.num_kv_heads):
                qh = q[:, h * group : (h + 1) * group].transpose(0, 1).unsqueeze(0)  # [1, G, L, D]
                kh = k[:, h : h + 1].transpose(0, 1).unsqueeze(0).expand(1, group, seq_len, self.head_dim).contiguous()
                vh = v[:, h : h + 1].transpose(0, 1).unsqueeze(0).expand(1, group, seq_len, self.head_dim).contiguous()
                if selected is None:
                    o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
                else:
                    mask = selected[h][:, pos // self.block_size] & (pos[:, None] >= pos[None, :])
                    o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[None, None])
                out.append(o[0].transpose(0, 1))  # [L, G, D]
        return torch.cat(out, dim=1)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, segments: list[tuple[int, int]]
    ) -> torch.Tensor:
        tokens = x.shape[0]
        q = qk_norm_rope(
            _rounded(self.q_proj, x).view(tokens, self.num_heads, self.head_dim),
            self.q_norm.weight,
            self.eps,
            cos,
            sin,
            self.rotary_dim,
        )
        k = qk_norm_rope(
            _rounded(self.k_proj, x).view(tokens, self.num_kv_heads, self.head_dim),
            self.k_norm.weight,
            self.eps,
            cos,
            sin,
            self.rotary_dim,
        )
        v = _rounded(self.v_proj, x).view(tokens, self.num_kv_heads, self.head_dim)
        with torch.no_grad():
            iq = qk_norm_rope(
                _rounded(self.index_q_proj, x).view(tokens, self.num_index_heads, self.index_head_dim),
                self.index_q_norm.weight,
                self.eps,
                cos,
                sin,
                self.rotary_dim,
            )
            ik = qk_norm_rope(
                _rounded(self.index_k_proj, x).view(tokens, 1, self.index_head_dim),
                self.index_k_norm.weight,
                self.eps,
                cos,
                sin,
                self.rotary_dim,
            )[:, 0]
        # Tokens outside every segment (right padding) attend to nothing; their
        # attention output is zero.
        outputs = []
        cursor = 0
        for start, end in segments:
            if start > cursor:
                outputs.append(q.new_zeros(start - cursor, self.num_heads, self.head_dim))
            length = end - start
            if length <= self.topk_blocks * self.block_size:
                selected = None
            else:
                selected = select_blocks(
                    iq[start:end], ik[start:end], self.block_size, self.topk_blocks, self.local_blocks, self.init_blocks
                )
            outputs.append(self._attend(q[start:end], k[start:end], v[start:end], selected))
            cursor = end
        if cursor < tokens:
            outputs.append(q.new_zeros(tokens - cursor, self.num_heads, self.head_dim))
        attn = torch.cat(outputs, dim=0).reshape(tokens, self.num_heads * self.head_dim)
        if not segments:
            # No valid token on this rank: keep the q/k/v projections (and their
            # adapters) in the graph with exact zero gradients, so every FSDP2 rank
            # reduce-scatters the same parameter set. Empty slices: no value enters.
            attn = attn + (q[:0].sum() + k[:0].sum() + v[:0].sum())
        return _rounded(self.o_proj, attn)


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
class MiniMaxM3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, args: MiniMaxM3Args, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.input_layernorm = MiniMaxM3RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxM3RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.self_attn = MiniMaxM3Attention(args)
        self.is_moe_layer = args.is_moe_layer(layer_id)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxM3MoE(args)
        else:
            self.mlp = MiniMaxM3MLP(args, args.dense_intermediate_size)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, segments=None, replay_ids=None
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.self_attn(self.input_layernorm(hidden_states), cos, sin, segments)
        hidden_states = residual + h
        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        return residual + (self.block_sparse_moe(x, replay_ids) if self.is_moe_layer else self.mlp(x))


class MiniMaxM3TextModel(nn.Module):
    def __init__(self, args: MiniMaxM3Args):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size, dtype=torch.bfloat16)
        self.layers = nn.ModuleList(MiniMaxM3DecoderLayer(args, i) for i in range(args.num_hidden_layers))
        self.norm = MiniMaxM3RMSNorm(args.hidden_size, args.rms_norm_eps)


class MiniMaxM3LanguageModel(nn.Module):
    def __init__(self, args: MiniMaxM3Args):
        super().__init__()
        self.model = MiniMaxM3TextModel(args)
        # No adapter on the head: a plain BF16 product (an FP32 one would double the logits' memory).
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False, dtype=torch.bfloat16)


def _segments_from_inputs(
    batch: int,
    seq: int,
    attention_mask: Optional[torch.Tensor],
    cu_seqlens_cpu: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor] = None,
) -> list[list[tuple[int, int]]]:
    """Independent sequences per batch row, as sorted, disjoint [start, end) ranges.

    Priority: ``cu_seqlens_cpu`` (packed batch of one row), then a right-padded
    ``attention_mask``, then, for a single packed row without either, the
    position-id resets (a new sequence starts wherever ``position_ids == 0``).
    """
    if cu_seqlens_cpu is not None:
        if batch != 1:
            raise ValueError("cu_seqlens requires a packed batch of size 1")
        bounds = [int(b) for b in cu_seqlens_cpu.tolist()]
        if bounds[0] != 0 or bounds[-1] > seq or any(b > a for a, b in zip(bounds[1:], bounds[:-1], strict=True)):
            raise ValueError(f"Invalid cu_seqlens {bounds} for {seq} tokens")
        return [[(s, e) for s, e in zip(bounds[:-1], bounds[1:], strict=True) if e > s]]
    if attention_mask is not None:
        lengths = attention_mask.sum(-1).tolist()
        expected = torch.arange(seq, device=attention_mask.device)[None, :] < attention_mask.sum(-1, keepdim=True)
        if not torch.equal(attention_mask.bool(), expected):
            raise ValueError("MiniMax M3 training model expects right-padded attention masks")
        return [[(0, int(n))] if n else [] for n in lengths]
    if position_ids is not None and batch == 1:
        starts = torch.nonzero(position_ids[0] == 0).flatten().tolist()
        if not starts or starts[0] != 0:
            starts = [0] + starts
        bounds = starts + [seq]
        return [[(s, e) for s, e in zip(bounds[:-1], bounds[1:], strict=True) if e > s]]
    return [[(0, seq)] for _ in range(batch)]


class MiniMaxM3ForCausalLM(PreTrainedModel):
    """MiniMax M3 text backbone with the LM head (vision/MTP are not modeled)."""

    base_model_prefix = "language_model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxM3DecoderLayer"]
    _skip_keys_device_placement = []
    # Checkpoint keys that the training graph does not contain.
    ignored_checkpoint_prefixes = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.", "mtp.")
    _verl_sharded_init = True
    # The indexer runs under no_grad (block ids are discrete); an adapter there
    # would never train. vLLM accepts their absence (LORA_CONTRACT.md §3).
    _verl_lora_forbidden_modules = ("index_q_proj", "index_k_proj")
    # Adapter tensor names vLLM's MiniMax M3 LoRA loader maps onto wrapped modules
    # (LORA_CONTRACT.md §1-2), after it strips ``base_model.model.``.
    _LORA_EXPORT_RE = re.compile(
        r"^base_model\.model\.language_model\.model\.layers\.(0|[1-9]\d*)\."
        r"(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj"
        r"|block_sparse_moe\.shared_experts\.(?:gate|up|down)_proj"
        r"|block_sparse_moe\.experts\.(0|[1-9]\d*)\.w[123])"
        r"\.lora_[AB]\.weight$"
    )

    def __init__(self, config, args: Optional[MiniMaxM3Args] = None):
        super().__init__(config)
        self.args = args or MiniMaxM3Args.from_text_config(_read_text_config(config), _training_overrides(config))
        self.language_model = MiniMaxM3LanguageModel(self.args)
        self.post_init()

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config)

    def _check_and_adjust_attn_implementation(self, attn_implementation=None, is_init_check=False, **kwargs):
        # Attention is implemented in this module (block-sparse SDPA); Transformers'
        # attention dispatch and its flash-attn availability checks do not apply.
        # verl's HFModelConfig requests flash_attention_2 by default.
        return attn_implementation or "sdpa"

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # PEFT's CAUSAL_LM wrapper binds this at construction; generation (KV cache)
        # is not supported by the training model.
        raise NotImplementedError("MiniMaxM3ForCausalLM is a training model; use vLLM for generation")

    @classmethod
    def verl_load_sharded(cls, model: nn.Module, path: str, device, seed: int = 0) -> dict[str, int]:
        """verl sharded-init hook: materialize ``model`` (meta, optionally PEFT +
        FSDP2-wrapped) from the released checkpoint at ``path``."""
        from .checkpoint import load_minimax_m3_sharded

        return load_minimax_m3_sharded(model, path, device, seed)

    @classmethod
    def verl_share_expert_outer_loras(cls, model: nn.Module) -> int:
        """Shared-outer expert LoRA: per MoE layer, all routed experts use one gate
        (w1) ``lora_A``, one up (w3) ``lora_A`` and one down (w2) ``lora_B``; the
        inner factors (w1/w3 ``lora_B``, w2 ``lora_A``) stay per expert.

        The factors are tied by module identity after PEFT wrapped the experts:
        ``parameters()`` yields each shared factor once (optimizer, FSDP2), while
        ``state_dict()`` keeps every per-expert name, so the adapter export is
        already expanded to the per-expert format the rollout engine loads by
        default. Returns the number of replaced factor modules.
        """
        tied = 0
        for moe in model.modules():
            if not isinstance(moe, MiniMaxM3MoE):
                continue
            first = moe.experts[0]
            for proj, factor in (("w1", "lora_A"), ("w3", "lora_A"), ("w2", "lora_B")):
                shared = getattr(getattr(first, proj), factor, None)
                if not isinstance(shared, nn.ModuleDict) or len(shared) == 0:
                    raise ValueError(f"Shared-outer LoRA needs an adapter on every routed expert {proj}")
                for expert in moe.experts[1:]:
                    own = getattr(getattr(expert, proj), factor, None)
                    if not isinstance(own, nn.ModuleDict) or set(own) != set(shared):
                        raise ValueError(f"Shared-outer LoRA needs an adapter on every routed expert {proj}")
                    for adapter in shared:
                        own[adapter] = shared[adapter]
                        tied += 1
        if tied == 0:
            raise ValueError("Shared-outer LoRA found no routed experts to tie")
        return tied

    @classmethod
    def verl_validate_lora_export(cls, names: list[str], args: Optional[MiniMaxM3Args] = None) -> None:
        """Fail loudly on adapter names vLLM would silently ignore.

        vLLM's tensor LoRA path does not report unmatched names: a wrong name
        just leaves that module's adapter at zero. Also, a layer with any
        routed-expert adapter needs all ``num_local_experts x {w1,w2,w3}``.
        """
        names = list(names)
        present = set(names)
        if len(present) != len(names):
            raise ValueError("Duplicate adapter tensor names in the LoRA export")
        unpaired = [
            name
            for name in names
            if name.replace(".lora_A.", ".lora_B.") not in present
            or name.replace(".lora_B.", ".lora_A.") not in present
        ]
        if unpaired:
            raise ValueError(
                f"{len(unpaired)} adapter tensors lack their lora_A/lora_B counterpart, e.g. {unpaired[:4]}"
            )
        bad = []
        experts: dict[int, set[tuple[int, str, str]]] = {}
        for name in names:
            match = cls._LORA_EXPORT_RE.match(name)
            if match is None:
                bad.append(name)
                continue
            layer, module = int(match.group(1)), match.group(2)
            if args is not None:
                if layer >= args.num_hidden_layers:
                    bad.append(name)
                    continue
                dense_only = module.startswith("mlp.")
                moe_only = module.startswith("block_sparse_moe.")
                if (dense_only and args.is_moe_layer(layer)) or (moe_only and not args.is_moe_layer(layer)):
                    bad.append(name)
                    continue
            if match.group(3) is not None:
                tail = name.rsplit(".", 3)  # [..., w1, lora_A, weight]
                experts.setdefault(layer, set()).add((int(match.group(3)), tail[-3], tail[-2]))
        if bad:
            raise ValueError(f"{len(bad)} adapter tensors do not match the vLLM MiniMax M3 LoRA schema, e.g. {bad[:4]}")
        if args is not None:
            full = {
                (expert, proj, ab)
                for expert in range(args.num_local_experts)
                for proj in ("w1", "w2", "w3")
                for ab in ("lora_A", "lora_B")
            }
            wrong = {layer: len(found ^ full) for layer, found in experts.items() if found != full}
            if wrong:
                raise ValueError(
                    "Routed-expert adapters must be exactly experts 0.."
                    f"{args.num_local_experts - 1} x w1/w2/w3 x A/B; layers with missing or extra tensors: {wrong}"
                )

    def _init_weights(self, module):
        # All weights come from the checkpoint loader; nothing to initialize here.
        return

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        routed_experts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("inputs_embeds is not supported")
        if kwargs.get("labels") is not None:
            raise ValueError("labels are not supported: the trainer computes the loss from the logits")
        batch, seq = input_ids.shape
        if position_ids is not None and position_ids.dim() != 2:
            raise ValueError(f"Expected [batch, seq] position_ids, got shape {tuple(position_ids.shape)}")
        if cu_seqlens_cpu is None and cu_seqlens is not None:
            cu_seqlens_cpu = cu_seqlens.cpu()
        segments_per_row = _segments_from_inputs(batch, seq, attention_mask, cu_seqlens_cpu, position_ids)
        # All rows are flattened into one token stream so every layer (an FSDP
        # unit) runs exactly once per forward; rows become disjoint segments.
        segments = [
            (row * seq + s, row * seq + e) for row, row_segs in enumerate(segments_per_row) for s, e in row_segs
        ]
        if position_ids is None:
            pos = torch.zeros(batch * seq, dtype=torch.long, device=input_ids.device)
            for s, e in segments:
                pos[s:e] = torch.arange(e - s, device=input_ids.device)
        else:
            pos = position_ids.reshape(-1)
            expected = torch.zeros_like(pos)
            covered = torch.zeros_like(pos, dtype=torch.bool)
            for s, e in segments:
                expected[s:e] = torch.arange(e - s, device=pos.device)
                covered[s:e] = True
            if not torch.equal(pos[covered], expected[covered]):
                raise ValueError("position_ids must restart at 0 and be contiguous inside every sequence")
        cos, sin = rope_cos_sin(pos, self.args.rotary_dim, self.args.rope_theta)
        inner = self.language_model.model
        h = inner.embed_tokens(input_ids.reshape(-1))
        if routed_experts is not None:
            # R3: [batch, seq, num_hidden_layers, top_k], indexed by absolute decoder layer.
            if routed_experts.shape[:3] != (batch, seq, len(inner.layers)):
                raise ValueError(
                    f"routed_experts {tuple(routed_experts.shape)} does not match "
                    f"[{batch}, {seq}, {len(inner.layers)}, top_k]"
                )
            routed_experts = routed_experts.reshape(batch * seq, len(inner.layers), -1)
        for layer in inner.layers:
            replay_ids = None
            if routed_experts is not None and layer.is_moe_layer:
                replay_ids = routed_experts[:, layer.layer_id]
            h = layer(h, cos, sin, segments=segments, replay_ids=replay_ids)
        hidden = inner.norm(h).view(batch, seq, -1)
        logits = self.language_model.lm_head(hidden)
        return CausalLMOutputWithPast(logits=logits)
