# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""MXFP4 quantization-aware training for DeepSeek-V4 routed experts.

Follows DeepSeek-V4 §5.2.1: the master weights are quantized to MXFP4 and
dequantized back for the forward, and the backward propagates straight through
to the unquantized weights (Jacob et al. 2018's STE). Training then sees the
same expert values the rollout engine will serve, instead of a bf16 view the
deployment never uses.

Why this matters here rather than only at deployment: in RL the actor's logprobs
are compared against the rollout's on every step, and a DSv4 rollout serves its
routed experts in packed MXFP4. Without QAT the actor computes in bf16 while the
sampler computes in 4-bit, and that gap is charged to the policy-divergence
metrics -- measured on this stack as k3_kl growing roughly 4x over fifteen steps
with routing replay off.

Bit-exactness with the exporter is the whole point of the implementation. The
paper's benefit only holds if the values seen in training are the values the
engine will hold, so the scale derivation and the E2M1 code boundaries below
mirror ``quantize_mxfp4_e2m1_like_scale`` exactly, and ``test_matches_exporter``
pins that agreement against the real quantize/dequantize pair.

What is deliberately not copied from the paper: DeepSeek dequantizes FP4 to FP8
and runs the existing FP8 GEMM, which makes the simulation free because their
trainer is already FP8. A bf16 trainer has no such path to ride, so this
dequantizes to the parameter dtype and pays for an extra round trip per forward.
The numerical effect on the weights is identical; only the speedup is missing.
"""

from __future__ import annotations

import torch

# E2M1 magnitudes and the midpoints between them. ``torch.bucketize`` against the
# midpoints is round-to-nearest, matching the exporter.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

FP4_E2M1_MAX = 6.0
MXFP4_BLOCK_K = 32

# The E8M0 scale exponent range; scales outside it cannot be represented.
_E8M0_MIN = 2.0**-127
_E8M0_MAX = 2.0**127


def mxfp4_round_trip(weight: torch.Tensor) -> torch.Tensor:
    """Quantize ``weight`` to MXFP4 and back, without ever packing the nibbles.

    Packing is what the wire format needs, not what the forward needs: the
    dequantized values are identical either way, and skipping it avoids a
    pack/unpack pair per step.

    The last dimension must be a multiple of 32 -- the MXFP4 scale block -- which
    holds for every DSv4 expert matrix.
    """
    if weight.shape[-1] % MXFP4_BLOCK_K:
        raise ValueError(
            f"MXFP4 QAT needs the last dim to be a multiple of {MXFP4_BLOCK_K}, got {tuple(weight.shape)}"
        )

    orig_dtype = weight.dtype
    w = weight.float()
    blocks = w.unflatten(-1, (-1, MXFP4_BLOCK_K))

    # Power-of-two scale per 32-element block, rounded up so the block max lands
    # inside E2M1's range rather than saturating at 6.0.
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP4_E2M1_MAX, torch.ones_like(amax))
    scale = torch.exp2(torch.ceil(torch.log2(scale.clamp(_E8M0_MIN, _E8M0_MAX))))

    normalized = blocks / scale
    boundaries = torch.tensor(_E2M1_BOUNDARIES, dtype=torch.float32, device=w.device)
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=w.device)
    codes = torch.bucketize(normalized.abs(), boundaries)
    dequantized = torch.sign(normalized) * magnitudes[codes] * scale

    return dequantized.flatten(-2).to(orig_dtype)


class _Mxfp4STE(torch.autograd.Function):
    """Round-trip in the forward, identity in the backward.

    The straight-through estimator is what makes this trainable: the rounding to
    sixteen levels has zero gradient almost everywhere, so a faithful backward
    would stop learning entirely. Passing the gradient to the unquantized weight
    is the standard substitute, and is what the paper describes as propagating
    "directly back to the FP32 master weights".
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor) -> torch.Tensor:
        return mxfp4_round_trip(weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def mxfp4_fake_quant(weight: torch.Tensor) -> torch.Tensor:
    """MXFP4 round trip that a gradient can flow through."""
    return _Mxfp4STE.apply(weight)


# --- Wiring into Megatron's grouped MoE experts ------------------------------

_SAVED_ATTR = "_verl_mxfp4_qat_saved"


def _grouped_expert_weight_names(linear: torch.nn.Module) -> list[str]:
    """Per-expert weight names on a TE grouped linear.

    ``TEGroupedMLP`` registers either one fused ``weight`` or one ``weight{i}``
    per local expert, depending on ``single_grouped_weight``; both spellings are
    accepted so the hook does not depend on which branch mcore took.
    """
    names = [n for n, _ in linear.named_parameters(recurse=False) if n.startswith("weight")]
    return [n for n in names if n == "weight" or n[len("weight") :].isdigit()]


def _swap_in_quantized(module: torch.nn.Module, _args) -> None:
    """Replace expert weights with their MXFP4 round trip for this forward."""
    saved: dict[str, torch.nn.Parameter] = {}
    for name in _grouped_expert_weight_names(module):
        param = module._parameters[name]
        if param is None:
            continue
        quantized = mxfp4_fake_quant(param)

        # With gradient_accumulation_fusion on -- the default wherever TE can
        # support it -- TE writes the weight gradient straight into
        # ``main_grad``, bypassing autograd. A derived tensor has no such buffer,
        # so TE would either fail or drop the gradient. Pointing it at the
        # parameter's own buffer is exactly right here because the STE makes the
        # gradient w.r.t. the quantized tensor identical to the gradient w.r.t.
        # the parameter.
        main_grad = getattr(param, "main_grad", None)
        if main_grad is not None:
            quantized.main_grad = main_grad

        saved[name] = param
        # Assigning a non-Parameter over a registered parameter name is rejected
        # by nn.Module, so drop the registration for the duration of the call and
        # put it back in the post-hook. Nothing walks named_parameters() inside a
        # forward, and keeping the registration intact everywhere else is what
        # stops this from disturbing the optimizer, the grad buffers, or the
        # checkpoint's name mapping.
        del module._parameters[name]
        setattr(module, name, quantized)

    setattr(module, _SAVED_ATTR, saved)


def _restore_parameters(module: torch.nn.Module, _args, _output) -> None:
    saved = getattr(module, _SAVED_ATTR, None) or {}
    for name, param in saved.items():
        module.__dict__.pop(name, None)
        module._parameters[name] = param
    if hasattr(module, _SAVED_ATTR):
        delattr(module, _SAVED_ATTR)


def enable_mxfp4_qat(model) -> int:
    """Quantize routed-expert weights to MXFP4 on every forward.

    Returns the number of grouped linears hooked. Shared experts and the dense
    linears are left alone: the checkpoint stores only the routed experts in
    MXFP4, so they are the only weights whose deployment precision the training
    has to anticipate.
    """
    hooked = 0
    modules = model if isinstance(model, (list, tuple)) else [model]
    for root in modules:
        for module in root.modules():
            if type(module).__name__ != "TEGroupedMLP":
                continue
            for linear in (getattr(module, "linear_fc1", None), getattr(module, "linear_fc2", None)):
                if linear is None or not _grouped_expert_weight_names(linear):
                    continue
                linear.register_forward_pre_hook(_swap_in_quantized)
                linear.register_forward_hook(_restore_parameters)
                hooked += 1
    return hooked
