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
"""Post-refit self-check for MXFP8 rollout layers.

Why. Both rollout engines derive kernel-specific copies of the MXFP8 weight scales
at load time (swizzled / packed layouts). A weight sync rewrites the canonical
``weight`` / ``weight_scale`` tensors; if the derived copy is not rebuilt, the
kernel pairs fresh weights with stale scales and generates garbage while every
sync-side check passes (observed on 2xB200 with SGLang 0.5.12 + FlashInfer
CUTLASS: held-out accuracy 0.0 from step 0, kl 7.2, entropy ~ log(vocab), exit
code 0). The failure is invisible to the sync path because the sync itself
succeeded; only the kernel's *output* is wrong.

What. After the engine-specific re-processing, run one MXFP8 linear layer's own
``apply`` on a small random input and compare against a plain bf16 reference
computed from the canonical fp8 weight and UE8M0 scale. The kernel quantizes the
activation to MXFP8 internally, so a healthy layer differs from the reference by
a few percent; a layer reading a stale or mis-laid-out scale differs by O(1).

Knobs (environment variables, read at call time):

- ``VERL_MXFP8_REFIT_CHECK`` — ``0`` disables the check (default enabled).
- ``VERL_MXFP8_REFIT_CHECK_TOL`` — relative-error threshold (default ``0.25``).
- ``VERL_MXFP8_REFIT_CHECK_MOE_TOL`` — threshold for the MoE expert probe
  (defaults to the linear threshold; the gated MLP compounds the activation
  quantization noise of two GEMMs).

MoE experts. The same stale-layout failure applies to the fused expert weights
``w13`` / ``w2`` (their scales are rewritten in place by the MoE runner), so one
local expert is probed the same way: every probe row is routed to that expert
with weight 1.0 and the layer's own forward is compared against
``silu(x @ w1^T) * (x @ w3^T) @ w2^T`` on the dequantized weights.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32
_DEFAULT_TOL = 0.25
_PROBE_ROWS = 8


def refit_check_enabled() -> bool:
    return os.environ.get("VERL_MXFP8_REFIT_CHECK", "1") != "0"


def refit_check_tolerance() -> float:
    return float(os.environ.get("VERL_MXFP8_REFIT_CHECK_TOL", str(_DEFAULT_TOL)))


def refit_check_moe_tolerance() -> float:
    return float(os.environ.get("VERL_MXFP8_REFIT_CHECK_MOE_TOL", str(refit_check_tolerance())))


def mxfp8_dequantize(qweight: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize a rowwise MXFP8 tensor to fp32.

    ``qweight`` is ``[..., K]`` (fp8 e4m3fn, or any dtype castable to fp32) and
    ``scale_u8`` is ``[..., K // 32]`` UE8M0: the multiplier is ``2 ** (u8 - 127)``.
    """
    k = qweight.shape[-1]
    if k % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(f"last dim {k} is not a multiple of {MXFP8_BLOCK_SIZE}")
    blocks = k // MXFP8_BLOCK_SIZE
    expected = (*qweight.shape[:-1], blocks)
    if tuple(scale_u8.shape) != expected:
        raise ValueError(f"scale shape {tuple(scale_u8.shape)} does not match canonical {expected}")
    q = qweight.to(torch.float32).reshape(*qweight.shape[:-1], blocks, MXFP8_BLOCK_SIZE)
    descale = torch.exp2(scale_u8.to(torch.float32) - 127.0).unsqueeze(-1)
    return (q * descale).reshape(qweight.shape)


@torch.no_grad()
def probe_mxfp8_linear(
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    qweight: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    rows: int = _PROBE_ROWS,
    seed: int = 0,
) -> float:
    """Relative error between the kernel's output and the dequantized reference.

    ``apply_fn(x)`` must run the layer's own quantized GEMM on ``x`` of shape
    ``[rows, K]`` (bf16) and return ``[rows, N]``.
    """
    n, k = qweight.shape[-2], qweight.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, k, generator=gen, dtype=torch.float32).to(device=qweight.device, dtype=torch.bfloat16)
    ref = x.to(torch.float32) @ mxfp8_dequantize(qweight, scale_u8).reshape(n, k).t()
    out = apply_fn(x)
    if isinstance(out, tuple):  # some engines return (output, bias)
        out = out[0]
    out = out.to(torch.float32).reshape(ref.shape)
    if not torch.isfinite(out).all():
        return float("inf")
    denom = ref.norm().item()
    return float((out - ref).norm().item() / denom) if denom > 0 else float(out.norm().item())


def assert_mxfp8_linear_matches(
    name: str,
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    qweight: torch.Tensor,
    scale_u8: torch.Tensor,
    *,
    engine: str,
) -> float:
    """Run the probe and raise ``RuntimeError`` with an actionable message when it fails.

    Returns the measured relative error. Errors inside the probe itself (an
    ``apply`` signature this engine version does not support, a CPU-only build)
    are logged and skipped rather than raised: the check must never be the
    reason a healthy refit fails.
    """
    tol = refit_check_tolerance()
    try:
        rel = probe_mxfp8_linear(apply_fn, qweight, scale_u8)
    except Exception as err:  # noqa: BLE001 - the probe is best-effort
        logger.warning("mxfp8 refit check skipped on %s (%s): %s", name, engine, err)
        return float("nan")
    if rel > tol:
        raise RuntimeError(
            f"MXFP8 refit self-check failed on {engine} layer '{name}': the layer's quantized GEMM disagrees "
            f"with the dequantized reference by {rel:.3f} relative error (tolerance {tol}). This is the "
            "signature of kernel scale layouts that were not re-derived after a weight sync (fresh weights "
            "paired with stale swizzled/packed scales), or of a scale/weight layout the engine version "
            "changed. Check the engine version against docs/low_precision/fp8.md, the selected MXFP8 GEMM "
            "backend, and that this layer is meant to be quantized at all. Set VERL_MXFP8_REFIT_CHECK=0 to "
            "bypass (not recommended) or VERL_MXFP8_REFIT_CHECK_TOL to loosen the threshold."
        )
    logger.debug("mxfp8 refit check on %s (%s): rel err %.4f <= %.2f", name, engine, rel, tol)
    return rel


# ---------------------------------------------------------------------------
# MoE experts
# ---------------------------------------------------------------------------


def mxfp8_moe_expert_reference(
    x: torch.Tensor,
    w13_q: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w2_q: torch.Tensor,
    w2_scale_u8: torch.Tensor,
) -> torch.Tensor:
    """fp32 reference of one gated expert on dequantized weights.

    ``w13_q`` is ``[2I, H]`` in the fused-MoE convention (rows ``[:I]`` = w1 /
    gate, rows ``[I:]`` = w3 / up), ``w2_q`` is ``[H, I]``. ``x`` is ``[rows, H]``.
    """
    w13 = mxfp8_dequantize(w13_q, w13_scale_u8)
    w2 = mxfp8_dequantize(w2_q, w2_scale_u8)
    h = x.to(torch.float32) @ w13.t()
    gate, up = h.chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w2.t()


@torch.no_grad()
def probe_mxfp8_moe_expert(
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    w13_q: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w2_q: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    *,
    reduce_ref: Callable[[torch.Tensor], torch.Tensor] | None = None,
    rows: int = _PROBE_ROWS,
    seed: int = 0,
) -> float:
    """Relative error between the MoE layer routed entirely to one expert and the dequantized reference.

    ``apply_fn(x)`` must run the layer's own forward with every row of ``x``
    (``[rows, H]`` bf16) routed to the probed expert with weight 1.0 and return
    ``[rows, H]``. With tensor parallelism the caller passes the local expert
    shards and ``reduce_ref`` (the same all-reduce the layer applies), so the
    reference is summed the way the kernel output is.
    """
    hidden = w13_q.shape[-1]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, hidden, generator=gen, dtype=torch.float32).to(device=w13_q.device, dtype=torch.bfloat16)
    ref = mxfp8_moe_expert_reference(x, w13_q, w13_scale_u8, w2_q, w2_scale_u8)
    if reduce_ref is not None:
        ref = reduce_ref(ref)
    out = apply_fn(x)
    if isinstance(out, tuple):
        out = out[0]
    out = out.to(torch.float32).reshape(ref.shape)
    if not torch.isfinite(out).all():
        return float("inf")
    denom = ref.norm().item()
    return float((out - ref).norm().item() / denom) if denom > 0 else float(out.norm().item())


def assert_mxfp8_moe_expert_matches(
    name: str,
    expert_id: int,
    apply_fn: Callable[[torch.Tensor], torch.Tensor],
    w13_q: torch.Tensor,
    w13_scale_u8: torch.Tensor,
    w2_q: torch.Tensor,
    w2_scale_u8: torch.Tensor,
    *,
    engine: str,
    reduce_ref: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    """MoE counterpart of ``assert_mxfp8_linear_matches``: raise on a mismatch, skip (warn) on probe errors."""
    tol = refit_check_moe_tolerance()
    try:
        rel = probe_mxfp8_moe_expert(apply_fn, w13_q, w13_scale_u8, w2_q, w2_scale_u8, reduce_ref=reduce_ref)
    except Exception as err:  # noqa: BLE001 - the probe is best-effort
        logger.warning("mxfp8 MoE refit check skipped on %s expert %d (%s): %s", name, expert_id, engine, err)
        return float("nan")
    if rel > tol:
        raise RuntimeError(
            f"MXFP8 refit self-check failed on {engine} MoE layer '{name}', expert {expert_id}: the layer routed "
            f"to this expert disagrees with the dequantized reference by {rel:.3f} relative error (tolerance "
            f"{tol}). This is the signature of expert scales (w13_/w2_weight_scale) whose kernel layout was "
            "not re-derived after a weight sync, or of a MoE runner whose scale layout the loader does not "
            "know. Check the engine version and MoE backend against docs/low_precision/fp8.md. Set "
            "VERL_MXFP8_REFIT_CHECK=0 to bypass (not recommended) or VERL_MXFP8_REFIT_CHECK_MOE_TOL to loosen."
        )
    logger.debug("mxfp8 MoE refit check on %s expert %d (%s): rel err %.4f <= %.2f", name, expert_id, engine, rel, tol)
    return rel
