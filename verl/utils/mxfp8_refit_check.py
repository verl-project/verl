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
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32
_DEFAULT_TOL = 0.25
_PROBE_ROWS = 8


def refit_check_enabled() -> bool:
    return os.environ.get("VERL_MXFP8_REFIT_CHECK", "1") != "0"


def refit_check_tolerance() -> float:
    return float(os.environ.get("VERL_MXFP8_REFIT_CHECK_TOL", str(_DEFAULT_TOL)))


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
