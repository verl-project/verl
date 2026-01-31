# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

"""
Group-wise helpers for RL training utilities.

Public API:
    - as_torch_index(index, device=None) -> torch.LongTensor
    - group_mean_std(scores, gidx, eps=1e-6, device=None) -> (mean_g, std_g, count_g)

Default device policy:
    - If `device` is None:
        * In pytest (detected by env "PYTEST_CURRENT_TEST"): use CPU.
        * Else if CUDA is available: use CUDA.
        * Else: use CPU.
    - You can override via env "VERL_FORCE_DEVICE" (e.g., "cuda:0" / "cpu").

Notes:
- as_torch_index: canonicalizes arbitrary group labels to a contiguous 1-D torch.long
  tensor in range [0..G-1]. Robust to torch/numpy/list/tuple, ints/floats/bools,
  numeric strings, UUIDs, mixed object arrays. Near-integer floats (|x-round(x)|<=1e-6)
  are rounded; otherwise factorization is applied.
- group_mean_std: pure-PyTorch per-group mean/std with Bessel correction for variance
  (denominator max(count-1, 1)). Singleton groups fallback to mean=0, std=1 for
  compatibility with common “native” conventions.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from verl.utils.device import get_device_name

__all__ = ["as_torch_index", "group_mean_std"]


def _resolve_device(explicit: Optional[torch.device | str]) -> torch.device:
    """
    Resolve device according to policy described in the module docstring.
    Priority:
      1) explicit argument
      2) VERL_FORCE_DEVICE env
      3) pytest detection -> cpu
      4) cuda if available, else cpu
    """
    if explicit is not None:
        return torch.device(explicit)

    forced = os.getenv("VERL_FORCE_DEVICE")
    if forced:
        return torch.device(forced)

    # Heuristic: pytest sets PYTEST_CURRENT_TEST
    if "PYTEST_CURRENT_TEST" in os.environ:
        return torch.device("cpu")

    return torch.device(get_device_name())


def _to_1d_numpy_object_array(x: Any) -> np.ndarray:
    """Best-effort: convert arbitrary input into a 1-D numpy array; fallback to object dtype."""
    try:
        arr = np.asarray(x)
    except Exception:
        try:
            arr = np.array(list(x), dtype=object)
        except Exception:
            arr = np.array([x], dtype=object)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def as_torch_index(index: Any, device: torch.device | str | None = None) -> torch.Tensor:
    """
    Convert arbitrary group labels to a contiguous 1-D torch.long tensor (0..G-1).

    Args:
        index: Any iterable of labels or tensor/ndarray.
        device: Target device; if None, resolved via _resolve_device().

    Returns:
        torch.LongTensor with shape (N,)
    """
    target = _resolve_device(device)

    # ---------- Fast path: torch.Tensor ----------
    if isinstance(index, torch.Tensor):
        t = index.reshape(-1)
        if t.dtype in (
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            getattr(torch, "uint8", torch.uint8),
            torch.bool,
        ):
            return t.to(device=target, dtype=torch.long)

        if t.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            t64 = t.to(dtype=torch.float64)
            rounded = torch.round(t64)
            if torch.allclose(t64, rounded, rtol=0.0, atol=1e-6):
                return rounded.to(device=target, dtype=torch.long)
            arr = np.array([str(x.item()) for x in t], dtype=object)
        else:
            arr = np.array([str(x.item()) if hasattr(x, "item") else str(x) for x in t], dtype=object)

    else:
        # ---------- Non-torch: go through numpy ----------
        arr = _to_1d_numpy_object_array(index)

        # Pure integers (incl. bool)
        if arr.dtype != object and np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr.astype(np.int64, copy=False)).to(device=target)

        # Floats nearly equal to integers
        if arr.dtype != object and np.issubdtype(arr.dtype, np.floating):
            arr64 = arr.astype(np.float64, copy=False)
            rounded = np.rint(arr64)
            if np.allclose(arr64, rounded, rtol=0.0, atol=1e-6):
                return torch.from_numpy(rounded.astype(np.int64)).to(device=target)
            # fall through

        # Try numeric string coercion
        try:
            coerced = arr.astype(np.int64)
            return torch.from_numpy(coerced).to(device=target)
        except Exception:
            pass

        if arr.dtype != object:
            arr = arr.astype(object)

    # ---------- Factorization (UUIDs / mixed types / arbitrary labels) ----------
    try:
        _, inv = np.unique(arr, return_inverse=True)
    except Exception:
        sarr = np.array([str(x) for x in arr], dtype=object)
        _, inv = np.unique(sarr, return_inverse=True)

    inv = inv.astype(np.int64, copy=False)
    return torch.from_numpy(inv).to(device=target)


@torch.no_grad()
def group_mean_std(
    scores: torch.Tensor,
    gidx: torch.Tensor,
    eps: float = 1e-6,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-group mean/std/count in pure PyTorch.

    mean_g = sum / count
    std_g  = sqrt( max( (sum2 - sum^2/count) / max(count-1, 1), eps ) )

    Singleton groups fallback to mean=0, std=1.

    Args:
        scores: (N,) float tensor.
        gidx  : (N,) long/int tensor with group indices (0..G-1).
        eps   : Numerical floor for variance.
        device: Target device; if None, resolved via _resolve_device().

    Returns:
        mean_g: (G,) float32
        std_g : (G,) float32
        count : (G,) float32
    """
    target = _resolve_device(device)

    scores = scores.reshape(-1).to(device=target, dtype=torch.float32)
    gidx = gidx.reshape(-1).to(device=target, dtype=torch.long)

    if scores.numel() != gidx.numel():
        raise ValueError(f"scores and gidx length mismatch: {scores.numel()} vs {gidx.numel()}")

    G = int(torch.max(gidx).item()) + 1 if gidx.numel() > 0 else 0
    if G == 0:
        # Return empty tensors on the selected device
        empty = torch.empty(0, device=target, dtype=torch.float32)
        return empty, empty, empty

    ones = torch.ones_like(scores, dtype=torch.float32)

    count = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, ones)
    s1 = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, scores)
    s2 = torch.zeros(G, device=target, dtype=torch.float32).index_add_(0, gidx, scores * scores)

    mean = s1 / count.clamp_min(1.0)
    var_num = s2 - (s1 * s1) / count.clamp_min(1.0)
    denom = (count - 1.0).clamp_min(1.0)
    var = var_num / denom
    std = torch.sqrt(torch.clamp(var, min=eps))

    # Singleton groups: mean=0, std=1
    single = count <= 1.0
    if torch.any(single):
        mean = mean.clone()
        std = std.clone()
        mean[single] = 0.0
        std[single] = 1.0

    return mean, std, count
