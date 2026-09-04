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
"""Sharded blockwise FP8 quantization: bitwise-identical to the whole-tensor
kernel, computed shard-locally with one collective.

The rollout side quantizes whole HF tensors (``scaled_fp8_blockwise``:
per-block absmax -> descale = absmax / FP8_MAX -> codes = clamp(x / descale)).
A trainer rank holds only a slice of the tensor, but the block grid -- and
therefore every scale -- is defined on the FULL tensor. The sharded scheme
splits the kernel at its natural seam:

1. every rank computes a PARTIAL absmax grid over its own rows, laid out on
   the GLOBAL block grid (zeros where the shard does not overlap a block);
2. one ``all_reduce(MAX)`` over the gather group turns partials into the
   global grid (tiny: 4 bytes per 128x128 block);
3. every rank quantizes its own rows locally with the global descales,
   replicating the kernel's exact fp32 op order (absmax/FP8_MAX, then
   1/descale, multiply, clamp, cast) so codes and descales match the
   whole-tensor kernel BIT FOR BIT.

Scope: dim-0 contiguous row shards (FSDP ``Shard(0)``; the mcore block case
rides the same helpers per touched block). Row offsets may fall mid-block --
partial overlaps contribute partial maxima, exactly like the kernel's padding
mask contributes zeros.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from verl.utils.kernel.fp8_kernel import FP8_DTYPE, FP8_MAX, ceil_div

logger = logging.getLogger(__name__)

# matches the triton kernel's numerical-stability floor for block absmax
_ABSMAX_EPS = 1e-10


def local_blockwise_absmax(
    shard: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
    row_offset: int,
    full_shape: tuple[int, int],
) -> torch.Tensor:
    """Partial per-block absmax of a dim-0 row shard, on the GLOBAL block grid.

    Returns a float32 ``(ceil(M/BM), ceil(N/BN))`` grid; blocks the shard does
    not overlap hold 0 (abs values are >= 0, so ``all_reduce(MAX)`` composes
    partials correctly).
    """
    bm, bn = int(weight_block_size[0]), int(weight_block_size[1])
    m_full, n_full = int(full_shape[0]), int(full_shape[1])
    rows, cols = shard.shape
    assert cols == n_full, f"row shard must span full dim-1: {cols} != {n_full}"
    n_br, n_bc = ceil_div(m_full, bm), ceil_div(n_full, bn)
    grid = torch.zeros(n_br, n_bc, dtype=torch.float32, device=shard.device)
    if rows == 0:
        return grid

    x = shard.to(torch.float32).abs()
    # NaN placeholders (mcore probe output: positions owned by OTHER ranks)
    # must not poison the partial max; zeros never win a legitimate max.
    x = torch.nan_to_num(x, nan=0.0)
    # pad dim-1 to the block grid once (zeros never win a max)
    pad_n = n_bc * bn - n_full
    if pad_n:
        x = torch.nn.functional.pad(x, (0, pad_n))
    first_block = row_offset // bm
    r = 0
    for br in range(first_block, ceil_div(row_offset + rows, bm)):
        take = min((br + 1) * bm - (row_offset + r), rows - r)
        seg = x[r : r + take]
        grid[br] = seg.view(take, n_bc, bn).amax(dim=(0, 2))
        r += take
    return grid


def quantize_shard_with_descale(
    shard: torch.Tensor,
    descale: torch.Tensor,
    weight_block_size: list[int] | tuple[int, int],
    row_offset: int,
) -> torch.Tensor:
    """Quantize a dim-0 row shard using GLOBAL per-block descales, replicating
    the kernel's fp32 op order (``s_inv = 1.0 / descale``; ``clamp(x * s_inv)``;
    cast) so the codes are bitwise-identical to the whole-tensor kernel."""
    bm, bn = int(weight_block_size[0]), int(weight_block_size[1])
    rows, cols = shard.shape
    n_bc = descale.shape[1]
    s_inv = 1.0 / descale  # matches the kernel's second fp32 division

    # NaN passes through multiply/clamp/cast untouched and lands as the fp8
    # NaN byte -- exactly the wire sentinel for "not this rank's position".
    x = shard.to(torch.float32)
    pad_n = n_bc * bn - cols
    if pad_n:
        x = torch.nn.functional.pad(x, (0, pad_n))
    # per-row block-row index -> expand descale rows to shard rows
    br_of_row = (torch.arange(row_offset, row_offset + rows, device=shard.device) // bm) - (row_offset // bm)
    first_block = row_offset // bm
    s_rows = s_inv[first_block + br_of_row]  # (rows, n_bc)
    x = x.view(rows, n_bc, bn)
    x = x * s_rows.unsqueeze(-1)
    x = x.clamp_(min=-FP8_MAX, max=FP8_MAX).to(FP8_DTYPE)
    x = x.view(rows, n_bc * bn)
    if pad_n:
        x = x[:, :cols].contiguous()
    return x


@dataclass
class QuantSpec:
    """Rollout-format request handed to a backend's ``get_per_tensor_param``.

    Deliberately rollout-agnostic: the caller (checkpoint engine) distills the
    serving engine's quantization config into a block shape plus a per-param
    predicate; the backend only honors the spec and never sees who asked.
    """

    weight_block_size: tuple[int, int]
    should_quantize: object  # Callable[[str], bool]
    # The checkpoint's scale dialect, from its quantization_config. DSv4 ships
    # "ue8m0" and sglang's loader requires it; None keeps the plain fp32 grid.
    scale_fmt: str | None = None
    # weight name -> the checkpoint's own scale grid (fp32, CPU). When present,
    # quantization is STICKY: a block keeps the checkpoint's scale as long as
    # its amax still fits under it, and only bumps to the tightest covering
    # power when the weights genuinely outgrew it. The checkpoint's scales
    # carry headroom on ~2% of blocks, and that headroom is unrecoverable from
    # the dequantized master -- recomputing from amax alone necessarily
    # tightens those blocks and changes their bytes.
    ckpt_scales: object | None = None  # dict[str, torch.Tensor] | None
    # name -> bool: params the CHECKPOINT stores in fp32 (DSv4's special
    # families). The wire keeps these fp32 instead of folding to the rollout
    # dtype; the predicate is checkpoint-derived so every rank -- including
    # ranks that do not own the param and see no tensor -- routes the slot
    # into the same wire group. None keeps the legacy fold-to-rollout-dtype.
    fp32_predicate: object | None = None  # Callable[[str], bool] | None


def sticky_ue8m0_descale(amax: torch.Tensor, ckpt_scale: torch.Tensor | None) -> torch.Tensor:
    """ue8m0 descale that PREFERS the checkpoint's scale wherever it still covers.

    A block's original scale is valid for any amax <= scale * FP8_MAX; keeping
    it makes unchanged weights reproduce the checkpoint's bytes exactly, which
    is what lets seed == disk AND keeps the steady verify quiet on blocks that
    never trained. Only blocks whose weights outgrew the old scale move -- to
    the tightest covering power, same dialect.
    """
    tight = ue8m0_descale(amax)
    if ckpt_scale is None:
        return tight
    assert ckpt_scale.shape == amax.shape, (
        f"ckpt scale grid {tuple(ckpt_scale.shape)} does not match the absmax grid "
        f"{tuple(amax.shape)}: the lookup matched the wrong tensor, refusing to guess"
    )
    ckpt_scale = ckpt_scale.to(amax.device)
    return torch.where(amax <= ckpt_scale * FP8_MAX, ckpt_scale, tight)


_CKPT_SCALES_CACHE: dict = {}


def load_ckpt_scales(ckpt_path: str) -> dict:
    """Read every ``<stem>.scale`` tensor from the checkpoint, keyed by the
    WEIGHT's name (``<stem>.weight``) for direct lookup at quantize time.

    Scale grids are tiny relative to the weights, so this loads once per
    process and stays on CPU. safetensors reads only the requested tensors,
    not the full shards.
    """
    got = _CKPT_SCALES_CACHE.get(ckpt_path)
    if got is not None:
        return got
    import json
    import os

    from safetensors import safe_open

    from verl.utils.fp8_ckpt_dtypes import canonical_ckpt_name

    with open(os.path.join(ckpt_path, "model.safetensors.index.json")) as index_file:
        idx = json.load(index_file)
    wm = idx["weight_map"]
    by_file: dict[str, list[str]] = {}
    for n, f in wm.items():
        if n.endswith(".scale"):
            by_file.setdefault(f, []).append(n)
    out: dict = {}
    stale: list[str] = []
    for f, names in by_file.items():
        with safe_open(os.path.join(ckpt_path, f), framework="pt", device="cpu") as fh:
            # A shard index can retain entries for tensors removed from the
            # actual safetensors file. DeepSeek-V4-Flash-FP8 does this for BF16
            # ``attn.wo_a`` scales. The index routes tensors to shards; it does
            # not prove a tensor exists in that shard.
            available = set(fh.keys())
            for n in names:
                if n not in available:
                    stale.append(n)
                    continue
                weight_name = n[: -len(".scale")] + ".weight"
                scale = fh.get_tensor(n).float()
                out[weight_name] = scale
                out.setdefault(canonical_ckpt_name(weight_name), scale)
    if stale:
        logger.warning(
            "ignored %d stale scale entries in %s (first=%s)",
            len(stale),
            ckpt_path,
            stale[0],
        )
    _CKPT_SCALES_CACHE[ckpt_path] = out
    return out


def ue8m0_descale(amax: torch.Tensor) -> torch.Tensor:
    """Power-of-two descale, byte-identical to the DSv4 nccl converter's formula.

    The exponent-only scale is what makes the trainer's dequant->requant round
    trip bit-exact: multiplying and dividing by 2^k shifts the fp8 exponent and
    never touches the mantissa. A plain amax/FP8_MAX scale is an arbitrary real,
    so the round trip rewrites the codes.
    """
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-10) / FP8_MAX)))


def quantize_hf_stream(weights, spec: QuantSpec):
    """Wrap a full HF ``(name, tensor)`` export with blockwise fp8 quantization:
    for every 2D weight the spec selects, yield ``(name, codes)`` +
    ``(name_scale_inv, descales)``; everything else passes through in bf16.
    Whole-tensor path (``group=None``) -- bitwise-identical to the sharded
    steady quantizer, which matters because fp32->fp8 tie rounding is
    implementation-sensitive across kernels.
    """
    block = list(spec.weight_block_size)
    for name, t in weights:
        if t.dim() != 2 or not spec.should_quantize(name):
            yield name, t
            continue
        # Fail loud on already-quantized input. Quantizing fp8 codes appears
        # numerically plausible but applies a second, destructive transform.
        assert t.element_size() > 1, (
            f"quantize_hf_stream got {t.dtype} for {name!r}: the input is already "
            "quantized codes, not a bf16 master. The export upstream must produce "
            "plain bf16 (pass explicit non-fp8 conversion_tasks to the bridge)."
        )
        t = t.to(torch.bfloat16)
        grid = local_blockwise_absmax(t, block, 0, tuple(t.shape))
        if getattr(spec, "scale_fmt", None) == "ue8m0":
            ck = getattr(spec, "ckpt_scales", None)
            descale = sticky_ue8m0_descale(grid, ck.get(name) if ck else None)
        else:
            descale = grid.clamp_(min=_ABSMAX_EPS) / FP8_MAX
        codes = quantize_shard_with_descale(t, descale, block, 0)
        yield name, codes
        yield name + "_scale_inv", descale
