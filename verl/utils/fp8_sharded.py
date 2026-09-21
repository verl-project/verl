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
    # Optional packed MXFP4 codec for routed experts.  The predicate sees the
    # logical (unpacked BF16) HF weight name.  Selected weights are encoded as
    # two E2M1 values per int8 byte with one E8M0 scale per row/K32 tile.
    # Keeping this beside the FP8 definition makes the spec a per-weight codec
    # plan rather than a model-wide dtype switch.
    mxfp4_predicate: object | None = None  # Callable[[str], bool] | None


def quantize_mxfp4_e2m1(weight: torch.Tensor, *, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a logical 2-D weight to the DSV4 checkpoint's MXFP4 layout.

    Returns packed int8 codes (low nibble first) and E8M0 power-of-two scales.
    The arithmetic intentionally matches Megatron Bridge's DSV4 exporter so
    the sharded and whole-tensor paths produce the same bytes.
    """
    if weight.ndim != 2:
        raise RuntimeError(f"MXFP4 export expects a 2-D weight, got {weight.ndim}D")
    rows, cols = weight.shape
    if cols % block_size or cols % 2:
        raise RuntimeError(
            f"MXFP4 export requires K divisible by {block_size} and 2, got shape={tuple(weight.shape)}"
        )
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is None:
        raise RuntimeError("this PyTorch build has no float8_e8m0fnu dtype required by DSV4 MXFP4")

    x = weight.to(torch.float32)
    scale_cols = cols // block_size
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((rows, scale_cols), dtype=torch.float32, device=weight.device)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=weight.device
    )
    max_chunk_elements = 16_000_000
    rows_per_chunk = max(1, min(rows, max_chunk_elements // max(cols, 1)))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = x[row_start:row_end].reshape(-1, scale_cols, block_size)
        amax = chunk.abs().amax(dim=-1)
        scale = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
        scale = torch.exp2(torch.ceil(torch.log2(scale.clamp(min=2.0**-127, max=2.0**127))))
        scales[row_start:row_end] = scale

        normalized = chunk / scale.unsqueeze(-1)
        codes = torch.bucketize(normalized.abs(), boundaries).to(torch.uint8)
        codes = (codes | ((normalized < 0).to(torch.uint8) * 8)).reshape(row_end - row_start, cols)
        lo = codes[:, 0::2].to(torch.int16)
        hi = codes[:, 1::2].to(torch.int16)
        packed[row_start:row_end] = (lo | (hi << 4)).to(torch.uint8)

    return packed.contiguous().view(torch.int8), scales.to(e8m0)


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


def load_ckpt_scales(ckpt_path: str, *, skip_mxfp4_experts: bool = False) -> dict:
    """Read every ``<stem>.scale`` tensor from the checkpoint, keyed by the
    WEIGHT's name (``<stem>.weight``) for direct lookup at quantize time.

    Scale grids are tiny relative to the weights, so this loads once per
    process and stays on CPU. safetensors reads only the requested tensors,
    not the full shards.
    """
    cache_key = (ckpt_path, bool(skip_mxfp4_experts))
    got = _CKPT_SCALES_CACHE.get(cache_key)
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
            # Standard DSV4 Flash stores routed-expert K32 E8M0 scales here.
            # They are gigabytes in aggregate and belong to the MXFP4 codec,
            # not FP8 sticky-scale reconstruction.  Loading/converting them to
            # fp32 in every trainer process would multiply that footprint by 4.
            if skip_mxfp4_experts and ".experts." in n:
                continue
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
    _CKPT_SCALES_CACHE[cache_key] = out
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
        mxfp4_pred = getattr(spec, "mxfp4_predicate", None)
        if t.dim() == 2 and mxfp4_pred is not None and mxfp4_pred(name):
            assert t.element_size() > 1, (
                f"quantize_hf_stream got {t.dtype} for MXFP4 weight {name!r}: "
                "the upstream export must provide the logical BF16 master"
            )
            codes, scales = quantize_mxfp4_e2m1(t.to(torch.bfloat16))
            yield name, codes
            yield name + "_scale_inv", scales
            continue
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
