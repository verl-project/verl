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
"""cuTile linear projection, selected log probability and token entropy."""

from typing import NamedTuple

import cuda.tile as ct
import torch
import torch.distributed as dist
from cuda.tile.tune import exhaustive_search

from verl.utils.device import get_torch_device

from ..linear_cross_entropy_config import BackwardEnum, _config

# 32 divides both the forward split and the backward chunk width.
_BLOCK = 32
_SPLIT = 1024
_CHUNK = 9504
_TOKEN_CHUNK_BYTES = 256 * 1024 * 1024


def _token_chunk_size(tokens, local_vocab, element_size, device):
    """Match Liger cuTile's default workspace budget and balance small tail chunks."""
    budget = _TOKEN_CHUNK_BYTES
    if element_size == 2 and tokens >= 4096 and local_vocab >= 131072:
        if get_torch_device().get_device_capability(device) in ((10, 0), (10, 3), (12, 0)):
            budget *= 2
    capacity = max(1, budget // (local_vocab * element_size))
    chunk = 1 << (capacity.bit_length() - 1)
    if tokens <= chunk:
        return tokens
    count = (tokens + chunk - 1) // chunk
    minimum = (tokens + capacity - 1) // capacity
    tail = tokens - (count - 1) * chunk
    if count > minimum or 2 * tail < chunk:
        return (tokens + minimum - 1) // minimum
    return chunk


def _load(a, index, shape):
    return ct.load(a, index, shape, padding_mode=ct.PaddingMode.ZERO)


def _projection(x, w, row, col, inv_temp, tm=_BLOCK, tn=_BLOCK, tk=_BLOCK):
    z = ct.full((tm, tn), 0, dtype=ct.float32)
    for k in range(ct.cdiv(x.shape[1], tk)):
        a = _load(x, (row, k), (tm, tk))
        b = _load(w, (col, k), (tn, tk))
        z = ct.mma(a, ct.transpose(b), z)
    return z * inv_temp


@ct.kernel
def _forward(
    x,
    w,
    labels,
    maxima,
    sums,
    moments,
    targets,
    rank_offset,
    inv_temp,
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
):
    row, split = ct.bid(0), ct.bid(1)
    rows = row * tm + ct.arange(tm, dtype=ct.int32)
    if labels.ndim == 1:
        ids = _load(labels, (row,), (tm,))
    maximum = ct.full((tm,), -float("inf"), dtype=ct.float32)
    total = ct.full((tm,), 0, dtype=ct.float32)
    moment = ct.full((tm,), 0, dtype=ct.float32)
    selected = ct.full((tm,), 0, dtype=ct.float32)
    first = split * (_SPLIT // tn)
    for col in range(first, min(first + _SPLIT // tn, ct.cdiv(w.shape[0], tn))):
        z = _projection(x, w, row, col, inv_temp, tm, tn, tk)
        vocab = col * tn + ct.arange(tn, dtype=ct.int32)
        valid = vocab[None, :] < w.shape[0]
        masked = ct.where(valid, z, -float("inf"))
        next_max = ct.maximum(maximum, ct.max(masked, axis=1))
        scale = ct.exp(maximum - next_max)
        exp_z = ct.exp(masked - next_max[:, None])
        total = total * scale + ct.sum(exp_z, axis=1)
        moment = moment * scale + ct.sum(z * exp_z, axis=1)
        if labels.ndim == 1:
            selected += ct.sum(ct.where(ids[:, None] == rank_offset + vocab[None, :], z, 0), axis=1)
        else:
            start = rank_offset + col * tn
            for q in range(labels.shape[1]):
                query = ct.gather(labels, (rows, q))
                value = ct.sum(ct.where(query[:, None] == rank_offset + vocab[None, :], z, 0), axis=1)
                owned = (query >= start) & (query < min(start + tn, rank_offset + w.shape[0]))
                ct.scatter(targets, (rows, q), value, mask=owned)
        maximum = next_max
    ct.store(maxima, (row, split), maximum[:, None])
    ct.store(sums, (row, split), total[:, None])
    ct.store(moments, (row, split), moment[:, None])
    if labels.ndim == 1:
        start = split * _SPLIT + rank_offset
        ct.scatter(targets, rows, selected, mask=(ids >= start) & (ids < min(start + _SPLIT, rank_offset + w.shape[0])))


# Each vocabulary tile divides the fixed split; the last split may be partial.
_FORWARD_TILES = (
    (32, 32, 32),
    (64, 64, 32),
    (64, 128, 32),
    (64, 128, 64),
    (64, 256, 64),
    (128, 128, 64),
    (128, 256, 32),
)
_forward_launch_cache = {}


def _launch_forward(stream, x, w, labels, maxima, sums, moments, targets, offset, inv_temp):
    m, k = x.shape
    splits = maxima.shape[1]
    key = (x.device, x.dtype, w.dtype, labels.dtype, labels.shape, m, w.shape[0], k)
    args = (x, w, labels, maxima, sums, moments, targets, offset, inv_temp)
    launch = _forward_launch_cache.get(key)
    if launch is None:

        def grid(tile):
            return ((m + tile[0] - 1) // tile[0], splits)

        tile = _FORWARD_TILES[0]
        device_module = get_torch_device()
        with device_module.device(x.device):
            capturing = device_module.is_current_stream_capturing()
            if not capturing:
                # Each trial overwrites the statistics and the same owned label
                # slots. Unowned TP targets retain their initial zeros.
                tile = exhaustive_search(
                    _FORWARD_TILES, stream, grid, _forward, lambda tile: (*args, *tile), quiet=True
                ).best.config
        launch = grid(tile), tile
        if not capturing:
            _forward_launch_cache[key] = launch
    launch_grid, tile = launch
    ct.launch(stream, launch_grid, _forward, (*args, *tile))


def _merge(maxima, local_maxima, sums, moments):
    row = ct.bid(0)
    maximum = ct.full((_BLOCK,), -float("inf"), dtype=ct.float32)
    total = ct.full((_BLOCK,), 0, dtype=ct.float32)
    moment = ct.full((_BLOCK,), 0, dtype=ct.float32)
    for block in range(ct.cdiv(maxima.shape[1], _BLOCK)):
        index = (row, block)
        valid = block * _BLOCK + ct.arange(_BLOCK, dtype=ct.int32) < maxima.shape[1]
        m = _load(maxima, index, (_BLOCK, _BLOCK))
        next_max = ct.maximum(maximum, ct.max(ct.where(valid[None, :], m, -float("inf")), axis=1))
        scale = ct.exp(maximum - next_max)
        local = _load(local_maxima, index, (_BLOCK, _BLOCK))
        coeff = ct.exp(ct.where(valid[None, :], local - next_max[:, None], -float("inf")))
        total = total * scale + ct.sum(coeff * _load(sums, index, (_BLOCK, _BLOCK)), axis=1)
        moment = moment * scale + ct.sum(coeff * _load(moments, index, (_BLOCK, _BLOCK)), axis=1)
        maximum = next_max
    return maximum, total, moment


def _finish(maximum, total, moment, targets, entropy, out, reduction):
    row = ct.bid(0)
    lse = maximum + ct.log(total)
    ct.store(entropy, (row,), lse - moment / total)
    if targets.ndim == 2:
        for q in range(ct.cdiv(targets.shape[1], _BLOCK)):
            lp = _load(targets, (row, q), (_BLOCK, _BLOCK)) - lse[:, None]
            ct.store(out, (row, q), lp)
        return
    lp = _load(targets, (row,), (_BLOCK,)) - lse
    if reduction == 0:
        ct.store(out, (row,), lp)
    else:
        rows = row * _BLOCK + ct.arange(_BLOCK, dtype=ct.int32)
        value = ct.sum(ct.where(rows < targets.shape[0], lp, 0), axis=0)
        if reduction == 2:
            value = value / targets.shape[0]
        ct.atomic_add(out, 0, value)


@ct.kernel
def _epilogue(maxima, sums, moments, maximum, total, mean, targets, entropy, out, reduction: ct.Constant[int]):
    m, s, b = _merge(maxima, maxima, sums, moments)
    row = ct.bid(0)
    ct.store(maximum, (row,), m)
    ct.store(total, (row,), s)
    ct.store(mean, (row,), b / s)
    _finish(m, s, b, targets, entropy, out, reduction)


@ct.kernel
def _epilogue_tp(maxima, local, sums, moments, maximum, total, moment):
    m, s, b = _merge(maxima, local, sums, moments)
    row = ct.bid(0)
    ct.store(maximum, (row,), m)
    ct.store(total, (row,), s)
    ct.store(moment, (row,), b)


@ct.kernel
def _finish_tp(maximum, total, moment, targets, entropy, out, reduction: ct.Constant[int]):
    row = ct.bid(0)
    m = _load(maximum, (row,), (_BLOCK,))
    s = _load(total, (row,), (_BLOCK,))
    b = _load(moment, (row,), (_BLOCK,))
    ct.store(moment, (row,), b / s)
    _finish(m, s, b, targets, entropy, out, reduction)


def _grad(
    x, w, labels, maximum, total, mean, g, h, row, col, offset, inv_temp, reduction, tm=_BLOCK, tn=_BLOCK, tk=_BLOCK
):
    z = _projection(x, w, row, col, inv_temp, tm, tn, tk)
    vocab = col * tn + ct.arange(tn, dtype=ct.int32)
    rows = row * tm + ct.arange(tm, dtype=ct.int32)
    inv_total = 1.0 / _load(total, (row,), (tm,))
    p = ct.exp(z - _load(maximum, (row,), (tm,))[:, None]) * inv_total[:, None]
    if labels.ndim == 1:
        ids = _load(labels, (row,), (tm,))
        upstream = _load(g, (row,), (tm,))[:, None] if reduction == 0 else ct.gather(g, 0)
        if reduction == 2:
            upstream = upstream / x.shape[0]
        grad = upstream * (ct.astype(ids[:, None] == offset + vocab[None, :], ct.float32) - p)
    else:
        selected = ct.full((tm, tn), 0, dtype=ct.float32)
        upstream_sum = ct.full((tm,), 0, dtype=ct.float32)
        for q in range(labels.shape[1]):
            ids = ct.gather(labels, (rows, q))
            upstream = ct.gather(g, (rows, q))
            upstream_sum += upstream
            selected += ct.where(ids[:, None] == offset + vocab[None, :], upstream[:, None], 0)
        grad = selected - upstream_sum[:, None] * p
    dh = _load(h, (row,), (tm,))
    mu = _load(mean, (row,), (tm,))
    grad += dh[:, None] * p * (mu[:, None] - z)
    return ct.where((rows[:, None] < x.shape[0]) & (vocab[None, :] < w.shape[0]), grad * inv_temp, 0)


@ct.kernel
def _dlogits(x, w, labels, maximum, total, mean, g, h, dst, offset, inv_temp, reduction: ct.Constant[int]):
    row, col = ct.bid(0), ct.bid(1)
    grad = _grad(x, w, labels, maximum, total, mean, g, h, row, col, offset, inv_temp, reduction)
    ct.store(dst, (row, col), ct.astype(grad, dst.dtype))


@ct.kernel
def _dlogits_chunk(
    x,
    w,
    labels,
    maximum,
    total,
    mean,
    g,
    h,
    dst,
    offset,
    inv_temp,
    reduction: ct.Constant[int],
    tm: ct.Constant[int],
    tn: ct.Constant[int],
    tk: ct.Constant[int],
):
    row, col = ct.bid(0), ct.bid(1)
    grad = _grad(x, w, labels, maximum, total, mean, g, h, row, col, offset, inv_temp, reduction, tm, tn, tk)
    ct.store(dst, (row, col), ct.astype(grad, dst.dtype))


class _ChunkConfig(NamedTuple):
    tm: int
    tn: int
    tk: int
    occupancy: int
    num_ctas: int = 1


_CHUNK_CONFIGS = (
    _ChunkConfig(128, 256, 32, 1),
    _ChunkConfig(128, 128, 32, 1),
    _ChunkConfig(128, 128, 64, 1),
    _ChunkConfig(128, 128, 64, 2),
    _ChunkConfig(256, 256, 64, 1, 2),
    _ChunkConfig(256, 256, 64, 1, 4),
    _ChunkConfig(64, 128, 64, 1),
    _ChunkConfig(64, 128, 64, 2),
    _ChunkConfig(64, 64, 32, 2),
    _ChunkConfig(32, 32, 32, 1),
)
_chunk_launch_cache = {}


def _launch_chunk(stream, x, w, labels, maximum, total, mean, g, h, dst, offset, inv_temp, reduction):
    m, k = x.shape
    n = w.shape[0]
    key = (x.device, x.dtype, labels.dtype, labels.shape, m, n, k, dst.stride(0), g.stride(0), h.stride(0), reduction)
    launch = _chunk_launch_cache.get(key)
    args = (x, w, labels, maximum, total, mean, g, h, dst, offset, inv_temp, reduction)
    if launch is None:

        def grid(cfg):
            return ((m + cfg.tm - 1) // cfg.tm, (n + cfg.tn - 1) // cfg.tn)

        cfg = _CHUNK_CONFIGS[-1]
        # Tuning synchronizes the device. A cold graph capture can use a static
        # launch; the next eager invocation will still tune this shape.
        device_module = get_torch_device()
        with device_module.device(x.device):
            capturing = device_module.is_current_stream_capturing()
            if not capturing:
                configs = [
                    cfg
                    for cfg in _CHUNK_CONFIGS
                    if cfg.num_ctas == 1
                    or (m >= cfg.tm and n >= cfg.tn and device_module.get_device_capability(x.device)[0] >= 9)
                ]
                result = exhaustive_search(
                    configs,
                    stream,
                    grid,
                    _dlogits_chunk,
                    lambda cfg: (*args, cfg.tm, cfg.tn, cfg.tk),
                    lambda cfg: {"occupancy": cfg.occupancy, "num_ctas": cfg.num_ctas},
                    quiet=True,
                )
                cfg = result.best.config
        launch = grid(cfg), _dlogits_chunk.replace_hints(occupancy=cfg.occupancy, num_ctas=cfg.num_ctas), cfg
        if not capturing:
            _chunk_launch_cache[key] = launch
    launch_grid, kernel, cfg = launch
    ct.launch(stream, launch_grid, kernel, (*args, cfg.tm, cfg.tn, cfg.tk))


def _add_gradient(dst, left, right, output_row, source_row):
    axis = ct.arange(_BLOCK, dtype=ct.int32)
    rows = output_row * _BLOCK + axis
    for k in range(ct.cdiv(right.shape[1], _BLOCK)):
        rhs = _load(right, (source_row, k), (_BLOCK, _BLOCK))
        value = ct.mma(left, ct.astype(rhs, ct.float32), ct.full((_BLOCK, _BLOCK), 0, dtype=ct.float32))
        ct.atomic_add(dst, (rows[:, None], (k * _BLOCK + axis)[None, :]), value)


@ct.kernel
def _fused_grad(x, w, labels, maximum, total, mean, g, h, dx, dw, offset, inv_temp, reduction: ct.Constant[int]):
    row, col = ct.bid(0), ct.bid(1)
    grad = _grad(x, w, labels, maximum, total, mean, g, h, row, col, offset, inv_temp, reduction)
    _add_gradient(dx, grad, w, row, col)
    _add_gradient(dw, ct.transpose(grad), x, col, row)


@ct.kernel
def _d_hidden(x, w, labels, maximum, total, mean, g, h, dst, offset, inv_temp, reduction: ct.Constant[int]):
    row, k = ct.bid(0), ct.bid(1)
    value = ct.full((_BLOCK, _BLOCK), 0, dtype=ct.float32)
    for col in range(ct.cdiv(w.shape[0], _BLOCK)):
        grad = _grad(x, w, labels, maximum, total, mean, g, h, row, col, offset, inv_temp, reduction)
        value = ct.mma(ct.astype(grad, w.dtype), _load(w, (col, k), (_BLOCK, _BLOCK)), value)
    ct.store(dst, (row, k), ct.astype(value, dst.dtype))


@ct.kernel
def _d_weight(x, w, labels, maximum, total, mean, g, h, dst, offset, inv_temp, reduction: ct.Constant[int]):
    col, k = ct.bid(0), ct.bid(1)
    value = ct.full((_BLOCK, _BLOCK), 0, dtype=ct.float32)
    for row in range(ct.cdiv(x.shape[0], _BLOCK)):
        grad = _grad(x, w, labels, maximum, total, mean, g, h, row, col, offset, inv_temp, reduction)
        value = ct.mma(ct.astype(ct.transpose(grad), x.dtype), _load(x, (row, k), (_BLOCK, _BLOCK)), value)
    ct.store(dst, (col, k), ct.astype(value, dst.dtype))


def _blocks(n):
    return (n + _BLOCK - 1) // _BLOCK


def efficient_entropy_forward(hidden, weight, labels, reduction, temperature, group):
    assert hidden.device == weight.device == labels.device
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    m, d = hidden.shape
    v, dw = weight.shape
    assert d == dw and d % 128 == 0 and labels.shape[0] == m
    splits = (v + _SPLIT - 1) // _SPLIT
    options = dict(device=hidden.device, dtype=torch.float32)
    maxima = torch.empty((m, splits), **options)
    sums = torch.empty_like(maxima)
    moments = torch.empty_like(maxima)
    targets = torch.zeros(labels.shape, **options)
    maximum = torch.empty(m, **options)
    # Keep the two row-statistic arrays aligned for tiled loads, including
    # when the token count is odd. TP reduces both rows in one collective.
    stats = torch.zeros((2, _blocks(m) * _BLOCK), **options)
    total, mean = stats[:, :m].unbind()
    entropy = torch.empty_like(maximum)
    out = targets if reduction == 0 else torch.zeros(1, **options)
    rank = 0 if group is None else dist.get_rank(group)
    device_module = get_torch_device()
    stream = device_module.current_stream(hidden.device)
    _launch_forward(stream, hidden, weight, labels, maxima, sums, moments, targets, rank * v, 1.0 / temperature)
    grid = (_blocks(m),)
    if group is None:
        ct.launch(
            stream, grid, _epilogue, (maxima, sums, moments, maximum, total, mean, targets, entropy, out, reduction)
        )
    else:
        local_maxima = maxima.clone()
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=group)
        aux = device_module.Stream(device=hidden.device)
        aux.wait_stream(stream)
        with device_module.stream(aux):
            dist.all_reduce(targets, group=group)
        ct.launch(stream, grid, _epilogue_tp, (maxima, local_maxima, sums, moments, maximum, total, mean))
        stream.wait_stream(aux)
        dist.all_reduce(stats, group=group)
        ct.launch(stream, grid, _finish_tp, (maximum, total, mean, targets, entropy, out, reduction))
    return out if reduction == 0 else out.view(()), entropy, maximum, total, mean


def efficient_entropy_backward(
    g, h, hidden, weight, labels, maximum, total, mean, reduction, fp32_grad, temperature, group
):
    m, d = hidden.shape
    v = weight.shape[0]
    stream = get_torch_device().current_stream(hidden.device)
    g = g.reshape_as(labels) if labels.ndim == 2 else g.reshape(-1)
    args = (hidden, weight, labels, maximum, total, mean, g, h)
    offset = 0 if group is None else dist.get_rank(group) * v
    inv_temp = 1.0 / temperature
    params = (offset, inv_temp, reduction)
    method = _config._backward
    if method == BackwardEnum._Total_Fuse_MN:
        dx = torch.zeros_like(hidden, dtype=torch.float32)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        ct.launch(stream, (_blocks(m), _blocks(v)), _fused_grad, (*args, dx, dw, *params))
        return dx, dw
    dtype = torch.float32 if fp32_grad else hidden.dtype
    if method == BackwardEnum._Split_Dlogits_M:
        chunk = _token_chunk_size(m, v, hidden.element_size(), hidden.device)
        workspace = hidden.new_empty((chunk, v))
        dx = torch.empty_like(hidden, dtype=dtype)
        dw = torch.empty_like(weight, dtype=torch.float32 if _config._split_m_fp32_dw else weight.dtype)
        if reduction == 2:
            # The kernel sees only the local rows. Normalize by the full token count once.
            g = g / m
            reduction = 1
        for first in range(0, m, chunk):
            end = min(first + chunk, m)
            tokens = slice(first, end)
            grad = workspace[: end - first]
            _launch_chunk(
                stream,
                hidden[tokens],
                weight,
                labels[tokens],
                maximum[tokens],
                total[tokens],
                mean[tokens],
                g[tokens] if reduction == 0 else g,
                h[tokens],
                grad,
                offset,
                inv_temp,
                reduction,
            )
            if grad.dtype == dx.dtype:
                torch.mm(grad, weight, out=dx[tokens])
            else:
                torch.mm(grad, weight, out_dtype=dtype, out=dx[tokens])
            beta = int(first != 0)
            if grad.dtype == dw.dtype:
                dw.addmm_(grad.T, hidden[tokens], beta=beta)
            else:
                torch.addmm(dw, grad.T, hidden[tokens], beta=beta, out_dtype=torch.float32, out=dw)
            del grad
        del workspace
        return dx, dw.to(dtype)

    dw = torch.empty_like(weight, dtype=dtype)
    if method == BackwardEnum._Total_Separate:
        dx = torch.empty_like(hidden, dtype=dtype)
        grad = hidden.new_empty((m, v))
        ct.launch(stream, (_blocks(m), _blocks(v)), _dlogits, (*args, grad, *params))
        torch.matmul(grad, weight, out=dx)
        torch.matmul(grad.T, hidden, out=dw)
    else:
        dx = torch.empty_like(hidden, dtype=torch.float32)
        workspace = hidden.new_empty((m, min(_CHUNK, v)))
        for first in range(0, v, _CHUNK):
            end = min(first + _CHUNK, v)
            grad = workspace[:, : end - first]
            # Slice before tiling: chunk starts need not be multiples of tn.
            _launch_chunk(
                stream,
                hidden,
                weight[first:end],
                labels,
                maximum,
                total,
                mean,
                g,
                h,
                grad,
                offset + first,
                inv_temp,
                reduction,
            )
            beta = int(first != 0)
            if grad.dtype == dx.dtype:
                dx.addmm_(grad, weight[first:end], beta=beta)
            else:
                torch.addmm(dx, grad, weight[first:end], beta=beta, out_dtype=torch.float32, out=dx)
            torch.matmul(grad.T, hidden, out=dw[first:end])
            del grad
        # Release the chunk buffer before allocating the returned low-precision gradient.
        del workspace
        dx = dx.to(dtype)
    return dx, dw
