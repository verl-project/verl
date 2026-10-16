# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""Bucket-packing alignment for mixed-dtype weight streams (CPU-only).

The checkpoint-engine wire packs tensors back-to-back into a uint8 bucket and
the receive path reinterprets each single-chunk slice with
``Tensor.view(dtype)``, which torch only allows when the slice starts at a
multiple of the dtype's element size. A homogeneous stream keeps that
invariant for free; a mixed-dtype stream does not (for example an odd-numel
bf16 tensor followed by an fp32 tensor). ``align_bucket_offset`` restores the
invariant at pack time.

These tests pin the helper's math, drive a pack simulation that mirrors the
senders' loop structure (align, overflow check, flush, place) through the
real receive-side ``merge_weight_chunks`` across bucket sizes that force
flushes and multi-chunk continuation, keep a control showing the pre-fix
back-to-back layout is rejected at the view call, and assert at the source
level that every engine's send loop calls the helper.
"""

import asyncio
import pathlib
import re

import pytest
import torch

from verl.checkpoint_engine.base import TensorMeta, align_bucket_offset, merge_weight_chunks, split_weight_chunks


@pytest.mark.parametrize(
    ("offset", "dtype", "expected"),
    [
        (0, torch.float32, 0),  # bucket start is aligned for every dtype
        (6, torch.float32, 8),  # the motivating case: odd-numel bf16 tail
        (8, torch.float32, 8),  # already aligned: no-op
        (1, torch.uint8, 1),  # itemsize 1 never pads
        (2, torch.bfloat16, 2),
        (3, torch.bfloat16, 4),
        (9, torch.int64, 16),
        (17, torch.complex128, 32),  # largest itemsize torch ships
    ],
)
def test_align_bucket_offset_math(offset, dtype, expected):
    assert align_bucket_offset(offset, dtype) == expected


def _mixed_dtype_inventory() -> list[tuple[str, torch.Tensor]]:
    """A stream shaped like the failure case: odd byte counts before wider dtypes."""
    generator = torch.Generator().manual_seed(1234)
    return [
        ("norm.weight", torch.randn(3, generator=generator).to(torch.bfloat16)),  # 6 bytes: breaks fp32 alignment
        ("router.weight", torch.randn(4, 4, generator=generator)),  # fp32
        ("flags", torch.randint(0, 2, (5,), generator=generator, dtype=torch.uint8)),  # odd length
        ("step", torch.randint(0, 100, (3,), generator=generator, dtype=torch.int64)),
        ("proj.weight", torch.randn(7, 3, generator=generator).to(torch.bfloat16)),
        ("big.weight", torch.randn(200, generator=generator)),  # 800 bytes: multi-chunk at small buckets
        ("tail.weight", torch.randn(3, generator=generator).to(torch.bfloat16)),  # packs after the bucket cut
    ]


async def _pack(inventory, aligned: bool, bucket_size: int):
    """Pack through the real ``split_weight_chunks``, mirroring the senders' loop.

    The engines' send loops share one structure: (optionally) align the
    offset, flush the bucket when the next chunk does not fit, place the
    chunk, advance. This simulation reproduces that structure over sealed
    CPU buckets; ``aligned=False`` reproduces the pre-fix back-to-back
    layout. It does not drive the NCCL/zmq transport itself; the engines'
    loops are additionally pinned by ``test_every_engine_send_loop_aligns``.
    """
    buckets: list[tuple[torch.Tensor, list[TensorMeta]]] = []
    bucket = torch.zeros(bucket_size, dtype=torch.uint8)
    metas: list[TensorMeta] = []
    offset = 0

    def _gen():
        yield from inventory

    async for tensor_meta, chunk in split_weight_chunks(_gen(), bucket_size):
        chunk = chunk.contiguous()
        if aligned:
            offset = align_bucket_offset(offset, tensor_meta.dtype)
        if offset + tensor_meta.chunk_size > bucket_size:
            buckets.append((bucket, metas))
            bucket = torch.zeros(bucket_size, dtype=torch.uint8)
            metas = []
            offset = 0
        tensor_meta.offset = offset
        bucket[offset : offset + tensor_meta.chunk_size] = chunk
        metas.append(tensor_meta)
        offset += tensor_meta.chunk_size
    buckets.append((bucket, metas))
    return buckets


async def _receive(buckets, bucket_size: int):
    """Drive the real receive-side merge over slices at the packed offsets."""

    async def chunks():
        for bucket, metas in buckets:
            for meta in metas:
                yield meta, bucket[meta.offset : meta.offset + meta.chunk_size]

    received = []
    async for name, weight in merge_weight_chunks(chunks(), bucket_size):
        received.append((name, weight))
    return received


async def _round_trip(inventory, aligned: bool, bucket_size: int):
    buckets = await _pack(inventory, aligned=aligned, bucket_size=bucket_size)
    return buckets, await _receive(buckets, bucket_size)


# 1 MiB: everything single-chunk, no flush. 512/300: flushes between tensors.
# 257: odd bucket size with flushes. 256: exact fill by big.weight's chunks
# plus multi-chunk continuation for every bucket size below 800 bytes.
# 70/101: alignment padding itself forces a flush (offset aligned past what
# the bucket can hold), pinning the align-before-overflow-check ordering:
# with the align applied after the check instead, placement overruns the
# bucket and the copy raises.
@pytest.mark.parametrize("bucket_size", [70, 101, 256, 257, 300, 512, 1 << 20])
def test_aligned_mixed_stream_round_trips_bit_exact(bucket_size):
    inventory = _mixed_dtype_inventory()
    buckets, received = asyncio.run(_round_trip(inventory, aligned=True, bucket_size=bucket_size))
    for _, metas in buckets:
        for meta in metas:
            if meta.chunk_offset == 0:  # tensor start; continuation chunks are raw byte copies
                assert meta.offset % meta.dtype.itemsize == 0
    assert [name for name, _ in received] == [name for name, _ in inventory]
    for (_, sent), (_, got) in zip(inventory, received, strict=True):
        assert got.dtype == sent.dtype and got.shape == sent.shape
        assert torch.equal(
            got.contiguous().view(-1).view(torch.uint8),
            sent.contiguous().view(-1).view(torch.uint8),
        )


def test_unaligned_mixed_stream_rejected_at_view():
    """Control: the pre-fix back-to-back layout fails at the receive-side view.

    Pins the bug mechanism (torch refuses a dtype view at a storage offset
    that the element size does not divide) and proves the round-trip test
    can fail.
    """
    inventory = _mixed_dtype_inventory()
    with pytest.raises(RuntimeError, match="divisible"):
        asyncio.run(_round_trip(inventory, aligned=False, bucket_size=1 << 20))


# 64 bytes forces flushes between the 30-byte tensors; 1 MiB is the no-flush case.
@pytest.mark.parametrize("bucket_size", [64, 1 << 20])
def test_aligned_homogeneous_stream_layout_unchanged(bucket_size):
    """On a single-dtype stream the fix is a byte-for-byte no-op: same bucket
    count, same offsets bucket by bucket, same bucket bytes."""
    generator = torch.Generator().manual_seed(99)
    inventory = [(f"w{i}", torch.randn(5, 3, generator=generator).to(torch.bfloat16)) for i in range(4)]
    aligned_buckets = asyncio.run(_pack(inventory, aligned=True, bucket_size=bucket_size))
    packed_buckets = asyncio.run(_pack(inventory, aligned=False, bucket_size=bucket_size))
    assert len(aligned_buckets) == len(packed_buckets)
    for (a_bucket, a_metas), (p_bucket, p_metas) in zip(aligned_buckets, packed_buckets, strict=True):
        assert [m.offset for m in a_metas] == [m.offset for m in p_metas]
        assert torch.equal(a_bucket, p_bucket)


def test_every_engine_send_loop_aligns():
    """Source-level guard: every send loop that packs a running offset aligns it.

    The affected loops (nccl, hccl, nixl today) require live transports, so
    this pins their call sites the way tests/special_sanity pins API usage.
    The file list is derived from the loop's signature line rather than
    hardcoded, so an engine added later with the same packing structure is
    checked automatically. Engines without the signature (mooncake casts to
    the rollout dtype before packing; kimi takes offsets from the external
    checkpoint-engine package) are out of scope by construction.
    """
    engine_dir = pathlib.Path(__file__).resolve().parents[2] / "verl" / "checkpoint_engine"
    packing_engines = sorted(f.name for f in engine_dir.glob("*.py") if "tensor_meta.offset = offset" in f.read_text())
    assert packing_engines, "no packing loops found; the signature line moved and this guard needs updating"
    for engine in packing_engines:
        source = (engine_dir / engine).read_text()
        assert re.search(
            r"^\s*offset = align_bucket_offset\(offset, tensor_meta\.dtype\)$", source, flags=re.MULTILINE
        ), f"{engine}: packing loop no longer aligns offsets; mixed-dtype streams will crash the receiver"
