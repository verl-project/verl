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
"""Unit tests for the delta wire checksum (order-sensitive 64-bit fold).

Runs on CPU anywhere torch is installed, and on NPU/GPU devices when visible;
the checksum is a pure function of tensor bytes, so these assertions hold on
every device.
"""

import pytest
import torch

import verl.checkpoint_engine.delta_sync.encode as encode
from verl.checkpoint_engine.delta_sync.encode import _wire_digest, checksum

_MASK64 = (1 << 64) - 1
_POS_MIX = 0x9E3779B97F4A7C15  # unsigned twin of encode._POS_MIX


def _ref_digest(raw: bytes) -> int:
    """Pure-python reference for ``_wire_digest`` (little-endian int64 words)."""
    n8 = len(raw) // 8 * 8
    fold = 0
    for i in range(n8 // 8):
        w = int.from_bytes(raw[i * 8 : (i + 1) * 8], "little")
        fold ^= (w + _POS_MIX * (i + 1)) & _MASK64
    tail = 0
    for i in range(n8, len(raw)):
        tail ^= raw[i] << (8 * (i - n8))
    return (fold ^ tail) & _MASK64


def test_empty_arguments_checksum_to_zero():
    pos = torch.empty(0, dtype=torch.uint8)
    val = torch.empty(0, dtype=torch.bfloat16)
    assert _wire_digest(pos) == 0
    assert checksum(pos, val) == 0


def test_matches_reference():
    for n in [1, 2, 3, 7, 8, 9, 31, 64, 100, 4097]:
        pos = torch.randint(0, 256, (n,), dtype=torch.uint8)
        val = torch.randn(n + 13).to(torch.bfloat16)
        assert _wire_digest(pos) == _ref_digest(bytes(pos.tolist()))
        assert _wire_digest(val) == _ref_digest(bytes(val.view(torch.uint8).tolist()))


def test_positions_and_values_combine_deterministically():
    pos = torch.tensor([1, 2, 3], dtype=torch.uint8)
    val = torch.tensor([4.0], dtype=torch.bfloat16)
    p = _ref_digest(bytes(pos.tolist()))
    v = _ref_digest(bytes(val.view(torch.uint8).tolist()))
    expected = (p ^ (((v << 1) | (v >> 63)) & _MASK64)) & _MASK64
    assert checksum(pos, val) == expected
    assert checksum(pos, val) == checksum(pos.clone(), val.clone())


def test_single_bit_flip_is_detected():
    for which in ["pos", "val"]:
        pos = torch.randint(0, 256, (97,), dtype=torch.uint8)
        val = torch.randn(33).to(torch.bfloat16)
        before = checksum(pos, val)
        if which == "pos":
            pos = pos.clone()
            pos[41] ^= 0x01
        else:
            val = val.clone()
            raw = val.view(torch.uint8)
            raw[5] ^= 0x80
        assert checksum(pos, val) != before, f"flip in {which} went undetected"


def test_reordering_is_detected():
    raw = torch.arange(64, dtype=torch.uint8)
    base = _wire_digest(raw)
    # Swapping two whole words keeps the byte multiset; only the positional
    # term can catch it.
    words = raw.view(torch.int64)
    swapped = words.clone()
    swapped[0], swapped[5] = words[5].clone(), words[0].clone()
    assert _wire_digest(swapped.view(torch.uint8)) != base
    # Swapping two bytes inside one word changes the word value itself.
    inside = raw.clone()
    inside[8], inside[9] = raw[9].item(), raw[8].item()
    assert _wire_digest(inside) != base


def test_chunked_fold_matches_single_pass(monkeypatch):
    # The fold accumulates over _FOLD_CHUNK_BYTES chunks; shrinking the chunk
    # must not change the digest.
    pos = torch.randint(0, 256, (4097,), dtype=torch.uint8)
    val = torch.randn(999).to(torch.bfloat16)
    single = (checksum(pos, val), _wire_digest(pos))
    monkeypatch.setattr(encode, "_FOLD_CHUNK_BYTES", 32)
    chunked = (checksum(pos, val), _wire_digest(pos))
    assert chunked == single
    assert chunked[1] == _ref_digest(bytes(pos.tolist()))


def test_offset_slice_input_is_handled():
    # A contiguous slice can carry a storage offset the int64 chunk views
    # reject; those chunks must be re-housed, not crash.
    raw = torch.arange(17, dtype=torch.uint8)
    assert _wire_digest(raw[1:17]) == _ref_digest(bytes(raw[1:17].tolist()))
    vals = torch.randn(40).to(torch.bfloat16)[3:]  # offset 3 elements = 6 bytes
    assert _wire_digest(vals) == _ref_digest(bytes(vals.view(torch.uint8).tolist()))


def test_offset_slice_chunked_matches_reference(monkeypatch):
    raw = torch.randint(0, 256, (130,), dtype=torch.uint8)
    monkeypatch.setattr(encode, "_FOLD_CHUNK_BYTES", 32)
    for lo in (0, 1, 3, 5):
        s = raw[lo:]
        assert _wire_digest(s) == _ref_digest(bytes(s.tolist()))


def test_dtype_view_invariance():
    # same bytes, different dtype interpretation -> same digest
    raw = torch.randint(0, 256, (64,), dtype=torch.uint8)
    as_bf16 = raw.view(torch.bfloat16)
    assert _wire_digest(raw) == _wire_digest(as_bf16)
    assert _wire_digest(as_bf16.contiguous()) == _wire_digest(as_bf16.reshape(-1, 4).contiguous())


def test_input_storage_not_mutated():
    pos = torch.randint(0, 256, (33,), dtype=torch.uint8)
    val = torch.randn(17).to(torch.bfloat16)
    pos_before, val_before = pos.clone(), val.clone()
    checksum(pos, val)
    assert torch.equal(pos, pos_before)
    assert torch.equal(val.view(torch.uint8), val_before.view(torch.uint8))


def test_large_tensor_against_reference():
    pos = torch.randint(0, 256, (1 << 20,), dtype=torch.uint8)
    val = torch.randn(1 << 18).to(torch.bfloat16)
    assert _wire_digest(pos) == _ref_digest(bytes(pos.tolist()))
    assert _wire_digest(val) == _ref_digest(bytes(val.view(torch.uint8).tolist()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_checksum_agrees_across_cpu_and_cuda():
    pos = torch.randint(0, 256, (1000,), dtype=torch.uint8)
    val = torch.randn(777).to(torch.bfloat16)
    assert checksum(pos, val) == checksum(pos.cuda(), val.cuda())


if __name__ == "__main__":
    pytest.main([__file__])
