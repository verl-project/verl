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
"""Unit tests for the delta wire checksum (order-sensitive 128-bit dual fold).

Runs on CPU anywhere torch is installed, and on NPU/GPU devices when visible;
the checksum is a pure function of tensor bytes, so these assertions hold on
every device.
"""

import pytest
import torch

import verl.checkpoint_engine.delta_sync.encode as encode
from verl.checkpoint_engine.delta_sync.encode import _wire_digest, checksum

_MASK64 = (1 << 64) - 1
_MASK128 = (1 << 128) - 1
_POS_MIX = 0x9E3779B97F4A7C15  # unsigned twin of encode._POS_MIX
_AVALANCHE_MIX = 0xFF51AFD7ED558CCD  # unsigned twin of encode._AVALANCHE_MIX
_AVALANCHE_MIX2 = 0xC4CEB9FE1A85EC53  # unsigned twin of encode._AVALANCHE_MIX2


def _ref_digest(raw: bytes) -> int:
    """Pure-python reference for ``_wire_digest`` (little-endian int64 words)."""
    n8 = len(raw) // 8 * 8
    fold = 0
    total = 0
    for i in range(n8 // 8):
        w = int.from_bytes(raw[i * 8 : (i + 1) * 8], "little")
        v = (w + _POS_MIX * (i + 1)) & _MASK64
        v = ((v ^ (v >> 33)) * _AVALANCHE_MIX) & _MASK64
        v = ((v ^ (v >> 29)) * _AVALANCHE_MIX2) & _MASK64
        v = v ^ (v >> 32)
        fold ^= v
        total = (total + v) & _MASK64
    tail = 0
    for i in range(n8, len(raw)):
        tail ^= raw[i] << (8 * (i - n8))
    return (((total + tail) & _MASK64) << 64) | ((fold ^ tail) & _MASK64)


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
    expected = (p ^ (((v << 1) | (v >> 127)) & _MASK128)) & _MASK128
    assert checksum(pos, val) == expected
    assert checksum(pos, val) == checksum(pos.clone(), val.clone())


def test_int64_sum_folds_wraparound_mod_2_64():
    # The sum fold relies on torch int64 reduction wrapping mod 2**64 (no
    # promotion, no overflow error); pin that semantics explicitly.
    v = torch.full((4,), 1 << 62, dtype=torch.int64)
    assert int(v.sum()) == 0  # wraps: 4 * 2**62 == 2**64 == 0 mod 2**64
    big = torch.randint(-(1 << 62), 1 << 62, (4097,), dtype=torch.int64)
    assert int(big.sum()) & _MASK64 == sum(big.tolist()) & _MASK64


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


def test_bit63_word_pair_swap_is_detected():
    # Deterministic regression: an affine-only position mix (w + K * i) maps
    # words that differ solely in bit 63 to mixed values that also differ
    # solely in bit 63 -- adding 2**63 mod 2**64 toggles exactly that bit --
    # so the XOR fold cancels such swaps. The avalanche step exists to break
    # this cancellation; keep this case pinned regardless of future mixing.
    def as_i64(u: int) -> int:
        return u - (1 << 64) if u >= (1 << 63) else u

    for w1 in (0, 1, 0x123456789ABCDEF, 0x7FFFFFFFFFFFFFFF):
        for i, j in ((0, 1), (0, 5), (1, 2), (3, 7)):
            words = torch.arange(8, dtype=torch.int64)
            words[i] = as_i64(w1)
            words[j] = as_i64(w1 ^ (1 << 63))
            swapped = words.clone()
            swapped[i], swapped[j] = words[j].clone(), words[i].clone()
            assert _wire_digest(swapped.view(torch.uint8)) != _wire_digest(words.view(torch.uint8)), (
                f"bit-63 pair swap at ({i},{j}) with w1={w1:#x} went undetected"
            )


def test_complementary_word_corruption_is_detected():
    # Deterministic regression: with an arithmetic (sign-extending) shift in
    # the avalanche, v ^ (v >> 31) always clears bit 63, so a word w and its
    # position-twinned complement (-1 - w - 2 * K * p at 1-based position p,
    # whose mixed values are bitwise complements) collide. The masked logical
    # shift keeps the avalanche bijective; keep this construction pinned.
    K = _POS_MIX

    def as_i64(u: int) -> int:
        return u - (1 << 64) if u >= (1 << 63) else u

    for w in (0, 1, 0x123456789ABCDEF, 0x7FFFFFFFFFFFFFFF):
        for p0 in (0, 1, 3, 7):
            words = torch.arange(10, dtype=torch.int64)
            words[p0] = as_i64(w)
            base = _wire_digest(words.view(torch.uint8))
            words[p0] = as_i64((-1 - w - 2 * K * (p0 + 1)) & _MASK64)
            assert _wire_digest(words.view(torch.uint8)) != base, (
                f"complementary corruption at position {p0} with w={w:#x} went undetected"
            )


def test_topbit_toggle_pair_swaps_are_detected():
    # Deterministic regression: a single-stage avalanche passes top-bit
    # toggles through the odd multiply unchanged, so swapping two words whose
    # position-mixed values differ by exactly the toggle preimage (bit 63:
    # 0x8000000100000002; bit 62: 0x4000000080000001, which our first odd
    # constant also passes because it is 1 mod 4) cancels in the XOR fold.
    # These exact pairs collided before the two-stage avalanche; keep them
    # pinned regardless of future mixing.
    def as_i64(u: int) -> int:
        return u - (1 << 64) if u >= (1 << 63) else u

    pinned = [
        ((0, 1), 0x8000000080000001, 0xFFFFFFFF7FFFFFFF),
        ((1, 2), 0x00000000FFFFFFFE, 0x8000000000000000),
        ((1, 2), 0x0000000100000000, 0x7FFFFFFFFFFFFFFE),
        ((0, 64), 0x000000007FFFFFFF, 0x4000000000000000),
        ((3, 7), 0x9E3779B9FF4A7C14, 0xDE3779B97F4A7C15),
        ((5, 69), 0x0000000000000001, 0x80000000FFFFFFFF),
    ]
    for (i, j), w, w2 in pinned:
        words = [0x1111111111111111] * 80
        words[i] = w
        words[j] = w2
        swapped = list(words)
        swapped[i], swapped[j] = w2, w
        base = _wire_digest(torch.tensor([as_i64(x) for x in words], dtype=torch.int64).view(torch.uint8))
        other = _wire_digest(torch.tensor([as_i64(x) for x in swapped], dtype=torch.int64).view(torch.uint8))
        assert base != other, f"toggle-family swap at ({i},{j}) {w:#x} <-> {w2:#x} went undetected"


def test_two_bit_shift_response_pair_swaps_are_detected():
    # Deterministic regression: these two-bit difference pairs cancel the XOR
    # fold alone (they fold into single-bit differences through the avalanche's
    # xor-shift stages); the sum fold detects them. Collided before the dual
    # fold; keep pinned regardless of future mixing.
    def as_i64(u: int) -> int:
        return u - (1 << 64) if u >= (1 << 63) else u

    pinned = [
        ((1, 17), 0x3C92AD063D838AB2, 0xBC92AD067D838AB2),
        ((1, 1000), 0x6E0EC2DA019187AD, 0xEE0EC2DA419187AD),
        ((3, 17), 0x20AD07FD7C99BCF9, 0xA0AD07FD3C99BCF9),
        ((3, 33), 0x7F8E25C075D83D25, 0xFF8E25C035D83D25),
        ((3, 1000), 0x5CA055D303876CD5, 0xDCA055D343876CD5),
        ((4, 8), 0x7BA1CDBE5ED0B285, 0xFBA1CDBE1ED0B285),
        ((5, 17), 0x936B87A02E3992A6, 0xD36B87A00E3992A6),
        ((5, 33), 0x8BFA4052DE8BA43F, 0x9BFA4052D68BA43F),
    ]
    for (i, j), w, w2 in pinned:
        words = [0x1111111111111111] * (j + 10)
        words[i - 1], words[j - 1] = w, w2
        base = _wire_digest(torch.tensor([as_i64(x) for x in words], dtype=torch.int64).view(torch.uint8))
        swapped = list(words)
        swapped[i - 1], swapped[j - 1] = w2, w
        other = _wire_digest(torch.tensor([as_i64(x) for x in swapped], dtype=torch.int64).view(torch.uint8))
        assert base != other, f"two-bit family swap at ({i},{j}) {w:#x} <-> {w2:#x} went undetected"


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
