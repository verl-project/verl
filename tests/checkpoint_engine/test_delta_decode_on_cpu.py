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
"""Unit tests for the device-agnostic DeltaFlush wire decoder.

Covers :func:`verl.checkpoint_engine.delta_sync.decode_delta_wire` (shared by
the vLLM CUDA consumer, the NPU HCCL receiver and the correctness-test mock
receiver). No vLLM / ray / accelerator required; the decoder is a pure
function of the wire payload and runs on CPU.
"""

import json

import pytest
import torch

from verl.checkpoint_engine.delta_sync import DeltaParam, checksum
from verl.checkpoint_engine.delta_sync.decode import decode_delta_wire


def _make_named(dtype=torch.bfloat16) -> list[tuple[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        ("layer.0.weight", torch.randn(64, 32, dtype=dtype)),
        ("layer.1.weight", torch.randn(32, 16, dtype=dtype)),
    ]


def _sparse_indices_flush(old_named, new_named):
    """Assemble one indices-encoding flush from a bytewise old/new diff --
    the same layout the sharded engines' ``_assemble_flush`` produces."""
    params, idx_pieces, val_pieces = [], [], []
    pos_off = val_off = 0
    for (name, old), (_, new) in zip(old_named, new_named, strict=True):
        fo, fn = old.reshape(-1), new.reshape(-1)
        changed = (fo.view(torch.uint8).view(fo.numel(), -1) != fn.view(torch.uint8).view(fn.numel(), -1)).any(dim=-1)
        idx = changed.nonzero(as_tuple=False).view(-1)
        # Keep zero-length manifest entries, like the real wire: unchanged params
        # still appear with pos_start == pos_end / val_start == val_end.
        nnz = int(idx.numel())
        idx_pieces.append(idx.to(torch.int32))
        val_pieces.append(fn[idx])
        params.append(
            DeltaParam(
                name=name,
                dtype=str(new.dtype).replace("torch.", ""),
                shape=list(new.shape),
                pos_start=pos_off,
                pos_end=pos_off + nnz * 4,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + nnz,
            )
        )
        pos_off += nnz * 4
        val_off += nnz
    positions = torch.cat(idx_pieces).contiguous().view(torch.uint8)
    values = torch.cat(val_pieces)
    return params, positions, values


def _named_tensors(params, positions, values, encoding="indices"):
    spec = {
        "encoding": encoding,
        "params": [vars(p) for p in params],
        "checksum": int(checksum(positions, values)),
    }
    spec_t = torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)
    out = [("__delta_spec__", spec_t), ("__values__", values.clone())]
    if positions.numel():
        out.insert(1, ("__positions__", positions.clone()))
    return out


def test_decode_indices_round_trip():
    named = _make_named()
    new_named = []
    for name, t in named:
        new = t.clone()
        flat = new.view(-1)
        idx = torch.tensor([1, 17, 200, 511], dtype=torch.int64) % flat.numel()
        flat[idx] = flat[idx] + 0.5
        new_named.append((name, new))

    params, positions, values = _sparse_indices_flush(named, new_named)
    encoding, patches = decode_delta_wire(_named_tensors(params, positions, values))

    assert encoding == "indices"
    assert [p.name for p in patches] == [n for n, _ in named]
    for patch, (_, expected) in zip(patches, new_named, strict=True):
        assert patch.shape == tuple(expected.shape)
        assert patch.dtype == expected.dtype
        assert patch.positions is not None
        # The decoded (positions, values) pairs must reconstruct exactly the bytewise diff.
        rebuilt = expected.reshape(-1).clone()
        baseline = dict(named)[patch.name].reshape(-1)
        changed = (baseline.view(torch.uint8).view(-1, 2) != expected.reshape(-1).view(torch.uint8).view(-1, 2)).any(-1)
        assert torch.equal(patch.positions.to(torch.int64), changed.nonzero(as_tuple=False).view(-1))
        assert torch.equal(patch.values.view(torch.int16), rebuilt[changed].view(torch.int16))


def test_decode_dense_round_trip():
    named = _make_named()
    params, val_off, values = [], 0, []
    for name, t in named:
        flat = t.contiguous().view(-1)
        params.append(
            DeltaParam(
                name=name,
                dtype=str(t.dtype).replace("torch.", ""),
                shape=list(t.shape),
                pos_start=0,
                pos_end=0,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + flat.numel(),
            )
        )
        val_off += flat.numel()
        values.append(flat)
    values = torch.cat(values)

    empty_pos = torch.empty(0, dtype=torch.uint8)
    encoding, patches = decode_delta_wire(_named_tensors(params, empty_pos, values, encoding="dense"))

    assert encoding == "dense"
    for patch, (name, expected) in zip(patches, named, strict=True):
        assert patch.name == name
        assert patch.shape == tuple(expected.shape)
        assert patch.positions is None
        assert torch.equal(patch.values.view(torch.int16), expected.reshape(-1).view(torch.int16))


def test_decode_indices_with_empty_param_keeps_manifest_entry():
    """A param whose delta is empty still appears in the manifest with zero-length slices."""
    named = _make_named()
    new_named = [(named[0][0], named[0][1].clone()), (named[1][0], named[1][1].clone())]
    new_named[0][1].view(-1)[5] += 1.0  # only the first param changes

    params, positions, values = _sparse_indices_flush(named, new_named)
    encoding, patches = decode_delta_wire(_named_tensors(params, positions, values))
    assert encoding == "indices"
    assert [p.name for p in patches] == [n for n, _ in named]
    assert patches[0].values.numel() == patches[0].positions.numel() == 1
    assert patches[1].values.numel() == patches[1].positions.numel() == 0


def test_decode_checksum_mismatch_raises():
    named = _make_named()
    new_named = [(n, t + 0.5) for n, t in named]
    named_tensors = _named_tensors(*_sparse_indices_flush(named, new_named))
    named_tensors[2][1].view(torch.uint8)[0] ^= 0xFF  # corrupt one value byte
    with pytest.raises(RuntimeError, match="checksum"):
        decode_delta_wire(named_tensors)


def test_decode_missing_spec_raises():
    named = _make_named()
    named_tensors = _named_tensors(*_sparse_indices_flush(named, [(n, t + 0.5) for n, t in named]))
    named_tensors = [(n, t) for n, t in named_tensors if n != "__delta_spec__"]
    with pytest.raises(ValueError, match="__delta_spec__"):
        decode_delta_wire(named_tensors)


def test_decode_indices_requires_positions():
    named = _make_named()
    params, positions, values = _sparse_indices_flush(named, [(n, t + 0.5) for n, t in named])
    named_tensors = _named_tensors(params, positions, values)
    named_tensors = [(n, t) for n, t in named_tensors if n != "__positions__"]
    with pytest.raises(ValueError, match="__positions__"):
        decode_delta_wire(named_tensors)


def test_decode_dense_rejects_positions():
    named = _make_named()
    values = torch.cat([t.reshape(-1) for _, t in named])
    params = [
        DeltaParam(
            name=name,
            dtype=str(t.dtype).replace("torch.", ""),
            shape=list(t.shape),
            pos_start=0,
            pos_end=0,
            pos_width=4,
            val_start=off,
            val_end=off + t.numel(),
        )
        for (name, t), off in zip(named, [0, named[0][1].numel()], strict=True)
    ]
    bogus_pos = torch.zeros(8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="must not contain positions"):
        decode_delta_wire(_named_tensors(params, bogus_pos, values, encoding="dense"))


def test_decode_unsupported_encoding_raises():
    named = _make_named()
    params, positions, values = _sparse_indices_flush(named, [(n, t + 0.5) for n, t in named])
    with pytest.raises(ValueError, match="encoding"):
        decode_delta_wire(_named_tensors(params, positions, values, encoding="rle"))


def test_decode_inconsistent_slice_lengths_raise():
    named = _make_named()
    new_named = [(n, t + 0.5) for n, t in named]
    params, positions, values = _sparse_indices_flush(named, new_named)
    # Truncate one param's position slice: pos bytes no longer match 4 * nnz values.
    params[0].pos_end -= 4
    with pytest.raises(ValueError, match="inconsistent lengths"):
        decode_delta_wire(_named_tensors(params, positions, values))


if __name__ == "__main__":
    pytest.main([__file__])
