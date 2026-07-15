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
"""Bit-identity round-trip tests for the delta sync sender / decoder.

These exercise the framework-agnostic core (``DeltaState`` + ``encode_chunk``
+ ``decode_chunk``) without requiring a delta-capable SGLang build. They
verify that:

* The first call seeds the snapshot and emits no flushes.
* The diff is dtype-agnostic and lossless: every changed element is sent
  with the exact trainer-side bytes, while unchanged elements are not in
  the payload at all.
* Both encodings (indices, gap-deltas) reconstruct the changed values to
  bit-identity at the receiver.
"""

from __future__ import annotations

import pytest
import torch

from verl.checkpoint_engine.delta_sync import (
    DeltaState,
    iter_delta_flushes,
)
from verl.checkpoint_engine.delta_sync.encode import decode_chunk


def _weights_gen(named: list[tuple[str, torch.Tensor]]):
    for n, t in named:
        yield n, t


def _round_trip_one_flush(flush, expected_changed: dict[str, torch.Tensor]) -> None:
    """Decode ``flush`` and verify changed positions match the trainer's bytes."""
    decoded = decode_chunk(
        flush.encoding,
        flush.positions_cpu.numpy().tobytes(),
        flush.values_gpu,
        flush.params,
    )
    for name, expected in expected_changed.items():
        assert name in decoded, f"{name} missing from receiver payload"
        recv = decoded[name]
        mask = ~torch.isnan(recv)
        # NaN means "not in payload"; for changed positions we must reconstruct
        # the exact trainer-side bytes.
        assert torch.equal(recv[mask].view(expected.dtype), expected[mask].view(expected.dtype))


def _make_named(dtype=torch.bfloat16) -> list[tuple[str, torch.Tensor]]:
    torch.manual_seed(0)
    return [
        ("layer.0.weight", torch.randn(64, 32, dtype=dtype)),
        ("layer.1.weight", torch.randn(32, 16, dtype=dtype)),
    ]


@pytest.mark.parametrize("encoding", ["indices", "deltas"])
def test_first_call_seeds_no_flushes(encoding: str):
    state = DeltaState()
    named = _make_named()
    flushes = list(
        iter_delta_flushes(
            _weights_gen(named),
            state,
            encoding=encoding,
            bucket_bytes=1 << 20,
        )
    )
    assert flushes == []
    assert state.seeded


@pytest.mark.parametrize("encoding", ["indices", "deltas"])
def test_round_trip_bit_identical(encoding: str):
    state = DeltaState()
    named = _make_named()

    # Seed.
    list(
        iter_delta_flushes(
            _weights_gen(named),
            state,
            encoding=encoding,
            bucket_bytes=1 << 20,
        )
    )

    # Mutate a sparse set of positions in each tensor.
    changed_truth: dict[str, torch.Tensor] = {}
    new_named = []
    for name, t in named:
        new = t.clone()
        flat = new.view(-1)
        idx = torch.tensor([1, 17, 200, 511], dtype=torch.int64) % flat.numel()
        flat[idx] = flat[idx] + 0.5
        new_named.append((name, new))
        changed_truth[name] = new

    # Real sender path.
    flushes = list(
        iter_delta_flushes(
            _weights_gen(new_named),
            state,
            encoding=encoding,
            bucket_bytes=1 << 20,
        )
    )
    assert flushes, "expected at least one flush after a real change"

    # Concatenate decoded views; later flushes overwrite earlier ones for the
    # same parameter (matches the receiver's apply-then-overwrite semantics).
    for flush in flushes:
        _round_trip_one_flush(flush, changed_truth)


def test_no_change_emits_no_flush():
    state = DeltaState()
    named = _make_named()
    list(
        iter_delta_flushes(
            _weights_gen(named),
            state,
            encoding="indices",
            bucket_bytes=1 << 20,
        )
    )
    # Identical second pass.
    flushes = list(
        iter_delta_flushes(
            _weights_gen(named),
            state,
            encoding="indices",
            bucket_bytes=1 << 20,
        )
    )
    assert flushes == []


def test_dtype_agnostic_diff_fp32():
    state = DeltaState()
    named = _make_named(dtype=torch.float32)
    list(
        iter_delta_flushes(
            _weights_gen(named),
            state,
            encoding="indices",
            bucket_bytes=1 << 20,
        )
    )
    new_named = []
    truth: dict[str, torch.Tensor] = {}
    for name, t in named:
        new = t.clone()
        new.view(-1)[3] += 1e-3
        new_named.append((name, new))
        truth[name] = new
    flushes = list(
        iter_delta_flushes(
            _weights_gen(new_named),
            state,
            encoding="indices",
            bucket_bytes=1 << 20,
        )
    )
    assert flushes
    for flush in flushes:
        _round_trip_one_flush(flush, truth)


def _apply_flushes_to_mirror(mirror: dict[str, torch.Tensor], flushes) -> None:
    """Reproduce DeltaCheckpointEngine.receive_weights' mirror-combine in-process.

    Decodes each flush and overwrites only the changed (non-NaN) positions into
    the running full-weight mirror -- exactly what the rollout worker does before
    handing full tensors to ``server_adapter.update_weights``. No GPU / NCCL /
    SGLang needed: the transport only moves bytes, so the lossless guarantee is
    fully exercised by decode + combine.
    """
    for flush in flushes:
        decoded = decode_chunk(
            flush.encoding,
            flush.positions_cpu.numpy().tobytes(),
            flush.values_gpu,
            flush.params,
        )
        for name, recv in decoded.items():
            mask = ~torch.isnan(recv)
            mirror[name][mask] = recv[mask]


@pytest.mark.parametrize("encoding", ["indices", "deltas"])
def test_delta_result_equals_full_sync(encoding: str):
    """The weights a rollout ends up with via delta sync must be byte-identical
    to what the old full-weight path delivers (i.e. the trainer's current W).

    full path  -> rollout receives every tensor verbatim  == W_new
    delta path -> rollout starts from W_old mirror, applies only the changed
                  positions                                  == W_old + delta
    Assert the two are bit-equal for every parameter, across multiple steps.
    """
    state = DeltaState()
    named = _make_named()

    # Seed the trainer snapshot from W0 and initialize the rollout mirror to the
    # same W0 (both sides loaded the identical init checkpoint).
    list(iter_delta_flushes(_weights_gen(named), state, encoding=encoding, bucket_bytes=1 << 20))
    mirror = {n: t.clone() for n, t in named}

    cur_named = named
    for step in range(3):
        # Trainer takes a step: mutate a sparse, step-dependent set of positions.
        new_named = []
        for i, (name, t) in enumerate(cur_named):
            new = t.clone()
            flat = new.view(-1)
            idx = torch.tensor([1 + step, 17, 200 + 5 * i, 511], dtype=torch.int64) % flat.numel()
            flat[idx] = flat[idx] + 0.25 * (step + 1)
            new_named.append((name, new))
        cur_named = new_named

        # full path reference: the rollout would just receive W_new in full.
        full_result = {n: t.clone() for n, t in new_named}

        # delta path: sender diffs vs snapshot, receiver applies onto its mirror.
        flushes = list(
            iter_delta_flushes(_weights_gen(new_named), state, encoding=encoding, bucket_bytes=1 << 20)
        )
        _apply_flushes_to_mirror(mirror, flushes)

        for name, full in full_result.items():
            assert torch.equal(mirror[name].view(torch.uint8), full.view(torch.uint8)), (
                f"delta != full at step {step} for {name}"
            )
