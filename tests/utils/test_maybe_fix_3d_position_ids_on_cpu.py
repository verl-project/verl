# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CPU regression tests for 3D nested ``position_ids`` handling."""

import pytest
import torch
from tensordict import TensorDict

from verl.utils.tensordict_utils import (
    chunk_tensordict,
    concat_tensordict,
    index_select_tensor_dict,
    maybe_fix_3d_position_ids,
)


def _make_ids(seq_lens):
    return torch.nested.as_nested_tensor(
        [torch.arange(seq_len, dtype=torch.long) for seq_len in seq_lens], layout=torch.jagged
    )


def _make_3d_position_ids(seq_lens, mrope=4):
    samples = [torch.arange(mrope * seq_len, dtype=torch.long).view(mrope, seq_len) for seq_len in seq_lens]
    values = torch.cat(samples, dim=-1)
    offsets = torch.zeros(len(seq_lens) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(torch.tensor(seq_lens, dtype=torch.long), dim=0)
    position_ids = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
    position_ids._ragged_idx = 2
    return position_ids


def _assert_seq_ragged_position_ids(position_ids, seq_lens, mrope=4):
    assert position_ids.is_nested
    assert position_ids._ragged_idx == 2
    assert position_ids.offsets().diff().tolist() == seq_lens
    assert position_ids.values().shape == (mrope, sum(seq_lens))


def test_noop_on_2d_position_ids():
    seq_lens = [3, 5]
    td = TensorDict(
        {"input_ids": _make_ids(seq_lens), "position_ids": _make_ids(seq_lens)},
        batch_size=[len(seq_lens)],
    )
    before = td["position_ids"]
    maybe_fix_3d_position_ids(td)
    # 2D nested tensor is unchanged.
    assert td["position_ids"] is before
    assert td["position_ids"].dim() == 2


def test_fast_path_flips_ragged_idx_when_storage_is_seq_major():
    seq_lens = [4, 7]
    # Multi-sample case: as_nested_tensor picks ragged_idx=2 automatically.
    samples = [torch.arange(4 * s, dtype=torch.long).view(4, s) for s in seq_lens]
    nt = torch.nested.as_nested_tensor(samples, layout=torch.jagged)
    assert nt._ragged_idx == 2  # sanity: upstream is fine
    # Simulate the "label got flipped back to 1 during pickle round-trip".
    nt._ragged_idx = 1
    td = TensorDict({"input_ids": _make_ids(seq_lens), "position_ids": nt}, batch_size=[len(seq_lens)])
    maybe_fix_3d_position_ids(td)
    fixed = td["position_ids"]
    assert fixed._ragged_idx == 2
    # Storage layout either (total_seq, mrope) or (mrope, total_seq): both
    # are valid as long as sum(seq_lens) appears somewhere in values.shape.
    assert sum(seq_lens) in fixed.values().shape


def test_single_sample_mislabeled_ragged_idx_is_fixed_by_label_flip():
    """Single-sample 3D ``position_ids`` should keep the seq axis ragged.

    ``as_nested_tensor([dense_3d], layout=jagged)`` picks ``_ragged_idx=1``
    for a single-sample input, even though the storage is seq-major. The
    downstream ``.values().unsqueeze(1)`` chain works correctly as long as
    the label gets flipped back to 2.
    """
    seq_len = 1283
    mrope = 4
    sample = torch.arange(mrope * seq_len, dtype=torch.long).view(mrope, seq_len)
    bad = torch.nested.as_nested_tensor([sample], layout=torch.jagged)
    assert bad._ragged_idx == 1
    assert bad.values().shape == (mrope, seq_len)

    td = TensorDict(
        {"input_ids": _make_ids([seq_len]), "position_ids": bad},
        batch_size=[1],
    )
    maybe_fix_3d_position_ids(td)
    fixed = td["position_ids"]
    assert fixed._ragged_idx == 2
    # Simulate the downstream reshape in FSDPEngine.prepare_model_inputs.
    rmpad = fixed.values().unsqueeze(1)
    assert rmpad.shape == (mrope, 1, seq_len)
    # Content preserved: rmpad[:, 0, :] == sample.
    assert torch.equal(rmpad[:, 0, :], sample)


def test_rebuild_raises_when_seq_data_lost():
    """If storage lost the seq dim (e.g. pickle dropped values), raise loudly.

    If ``_values`` no longer contains the true seq length from ``input_ids``,
    there is no way to reconstruct the original data.
    """
    fake_mrope_axes = 4
    fake_seq = 32
    true_seq = 1283
    # Build a nested tensor whose per-sample shape (4, 32) doesn't match the
    # sibling input_ids seq_len=1283.
    bad_values = torch.arange(fake_mrope_axes * fake_seq, dtype=torch.float32).view(fake_mrope_axes, fake_seq)
    broken = torch.nested.as_nested_tensor([bad_values], layout=torch.jagged)
    td = TensorDict(
        {"input_ids": _make_ids([true_seq]), "position_ids": broken},
        batch_size=[1],
    )
    with pytest.raises(RuntimeError, match="Invalid 3D nested position_ids storage"):
        maybe_fix_3d_position_ids(td)


def test_index_select_keeps_equal_length_3d_position_ids_seq_ragged():
    seq_lens = [1283] * 8
    td = TensorDict(
        {"input_ids": _make_ids(seq_lens), "position_ids": _make_3d_position_ids(seq_lens)},
        batch_size=[len(seq_lens)],
    )

    selected = index_select_tensor_dict(td, list(range(len(seq_lens))))

    _assert_seq_ragged_position_ids(selected["position_ids"], seq_lens)
    maybe_fix_3d_position_ids(selected)


def test_chunk_tensordict_keeps_equal_length_3d_position_ids_seq_ragged():
    seq_lens = [1283] * 16
    td = TensorDict(
        {"input_ids": _make_ids(seq_lens), "position_ids": _make_3d_position_ids(seq_lens)},
        batch_size=[len(seq_lens)],
    )

    chunks = chunk_tensordict(td, chunks=2)

    assert len(chunks) == 2
    for chunk in chunks:
        _assert_seq_ragged_position_ids(chunk["position_ids"], [1283] * 8)
        maybe_fix_3d_position_ids(chunk)


def test_concat_tensordict_keeps_equal_length_3d_position_ids_seq_ragged():
    seq_lens = [1283] * 4
    td_a = TensorDict(
        {"input_ids": _make_ids(seq_lens), "position_ids": _make_3d_position_ids(seq_lens)},
        batch_size=[len(seq_lens)],
    )
    td_b = TensorDict(
        {"input_ids": _make_ids(seq_lens), "position_ids": _make_3d_position_ids(seq_lens)},
        batch_size=[len(seq_lens)],
    )

    merged = concat_tensordict([td_a, td_b])

    _assert_seq_ragged_position_ids(merged["position_ids"], seq_lens + seq_lens)
    maybe_fix_3d_position_ids(merged)
