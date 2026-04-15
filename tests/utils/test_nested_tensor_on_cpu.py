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

import pytest
import torch
from tensordict import TensorDict

from verl.utils.nested_tensor import (
    MaskNestingSpec,
    mask_to_rle,
    rle_to_mask,
    unnest_tensor_by_rle,
)


def _unbind_nested(nt: torch.Tensor) -> list[torch.Tensor]:
    """Unbind a nested tensor into a list of regular tensors."""
    return list(nt.unbind())


def _assert_rle_roundtrip(mask: torch.Tensor):
    """Assert mask_to_rle -> rle_to_mask roundtrip, allowing trailing-False truncation.

    ``rle_to_mask`` without ``shape`` infers ``seq_len`` as
    ``max(offset + length)``, so trailing all-False columns from the original
    mask are not preserved. This helper accounts for that by only comparing up
    to the recovered width and verifying that any truncated columns were indeed
    all-False.
    """
    offsets, lengths = mask_to_rle(mask)
    recovered = rle_to_mask(offsets, lengths)
    rec_cols = recovered.shape[1]
    orig_cols = mask.shape[1]
    assert torch.equal(recovered, mask[:, :rec_cols])
    if rec_cols < orig_cols:
        assert not mask[:, rec_cols:].any(), "Trailing columns contain True values but were lost"


# ---------------------------------------------------------------------------
# mask_to_rle
# ---------------------------------------------------------------------------


class TestMaskToRle:
    def test_basic_example(self):
        """Docstring example: mixed segments across two rows."""
        mask = torch.tensor([[False, False, True, True, False], [False, True, True, False, True]])
        offsets, lengths = mask_to_rle(mask)

        off = _unbind_nested(offsets)
        ln = _unbind_nested(lengths)

        assert torch.equal(off[0], torch.tensor([2]))
        assert torch.equal(off[1], torch.tensor([1, 4]))
        assert torch.equal(ln[0], torch.tensor([2]))
        assert torch.equal(ln[1], torch.tensor([2, 1]))

    def test_all_true_row(self):
        """Entirely True row produces a single full-width segment."""
        mask = torch.ones(1, 6, dtype=torch.bool)
        offsets, lengths = mask_to_rle(mask)

        assert torch.equal(_unbind_nested(offsets)[0], torch.tensor([0]))
        assert torch.equal(_unbind_nested(lengths)[0], torch.tensor([6]))

    def test_all_false_row(self):
        """Entirely False row produces zero segments."""
        mask = torch.zeros(1, 4, dtype=torch.bool)
        offsets, lengths = mask_to_rle(mask)

        assert _unbind_nested(offsets)[0].numel() == 0
        assert _unbind_nested(lengths)[0].numel() == 0

    def test_alternating(self):
        """Alternating True/False gives single-element segments."""
        mask = torch.tensor([[True, False, True, False, True, False]])
        offsets, lengths = mask_to_rle(mask)

        assert torch.equal(_unbind_nested(offsets)[0], torch.tensor([0, 2, 4]))
        assert torch.equal(_unbind_nested(lengths)[0], torch.tensor([1, 1, 1]))

    def test_edges_true(self):
        """Segments touching both edges of the row."""
        mask = torch.tensor([[True, True, False, False, True]])
        offsets, lengths = mask_to_rle(mask)

        assert torch.equal(_unbind_nested(offsets)[0], torch.tensor([0, 4]))
        assert torch.equal(_unbind_nested(lengths)[0], torch.tensor([2, 1]))

    def test_multiple_rows_mixed(self):
        """Rows with varying segment counts including an empty row."""
        mask = torch.tensor(
            [
                [True, True, False, True],
                [False, False, False, False],
                [True, False, True, True],
            ]
        )
        offsets, lengths = mask_to_rle(mask)
        off = _unbind_nested(offsets)
        ln = _unbind_nested(lengths)

        assert torch.equal(off[0], torch.tensor([0, 3]))
        assert torch.equal(ln[0], torch.tensor([2, 1]))
        assert off[1].numel() == 0
        assert ln[1].numel() == 0
        assert torch.equal(off[2], torch.tensor([0, 2]))
        assert torch.equal(ln[2], torch.tensor([1, 2]))

    def test_rejects_1d(self):
        with pytest.raises(AssertionError, match="Expected 2D mask"):
            mask_to_rle(torch.tensor([True, False]))

    def test_rejects_3d(self):
        with pytest.raises(AssertionError, match="Expected 2D mask"):
            mask_to_rle(torch.zeros(2, 3, 4, dtype=torch.bool))


# ---------------------------------------------------------------------------
# rle_to_mask
# ---------------------------------------------------------------------------


class TestRleToMask:
    def test_roundtrip_basic(self):
        """mask_to_rle -> rle_to_mask should recover the original mask."""
        mask = torch.tensor([[False, False, True, True, False], [False, True, True, False, True]])
        _assert_rle_roundtrip(mask)

    def test_roundtrip_all_true(self):
        mask = torch.ones(1, 5, dtype=torch.bool)
        _assert_rle_roundtrip(mask)

    def test_roundtrip_all_false(self):
        """All-false mask: recovered mask has seq_len=0 without explicit shape."""
        mask = torch.zeros(1, 3, dtype=torch.bool)
        offsets, lengths = mask_to_rle(mask)
        recovered = rle_to_mask(offsets, lengths)
        assert recovered.shape == (1, 0)

    def test_roundtrip_multiple_rows(self):
        mask = torch.tensor(
            [
                [True, True, False, True],
                [False, False, False, False],
                [True, False, True, True],
            ]
        )
        _assert_rle_roundtrip(mask)

    def test_roundtrip_alternating(self):
        mask = torch.tensor([[True, False, True, False, True, False]])
        _assert_rle_roundtrip(mask)

    def test_roundtrip_last_col_true(self):
        """When the last column is True, recovered seq_len exactly matches."""
        mask = torch.tensor([[True, False, False, True], [False, True, True, True]])
        offsets, lengths = mask_to_rle(mask)
        recovered = rle_to_mask(offsets, lengths)
        assert torch.equal(recovered, mask)

    def test_roundtrip_random(self):
        """Fuzz-style: random boolean masks should roundtrip correctly."""
        gen = torch.Generator().manual_seed(42)
        for _ in range(10):
            rows = torch.randint(1, 6, (1,), generator=gen).item()
            cols = torch.randint(1, 20, (1,), generator=gen).item()
            mask = torch.randint(0, 2, (rows, cols), dtype=torch.bool, generator=gen)
            _assert_rle_roundtrip(mask)

    # -- shape parameter tests -----------------------------------------------

    def test_shape_exact_recovery(self):
        """Passing the original shape recovers trailing-False columns."""
        mask = torch.tensor([[True, False, True, False, True, False]])
        offsets, lengths = mask_to_rle(mask)
        recovered = rle_to_mask(offsets, lengths, shape=mask.shape)
        assert torch.equal(recovered, mask)

    def test_shape_all_false(self):
        """All-false mask is fully recovered when shape is given."""
        mask = torch.zeros(2, 5, dtype=torch.bool)
        offsets, lengths = mask_to_rle(mask)
        recovered = rle_to_mask(offsets, lengths, shape=mask.shape)
        assert torch.equal(recovered, mask)

    def test_shape_roundtrip_random(self):
        """Fuzz-style: with shape, roundtrip always recovers the exact mask."""
        gen = torch.Generator().manual_seed(77)
        for _ in range(10):
            rows = torch.randint(1, 6, (1,), generator=gen).item()
            cols = torch.randint(1, 20, (1,), generator=gen).item()
            mask = torch.randint(0, 2, (rows, cols), dtype=torch.bool, generator=gen)
            offsets, lengths = mask_to_rle(mask)
            recovered = rle_to_mask(offsets, lengths, shape=mask.shape)
            assert torch.equal(recovered, mask), f"shape roundtrip failed for {mask.shape}"

    def test_shape_batch_mismatch_raises(self):
        """shape[0] != batch_size from offsets should raise."""
        mask = torch.tensor([[True, False]])
        offsets, lengths = mask_to_rle(mask)
        with pytest.raises(AssertionError, match="does not match batch_size"):
            rle_to_mask(offsets, lengths, shape=(3, 2))

    def test_shape_as_torch_size(self):
        """torch.Size should be accepted as shape."""
        mask = torch.tensor([[True, True, False, False]])
        offsets, lengths = mask_to_rle(mask)
        recovered = rle_to_mask(offsets, lengths, shape=torch.Size([1, 4]))
        assert torch.equal(recovered, mask)


# ---------------------------------------------------------------------------
# unnest_tensor_by_rle
# ---------------------------------------------------------------------------


class TestUnnestTensorByRle:
    def _roundtrip(self, tensor: torch.Tensor, mask: torch.Tensor, pad_value: float = 0.0):
        """Helper: masked_select -> unnest_tensor_by_rle should recover masked positions.

        The recovered tensor may be narrower than the original when trailing
        columns are all-False, because ``unnest_tensor_by_rle`` infers
        ``seq_len`` from ``max(offset + length)``.
        """
        offsets, lengths = mask_to_rle(mask)
        nested = torch.nested.masked_select(tensor, mask)
        recovered = unnest_tensor_by_rle(nested, offsets, lengths, pad_value=pad_value)

        rec_cols = recovered.shape[1]
        orig_cols = mask.shape[1]
        trimmed_mask = mask[:, :rec_cols]

        # Masked positions within recovered range must match
        assert torch.equal(recovered[trimmed_mask], tensor[:, :rec_cols][trimmed_mask])
        # Non-masked positions must be filled with pad_value
        assert (recovered[~trimmed_mask] == pad_value).all()
        # Anything beyond recovered width must be all-False in original mask
        if rec_cols < orig_cols:
            assert not mask[:, rec_cols:].any()

    def test_basic_1d_values(self):
        """Basic 2D tensor with a simple mask."""
        tensor = torch.arange(10).reshape(2, 5).float()
        mask = torch.tensor([[False, False, True, True, False], [False, True, True, False, True]])
        self._roundtrip(tensor, mask)

    def test_all_true(self):
        """All-true mask should recover the full tensor."""
        tensor = torch.arange(8).reshape(2, 4).float()
        mask = torch.ones(2, 4, dtype=torch.bool)
        self._roundtrip(tensor, mask)

    def test_all_false(self):
        """All-false mask: output should be entirely pad_value."""
        tensor = torch.arange(6).reshape(2, 3).float()
        mask = torch.zeros(2, 3, dtype=torch.bool)

        offsets, lengths = mask_to_rle(mask)
        nested = torch.nested.masked_select(tensor, mask)
        recovered = unnest_tensor_by_rle(nested, offsets, lengths, pad_value=-1.0)

        # seq_len should be 0 when no segments exist, so output is (2, 0)
        assert recovered.shape[0] == 2
        assert recovered.shape[1] == 0

    def test_custom_pad_value(self):
        """Pad value should fill non-masked positions."""
        tensor = torch.tensor([[10.0, 20.0, 30.0]])
        mask = torch.tensor([[True, False, True]])
        pad_value = -999.0
        self._roundtrip(tensor, mask, pad_value=pad_value)

    def test_mixed_empty_and_nonempty_rows(self):
        """Rows with zero segments mixed with rows that have segments."""
        tensor = torch.randn(3, 4)
        mask = torch.tensor(
            [
                [True, True, False, False],
                [False, False, False, False],
                [False, True, True, False],
            ]
        )
        self._roundtrip(tensor, mask)

    def test_exact_recovery_when_last_col_true(self):
        """When the last column has True in some row, recovered width matches exactly."""
        tensor = torch.arange(12).reshape(3, 4).float()
        mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, False, False],
                [True, True, True, True],
            ]
        )
        offsets, lengths = mask_to_rle(mask)
        nested = torch.nested.masked_select(tensor, mask)
        recovered = unnest_tensor_by_rle(nested, offsets, lengths)
        # Exact shape match since last column has True
        assert recovered.shape == tensor.shape
        assert torch.equal(recovered[mask], tensor[mask])
        assert (recovered[~mask] == 0).all()

    def test_random_roundtrip(self):
        """Fuzz-style: random tensors and masks should roundtrip correctly."""
        gen = torch.Generator().manual_seed(123)
        for _ in range(10):
            rows = torch.randint(1, 6, (1,), generator=gen).item()
            cols = torch.randint(1, 15, (1,), generator=gen).item()
            tensor = torch.randn(rows, cols, generator=gen)
            mask = torch.randint(0, 2, (rows, cols), dtype=torch.bool, generator=gen)
            if mask.any():
                self._roundtrip(tensor, mask)


# ---------------------------------------------------------------------------
# MaskNestingSpec
# ---------------------------------------------------------------------------


class TestMaskNestingSpec:
    def _make_td(self, batch_size: int, seq_len: int, *, seed: int = 0):
        """Create a TensorDict with a mask and a few data fields.

        Sets the first *and last* column to True so that ``rle_to_mask``
        recovers the exact same ``seq_len`` (no trailing-False truncation).
        """
        gen = torch.Generator().manual_seed(seed)
        mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool, generator=gen)
        mask[:, 0] = True
        mask[:, -1] = True
        td = TensorDict(
            {
                "response_mask": mask,
                "values": torch.randn(batch_size, seq_len, generator=gen),
                "logprobs": torch.randn(batch_size, seq_len, generator=gen),
            },
            batch_size=[batch_size],
        )
        return td, mask

    def test_default_field_names(self):
        """Offsets/lengths fields should default to <mask_field>_offsets/_lengths."""
        spec = MaskNestingSpec(mask_field="response_mask", data_field_to_pad_value={"values": 0})
        assert spec.offsets_field == "response_mask_offsets"
        assert spec.lengths_field == "response_mask_lengths"

    def test_custom_field_names(self):
        spec = MaskNestingSpec(
            mask_field="m",
            offsets_field="my_off",
            lengths_field="my_len",
        )
        assert spec.offsets_field == "my_off"
        assert spec.lengths_field == "my_len"

    def test_nest_removes_mask_adds_rle(self):
        """After nesting, the mask field is removed and offsets/lengths are present."""
        td, _ = self._make_td(2, 6)
        spec = MaskNestingSpec(mask_field="response_mask", data_field_to_pad_value={"values": 0, "logprobs": 0})
        spec.nest_in_td(td)

        assert "response_mask" not in td.keys()
        assert "response_mask_offsets" in td.keys()
        assert "response_mask_lengths" in td.keys()
        # Data fields should now be nested tensors
        assert td["values"].is_nested
        assert td["logprobs"].is_nested

    def test_nest_unnest_roundtrip(self):
        """nest_in_td followed by unnest_in_td should recover original data at masked positions."""
        td, original_mask = self._make_td(3, 8, seed=7)
        original_values = td["values"].clone()
        original_logprobs = td["logprobs"].clone()

        spec = MaskNestingSpec(mask_field="response_mask", data_field_to_pad_value={"values": 0, "logprobs": 0})
        spec.nest_in_td(td)
        spec.unnest_in_td(td)

        # Mask should be recovered exactly (last column is True, so no truncation)
        recovered_mask = td["response_mask"]
        assert torch.equal(recovered_mask, original_mask)

        # Data at masked positions should match
        assert torch.equal(td["values"][original_mask], original_values[original_mask])
        assert torch.equal(td["logprobs"][original_mask], original_logprobs[original_mask])

        # Non-masked positions should be zero (default pad_value)
        assert (td["values"][~original_mask] == 0).all()
        assert (td["logprobs"][~original_mask] == 0).all()

    def test_nest_unnest_no_data_fields(self):
        """Spec with no data fields should still roundtrip the mask via RLE."""
        td, original_mask = self._make_td(2, 5)
        spec = MaskNestingSpec(mask_field="response_mask")

        spec.nest_in_td(td)
        assert "response_mask" not in td.keys()

        spec.unnest_in_td(td)
        assert torch.equal(td["response_mask"], original_mask)
