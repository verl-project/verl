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

"""Utilities for working with nested (jagged) tensors via Run-Length Encoding (RLE).

Provides efficient compression and decompression of boolean masks by encoding
contiguous ``True`` runs as (offset, length) pairs stored in PyTorch nested
tensors.  This is useful for variable-length sequence packing and transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from tensordict import TensorDictBase


def mask_to_rle(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a boolean mask into Run-Length Encoding (RLE) of the ``True`` runs along the last dimension.

    RLE compresses a mask by recording only the start offset and length of each contiguous
    run of ``True`` values, rather than storing every element. Each row in the batch may have
    a different number of runs, so the results are returned as nested tensors.

    Currently only supports a single batch dimension (i.e. 2D mask input).

    Args:
        mask (torch.Tensor): The boolean mask to encode. Shape: ``(batch_size, seq_len)``.
    Returns:
        offsets (torch.Tensor): Nested tensor of run start positions. Shape: ``(batch_size, j1)``.
        lengths (torch.Tensor): Nested tensor of run lengths. Shape: ``(batch_size, j1)``.
    Example:
        >>> mask = torch.tensor(
        ...     [[False, False, True, True, False],
        ...      [False, True, True, False, True]])
        >>> offsets, lengths = mask_to_rle(mask)
        >>> print([component for component in offsets])
        [tensor([2]), tensor([1, 4])]
        >>> print([component for component in lengths])
        [tensor([2]), tensor([2, 1])]

    """
    assert mask.ndim == 2, f"Expected 2D mask (batch_size, last_dim), got {mask.ndim}D with shape {mask.shape}"
    last_dim = mask.shape[-1]
    flat_mask = mask.reshape(-1, last_dim).bool()
    num_rows = flat_mask.shape[0]

    # Pad each row with False on both sides to detect transitions at boundaries
    padded = torch.nn.functional.pad(flat_mask.int(), (1, 1), value=0)
    diff = padded[:, 1:] - padded[:, :-1]

    # Segment starts where diff == 1 (False -> True),
    # segment ends where diff == -1 (True -> False)
    start_rows, start_cols = (diff == 1).nonzero(as_tuple=True)
    _, end_cols = (diff == -1).nonzero(as_tuple=True)

    seg_lengths = end_cols - start_cols

    # Group by row: count segments per row, then split into per-row tensors
    counts = torch.bincount(start_rows, minlength=num_rows).tolist()
    offsets = torch.nested.nested_tensor(list(torch.split(start_cols, counts)), layout=torch.jagged)
    lengths = torch.nested.nested_tensor(list(torch.split(seg_lengths, counts)), layout=torch.jagged)
    return offsets, lengths


def _rle_scatter_indices(
    offsets: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build vectorized scatter indices from nested RLE offsets/lengths.

    Returns ``(row_indices, col_indices, flat_offsets_vals, seq_len)`` where
    ``row_indices[k]`` and ``col_indices[k]`` identify the output position
    for the k-th element across all rows/segments. ``seq_len`` is the
    inferred output width (``max(offset + length)``).
    """
    flat_off = offsets.values()  # (total_segments,)
    flat_len = lengths.values()  # (total_segments,)
    device = flat_off.device

    total_segments = flat_off.shape[0]
    if total_segments == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, flat_off, 0

    seq_len: int = (flat_off + flat_len).max().item()

    # Segments-per-row from the jagged offsets buffer
    nt_offsets = offsets._offsets  # (batch_size + 1,)
    segs_per_row = nt_offsets[1:] - nt_offsets[:-1]  # (batch_size,)
    batch_size = segs_per_row.shape[0]

    total_elements = flat_len.sum().item()

    # Map each element to its destination column:
    #   dest_col[k] = flat_off[seg_of_k] + (k - seg_start_of_k)
    seg_cumlen = flat_len.cumsum(0)
    seg_starts = seg_cumlen - flat_len

    within_seg = torch.arange(total_elements, device=device) - seg_starts.repeat_interleave(flat_len)
    col_indices = flat_off.repeat_interleave(flat_len) + within_seg

    # Map each element to its batch row
    row_of_seg = torch.arange(batch_size, device=device).repeat_interleave(segs_per_row)
    row_indices = row_of_seg.repeat_interleave(flat_len)

    return row_indices, col_indices, flat_off, seq_len


def rle_to_mask(
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    shape: Sequence[int, ...] | torch.Size | None = None,
) -> torch.Tensor:
    """
    Decode a Run-Length Encoding (RLE) of the ``True`` runs along the last dimension into a boolean mask.

    When ``shape`` is not provided the output ``seq_len`` is inferred as
    ``max(offset + length)`` across all rows, so trailing all-False columns
    from the original mask are not preserved. Pass the original mask shape
    to recover the exact dimensions.

    Args:
        offsets (torch.Tensor): Nested tensor of run start positions. Shape: ``(batch_size, j1)``.
        lengths (torch.Tensor): Nested tensor of run lengths. Shape: ``(batch_size, j1)``.
        shape (Sequence[int, ...] | torch.Size | None): If given, the output mask
            is created with this shape instead of inferring ``(batch_size, seq_len)``.
            Must be compatible with ``batch_size`` from the offsets tensor.
    Returns:
        mask (torch.Tensor): Boolean mask. Shape: ``shape`` or ``(batch_size, seq_len)``.
    """
    assert offsets.ndim == 2, f"Expected 2D offsets (batch_size, j1), got {offsets.ndim}D with shape {offsets.shape}"
    assert lengths.ndim == 2, f"Expected 2D lengths (batch_size, j1), got {lengths.ndim}D with shape {lengths.shape}"
    batch_size = offsets.shape[0]

    row_idx, col_idx, _, seq_len = _rle_scatter_indices(offsets, lengths)

    if shape is not None:
        assert shape[0] == batch_size, f"shape[0]={shape[0]} does not match batch_size={batch_size} from offsets"
        mask = torch.zeros(shape, dtype=torch.bool)
    else:
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    if row_idx.numel() > 0:
        mask[row_idx, col_idx] = True
    return mask


def unnest_tensor_by_rle(
    tensor: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor, pad_value: float = 0.0
) -> torch.Tensor:
    """
    Unnest a nested tensor by the nested RLE of the mask.
    Inversion of ``torch.nested.masked_select(tensor, mask)`` & ``offsets, lengths = mask_to_rle(mask)``.

    Reconstructs a dense tensor from a nested tensor by placing each row's
    elements back at the positions described by the RLE segments. Positions
    not covered by any segment are filled with ``pad_value``.

    Uses vectorized index computation with a single advanced-indexing scatter
    instead of Python loops, so performance scales with tensor ops rather than
    the number of segments.

    Args:
        tensor (torch.Tensor): Nested tensor produced by ``masked_select``.
            Shape: ``(batch_size, j1, *trailing)``.
        offsets (torch.Tensor): Nested tensor of run start positions from ``mask_to_rle``.
            Shape: ``(batch_size, k1)``.
        lengths (torch.Tensor): Nested tensor of run lengths from ``mask_to_rle``.
            Shape: ``(batch_size, k1)``.
        pad_value (float): Value to fill positions not covered by any segment. Default: ``0.0``.
    Returns:
        torch.Tensor: Dense tensor of shape ``(batch_size, seq_len, *trailing)``
            where ``seq_len = max(offset + length)`` across all rows.
    """
    batch_size = tensor.shape[0]
    row_idx, col_idx, _, seq_len = _rle_scatter_indices(offsets, lengths)

    data_flat = tensor.values()  # (total_elements, *trailing)
    trailing_shape = data_flat.shape[1:]

    output = torch.full((batch_size, seq_len, *trailing_shape), pad_value, dtype=tensor.dtype, device=tensor.device)
    if row_idx.numel() > 0:
        output[row_idx, col_idx] = data_flat
    return output


@dataclass
class MaskNestingSpec:
    """Specification for nesting a group of data tensors with mask in a ``TensorDict``."""

    mask_field: str
    mask_shape: Sequence[int] | None = None
    offsets_field: str | None = None
    lengths_field: str | None = None
    data_field_to_pad_value: dict[str, int | float | bool] = field(default_factory=dict)

    def __post_init__(self):
        if self.offsets_field is None:
            self.offsets_field = f"{self.mask_field}_offsets"
        if self.lengths_field is None:
            self.lengths_field = f"{self.mask_field}_lengths"

    def nest_in_td(self, td: TensorDictBase) -> TensorDictBase:
        """Nest every data field in *td* by the mask.

        Each data tensor must have shape
        ``(*batch_dims, *sample_mask_dims, *feature_dims)``, where
        ``(*batch_dims, *sample_mask_dims) == mask.shape`` — the leading
        dims are the outer batch (TensorDict ``batch_size``), the next dims
        are what the per-sample mask actually selects over, and any
        trailing ``feature_dims`` (e.g. multi-head, hidden) pass through
        unchanged. This generalises ``torch.nested.masked_select`` which
        only handles same-shape mask + data.
        """
        mask: torch.Tensor = td.pop(self.mask_field).bool()
        if self.mask_shape is None:
            self.mask_shape = mask.shape

        # Per-row valid token counts → jagged offsets reused for every field.
        # mask is 2-D (batch_size, seq_len); flatten leading dims if higher-D.
        flat_mask = mask.reshape(mask.shape[0], -1)
        counts = flat_mask.sum(dim=-1)
        nt_offsets = torch.zeros(mask.shape[0] + 1, dtype=torch.long, device=mask.device)
        nt_offsets[1:] = counts.cumsum(0)

        for data_field in self.data_field_to_pad_value:
            t = td[data_field]
            assert t.shape[: mask.ndim] == mask.shape, (
                f"data field {data_field!r} has shape {tuple(t.shape)}, "
                f"expected leading dims to match mask shape {tuple(mask.shape)}"
            )
            # Fancy indexing broadcasts the mask over trailing dims, returning
            # a tensor of shape (n_true, *batch_dims).
            flat_values = t[mask]
            td[data_field] = torch.nested.nested_tensor_from_jagged(flat_values, offsets=nt_offsets)

        offsets, lengths = mask_to_rle(mask)
        td[self.offsets_field] = offsets
        td[self.lengths_field] = lengths

        return td

    def unnest_in_td(self, td: TensorDictBase) -> TensorDictBase:
        """Inversion of ``nest_in_td``.

        Computes the RLE scatter indices once and reuses them for every data
        field and the mask, avoiding redundant work when multiple fields share
        the same mask.

        The batch dimension always comes from the live RLE offsets
        (``offsets.shape[0]``), **not** from ``self.mask_shape[0]``. This
        matters when the TensorDict has been chunked (e.g. per-worker
        dispatch): the stashed ``mask_shape`` carries the **original**
        full-batch dim, while the offsets have been split to the local
        per-chunk batch. Only the trailing dims of ``mask_shape`` are
        load-bearing — they let us preserve trailing all-False columns
        that RLE inference alone would drop.
        """
        offsets = td.pop(self.offsets_field)
        lengths = td.pop(self.lengths_field)

        batch_size = offsets.shape[0]
        row_idx, col_idx, _, seq_len = _rle_scatter_indices(offsets, lengths)

        # Construct the local mask shape using:
        #   - batch_size from offsets (per-chunk correct), and
        #   - trailing dims from self.mask_shape if available (to honour
        #     trailing all-False columns), else from RLE-inferred seq_len.
        if self.mask_shape is not None:
            local_mask_shape = (batch_size, *tuple(self.mask_shape)[1:])
        else:
            local_mask_shape = (batch_size, seq_len)
        out_seq_len = local_mask_shape[-1]

        for data_field, pad_value in self.data_field_to_pad_value.items():
            nt = td[data_field]
            data_flat = nt.values()  # (total_elements, *trailing)
            trailing_shape = data_flat.shape[1:]
            out = torch.full((batch_size, out_seq_len, *trailing_shape), pad_value, dtype=nt.dtype, device=nt.device)
            if row_idx.numel() > 0:
                out[row_idx, col_idx] = data_flat
            td[data_field] = out

        mask = torch.zeros(local_mask_shape, dtype=torch.bool)
        if row_idx.numel() > 0:
            mask[row_idx, col_idx] = True
        td[self.mask_field] = mask
        return td
