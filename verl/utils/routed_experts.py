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
"""Wire format for rollout ``routed_experts``.

Driver batches cannot store a jagged NestedTensor in the TensorDict: ``cat``,
``reorder`` and ``chunk`` all require a regular dim-0. Per-sequence numpy
blocks in ``non_tensor_batch`` support those ops and only carry attended
tokens. Consumers that still expect a tensor go through
:func:`ragged_routed_experts_to_nested` (or the dense unpad adapter) at the
worker boundary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

ROUTER_REPLAY_UNRECORDED = -1


def unrecorded_fill_value(dtype: torch.dtype) -> int:
    """Sentinel for a token the rollout never ran through MoE.

    Unsigned dtypes cannot hold ``-1``; 0 is the only value that does not wrap
    into a legal expert id.
    """
    if dtype.is_floating_point:
        return ROUTER_REPLAY_UNRECORDED
    return ROUTER_REPLAY_UNRECORDED if torch.iinfo(dtype).min < 0 else 0


def pack_padded_routed_experts(rows: list[Any], attention_mask: torch.Tensor) -> np.ndarray:
    """Strip config padding from per-sample routed-experts tensors.

    Each input row is ``[1, seq, layers, topk]`` or ``[seq, layers, topk]`` with
    ``seq == attention_mask.shape[1]``. The result is an object array of
    ``[attended_i, layers, topk]`` numpy blocks, one per sample.
    """
    if any(row is None for row in rows):
        raise ValueError(
            "routed_experts is present on some samples of this chunk but not all; "
            "a partial routing payload cannot be replayed consistently."
        )
    seq_len = int(attention_mask.shape[1])
    packed = np.empty(len(rows), dtype=object)
    for i, row in enumerate(rows):
        row_tensor = torch.as_tensor(row)
        if row_tensor.ndim == 4 and row_tensor.shape[0] == 1:
            row_tensor = row_tensor[0]
        if row_tensor.ndim != 3 or row_tensor.shape[0] != seq_len:
            raise ValueError(
                f"routed_experts sample {i} has shape {tuple(row_tensor.shape)}; expected "
                f"(1, {seq_len}, layers, topk) or ({seq_len}, layers, topk)."
            )
        packed[i] = row_tensor[attention_mask[i].bool()].cpu().numpy()

    attended = attention_mask.sum(dim=1).tolist()
    mismatched = [
        (i, int(packed[i].shape[0]), int(n)) for i, n in enumerate(attended) if int(packed[i].shape[0]) != int(n)
    ]
    if mismatched:
        raise AssertionError(
            "ragged routed_experts row count != attended tokens for "
            f"{len(mismatched)} of {len(attended)} sequences "
            f"(first offenders, as (index, rows, attended): {mismatched[:5]})."
        )
    return packed


def _as_row_list(value: Any) -> list[Any] | None:
    """Per-sequence rows from a ragged container, or None if this is not one."""
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value)
    if isinstance(value, list | tuple):
        return list(value)
    return None


def ragged_routed_experts_to_nested(routed_experts: Any, cu_seqlens: torch.Tensor) -> Any:
    """Rebuild ragged ``routed_experts`` into a jagged nested tensor.

    Dense tensors (including already-nested ones) are returned unchanged so the
    caller can keep the historical unpad path. ``cu_seqlens`` must match the
    ``unpad_input`` offsets for ``input_ids``; a row-count mismatch would replay
    every later token against the wrong experts.
    """
    if routed_experts is None or isinstance(routed_experts, torch.Tensor):
        return routed_experts

    try:
        from tensordict.tensorclass import NonTensorData, NonTensorStack
    except ImportError:  # pragma: no cover
        NonTensorData = ()  # type: ignore[assignment]
        NonTensorStack = ()  # type: ignore[assignment]

    if isinstance(routed_experts, NonTensorStack):
        rows = routed_experts.tolist()
    elif isinstance(routed_experts, NonTensorData):
        rows = _as_row_list(routed_experts.data)
    else:
        rows = _as_row_list(routed_experts)
    if rows is None:
        raise TypeError(f"unsupported routed_experts type: {type(routed_experts)}")
    if not rows:
        return routed_experts

    expected = cu_seqlens.diff().tolist()
    if len(rows) != len(expected):
        raise ValueError(
            f"ragged routed_experts has {len(rows)} sequences but cu_seqlens describes "
            f"{len(expected)}; the rollout payload does not match this batch."
        )
    actual = [int(np.asarray(row).shape[0]) for row in rows]
    if actual != [int(n) for n in expected]:
        bad = [(i, a, int(e)) for i, (a, e) in enumerate(zip(actual, expected, strict=True)) if a != int(e)]
        raise ValueError(
            f"ragged routed_experts row counts disagree with cu_seqlens for {len(bad)} of "
            f"{len(rows)} sequences (first offenders, as (index, rows, expected): {bad[:5]})."
        )

    values = np.concatenate([np.asarray(row) for row in rows], axis=0)
    if not values.flags.writeable:
        values = values.copy()
    # Driver rows are host numpy; unpad offsets inherit input_ids.device.
    values_t = torch.from_numpy(values).to(device=cu_seqlens.device)
    return torch.nested.nested_tensor_from_jagged(values_t, offsets=cu_seqlens)


def has_rollout_routed_experts(batch: Any) -> bool:
    """True when the batch carries rollout routing, in either representation."""
    tensor_batch = getattr(batch, "batch", None)
    if tensor_batch is not None and "routed_experts" in tensor_batch.keys():
        return True
    non_tensor = getattr(batch, "non_tensor_batch", None) or {}
    return "routed_experts" in non_tensor


def rollout_and_actor_routed_experts_conflict(batch: Any, old_log_prob: Any) -> bool:
    """R2 (actor-recorded) and R3 (rollout-recorded) routing present together."""
    actor_batch = getattr(old_log_prob, "batch", None)
    actor_has = actor_batch is not None and "routed_experts" in actor_batch.keys()
    return has_rollout_routed_experts(batch) and actor_has


def get_routed_experts_from_data_proto(data: Any) -> Any:
    """Prefer the TensorDict copy, then the ragged ``non_tensor_batch`` copy."""
    tensor_batch = getattr(data, "batch", None)
    if tensor_batch is not None and "routed_experts" in tensor_batch.keys():
        return tensor_batch.get("routed_experts")
    non_tensor = getattr(data, "non_tensor_batch", None) or {}
    return non_tensor.get("routed_experts", None)
