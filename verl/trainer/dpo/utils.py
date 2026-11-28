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
"""Helpers shared by the DPO loss, the DPO trainer and the engine's micro-batching.

DPODataset packs each pair into one row, ``[chosen tokens, rejected tokens]``, with ``boundary_offset``
marking where the rejected part starts. Keeping a pair in one row means dynamic micro-batching cannot
separate or reorder it. ``expand_dpo_micro_batch`` splits each row back into two sequences after
micro-batching, interleaved as ``[chosen_0, rejected_0, chosen_1, rejected_1, ...]``, which is the
layout ``dpo_loss`` reads.
"""

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu

# Non-tensor flag the DPO trainer sets on a batch so that prepare_micro_batches expands packed pairs.
DPO_EXPAND_FLAG = "dpo_expand_packed_sequences"


def get_boundary_offsets(data) -> list[int]:
    """Return ``boundary_offset`` of a packed batch as a list of ints, one per pair."""
    raw = data["boundary_offset"]
    if isinstance(raw, torch.Tensor):
        if raw.dim() != 1:
            raise ValueError(f"boundary_offset must be a (B,) tensor, got shape {tuple(raw.shape)}")
        return raw.tolist()
    raise TypeError(f"boundary_offset must be a (B,) torch.Tensor, got {type(raw)}")


def expand_packed_sequences(packed_input_ids, packed_loss_mask, boundary_offsets) -> dict[str, torch.Tensor]:
    """Split packed pairs into 2B interleaved sequences.

    Args:
        packed_input_ids: list of B 1-D tensors, each ``[chosen tokens, rejected tokens]``.
        packed_loss_mask: list of B 1-D tensors aligned with ``packed_input_ids``.
        boundary_offsets: list of B ints, the index where each rejected part starts.

    Returns:
        dict of NestedTensors with 2B rows: "input_ids", "loss_mask" and "position_ids", where
        position ids restart at 0 for every sequence.
    """
    if not len(packed_input_ids) == len(packed_loss_mask) == len(boundary_offsets):
        raise ValueError("packed_input_ids, packed_loss_mask and boundary_offsets must have the same length")
    input_ids, loss_mask, position_ids = [], [], []
    for ids, mask, offset in zip(packed_input_ids, packed_loss_mask, boundary_offsets, strict=True):
        if not 0 < offset < ids.shape[0]:
            raise ValueError(f"boundary_offset {offset} is outside the packed sequence of length {ids.shape[0]}")
        for part in (slice(None, offset), slice(offset, None)):
            input_ids.append(ids[part])
            loss_mask.append(mask[part])
            position_ids.append(torch.arange(input_ids[-1].shape[0], dtype=torch.long, device=ids.device))

    def nested(tensors):
        return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    return {"input_ids": nested(input_ids), "loss_mask": nested(loss_mask), "position_ids": nested(position_ids)}


def expand_dpo_micro_batch(micro_batch: TensorDict) -> TensorDict:
    """Expand a micro-batch of B packed pairs into 2B interleaved sequences.

    Also expands "ref_logps" from (B, 2) ``[chosen, rejected]`` to interleaved (2B,), and copies the
    non-tensor metadata except ``boundary_offset`` and the expansion flag.
    """
    expanded = expand_packed_sequences(
        list(micro_batch["input_ids"].unbind()),
        list(micro_batch["loss_mask"].unbind()),
        get_boundary_offsets(micro_batch),
    )
    if "ref_logps" in micro_batch.keys():
        expanded["ref_logps"] = micro_batch["ref_logps"].reshape(-1)
    non_tensor = {key: val for key, val in micro_batch.non_tensor_items() if key != DPO_EXPAND_FLAG}
    return tu.get_tensordict(tensor_dict=expanded, non_tensor_dict=non_tensor)


def sum_logps_per_sequence(log_probs: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Sum per-token log-probs over the response tokens of each sequence.

    The engine returns ``log_probs[t] = log p(token[t + 1] | token[:t + 1])``, so the log-prob of
    token ``t`` sits at position ``t - 1``. The loss mask is therefore shifted left by one within
    each sequence, and the last position of every sequence (which predicts past its end) is dropped.

    Args:
        log_probs: NestedTensor (jagged) of shape (N, j) with per-token log-probs.
        loss_mask: NestedTensor (jagged) of shape (N, j), 1 for response tokens and 0 for prompt tokens.

    Returns:
        Tensor of shape (N,) with the summed response log-prob of each sequence.
    """
    if not (log_probs.is_nested and loss_mask.is_nested):
        raise NotImplementedError("sum_logps_per_sequence expects NestedTensors (pad_mode=no_padding)")

    lp_flat = log_probs.values()
    mask_flat = loss_mask.values().to(lp_flat.dtype)
    offsets = log_probs.offsets().to(lp_flat.device)
    lengths = offsets.diff()
    num_seqs = lengths.numel()

    shifted_mask = torch.roll(mask_flat, shifts=-1, dims=0)
    # Position offsets[i + 1] - 1 is the last token of sequence i: after the roll it would carry
    # the mask of the next sequence's first token, and it predicts nothing inside sequence i.
    shifted_mask[offsets[1:] - 1] = 0

    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, device=lp_flat.device), lengths)
    per_seq = torch.zeros(num_seqs, dtype=lp_flat.dtype, device=lp_flat.device)
    return per_seq.index_add(0, seq_ids, lp_flat * shifted_mask)
