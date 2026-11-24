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
"""Helpers shared by the DPO loss and the DPO trainer."""

import torch


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
