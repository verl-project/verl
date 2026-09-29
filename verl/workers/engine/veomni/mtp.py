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

"""Align VeOmni MTP targets with verl's unpadded sequences."""

import torch
import torch.nn.functional as F
from tensordict import TensorDict

IGNORE_INDEX = -100


def _token_masks(data: TensorDict) -> list[torch.Tensor]:
    """Return full-sequence masks in token coordinates (before any causal shift).

    RL loss_mask spans responses only. Its zeroes include tool observations,
    so response_mask.sum() must never be used as a response sequence length.
    SFT supplies a full-sequence jagged loss_mask instead.
    """
    input_ids = data["input_ids"]
    if not input_ids.is_nested:
        raise ValueError("VeOmni MTP expects verl NO_PADDING nested input_ids.")
    tokens = list(input_ids.unbind())
    loss_masks = list(data["loss_mask"].unbind())
    response_mask = data.get("response_mask")
    attention_mask = data.get("attention_mask")
    masks = []
    for i, (sample, mask) in enumerate(zip(tokens, loss_masks, strict=True)):
        mask = mask.to(device=sample.device, dtype=torch.bool)
        if response_mask is None:
            if mask.shape != sample.shape:
                raise ValueError("MTP SFT loss_mask must align with each full unpadded sequence.")
            masks.append(mask)
            continue

        if not response_mask.is_nested:
            if attention_mask is None or attention_mask.is_nested:
                raise ValueError("MTP with padded response_mask requires the original dense attention_mask.")
            response_width = response_mask.shape[-1]
            if mask.numel() != response_width:
                raise ValueError("MTP RL loss_mask must have the same response width as response_mask.")
            # The full original attention mask has left-padded prompts and
            # right-padded responses. Boolean indexing also handles interior holes.
            response_attention = attention_mask[i, attention_mask.shape[-1] - response_width :].to(
                device=sample.device, dtype=torch.bool
            )
            if int(attention_mask[i].sum().item()) != sample.numel():
                raise ValueError("MTP attention_mask does not describe the unpadded input_ids.")
            mask = mask[response_attention]
        elif mask.numel() != response_mask[i].numel():
            raise ValueError("MTP jagged loss_mask must align with the response sequence.")

        if mask.numel() > sample.numel():
            raise ValueError("MTP response sequence is longer than its full input sequence.")
        masks.append(F.pad(mask, (sample.numel() - mask.numel(), 0), value=False))
    return masks


def count_mtp_targets(data: TensorDict, num_depths: int) -> torch.Tensor:
    """Count valid targets before splitting a DP rank's batch into microbatches."""
    masks = _token_masks(data)
    count = data["input_ids"].values().new_zeros(())
    for mask in masks:
        for depth in range(num_depths):
            count = count + mask[depth + 2 :].sum()
    return count


def build_mtp_labels(
    data: TensorDict, num_depths: int, *, packed: bool, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [B,D,L] labels; depth d predicts token i+d+2, within one sample.

    Mask targets *before* shifting. This preserves prompt exclusion, tool-turn
    masks, and sample boundaries even when all samples share one packed row.
    """
    rows = []
    for tokens, mask in zip(data["input_ids"].unbind(), _token_masks(data), strict=True):
        targets = tokens.masked_fill(~mask, IGNORE_INDEX)
        labels = targets.new_full((num_depths, tokens.numel()), IGNORE_INDEX)
        for depth in range(num_depths):
            shift = depth + 2
            if shift < tokens.numel():
                labels[depth, :-shift] = targets[shift:]
        rows.append(labels)

    if packed:
        labels = torch.cat(rows, dim=-1).unsqueeze(0)
        if labels.shape[-1] > sequence_length:
            raise ValueError("MTP labels exceed the model's packed sequence length.")
        labels = F.pad(labels, (0, sequence_length - labels.shape[-1]), value=IGNORE_INDEX)
    else:
        if any(row.shape[-1] > sequence_length for row in rows):
            raise ValueError("MTP labels exceed the model's padded sequence length.")
        labels = torch.stack([F.pad(row, (0, sequence_length - row.shape[-1]), value=IGNORE_INDEX) for row in rows])
    return labels, (labels != IGNORE_INDEX).sum()
