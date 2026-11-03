# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Tensor operations shared by router replay backends."""

import torch


def build_r3_replay_mask(input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Build a full-sequence replay mask for rollout-captured routes.

    Response logprobs are read from shifted model rows, but those rows attend to
    earlier hidden states whose MoE routing also affects the result. Replay every
    causal row that can influence response logprobs and skip only the final
    response row, whose logits are not used by ``no_padding_2_padding``.
    """
    total_lens = input_ids.offsets().diff()
    response_lens = response_mask.sum(dim=-1).to(device=total_lens.device, dtype=total_lens.dtype)
    bs = total_lens.size(0)
    values = torch.tensor([True, False], dtype=torch.bool, device=total_lens.device).repeat(bs)
    replay_lens = torch.where(response_lens > 0, total_lens - 1, torch.zeros_like(total_lens))
    suffix_lens = total_lens - replay_lens
    counts = torch.stack([replay_lens, suffix_lens], dim=1).flatten()
    mask_values = torch.repeat_interleave(values, counts)
    return torch.nested.nested_tensor_from_jagged(mask_values, offsets=input_ids.offsets())
