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

import torch


def select_response_ids_for_reward(
    response_ids: torch.Tensor,
    response_attention_mask: torch.Tensor,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Select model-generated response ids and the token index that should receive reward."""
    valid_response_length = int(response_attention_mask.sum().item())
    valid_response_ids = response_ids[:valid_response_length]
    reward_index = max(valid_response_length - 1, 0)

    if response_mask is None:
        return valid_response_ids, reward_index

    valid_model_mask = response_mask[:valid_response_length].bool()
    if not torch.any(valid_model_mask):
        return valid_response_ids, reward_index

    valid_response_ids = valid_response_ids[valid_model_mask]
    reward_index = int(valid_model_mask.nonzero(as_tuple=False)[-1].item())
    return valid_response_ids, reward_index
