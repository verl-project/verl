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

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import pad_input, unpad_input


def left_right_2_no_padding(data: TensorDict) -> TensorDict:
    """
    Convert TensorDict from left-right padding to no-padding format.

    Args:
        data: TensorDict with "input_ids", "attention_mask", "response_mask", "position_ids"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors like "input_ids", "loss_mask", "position_ids"
        - NonTensorData includes "max_seq_len", "max_response_len", "indices"

    Note:
    1. the return input_ids/position_ids/loss_mask are nested tensor.
    2. we will remove "attention_mask", "response" in the return data, but "response_mask" is kept.
    """
    assert "input_ids" in data, "input_ids is required in left-right padding data"
    assert "attention_mask" in data, "attention_mask is required in left-right padding data"
    assert "response_mask" in data, "response_mask is required in left-right padding data"
    assert "position_ids" in data, "position_ids is required in left-right padding data"

    input_ids = data.pop("input_ids")
    attention_mask = data["attention_mask"]
    response_mask = data["response_mask"]
    position_ids = data["position_ids"]  # (bs, seq_len) or # (bs, 4, seq_len)

    max_seq_len, max_response_len = input_ids.shape[1], response_mask.shape[1]
    tu.assign_non_tensor_data(data, "max_seq_len", max_seq_len)
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    tu.assign_non_tensor_data(data, "indices", indices)

    input_ids_nested = torch.nested.nested_tensor_from_jagged(input_ids_rmpad.squeeze(-1), offsets=cu_seqlens)

    position_ids_list = []
    for i in range(attention_mask.shape[0]):
        curr_mask = attention_mask[i].bool()
        curr_pos_ids = position_ids[i]
        if curr_pos_ids.dim() == 1:  # (seq_len,)
            valid_ids = curr_pos_ids[curr_mask]
        else:  # (4, seq_len)
            valid_ids = curr_pos_ids[:, curr_mask]
        position_ids_list.append(valid_ids)
    position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)

    data["input_ids"] = input_ids_nested
    data["position_ids"] = position_ids_nested
    data["loss_mask"] = data["response_mask"]

    return data


def no_padding_2_padding(nested_tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """
    Convert NestedTensor from no-padding to right padding format.

    Args:
        nested_tensor: NestedTensor with no-padding format
        data: TensorDict with
        - Tensor includes NestedTensors like "input_ids", "loss_mask", "position_ids"
        - NonTensorData includes "max_seq_len", "max_response_len", "indices"

    Returns:
        values: regular tensor right padded to max_response_len
    """
    assert "indices" in data, "indices is required in left-right padding data"
    assert "max_seq_len" in data, "max_seq_len is required in left-right padding data"
    assert "max_response_len" in data, "max_response_len is required in left-right padding data"

    indices = tu.get_non_tensor_data(data=data, key="indices", default=None)
    max_seq_len = tu.get_non_tensor_data(data=data, key="max_seq_len", default=2048)
    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=1024)
    batch_size = nested_tensor.size(0)

    values = nested_tensor.values()
    full_values = pad_input(
        hidden_states=values.unsqueeze(-1),
        indices=indices,
        batch=batch_size,
        seqlen=max_seq_len,
    )
    values = full_values.squeeze(-1)[:, -max_response_len - 1 : -1]  # (bsz, response_length)

    return values
