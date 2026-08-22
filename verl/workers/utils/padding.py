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
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, unpad_input
from verl.utils.device import get_device_name
from verl.workers.utils.tpu_static_packing import get_packed_sequence_offsets_and_lens_tpu


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

    routed_experts = data.get("routed_experts", None)
    if routed_experts is not None and not routed_experts.is_nested:
        routed_experts_rmpad = index_first_axis(routed_experts.unsqueeze(-1).flatten(0, 1), indices)
        routed_experts_nested = torch.nested.nested_tensor_from_jagged(
            routed_experts_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        data["routed_experts"] = routed_experts_nested

    # (bsz, seqlen, topk)
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    if teacher_logprobs is not None and teacher_ids is not None:
        teacher_logprobs_rmpad = index_first_axis(teacher_logprobs.unsqueeze(-1).flatten(0, 1), indices)
        teacher_ids_rmpad = index_first_axis(teacher_ids.unsqueeze(-1).flatten(0, 1), indices)
        teacher_logprobs_nested = torch.nested.nested_tensor_from_jagged(
            teacher_logprobs_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        teacher_ids_nested = torch.nested.nested_tensor_from_jagged(teacher_ids_rmpad.squeeze(-1), offsets=cu_seqlens)
        data["teacher_logprobs"] = teacher_logprobs_nested
        data["teacher_ids"] = teacher_ids_nested

    return data


def get_packed_sequence_offsets_and_lens(
    tensor: torch.Tensor, data: TensorDict, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract and compute sequence offsets, response lengths, and maximum response length from packed sequences.

    Handles both TPU static binned packing and PyTorch nested/jagged tensor representations.

    Args:
        tensor (torch.Tensor): The input token or sequence tensor (nested/jagged or dense).
        data (TensorDict): Data batch containing prompts, responses, or mask attributes.
        values (torch.Tensor): Reference value tensor used to infer target device and dtypes.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]:
            - sequence_offsets (torch.Tensor): Cumulative sequence end offsets.
            - response_lens (torch.Tensor): Length of the response section for each sequence.
            - max_response_len (int): Maximum response length across the batch.
    """
    is_tpu = get_device_name() == "tpu" or "tpu_custom_attention_mask" in data.keys()
    if is_tpu and not (getattr(tensor, "is_nested", False) or hasattr(tensor, "offsets")):
        return get_packed_sequence_offsets_and_lens_tpu(tensor, data, values)

    if "prompts" in data.keys() and "responses" in data.keys():
        prompt_ids = data["prompts"]
        response_ids = data["responses"]

        if isinstance(prompt_ids, NonTensorData):
            prompt_ids = prompt_ids.data
        if isinstance(response_ids, NonTensorData):
            response_ids = response_ids.data
    else:
        prompt_ids = None
        response_ids = None

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    if isinstance(max_response_len, list | tuple):
        max_response_len = max(max_response_len) if len(max_response_len) > 0 else -1

    if getattr(tensor, "is_nested", False) or hasattr(tensor, "offsets"):
        sequence_offsets = tensor.offsets()[1:].to(device=values.device)
        sequence_lens = tensor.offsets().diff().to(device=values.device)

        if response_ids is not None and (getattr(response_ids, "is_nested", False) or hasattr(response_ids, "offsets")):
            response_lens = response_ids.offsets().diff().to(device=values.device)
        elif response_ids is not None and isinstance(response_ids, list | tuple):
            response_lens = torch.tensor(
                [r.shape[-1] if hasattr(r, "shape") else len(r) for r in response_ids],
                device=values.device,
                dtype=torch.int64,
            )
        elif response_ids is not None and isinstance(response_ids, torch.Tensor) and response_ids.ndim >= 2:
            if "response_mask" in data.keys():
                response_lens = data["response_mask"].sum(dim=-1).to(device=values.device, dtype=torch.int64)
            else:
                response_lens = torch.tensor(
                    [response_ids.shape[1]] * response_ids.shape[0], device=values.device, dtype=torch.int64
                )
        else:
            mask_key = (
                "response_mask"
                if "response_mask" in data.keys()
                else ("loss_mask" if "loss_mask" in data.keys() else None)
            )
            if mask_key is not None:
                mask_val = data[mask_key]
                if getattr(mask_val, "is_nested", False) or hasattr(mask_val, "offsets"):
                    response_lens = torch.tensor(
                        [int(x.sum().item()) for x in mask_val],
                        device=values.device,
                        dtype=torch.int64,
                    )
                else:
                    response_lens = mask_val.sum(dim=-1).to(device=values.device, dtype=torch.int64)
            else:
                response_lens = sequence_lens // 2

        if max_response_len < 0:
            max_response_len = int(response_lens.max().item())
    elif prompt_ids is not None and (getattr(prompt_ids, "is_nested", False) or hasattr(prompt_ids, "offsets")):
        prompt_lens = prompt_ids.offsets().diff().to(device=values.device)
        response_lens = response_ids.offsets().diff().to(device=values.device)
        if max_response_len < 0:
            max_response_len = int(response_lens.max().item())
        sequence_lens = prompt_lens + response_lens
        sequence_offsets = sequence_lens.cumsum(dim=0)
    elif prompt_ids is not None and isinstance(prompt_ids, list | tuple):
        prompt_lens = torch.tensor(
            [p.shape[-1] if hasattr(p, "shape") else len(p) for p in prompt_ids],
            device=values.device,
            dtype=torch.int64,
        )
        response_lens = torch.tensor(
            [r.shape[-1] if hasattr(r, "shape") else len(r) for r in response_ids],
            device=values.device,
            dtype=torch.int64,
        )
        if max_response_len < 0:
            max_response_len = int(response_lens.max().item())
        sequence_lens = prompt_lens + response_lens
        sequence_offsets = sequence_lens.cumsum(dim=0)
    else:
        return get_packed_sequence_offsets_and_lens_tpu(tensor, data, values)

    return sequence_offsets, response_lens, max_response_len


def no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    Args:
        tensor: a nested tensor or a tensor of shape (total_nnz,*),
            total_nnz is the total number of tokens across all sequences in the batch

        data: TensorDict with "prompts", "responses", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len, *]
    """
    values = tensor.values() if getattr(tensor, "is_nested", False) else tensor
    sequence_offsets, response_lens, max_response_len = get_packed_sequence_offsets_and_lens(tensor, data, values)

    response_list = []
    # Skip padding dimensions after sequence dimensions, if any.
    skip_padding = (0, 0) * (values.ndim - 1)
    prev_offset = 0
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        resp_len_item = int(resp_len.item()) if isinstance(resp_len, torch.Tensor) else int(resp_len)
        seq_offset_item = int(seq_offset.item()) if isinstance(seq_offset, torch.Tensor) else int(seq_offset)
        pad_size = max(0, max_response_len - resp_len_item)

        start_idx = seq_offset_item - resp_len_item - 1
        end_idx = seq_offset_item - 1
        if start_idx < prev_offset:
            start_idx = seq_offset_item - resp_len_item
            end_idx = seq_offset_item

        response_list.append(F.pad(values[start_idx:end_idx], (*skip_padding, 0, pad_size)))
        prev_offset = seq_offset_item

    output = torch.stack(response_list, dim=0)
    return output


def build_attention_mask_from_nested(input_ids: torch.Tensor, max_seq_len: int | None = None) -> torch.Tensor:
    """Build a padded full-sequence attention mask from nested input ids."""
    assert input_ids.is_nested, "input_ids must be a nested tensor"
    device = input_ids.values().device
    seq_lens = input_ids.offsets().diff().to(device=device)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
    return (positions < seq_lens.unsqueeze(1)).to(torch.int32)


def embeds_padding_2_no_padding(data: TensorDict) -> TensorDict:
    """
    Convert TensorDict from prompt embeds with padding to no-padding format.

    Currently we expect the prompt embedding mask to be [1111000...] format,
    which means the valid tokens are continuous and start from the left.

    Args:
        data: TensorDict with "prompt_embeds", "prompt_embeds_mask",
              "negative_prompt_embeds", "negative_prompt_embeds_mask"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors "prompt_embeds", "prompt_embeds_mask",
          "negative_prompt_embeds", "negative_prompt_embeds_mask"
    """

    def _to_nested(embeds: torch.Tensor, mask: torch.Tensor):
        """Strip padding from (bs, seq_len, dim) embeds using the boolean mask and return nested tensors."""
        embeds_list, mask_list = [], []
        for i in range(mask.shape[0]):
            curr_mask = mask[i].bool()
            embeds_list.append(embeds[i, curr_mask, :])
            mask_list.append(curr_mask[curr_mask])
        return (
            torch.nested.as_nested_tensor(embeds_list, layout=torch.jagged),
            torch.nested.as_nested_tensor(mask_list, layout=torch.jagged),
        )

    data["prompt_embeds"], data["prompt_embeds_mask"] = _to_nested(data["prompt_embeds"], data["prompt_embeds_mask"])

    if isinstance(data.get("negative_prompt_embeds", None), torch.Tensor):
        data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"] = _to_nested(
            data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"]
        )

    return data


def response_from_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Extract response from nested model output.

    Args:
        tensor: a nested tensor with shape (bsz, prompt_len + response_len)
        response_mask: a nested tensor with shape (bsz, response_len)

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    values, offsets = tensor.values(), tensor.offsets()
    response_lens = response_mask.offsets().diff()
    response_list = []
    for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
        # left-shift model output by one token for log_probs/values
        response_list.append(values[seq_offset - resp_len - 1 : seq_offset - 1])
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)


def response_to_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Convert padded response tensor to nested tensor.

    Args:
        tensor: a tensor with shape (bsz, response_len)
        response_mask: a nested tensor with shape (bsz, response_len)

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    assert response_mask.is_nested
    response_lens = response_mask.offsets().diff()
    response_list = []
    for i in range(tensor.shape[0]):
        response_list.append(tensor[i, : response_lens[i]])

    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)
