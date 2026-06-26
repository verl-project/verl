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

import os

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, unpad_input


def r3_speedup_enabled() -> bool:
    """Return whether the R3 per-trajectory routing trace fast path is enabled."""
    raw = os.environ.get("VERL_R3_SPEEDUP", "0").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    try:
        return int(raw) >= 1
    except ValueError:
        return False


def narrow_routed_experts(routed_experts):
    """Convert routed expert ids to the smallest integer dtype that preserves their values."""
    if routed_experts is None:
        return None

    if isinstance(routed_experts, torch.Tensor):
        if routed_experts.numel() == 0:
            return routed_experts.to(torch.uint8)
        min_id = routed_experts.min().item()
        max_id = routed_experts.max().item()
        if min_id >= 0 and max_id <= torch.iinfo(torch.uint8).max:
            return routed_experts.to(torch.uint8)
        if min_id >= torch.iinfo(torch.int16).min and max_id <= torch.iinfo(torch.int16).max:
            return routed_experts.to(torch.int16)
        if min_id >= torch.iinfo(torch.int32).min and max_id <= torch.iinfo(torch.int32).max:
            return routed_experts.to(torch.int32)
        return routed_experts.to(torch.int64)

    arr = np.asarray(routed_experts)
    if arr.size == 0:
        return arr.astype(np.uint8, copy=False)
    min_id = arr.min()
    max_id = arr.max()
    if min_id >= 0 and max_id <= np.iinfo(np.uint8).max:
        return arr.astype(np.uint8, copy=False)
    if min_id >= np.iinfo(np.int16).min and max_id <= np.iinfo(np.int16).max:
        return arr.astype(np.int16, copy=False)
    if min_id >= np.iinfo(np.int32).min and max_id <= np.iinfo(np.int32).max:
        return arr.astype(np.int32, copy=False)
    return arr.astype(np.int64, copy=False)


def routed_experts_to_tensor(routed_experts):
    if routed_experts is None:
        return None
    if isinstance(routed_experts, np.ndarray):
        if not routed_experts.flags.writeable:
            routed_experts = routed_experts.copy()
        routed_experts = torch.from_numpy(routed_experts)
    elif not isinstance(routed_experts, torch.Tensor):
        routed_experts = torch.as_tensor(routed_experts)
    return narrow_routed_experts(routed_experts)


def make_r3_per_traj_object_array(per_traj_items):
    per_traj_items = list(per_traj_items)
    output = np.empty(len(per_traj_items), dtype=object)
    for i, item in enumerate(per_traj_items):
        output[i] = item
    return output


def _normalize_r3_object_array_item(item):
    if hasattr(item, "data") and not isinstance(item, np.ndarray | torch.Tensor):
        item = item.data
    if isinstance(item, np.ndarray) and item.dtype == object:
        item = np.asarray(item.tolist())
    if isinstance(item, np.ndarray):
        return narrow_routed_experts(item)
    return item


def normalize_r3_per_traj_list(per_traj_stack):
    if isinstance(per_traj_stack, np.ndarray):
        return [_normalize_r3_object_array_item(item) for item in per_traj_stack]
    return [_normalize_r3_object_array_item(item) for item in list(per_traj_stack)]


def _r3_per_traj_item_to_tensor(item):
    if hasattr(item, "data") and not isinstance(item, np.ndarray | torch.Tensor):
        item = item.data
    if isinstance(item, memoryview):
        item = np.asarray(item)
    if isinstance(item, np.ndarray):
        if item.dtype == object:
            item = _normalize_r3_object_array_item(item)
        if not item.flags.writeable:
            item = item.copy()
        return torch.from_numpy(item)
    if isinstance(item, torch.Tensor):
        return item
    return torch.as_tensor(item)


def _align_r3_per_traj_tensors(per_traj_list, seqlens, target_device):
    aligned = []
    for i, item in enumerate(per_traj_list):
        tensor = _r3_per_traj_item_to_tensor(item).to(device=target_device)
        target_len = int(seqlens[i].item())
        if tensor.shape[0] > target_len:
            tensor = tensor[:target_len]
        elif tensor.shape[0] < target_len:
            pad_shape = (target_len - tensor.shape[0], *tensor.shape[1:])
            pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=0)
        aligned.append(tensor)
    return aligned


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
    if r3_speedup_enabled() and "routed_experts_per_traj" in data.keys():
        per_traj_stack = data.pop("routed_experts_per_traj")
        per_traj_list = normalize_r3_per_traj_list(per_traj_stack)
        tensors = _align_r3_per_traj_tensors(per_traj_list, attention_mask.sum(dim=-1), input_ids_rmpad.device)
        flat = narrow_routed_experts(torch.cat(tensors, dim=0))
        data["routed_experts"] = torch.nested.nested_tensor_from_jagged(flat, offsets=cu_seqlens)
    elif routed_experts is not None and not routed_experts.is_nested:
        if r3_speedup_enabled():
            routed_experts = narrow_routed_experts(routed_experts)
        elif routed_experts.max() <= 255:
            routed_experts = routed_experts.to(torch.uint8)
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


def no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """Slice response from unpad model output.

    Args:
        tensor: a nested tensor or a tensor of shape (total_nnz,*),
            total_nnz is the total number of tokens across all sequences in the batch

        data: TensorDict with "prompts", "responses", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len, *]
    """
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        attention_mask = data["attention_mask"]
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]
    assert not prompt_lens.eq(0).any(), f"seq_offset - resp_len - 1 assumes prompt_len > 0. Got {prompt_lens}"

    response_list = []
    # Skip padding dimensions after sequence dimensions, if any.
    skip_padding = (0, 0) * (values.ndim - 1)
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift model output by one token for log_probs/values
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (*skip_padding, 0, pad_size)))

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
