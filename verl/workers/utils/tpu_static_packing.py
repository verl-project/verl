# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from tensordict.tensorclass import NonTensorData

from verl.utils import tensordict_utils as tu


def unpack_tpu_packed_data(data: TensorDict) -> TensorDict:
    """Reconstruct NestedTensors from packed flat sequence keys on TPU.

    In TPU binned sequence packing, standard sequence keys (like old_log_probs,
    advantages, response_mask) are packed into 2D flat tensors. Reconstructing them
    as device NestedTensors using cu_seqlens ensures they can be correctly
    padded/sliced to [batch_size, max_response_len].
    """
    if "tpu_custom_attention_mask" not in data.keys():
        return data

    input_ids = data["input_ids"]
    mbsz = input_ids.shape[0]

    cu_seqlens_list = data["tpu_cu_seqlens"]
    if hasattr(cu_seqlens_list, "data"):
        cu_seqlens_list = cu_seqlens_list.data
    if not isinstance(cu_seqlens_list, list | tuple):
        cu_seqlens_list = [cu_seqlens_list]

    num_seqs_list = data["tpu_num_seqs"]
    if hasattr(num_seqs_list, "data"):
        num_seqs_list = num_seqs_list.data
    if not isinstance(num_seqs_list, list | tuple):
        num_seqs_list = [num_seqs_list]

    # Include "loss_mask" in the unpack keys list to unpack SFT training loss mask correctly.
    sequence_keys_to_unpack = [
        "old_log_probs",
        "advantages",
        "response_mask",
        "values",
        "returns",
        "ref_log_prob",
        "rollout_is_weights",
        "loss_mask",
    ]

    for k in sequence_keys_to_unpack:
        if k in data.keys():
            val = data[k]
            if isinstance(val, torch.Tensor) and not getattr(val, "is_nested", False):
                cumulative_offset = 0
                combined_cu_seqlens = [0]
                combined_val_rmpad_list = []

                for row_idx in range(mbsz):
                    cu_seqlens_row = cu_seqlens_list[row_idx]
                    num_seqs_row = num_seqs_list[row_idx]

                    if isinstance(cu_seqlens_row, list):
                        cu_seqlens_row = cu_seqlens_row[0]
                    if not isinstance(cu_seqlens_row, torch.Tensor):
                        cu_seqlens_row = torch.tensor(cu_seqlens_row, dtype=torch.int32)

                    cu_seqlens_active = cu_seqlens_row[: num_seqs_row + 1]
                    active_len = int(cu_seqlens_active[-1].item())

                    if val.dim() >= 2:
                        val_row = val[row_idx, :active_len]
                    else:
                        val_row = val[:active_len]

                    combined_val_rmpad_list.append(val_row)

                    row_offsets = cu_seqlens_active[1:] - cu_seqlens_active[0] + cumulative_offset
                    combined_cu_seqlens.extend(row_offsets.tolist())
                    cumulative_offset += active_len

                combined_val_rmpad = torch.cat(combined_val_rmpad_list, dim=0).to(val.device)
                combined_cu_seqlens_tensor = torch.tensor(combined_cu_seqlens, dtype=torch.int32, device=val.device)
                nested_val = torch.nested.nested_tensor_from_jagged(combined_val_rmpad, combined_cu_seqlens_tensor)
                data._tensordict[k] = nested_val

    return data


def nested_to_padded_tpu(val: torch.Tensor, fill_val=0.0) -> torch.Tensor:
    """Safely converts a PyTorch NestedTensor to a padded 2D dense tensor on TPU."""
    if not isinstance(val, torch.Tensor) or not getattr(val, "is_nested", False):
        return val
    try:
        return val.to_padded_tensor(fill_val)
    except Exception:
        values = val.values()
        offsets = val.offsets()
        bsz = offsets.shape[0] - 1
        lens = offsets.diff().tolist()
        max_len = max(lens) if len(lens) > 0 else 0
        out = torch.full(
            (bsz, max_len, *values.shape[1:]), fill_value=fill_val, dtype=values.dtype, device=values.device
        )
        for i in range(bsz):
            st, en = int(offsets[i]), int(offsets[i + 1])
            row_data = values[st:en]
            row_len = min(en - st, row_data.shape[0])
            if row_len > 0:
                out[i, :row_len] = row_data[:row_len]
        return out


def pad_to_2d_tensor(val, target_len=-1, fill_val=0.0, target_tensor=None):
    """Pads standard and nested tensors to 2D dense tensors of target length."""
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        val = nested_to_padded_tpu(val, fill_val)
        if val.ndim == 1:
            val = val.unsqueeze(0)
    elif isinstance(val, list | tuple):
        t_list = []
        for x in val:
            if isinstance(x, torch.Tensor):
                x = nested_to_padded_tpu(x, fill_val)
                if x.ndim == 0:
                    x = x.unsqueeze(0)
                elif x.ndim > 1:
                    x = x.flatten()
            else:
                x = torch.as_tensor(x)
                if x.ndim == 0:
                    x = x.unsqueeze(0)
            t_list.append(x)

        if len(t_list) > 0:
            max_item_len = max(t.shape[0] for t in t_list)
            if target_len > 0:
                max_item_len = max(max_item_len, target_len)
            aligned = []
            for t in t_list:
                if t.shape[0] < max_item_len:
                    pad = torch.full((max_item_len - t.shape[0],), fill_value=fill_val, dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad], dim=0)
                elif t.shape[0] > max_item_len:
                    t = t[:max_item_len]
                aligned.append(t)
            val = torch.stack(aligned, dim=0)
        else:
            val = torch.empty((0, target_len if target_len > 0 else 0))
    else:
        val = torch.as_tensor(val)
        if val.ndim == 1:
            val = val.unsqueeze(0)

    if isinstance(val, torch.Tensor):
        val = nested_to_padded_tpu(val, fill_val)
    if isinstance(target_tensor, torch.Tensor):
        target_tensor = nested_to_padded_tpu(target_tensor, 0.0)

    if isinstance(val, torch.Tensor) and val.ndim >= 2:
        if target_len > 0:
            curr_len = int(val.shape[1])
            if curr_len < target_len:
                pad_shape = list(val.shape)
                pad_shape[1] = target_len - curr_len
                pad = torch.full(pad_shape, fill_value=fill_val, dtype=val.dtype, device=val.device)
                val = torch.cat([val, pad], dim=1)
            elif curr_len > target_len:
                val = val[:, :target_len]

    if target_tensor is not None and isinstance(target_tensor, torch.Tensor) and isinstance(val, torch.Tensor):
        if val.device != target_tensor.device or val.dtype != target_tensor.dtype:
            if not val.dtype.is_floating_point and target_tensor.dtype.is_floating_point:
                val = val.to(device=target_tensor.device, dtype=target_tensor.dtype)
            else:
                val = val.to(device=target_tensor.device)
        val_s0, target_s0 = int(val.shape[0]), int(target_tensor.shape[0])
        val_s1, target_s1 = int(val.shape[1]), int(target_tensor.shape[1])
        if (val_s0, val_s1) != (target_s0, target_s1):
            if val_s0 != target_s0:
                if val_s0 < target_s0:
                    rep = (target_s0 + val_s0 - 1) // max(1, val_s0)
                    val = val.repeat(rep, 1)[:target_s0]
                else:
                    val = val[:target_s0]
            if val_s1 != target_s1:
                if val_s1 < target_s1:
                    pad_shape = list(val.shape)
                    pad_shape[1] = target_s1 - val_s1
                    pad = torch.full(pad_shape, fill_value=fill_val, dtype=val.dtype, device=val.device)
                    val = torch.cat([val, pad], dim=1)
                else:
                    val = val[:, :target_s1]
    return val


def select_and_pad_tpu_data(data: TensorDict, *fields, target_tensor=None) -> dict:
    """Select and pad TensorDict fields on TPU."""
    from verl.workers.utils.padding import no_padding_2_padding  # Avoid circular import with padding.py

    is_tpu_packed = "tpu_custom_attention_mask" in data.keys()
    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)
    if isinstance(max_response_len, list | tuple):
        max_response_len = max(max_response_len) if len(max_response_len) > 0 else -1

    padded_dict = {}
    for k in fields:
        if k not in data.keys():
            continue
        val = data[k]
        if isinstance(val, NonTensorData):
            val = val.data

        fill_val = (
            1.0
            if k == "rollout_is_weights"
            else (True if k == "response_mask" and getattr(val, "dtype", None) == torch.bool else 0.0)
        )

        padded = None
        if is_tpu_packed:
            if isinstance(val, list | tuple):
                try:
                    val_t = torch.as_tensor(val)
                    padded = no_padding_2_padding(val_t, data)
                except Exception:
                    pass
            if padded is None:
                try:
                    padded = no_padding_2_padding(val, data)
                except Exception:
                    pass

        padded = pad_to_2d_tensor(
            padded if padded is not None else val,
            target_len=max_response_len,
            fill_val=fill_val,
            target_tensor=target_tensor,
        )
        padded_dict[k] = padded

    return padded_dict


def flatten_tpu_loss_mask(loss_mask: torch.Tensor, data: TensorDict, log_prob_flatten: torch.Tensor) -> torch.Tensor:
    """Flatten loss_mask using TPU binned packing active seqlens.

    Called in `verl/workers/utils/losses.py` during loss calculation when data.pad_mode=no_padding on TPU.
    """
    if isinstance(loss_mask, torch.Tensor) and not getattr(loss_mask, "is_nested", False):
        mbsz = loss_mask.shape[0]
        cu_seqlens_list = data["tpu_cu_seqlens"]
        if hasattr(cu_seqlens_list, "data"):
            cu_seqlens_list = cu_seqlens_list.data
        if not isinstance(cu_seqlens_list, list | tuple):
            cu_seqlens_list = [cu_seqlens_list]

        num_seqs_list = data["tpu_num_seqs"]
        if hasattr(num_seqs_list, "data"):
            num_seqs_list = num_seqs_list.data
        if not isinstance(num_seqs_list, list | tuple):
            num_seqs_list = [num_seqs_list]

        loss_mask_parts = []
        for row_idx in range(mbsz):
            cu_seqlens = cu_seqlens_list[row_idx]
            if isinstance(cu_seqlens, list):
                cu_seqlens = cu_seqlens[0]
            if not isinstance(cu_seqlens, torch.Tensor):
                cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

            num_seqs = num_seqs_list[row_idx]
            if isinstance(num_seqs, list):
                num_seqs = num_seqs[0]

            cu_seqlens_active = cu_seqlens[: num_seqs + 1]
            active_len = int(cu_seqlens_active[-1].item())

            loss_mask_parts.append(loss_mask[row_idx, :active_len])

        return torch.cat(loss_mask_parts, dim=0).to(log_prob_flatten.device)
    else:
        return loss_mask.values()


def get_packed_sequence_offsets_and_lens_tpu(
    tensor: torch.Tensor, data: TensorDict, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Extract and compute sequence offsets, response lengths, and maximum response length for packed data on TPU."""
    # Handle SFT fallback when "prompts" or "responses" are not in the TensorDict keys.
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
            # SFT fallback: response_lens is the sum of loss_mask/response_mask within each sequence
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
        # SFT Fallback: tensor is not nested, standard 2D tensor or packed 2D tensor
        mask_key = (
            "response_mask" if "response_mask" in data.keys() else ("loss_mask" if "loss_mask" in data.keys() else None)
        )
        if mask_key is not None:
            mask_val = data[mask_key]
            response_lens = mask_val.sum(dim=-1).to(device=values.device, dtype=torch.int64)
            if "input_ids" in data.keys():
                input_ids = data["input_ids"]
                prompt_lens = (
                    torch.tensor([input_ids.shape[1]] * input_ids.shape[0], device=values.device) - response_lens
                )
            else:
                prompt_lens = response_lens
            max_response_len = int(response_lens.max().item())
        else:
            prompt_lens = torch.tensor([tensor.shape[1] // 2] * tensor.shape[0], device=values.device)
            response_lens = torch.tensor([tensor.shape[1] // 2] * tensor.shape[0], device=values.device)
            max_response_len = tensor.shape[1] // 2

        sequence_lens = prompt_lens + response_lens
        sequence_offsets = sequence_lens.cumsum(dim=0)

    return sequence_offsets, response_lens, max_response_len
