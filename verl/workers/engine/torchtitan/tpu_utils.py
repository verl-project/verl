# Copyright 2024-2025 BAAI and Google LLC
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
"""TPU sequence packing utilities, attention monkey patches, and memory layout helpers."""

import logging
import os
from typing import Any

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.torch_functional import logprobs_from_logits

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default sequence packing constants
DEFAULT_MAX_BIN_LEN = 1024
DEFAULT_MAX_PROMPT_LEN = 512
DEFAULT_MAX_RESPONSE_LEN = 512
SEQUENCE_ALIGNMENT_MULTIPLE = 128


def slice_metadata_item(data: TensorDict, k: str, i: int, batch_size: int, attn_mask: torch.Tensor = None):
    """Slices a metadata item from a TensorDict for the i-th batch element."""
    val = data.get(k, None)
    if val is None:
        return None
    if isinstance(val, NonTensorData):
        val = val.data

    # Handle NestedTensor or structures with offsets
    if getattr(val, "is_nested", False) or hasattr(val, "offsets"):
        try:
            return val.unbind()[i]
        except Exception:
            try:
                return val[i]
            except Exception:
                pass

    # Slice prompts and responses to exact active lengths if standard padded 2D tensors
    if k == "prompts" and isinstance(val, torch.Tensor) and not getattr(val, "is_nested", False):
        if attn_mask is not None and val.ndim >= 2:
            prompt_len_active = int(attn_mask[i, : val.shape[-1]].sum().item())
            return val[i, :prompt_len_active]
        elif val.ndim >= 2:
            return val[i]
    elif k == "responses" and isinstance(val, torch.Tensor) and not getattr(val, "is_nested", False):
        if attn_mask is not None and val.ndim >= 2:
            prompt_val = data.get("prompts", None)
            if isinstance(prompt_val, NonTensorData):
                prompt_val = prompt_val.data
            prompt_len_i = 0
            if getattr(prompt_val, "is_nested", False) or hasattr(prompt_val, "offsets"):
                prompt_len_i = prompt_val.unbind()[i].shape[-1]
            elif isinstance(prompt_val, torch.Tensor) and prompt_val.ndim >= 2:
                prompt_len_i = int(attn_mask[i, : prompt_val.shape[-1]].sum().item())

            resp_mask = attn_mask[i, prompt_len_i : prompt_len_i + val.shape[-1]]
            response_len_active = int(resp_mask.sum().item())
            return val[i, :response_len_active]
        elif val.ndim >= 2:
            return val[i]

    # Handle standard PyTorch Tensor
    if isinstance(val, torch.Tensor):
        if val.ndim > 0 and val.shape[0] == batch_size:
            return val[i]
        return val

    # Handle list or tuple
    if isinstance(val, list | tuple) and len(val) == batch_size:
        return val[i]

    raw_val = tu.get_non_tensor_data(data=data, key=k, default=None)
    if isinstance(raw_val, NonTensorData):
        raw_val = raw_val.data
    if getattr(raw_val, "is_nested", False) or hasattr(raw_val, "offsets"):
        try:
            return raw_val.unbind()[i]
        except Exception:
            try:
                return raw_val[i]
            except Exception:
                pass
    if isinstance(raw_val, list | tuple) and len(raw_val) == batch_size:
        return raw_val[i]
    return raw_val


def tpu_binned_pack_tensordict(
    data: TensorDict, max_bin_len: int = DEFAULT_MAX_BIN_LEN
) -> tuple[list[TensorDict], list[list[int]]]:
    """Packs variable-length sequence data into static micro-batches of shape [1, max_bin_len].

    On Google TPU execution, variable-length sequence shapes trigger dynamic graph recompilations.
    This host-side CPU sequence packer bins active sequence tokens into static micro-batches of
    shape [1, max_bin_len] to eliminate recompilation overhead.
    """
    data = data.cpu()
    batch_size = data.batch_size[0]

    attention_mask_key = "attention_mask" if "attention_mask" in data.keys() else "loss_mask"
    attn_mask = data[attention_mask_key]

    active_lengths = [int(attn_mask[i].sum().item()) for i in range(batch_size)]

    keys_to_pack = []
    keys_to_preserve = []

    sequence_keys = {
        "input_ids",
        "position_ids",
        "attention_mask",
        "loss_mask",
        "labels",
        "log_prob",
        "log_probs",
        "old_log_prob",
        "old_log_probs",
        "ref_log_prob",
        "advantages",
        "returns",
        "values",
        "response_mask",
        "rollout_is_weights",
    }

    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[0] == batch_size:
            if k in sequence_keys:
                keys_to_pack.append(k)
                continue
        keys_to_preserve.append(k)

    sliced_sequences = []
    for i in range(batch_size):
        length = active_lengths[i]

        seq_data = {}
        for k in keys_to_pack:
            val = data[k]
            if getattr(val, "is_nested", False):
                seq_tensor = val.unbind()[i]
                if seq_tensor.dim() == 1:
                    seq_data[k] = seq_tensor[:length]
                else:
                    seq_data[k] = seq_tensor[..., :length]
            else:
                if val.dim() == 2:
                    seq_data[k] = val[i, :length]
                else:
                    seq_data[k] = val[i, ..., :length]

        metadata_dict = {}
        for k in keys_to_preserve:
            metadata_dict[k] = slice_metadata_item(data, k, i, batch_size, attn_mask=attn_mask)

        sliced_sequences.append(
            {
                "seq_data": seq_data,
                "metadata": metadata_dict,
                "length": length,
                "original_idx": i,
            }
        )

    # Sort sequences descending by length for dense bin-packing
    sorted_seqs = sorted(sliced_sequences, key=lambda s: s["length"], reverse=True)

    bins: list[list[dict]] = []
    bin_lengths: list[int] = []

    for seq in sorted_seqs:
        seq_len_active = seq["length"]
        if seq_len_active == 0:
            continue
        assert seq_len_active <= max_bin_len, f"Sequence length {seq_len_active} exceeds max_bin_len {max_bin_len}"

        placed = False
        for i, current_len in enumerate(bin_lengths):
            if current_len + seq_len_active <= max_bin_len:
                bins[i].append(seq)
                bin_lengths[i] += seq_len_active
                placed = True
                break

        if not placed:
            bins.append([seq])
            bin_lengths.append(seq_len_active)

    logger.debug(f"TPU Binned Packer: input batch_size={batch_size}, total bins={len(bins)}")
    packed_micro_batches = []
    batch_idx_list = []
    for bin_idx, (bin_seqs, active_len) in enumerate(zip(bins, bin_lengths, strict=False)):
        batch_idx_list.append([seq["original_idx"] for seq in bin_seqs])
        logger.debug(f"Bin #{bin_idx}: {len(bin_seqs)} sequences, active tokens={active_len}/{max_bin_len}")

        concatenated_td = {}
        for k in keys_to_pack:
            first_seq = bin_seqs[0]["seq_data"][k]
            concat_dim = -1 if first_seq.dim() > 1 else 0
            concatenated_td[k] = torch.cat([seq["seq_data"][k] for seq in bin_seqs], dim=concat_dim)

        slack_len = max_bin_len - active_len
        offsets = [0]
        curr = 0
        for seq in bin_seqs:
            curr += seq["length"]
            offsets.append(curr)

        if slack_len > 0:
            offsets.append(max_bin_len)
            for k in keys_to_pack:
                val = concatenated_td[k]
                pad_val = 0
                if k == "input_ids":
                    pad_val = int(tu.get_non_tensor_data(data=data, key="pad_token_id", default=0))

                if val.dim() == 1:
                    pad_tensor = torch.full((slack_len,), pad_val, dtype=val.dtype)
                    concatenated_td[k] = torch.cat([val, pad_tensor], dim=0)
                else:
                    pad_shape = list(val.shape)
                    pad_shape[-1] = slack_len
                    pad_tensor = torch.full(pad_shape, pad_val, dtype=val.dtype)
                    concatenated_td[k] = torch.cat([val, pad_tensor], dim=-1)

        cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
        positions = torch.arange(max_bin_len)
        seq_indices = (positions.unsqueeze(1) >= cu_seqlens.unsqueeze(0)).sum(dim=1) - 1
        same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
        causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)

        padding_seq_idx = (len(cu_seqlens) - 2) if slack_len > 0 else (len(cu_seqlens) - 1)
        tpu_num_seqs_val = padding_seq_idx

        non_pad_mask = seq_indices < padding_seq_idx

        attention_mask = same_seq_mask & causal_mask & non_pad_mask.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        mb_td = TensorDict({k: v.unsqueeze(0) for k, v in concatenated_td.items()}, batch_size=[1])

        mb_td["tpu_custom_attention_mask"] = attention_mask
        mb_td["tpu_cu_seqlens"] = NonTensorData([cu_seqlens])
        mb_td["tpu_num_seqs"] = NonTensorData([tpu_num_seqs_val])

        for k in keys_to_preserve:
            vals = [seq["metadata"][k] for seq in bin_seqs]
            if k in ["prompts", "responses"]:
                mb_td[k] = NonTensorData(vals)
            elif all(isinstance(v, torch.Tensor) for v in vals):
                shapes = [v.shape for v in vals]
                if all(s == shapes[0] for s in shapes):
                    stacked = torch.stack(vals, dim=0)
                    mb_td[k] = stacked.unsqueeze(0)
                else:
                    mb_td[k] = NonTensorData(vals)
            else:
                mb_td[k] = NonTensorData(vals)

        packed_micro_batches.append(mb_td)

    return packed_micro_batches, batch_idx_list


def prepare_tpu_binned_pack_micro_batches(data: TensorDict) -> tuple[list[TensorDict], list[list[int]]]:
    """Prepares static sequence-packed micro-batches synchronized across data-parallel ranks on TPU."""
    max_length = tu.get_non_tensor_data(data=data, key="max_length", default=None)
    if max_length is not None:
        max_bin_len = max_length
    else:
        max_token_len = tu.get_non_tensor_data(data=data, key="max_token_len_per_gpu", default=None)
        if max_token_len is not None:
            max_bin_len = max_token_len
        else:
            max_prompt_length = tu.get_non_tensor_data(
                data=data, key="max_prompt_length", default=DEFAULT_MAX_PROMPT_LEN
            )
            max_response_length = tu.get_non_tensor_data(
                data=data, key="max_response_length", default=DEFAULT_MAX_RESPONSE_LEN
            )
            max_bin_len = max_prompt_length + max_response_length

    micro_batches, batch_idx_list = tpu_binned_pack_tensordict(data=data, max_bin_len=max_bin_len)

    num_bins = len(micro_batches)
    if torch.distributed.is_initialized():
        num_bins_tensor = torch.tensor([num_bins], dtype=torch.int32)
        torch.distributed.all_reduce(num_bins_tensor, op=torch.distributed.ReduceOp.MAX)
        global_max_bins = num_bins_tensor.item()
    else:
        global_max_bins = num_bins

    mbsz = tu.get_non_tensor_data(data=data, key="micro_batch_size_per_gpu", default=1)
    if global_max_bins % mbsz != 0:
        global_max_bins = ((global_max_bins // mbsz) + 1) * mbsz

    if num_bins < global_max_bins:
        last_batch = micro_batches[-1]
        last_indices = batch_idx_list[-1]
        for _ in range(global_max_bins - num_bins):
            dummy_batch = last_batch.clone()
            if "loss_mask" in dummy_batch.keys():
                dummy_batch["loss_mask"] = torch.zeros_like(dummy_batch["loss_mask"])
            micro_batches.append(dummy_batch)
            batch_idx_list.append([-1] * len(last_indices))

    if mbsz > 1:
        grouped_batches = []
        grouped_idx_list = []
        for idx in range(0, len(micro_batches), mbsz):
            batch_slice = micro_batches[idx : idx + mbsz]
            idx_slice = batch_idx_list[idx : idx + mbsz]

            keys_to_pop = []
            for k in list(batch_slice[0].keys()):
                if k in ("tpu_cu_seqlens", "tpu_num_seqs"):
                    keys_to_pop.append(k)
                    continue

                all_tensors = True
                shapes = []
                for mb in batch_slice:
                    v = mb.get(k, None)
                    if v is None or not isinstance(v, torch.Tensor) or isinstance(v, NonTensorData):
                        all_tensors = False
                        break
                    shapes.append(v.shape)

                if not all_tensors:
                    keys_to_pop.append(k)
                else:
                    first_shape_suffix = shapes[0][1:]
                    for sh in shapes[1:]:
                        if sh[1:] != first_shape_suffix:
                            keys_to_pop.append(k)
                            break

            combined_metadata = {}
            for k in keys_to_pop:
                combined_list = []
                for mb in batch_slice:
                    mb_val = mb.pop(k, None)
                    if mb_val is not None:
                        if isinstance(mb_val, NonTensorData):
                            raw = mb_val.data
                            if isinstance(raw, list | tuple):
                                combined_list.extend(raw)
                            else:
                                combined_list.append(raw)
                        else:
                            if isinstance(mb_val, list | tuple):
                                combined_list.extend(mb_val)
                            else:
                                combined_list.append(mb_val)
                combined_metadata[k] = combined_list

            grouped_td = torch.cat(batch_slice, dim=0)

            for k, combined_list in combined_metadata.items():
                if len(combined_list) > 0:
                    grouped_td[k] = NonTensorData(combined_list)

            grouped_batches.append(grouped_td)
            grouped_idx_list.append([i for sub in idx_slice for i in sub])
        micro_batches = grouped_batches
        batch_idx_list = grouped_idx_list

    return micro_batches, batch_idx_list


def prepare_tpu_packed_outputs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    micro_batch: TensorDict,
    temperature: float,
    calculate_entropy: bool,
    entropy_checkpointing: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reconstructs log_probs and entropy NestedTensors from sliced active tokens under TPU static sequence packing."""
    mbsz = logits.shape[0]

    cu_seqlens_list = micro_batch["tpu_cu_seqlens"]
    if hasattr(cu_seqlens_list, "data"):
        cu_seqlens_list = cu_seqlens_list.data
    if not isinstance(cu_seqlens_list, list | tuple):
        cu_seqlens_list = [cu_seqlens_list]

    num_seqs_list = micro_batch["tpu_num_seqs"]
    if hasattr(num_seqs_list, "data"):
        num_seqs_list = num_seqs_list.data
    if not isinstance(num_seqs_list, list | tuple):
        num_seqs_list = [num_seqs_list]

    log_probs_all_rows = []
    entropy_all_rows = []
    combined_offsets = [0]
    current_offset = 0

    for row_idx in range(mbsz):
        cu_seqlens = cu_seqlens_list[row_idx]
        if hasattr(cu_seqlens, "data"):
            cu_seqlens = cu_seqlens.data
        if isinstance(cu_seqlens, list):
            cu_seqlens = (
                cu_seqlens[0] if len(cu_seqlens) > 0 else torch.tensor([0, DEFAULT_MAX_BIN_LEN], dtype=torch.int32)
            )
        if hasattr(cu_seqlens, "data"):
            cu_seqlens = cu_seqlens.data
        if not isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)

        num_seqs = num_seqs_list[row_idx]
        if hasattr(num_seqs, "data"):
            num_seqs = num_seqs.data
        if isinstance(num_seqs, list):
            num_seqs = num_seqs[0] if len(num_seqs) > 0 else 1
        if hasattr(num_seqs, "data"):
            num_seqs = num_seqs.data

        cu_seqlens_active = cu_seqlens[: num_seqs + 1]
        active_len = int(cu_seqlens_active[-1].item())

        logits_rmpad = logits[row_idx, :active_len, :] / temperature
        labels_rmpad = labels[row_idx, :active_len]

        log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=labels_rmpad)
        log_probs_all_rows.append(log_probs_rmpad)

        if calculate_entropy:
            if not entropy_checkpointing:
                entropy_rmpad = verl_F.entropy_from_logits(logits_rmpad)
            else:
                entropy_rmpad = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits_rmpad)
            entropy_all_rows.append(entropy_rmpad)

        for i in range(1, len(cu_seqlens_active)):
            combined_offsets.append(current_offset + int(cu_seqlens_active[i].item()))
        current_offset += active_len

    log_probs_flat_cat = torch.cat(log_probs_all_rows, dim=0)
    cu_seqlens_combined = torch.tensor(combined_offsets, dtype=torch.int32, device=log_probs_flat_cat.device)

    log_probs = torch.nested.nested_tensor_from_jagged(log_probs_flat_cat, cu_seqlens_combined)

    entropy = None
    if calculate_entropy:
        entropy_flat_cat = torch.cat(entropy_all_rows, dim=0)
        entropy = torch.nested.nested_tensor_from_jagged(entropy_flat_cat, cu_seqlens_combined)

    return log_probs, entropy


def monkey_patch_varlen_attention_tpu():
    """Patches TorchTitan's VarlenAttention forward method to use native scaled_dot_product_attention on TPU."""
    try:
        from torchtitan.models.common.attention import VarlenAttention

        def tpu_varlen_forward(self, xq, xk, xv, *, attention_masks, scale=None, **kwargs):
            if attention_masks.dim() == 4:
                mask = attention_masks.to(torch.bool)
            elif hasattr(attention_masks, "cu_seq_q"):
                total_tokens = xq.shape[1]
                cu_seqs = attention_masks.cu_seq_q

                positions = torch.arange(total_tokens, device=xq.device)
                seq_indices = (positions.unsqueeze(1) >= cu_seqs.unsqueeze(0)).sum(dim=1) - 1
                same_seq_mask = seq_indices.unsqueeze(1) == seq_indices.unsqueeze(0)
                causal_mask = positions.unsqueeze(1) >= positions.unsqueeze(0)

                mask = same_seq_mask & causal_mask
                mask = mask.unsqueeze(0).unsqueeze(0)
            else:
                seq_len = xq.shape[1]
                positions = torch.arange(seq_len, device=xq.device)
                causal_mask = (positions.unsqueeze(1) >= positions.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

                padding_mask = attention_masks.unsqueeze(1).unsqueeze(2).to(torch.bool)
                mask = causal_mask & padding_mask

            q = xq.transpose(1, 2)
            k = xk.transpose(1, 2)
            v = xv.transpose(1, 2)

            if q.shape[1] != k.shape[1]:
                num_repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(num_repeat, dim=1)
                v = v.repeat_interleave(num_repeat, dim=1)

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            return attn_out.transpose(1, 2)

        VarlenAttention.forward = tpu_varlen_forward
        logger.info("Successfully patched VarlenAttention.forward for TPU execution.")
    except Exception as e:
        logger.warning(f"Failed to patch VarlenAttention: {e}")


def compute_global_batch_num_tokens(data: TensorDict, dp_group, tp_size: int) -> Any:
    """Computes global batch token count for loss normalization on TPU.

    On TPU, performing CPU-side all-reduce for loss normalization avoids graph desynchronization
    and JIT compilation lockups across ranks.
    """
    batch_num_tokens = data["loss_mask"].sum().cpu()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM)
        batch_num_tokens = batch_num_tokens / tp_size
    return batch_num_tokens.item()


def synchronize_tpu_loss(loss: torch.Tensor):
    """Materializes forward graph loss without blocking to split XLA forward and backward compilation passes."""
    try:
        from torch_tpu._internal.sync import synchronize

        synchronize(loss, wait=False)
    except ImportError:
        pass


def compute_tpu_max_seq_len(input_ids: torch.Tensor) -> int:
    """Rounds up sequence length to a multiple of 128 to ensure memory stride alignment on TPU."""
    offsets = input_ids.offsets()
    offsets_tpu = torch.empty(offsets.shape, dtype=offsets.dtype, device=offsets.device)
    offsets_tpu.copy_(offsets)
    max_seq_len = int(max(offsets_tpu.cpu().diff()))
    max_seq_len = (
        (max_seq_len + SEQUENCE_ALIGNMENT_MULTIPLE - 1) // SEQUENCE_ALIGNMENT_MULTIPLE
    ) * SEQUENCE_ALIGNMENT_MULTIPLE
    return max_seq_len


def reconstruct_tpu_packed_metadata_tensors(micro_batch: TensorDict, device_id: Any):
    """Reconstructs nested prompts/responses tensors on device from host NonTensorData lists."""
    for k in ["prompts", "responses"]:
        if k in micro_batch.keys():
            val = micro_batch[k]
            if isinstance(val, NonTensorData):
                val = val.data
            if isinstance(val, list):
                flat_vals = [v.values() if getattr(v, "is_nested", False) else v for v in val]
                flat_vals = [f.to(device_id) for f in flat_vals]
                nested_val = torch.nested.as_nested_tensor(flat_vals, layout=torch.jagged)
                micro_batch._tensordict[k] = nested_val


def unwrap_metadata(val):
    """Recursively unwraps metadata values (lists, single-element tensors) to standard Python types."""
    if isinstance(val, list):
        if len(val) > 0:
            return unwrap_metadata(val[0])
        return None
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            return val.item()
        elif val.numel() > 1:
            return val.flatten()[0].item()
    return val


def prepare_tpu_model_outputs_if_packed(
    logits: torch.Tensor,
    labels: torch.Tensor,
    micro_batch: TensorDict,
    temperature: float,
    calculate_entropy: bool,
    entropy_checkpointing: bool,
):
    """If the micro-batch uses TPU static sequence packing, reconstructs log_probs and entropy NestedTensors."""
    if "tpu_custom_attention_mask" in micro_batch.keys():
        log_probs, entropy = prepare_tpu_packed_outputs(
            logits=logits,
            labels=labels,
            micro_batch=micro_batch,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
            entropy_checkpointing=entropy_checkpointing,
        )
        return log_probs, entropy
    return None


def safe_to_padded_tensor(nt: Any, padding: Any = 0, output_size: Any = None) -> torch.Tensor:
    """Safely converts a NestedTensor to a padded dense tensor on TPU.

    PyTorch TPU currently lacks native C++ kernel support for `aten::_jagged_to_padded_dense_forward`.
    Falling back to `unbind()` + tensor slice assignment avoids operator runtime errors on TPU.
    """
    if not getattr(nt, "is_nested", False):
        return nt
    try:
        return torch.nested.to_padded_tensor(nt, padding=padding, output_size=output_size)
    except Exception:
        tensors = nt.unbind()
        if not tensors:
            return torch.empty(output_size, device=nt.device, dtype=nt.dtype)
        if output_size is not None and len(output_size) == 3:
            out = torch.full(output_size, padding, device=tensors[0].device, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                out[i, :, : t.shape[-1]] = t
        elif output_size is not None:
            out = torch.full(output_size, padding, device=tensors[0].device, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                out[i, : t.shape[-1]] = t
        else:
            batch_size = len(tensors)
            max_len = max(t.shape[-1] for t in tensors)
            out = torch.full((batch_size, max_len), padding, device=tensors[0].device, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                out[i, : t.shape[-1]] = t
        return out
