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
"""
Contain small torch utilities
"""

import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizer

from verl.utils.device import get_device_name, get_torch_device

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


try:
    import torch_npu

    NPU_CROSS_ENTROPY_LOSS_AVAILABLE = hasattr(torch_npu, "npu_cross_entropy_loss")
except ImportError:
    NPU_CROSS_ENTROPY_LOSS_AVAILABLE = False


def gather_from_labels(data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Gather values from data tensor at positions specified by label indices.

    Selects elements from the last dimension of `data` based on indices in `label`.
    Commonly used to extract log-probabilities for specific token IDs from a
    vocabulary distribution.

    Args:
        data: Input tensor of shape (..., vocab_size) containing values to gather from.
        label: Index tensor of shape (...,) with values in range [0, vocab_size).

    Returns:
        torch.Tensor: Gathered values with shape (...,), same as label shape.

    Example:
        >>> logits = torch.randn(2, 3, 100)  # [batch, seq, vocab]
        >>> labels = torch.randint(0, 100, (2, 3))  # [batch, seq]
        >>> gathered = gather_from_labels(logits, labels)  # [batch, seq]
    """
    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output


def logprobs_from_logits(logits, labels, inplace_backward=True):
    """
    Compute per-token log-probabilities for the given labels.

    Uses a Flash-Attention–based cross-entropy (if available) for efficient backward,
    otherwise falls back to a standard log-softmax+gather approach.

    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.

    Returns:
        Tensor: Log-probabilities of the target labels, shape logits.shape[:-1].
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward=inplace_backward)
        output = output.view(*batch_dim)
    elif NPU_CROSS_ENTROPY_LOSS_AVAILABLE:
        output = logprobs_from_logits_torch_npu(logits, labels)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(
    logits: torch.Tensor, labels: torch.Tensor, inplace_backward: bool = True
) -> torch.Tensor:
    """Compute log-probabilities using Flash Attention's optimized cross-entropy.

    Uses the Flash Attention library's Triton-based cross-entropy implementation
    for efficient computation on NVIDIA GPUs.

    Args:
        logits: Model output logits of shape (batch_size, vocab_size).
        labels: Target token indices of shape (batch_size,).
        inplace_backward: If True, perform backward pass in-place for memory efficiency.

    Returns:
        torch.Tensor: Log-probabilities for target labels, shape (batch_size,).

    Raises:
        AssertionError: If flash-attn version < 2.4.3 (different return format).
    """
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(output, tuple), (
        "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    )
    return -output[0]


def logprobs_from_logits_torch_npu(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log-probabilities using Ascend NPU's optimized cross-entropy.

    Uses torch_npu's native cross-entropy implementation for efficient
    computation on Huawei Ascend NPU devices.

    Args:
        logits: Model output logits of shape (..., vocab_size).
        labels: Target token indices of shape (...,).

    Returns:
        torch.Tensor: Log-probabilities for target labels, same shape as labels.
    """
    batch_dim = logits.shape[:-1]
    logits = logits.reshape(-1, logits.shape[-1])
    loss, _, _, _ = torch_npu.npu_cross_entropy_loss(logits, labels.reshape(-1), reduction="none")
    return -loss.view(*batch_dim)


def logprobs_from_logits_naive(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log-probabilities using standard log-softmax approach.

    Simple implementation using PyTorch's log_softmax followed by gathering.
    Less memory-efficient than specialized implementations but works on all devices.

    Args:
        logits: Model output logits of shape (..., vocab_size).
        labels: Target token indices of shape (...,).

    Returns:
        torch.Tensor: Log-probabilities for target labels, same shape as labels.
    """
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels: torch.Tensor) -> torch.Tensor:
    """Memory-efficient log-probability computation using row-wise processing.

    Computes log-probabilities by processing one row at a time to reduce peak
    memory consumption. Uses logsumexp for float32/float64, falls back to
    log_softmax for bfloat16 due to numerical stability concerns.

    The mathematical identity used is: log_softmax(x_i) = x_i - logsumexp(x)

    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
            or (batch_size, vocab_size).
        labels: Target token indices matching logits shape without vocab dimension.

    Returns:
        torch.Tensor: Log-probabilities for target labels.

    Note:
        This implementation trades compute for memory by iterating over batch
        dimension, making it suitable for large vocabulary sizes.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels, strict=True):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels


def clip_by_value(x: torch.Tensor, tensor_min: torch.Tensor, tensor_max: torch.Tensor) -> torch.Tensor:
    """Clip tensor values to a range defined by tensor bounds.

    Extension of torch.clamp that supports tensor-valued min/max bounds
    instead of only scalar bounds.

    Args:
        x: Input tensor to clip.
        tensor_min: Minimum bound tensor (broadcastable to x).
        tensor_max: Maximum bound tensor (broadcastable to x).

    Returns:
        torch.Tensor: Clipped tensor with values in [tensor_min, tensor_max].

    See Also:
        https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate Shannon entropy from unnormalized logits.

    Computes H(p) = -sum(p * log(p)) using the numerically stable formula:
    entropy = logsumexp(logits) - sum(softmax(logits) * logits)

    Args:
        logits: Unnormalized log-probabilities of shape (..., vocab_size).

    Returns:
        torch.Tensor: Entropy values with shape (...,), one per distribution.
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def entropy_from_logits_with_chunking(logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Memory-efficient entropy calculation using chunked processing.

    Computes entropy by processing the batch in chunks to reduce peak memory
    usage. Useful for large batch sizes or when memory is constrained.

    Args:
        logits: Unnormalized log-probabilities of shape (batch_size, vocab_size).
        chunk_size: Number of samples to process at once. Defaults to 2048.

    Returns:
        torch.Tensor: Entropy values with shape (batch_size,).

    Note:
        Converts chunks to float32 for numerical stability during computation.
    """
    entropy = torch.zeros(logits.shape[0], device=logits.device)
    for i in range(0, logits.shape[0], chunk_size):
        logits_chunk = logits[i : i + chunk_size].float()
        pd_chunk = torch.nn.functional.softmax(logits_chunk, dim=-1)
        entropy_chunk = torch.logsumexp(logits_chunk, dim=-1) - torch.sum(pd_chunk * logits_chunk, dim=-1)
        entropy[i : i + chunk_size] = entropy_chunk
    return entropy


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Compute sum of tensor values where mask is True.

    NaN values outside the mask are replaced with zeros to prevent
    contaminating the sum.

    Args:
        values: Input tensor containing values to sum.
        mask: Boolean or numeric mask tensor (same shape as values).
            Non-zero values indicate elements to include.
        axis: Dimension(s) along which to sum. None sums all elements.

    Returns:
        torch.Tensor: Sum of masked values, reduced along specified axis.
    """
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions)
    valid_values = torch.where(mask.bool(), values, 0.0)
    return (valid_values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """
    Whiten `values` by normalizing with mean and variance computed over `mask`.

    Args:
        values (torch.Tensor): Input tensor.
        mask (torch.Tensor): Boolean tensor of same shape, selects elements for stats.
        shift_mean (bool): If True (default), output is zero-mean;
                           if False, the original mean is re-added after scaling.

    Returns:
        torch.Tensor: Whitened tensor of same shape as `values`.
    """
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_response_mask(response_id: torch.Tensor, eos_token: int | list[int] = 2, dtype=torch.int64):
    """
    end of sentence token can be int or list: 1 or [1, 2]
    e.g.
    response_id = torch.tensor([[20, 10, 34, 1, 0, 0, 0],
                                [78, 0, 76, 2, 1, 0, 0],
                                [23, 98, 1, 0, 0, 0, 0],
                                [33, 3, 98, 45, 1, 0, 0]])
    #eos_token=1
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    #eos_token=[1,2]
    response_mask:  tensor([[1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0]])
    """
    eos_mask = torch.isin(response_id, torch.tensor(eos_token, device=response_id.device)).int()
    return (eos_mask.cumsum(dim=1) - eos_mask).eq(0).to(dtype)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute the squared L2 norm of all gradients in a model.

    Sums the squared values of all gradient tensors across all parameters.
    Useful for monitoring gradient magnitudes during training.

    Args:
        model: PyTorch model with computed gradients.

    Returns:
        float: Sum of squared gradient values (not the square root).

    Note:
        Returns the squared norm, not the norm itself. To get the actual
        L2 norm, take the square root of the returned value.
    """
    total_grad_square = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    return total_grad_square


def broadcast_dict_tensor(tensors: dict[str, torch.Tensor] | TensorDict, src: int, group) -> None:
    """Broadcast all tensors in a dictionary from source rank to all ranks.

    Iterates over all tensors in the dictionary and broadcasts each one
    from the source rank to all other ranks in the process group.

    Args:
        tensors: Dictionary or TensorDict containing tensors to broadcast.
        src: Source rank from which to broadcast.
        group: Process group for the broadcast operation.

    Note:
        This implementation broadcasts tensors one at a time. Could be optimized
        to use a single broadcast with packed tensors.
    """
    for key in tensors.sorted_keys:
        torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=False)


def allgather_dict_tensors(
    tensors: dict[str, torch.Tensor] | TensorDict, size: int, group, dim: int = 0
) -> dict[str, torch.Tensor] | TensorDict:
    """Gather tensors from all ranks and concatenate them.

    Performs all_gather on each tensor in the dictionary and concatenates
    the results along the specified dimension.

    Args:
        tensors: Dictionary or TensorDict containing tensors to gather.
        size: Number of ranks in the process group.
        group: Process group for the all_gather operation.
        dim: Dimension along which to concatenate gathered tensors. Defaults to 0.

    Returns:
        Dictionary or TensorDict (matching input type) with gathered and
        concatenated tensors. Each tensor's size along `dim` is multiplied by `size`.

    Note:
        This implementation gathers tensors one at a time synchronously.
        Could be optimized using async ops or packed all_gather.
    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        val = tensors_as_dict[key]
        output[key] = [torch.empty_like(val) for _ in range(size)]
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output


def allgather_dict_into_dict(data: dict, group=None) -> dict:
    """allgather a dict into a dict of list

    Args:
        data: a dict
        group: the process group to allgather

    Returns: dict containing a list of the results from allgather

    """
    assert isinstance(data, dict), f"Expect data to be a dictionary, Got {type(data)}"

    group_size = torch.distributed.get_world_size(group=group)

    final_metrics = {}
    all_metrics_lst = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_metrics_lst, data, group=group)

    for all_metrics in all_metrics_lst:
        for key, val in all_metrics.items():
            if key not in final_metrics:
                final_metrics[key] = []
            final_metrics[key].append(val)
    return final_metrics


def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> list[TensorDict]:
    assert tensors.batch_size[0] % batch_size == 0, (
        f"input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}"
    )
    return tensors.split(batch_size)


def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    response_length = max(len(sub_list) for sub_list in response)
    target_length = max_length if max_length is not None and max_length > response_length else response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    # (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


def postprocess_data(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    """Process tokenizer outputs to consistent shapes via padding/truncation.

    Args:
        input_ids: Token indices [batch_size, seq_len]
        attention_mask: Mask [batch_size, seq_len]
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: "left", "right", "middle" or "error"

    Returns:
        (input_ids, attention_mask) padded/truncated to max_length
    """
    assert truncation in ["left", "right", "middle", "error"]
    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            input_ids = torch.cat([input_ids[:, :left_half], input_ids[:, -right_half:]], dim=-1)
            attention_mask = torch.cat([attention_mask[:, :left_half], attention_mask[:, -right_half:]], dim=-1)
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask


def tokenize_and_postprocess_data(
    prompt: str, tokenizer: PreTrainedTokenizer, max_length: int, pad_token_id: int, left_pad=True, truncation="error"
):
    """Tokenize text and process outputs to consistent tensor shapes.

    Args:
        prompt: Input text to tokenize
        tokenizer: HuggingFace tokenizer instance
        max_length: Target sequence length
        pad_token_id: Padding token ID
        left_pad: Pad left if True
        truncation: Truncation strategy ("left"/"right"/"error")

    Returns:
        Tuple of (input_ids, attention_mask) from postprocess_data
    """
    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]

    return postprocess_data(input_ids, attention_mask, max_length, pad_token_id, left_pad, truncation)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask, strict=True):
        no_padding_batch.append((ids[len(ids) - mask.sum() :]).cpu().numpy().tolist())
    return no_padding_batch


def log_probs_from_logits_response(input_ids, logits, response_length):
    """Compute the response log_probs from full logits. Note that logits = model(input_ids)

    Args:
        input_ids: [batch_size, seqlen]
        logits: [batch_size, seqlen, vocab_size]

    Returns:
        response_log_prob:
    """
    response_logits = logits[:, -response_length - 1 : -1]
    response = input_ids[:, -response_length:]
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    return response_log_prob


def log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Compute the log_probs from logits with rmpad logits and pad input. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size]
        response_length: int
    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output


def log_probs_from_logits_all_rmpad(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length):
    """Compute the log_probs from logits with rmpad input_ids and logits. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids_rmpad: [1, total_nnz]
        logits_rmpad: [total_nnz, vocab_size]
        indices: [total_nnz]
        batch_size: int
        seqlen: int
        response_length: int
    """
    from flash_attn.bert_padding import pad_input

    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # transpose back to [total_nnz, 1]
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output


def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    if temperature != 1.0:
        logits = logits.div_(temperature)  # inplace operation to avoid OOM
    # TODO: add them back
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    return logits


def calculate_sum_pi_squared_from_logits(logits: torch.Tensor):
    """
    Compute exact sum of squared probabilities from logits.
    Formula: Σπ² = exp(logsumexp(2*logits) - 2*logsumexp(logits))

    Used for optimal baseline variance reduction as described in
    "What Matters for Model Merging at Scale?" (arXiv:2410.03617)

    Args:
        logits: Logits tensor (..., vocab_size).

    Returns:
        Sum of squared probabilities tensor (...).
    """
    return torch.exp(torch.logsumexp(2.0 * logits, dim=-1) - 2.0 * torch.logsumexp(logits, dim=-1))


"""
Optimizer related
"""


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    init_lr_ratio: float = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        init_lr_ratio (:obj:`float`, `optional`, defaults to None):
            The initial lr ratio w.r.t the maximum.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    min_lr_ratio = 0.0 if min_lr_ratio is None else min_lr_ratio
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.0
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    init_lr_ratio = 0.0 if init_lr_ratio is None else init_lr_ratio
    assert init_lr_ratio >= 0 and init_lr_ratio <= 1.0

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return init_lr_ratio + (1.0 - init_lr_ratio) * (float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(min_lr_ratio, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    """
    Create a constant LR schedule with a linear warmup phase.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps (int): Number of steps to ramp up the LR from 0 to initial value.
        last_epoch (int, optional): The index of the last epoch when resuming training. Defaults to -1.

    Returns:
        LambdaLR: Scheduler that increases LR linearly during warmup, then holds it constant.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def get_wsd_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    stable_ratio: float = 0.9,
):
    """
    Create a Warmup-Stable-Decay learning rate scheduler.

    The schedule follows three phases:
    1. Warmup: Learning rate increases linearly from 0 to the initial LR
    2. Stable: Learning rate remains constant at the initial LR
    3. Decay: Learning rate decreases following a cosine curve to min_lr_ratio * initial LR

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum learning rate ratio w.r.t the initial learning rate.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule during decay phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        stable_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio of non-warmup steps that should maintain a constant learning rate.
            Set to 0.0 to behave exactly like cosine schedule.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    remaining_steps = max(0, num_training_steps - num_warmup_steps)
    num_stable_steps = int(remaining_steps * stable_ratio)
    num_decay_steps = remaining_steps - num_stable_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        if current_step < num_training_steps:
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


@contextmanager
def check_device_is_available():
    """
    Some modules must be imported after CUDA is initialized. Such as sglang's sharding manager.

    This context manager checks if CUDA is available and raises an error if it is not.
    """
    if not get_torch_device().is_available():
        raise RuntimeError("Device {} must be initialized before importing this module.".format(get_device_name()))

    yield


def distributed_mean_max_min_std(local_tensor, compute_max=True, compute_min=True, compute_std=True):
    """Compute distributed statistics across all processes.

    Args:
        local_tensor: Tensor containing local values
        compute_max: Include maximum value calculation
        compute_min: Include minimum value calculation
        compute_std: Include standard deviation calculation

    Returns:
        Tuple containing (mean, max, min, std) in this order. None for disabled metrics.
    """
    # Sum the local tensor across all processes
    local_sum = torch.sum(local_tensor)
    local_num = torch.tensor(torch.numel(local_tensor), device=get_device_name())

    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_num, op=torch.distributed.ReduceOp.SUM)

    global_mean = local_sum / local_num

    if compute_max:
        local_max = torch.max(local_tensor)
        torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX)
    else:
        local_max = None

    if compute_min:
        local_min = torch.min(local_tensor)
        torch.distributed.all_reduce(local_min, op=torch.distributed.ReduceOp.MIN)
    else:
        local_min = None

    if compute_std:
        square_diff = torch.sum(torch.pow(local_tensor - global_mean, 2))
        torch.distributed.all_reduce(square_diff, op=torch.distributed.ReduceOp.SUM)
        global_std = torch.sqrt(square_diff / (local_num - 1))
    else:
        global_std = None

    return global_mean, local_max, local_min, global_std


def distributed_masked_mean(local_tensor, local_mask):
    """Compute global mean of non-masked elements across distributed processes.

    Args:
        local_tensor (torch.Tensor): Input tensor with local values
        local_mask (torch.Tensor): Binary mask (1=valid, 0=ignore) matching local_tensor shape

    Returns:
        torch.Tensor: Global mean of all valid elements across processes
    """
    local_tensor = local_tensor * local_mask

    local_sum = torch.sum(local_tensor)
    local_num = torch.sum(local_mask)

    torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(local_num, op=torch.distributed.ReduceOp.SUM)

    global_mean = local_sum / local_num
    return global_mean


def expand_as_nested(tensor: torch.Tensor, nested_tensor: torch.Tensor) -> torch.Tensor:
    """

    Args:
        tensor: a tensor with shape (bsz,)
        nested_tensor: a nested tensor with shape (bsz, xxx)

    Returns:
        a tensor with the same shape as nested_tensor

    """
    assert nested_tensor.is_nested, "nested_tensor must be nested"
    assert tensor.shape[0] == nested_tensor.shape[0], (
        f"The batch shape must be the same. Got {tensor.shape[0]} vs {nested_tensor.shape[0]}"
    )
    assert len(tensor.shape) == 1, "The ndim of tensor must be 1"
    assert len(nested_tensor.shape) == 2, "The ndim of nested_tensor must be 2"

    offsets = nested_tensor.offsets()
    seqlens = offsets.diff()
    output = torch.repeat_interleave(tensor, seqlens, dim=0)
    output = torch.nested.nested_tensor_from_jagged(values=output, offsets=offsets)
    return output


@contextmanager
def use_original_torch_compile():
    """torch.compile might be replaced by mindspeed on NPU, this contextmanager
    can revert torch.compile temporarily.
    """
    try:
        from mindspeed.patch_utils import MindSpeedPatchesManager

        compile_patch = None
        for patch in MindSpeedPatchesManager.patches_info.values():
            if patch.orig_module_name == "torch" and patch.orig_func_name == "compile":
                if patch.is_applied():
                    compile_patch = patch
                break
        if compile_patch is not None:
            compile_patch.remove_patch()
            yield
            compile_patch.apply_patch()
        else:
            yield
    except Exception:
        yield
