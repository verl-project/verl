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

from __future__ import annotations

import torch
import torch.nn.functional as F
from prefix_grouper import PrefixGrouper
from tensordict.tensorclass import NonTensorData

from verl.utils.torch_functional import logprobs_from_logits


def _to_padded_tensor(tensor: torch.Tensor, padding_value: int | bool) -> torch.Tensor:
    if tensor.is_nested:
        return tensor.to_padded_tensor(padding=padding_value)
    return tensor


def build_position_ids_for_prefix_grouper(prefix_grouper: PrefixGrouper) -> torch.Tensor:
    """Build position_ids for PrefixGrouper where each response restarts from prefix_len."""
    num_samples = len(prefix_grouper.group_info)
    max_len = prefix_grouper.padding_mask.size(1)
    device = prefix_grouper.padding_mask.device

    position_ids = torch.zeros(num_samples, max_len, dtype=torch.long, device=device)

    for i, group in enumerate(prefix_grouper.group_info):
        prefix_len = group.prefix_len

        position_ids[i, :prefix_len] = torch.arange(prefix_len, device=device)
        cur_pos = prefix_len
        for suffix_len in group.suffix_lens:
            if suffix_len > 0:
                position_ids[i, cur_pos : cur_pos + suffix_len] = torch.arange(
                    prefix_len, prefix_len + suffix_len, device=device
                )
                cur_pos += suffix_len

    return position_ids


def build_pg_from_micro_batch(
    micro_batch: dict,
    pad_token_id: int,
    padding_mode: str = "right",
):
    """Build PrefixGrouper from micro_batch dict containing prompts, responses, response_mask, uid."""
    prompts = _to_padded_tensor(micro_batch["prompts"], pad_token_id)
    responses = _to_padded_tensor(micro_batch["responses"], pad_token_id)
    response_mask = _to_padded_tensor(micro_batch["response_mask"], False)
    uids = micro_batch["uid"]
    # Engine dispatch wraps non-tensor fields; unwrap UIDs before comparing group boundaries.
    if isinstance(uids, NonTensorData):
        uids = uids.data

    bs = responses.size(0)

    group_sizes = []
    cur = 1
    for i in range(1, bs):
        if uids[i] == uids[i - 1]:
            cur += 1
        else:
            group_sizes.append(cur)
            cur = 1
    group_sizes.append(cur)

    prefix_indices = []
    cursor = 0
    for gs in group_sizes:
        prefix_indices.append(cursor)
        cursor += gs
    prefix_indices = torch.tensor(prefix_indices, device=prompts.device)

    prefix_ids = prompts.index_select(0, prefix_indices)
    prefix_mask = prefix_ids.ne(pad_token_id)

    prefix_grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=group_sizes,
        padding_mode=padding_mode,
        device=prompts.device,
    )

    concat_input_ids = prefix_grouper.concat_input(prefix_ids, prefix_mask, responses, response_mask)

    attention_mask = prefix_grouper.padding_mask

    position_ids = build_position_ids_for_prefix_grouper(prefix_grouper)

    return (
        prefix_grouper,
        concat_input_ids,
        attention_mask,
        position_ids,
        responses,
        response_mask,
    )


def attach_prefix_grouper_forward_args(
    prefix_grouper: PrefixGrouper,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    temperature: float,
    calculate_entropy: bool,
) -> None:
    """Attach response-only fused projection inputs to a PrefixGrouper call."""
    prefix_grouper._verl_completion_ids = completion_ids
    prefix_grouper._verl_completion_mask = completion_mask
    prefix_grouper._verl_temperature = max(float(temperature), 1e-8)
    prefix_grouper._verl_calculate_entropy = calculate_entropy


def pg_forward(
    model,
    prefix_grouper,
    concat_input_ids,
    attention_mask,
    position_ids,
    completion_ids,
    completion_mask,
    *,
    temperature=1.0,
    padding_mode="right",
    include_prefix_last=1,
    calculate_entropy=False,
    entropy_fn=None,
):
    logits = model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        prefix_grouper=prefix_grouper,
    ).logits

    prefix_out, prefix_mask, suffix_out_raw, suffix_mask_raw = prefix_grouper.split_output(
        logits, include_prefix_last=include_prefix_last
    )

    completion_ids_right = prefix_grouper.convert_padding(
        completion_ids,
        completion_mask,
        padding_mode=padding_mode,
    )

    suffix_out = suffix_out_raw[:, :-1].float()
    suffix_mask = suffix_mask_raw[:, 1:]

    suffix_out /= temperature

    log_probs = logprobs_from_logits(suffix_out, completion_ids_right)

    entropy = None
    if calculate_entropy and entropy_fn is not None:
        entropy = entropy_fn(suffix_out)

    return log_probs, entropy, suffix_mask


def response_output_to_nested(
    output: torch.Tensor,
    response_mask: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Place response outputs into full-sequence jagged tensors.

    `no_padding_2_padding` expects model outputs aligned to every original
    prompt+response sequence. PrefixGrouper computes response positions only,
    so prepend prompt placeholders and append one final placeholder.
    """
    if not input_ids.is_nested:
        raise ValueError("PrefixGrouper requires jagged input_ids")

    sequence_lens = input_ids.offsets().diff()
    response_lens = response_mask.sum(dim=-1).to(sequence_lens.device)
    prompt_lens = sequence_lens - response_lens
    if torch.any(prompt_lens <= 0):
        raise ValueError(f"PrefixGrouper requires non-empty prompts, got {prompt_lens}")

    rows = []
    for row, (prompt_len, response_len) in enumerate(zip(prompt_lens.tolist(), response_lens.tolist(), strict=True)):
        response_output = output[row, :response_len]
        rows.append(F.pad(response_output, (prompt_len - 1, 1)))

    values = torch.cat(rows, dim=0)
    return torch.nested.nested_tensor_from_jagged(values, input_ids.offsets())


def forward_micro_batch_with_prefix_grouper(
    micro_batch: dict,
    model,
    temperature: float,
    calculate_entropy: bool,
    device_name: str,
    param_dtype,
    use_chunking_entropy: bool = False,
):
    """
    Forward pass using PrefixGrouper for shared-prefix optimization.

    Args:
        micro_batch: Dict containing prompts, responses, response_mask, uid, etc.
        model: The actor module.
        temperature: Temperature for logits scaling.
        calculate_entropy: Whether to compute entropy.
        device_name: Device name for autocast.
        param_dtype: Parameter dtype for autocast.
        use_chunking_entropy: Whether to use chunking entropy function.

    Returns:
        tuple: (entropy, log_probs) where entropy may be None if not calculated.
    """
    import verl.utils.torch_functional as verl_F

    entropy_fn = None
    if calculate_entropy:
        if use_chunking_entropy:
            entropy_fn = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_fn = verl_F.entropy_from_logits

    pad_token_id = micro_batch.get("pad_token_id", 0)

    (
        prefix_grouper,
        concat_input_ids,
        attention_mask,
        position_ids,
        responses,
        response_mask,
    ) = build_pg_from_micro_batch(
        micro_batch,
        pad_token_id=pad_token_id,
        padding_mode="right",
    )

    with torch.autocast(device_type=device_name, dtype=param_dtype):
        log_probs, entropy, suffix_mask_from_pg = pg_forward(
            model=model,
            prefix_grouper=prefix_grouper,
            concat_input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            completion_ids=responses,
            completion_mask=response_mask,
            temperature=temperature,
            padding_mode="right",
            include_prefix_last=1,
            calculate_entropy=calculate_entropy,
            entropy_fn=entropy_fn,
        )

    # Zero out padding positions
    padding_mask = suffix_mask_from_pg == 0
    log_probs = log_probs.masked_fill(padding_mask, 0.0)
    if entropy is not None:
        entropy = entropy.masked_fill(padding_mask, 0.0)

    # Pad to target response length if needed
    target_response_length = responses.size(1)
    if log_probs.size(1) != target_response_length:
        batch_size = log_probs.size(0)
        current_len = log_probs.size(1)

        full_log_probs = log_probs.new_zeros(batch_size, target_response_length)
        full_log_probs[:, :current_len] = log_probs
        log_probs = full_log_probs

        if entropy is not None:
            full_entropy = entropy.new_zeros(batch_size, target_response_length)
            full_entropy[:, :current_len] = entropy
            entropy = full_entropy

    return entropy, log_probs
