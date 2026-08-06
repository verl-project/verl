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
"""Batch padding utilities for multi-trajectory training with TransferQueue.

When the number of trajectories per prompt varies, the global batch size may
not be divisible by ``dp_size`` or ``mini_batch_size``.  The helpers here
append minimal synthetic samples so that downstream training steps can
partition the batch evenly.
"""

from __future__ import annotations

import copy
import logging
import uuid
from typing import Any

import torch

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)


def build_padding_position_ids(source_position_ids: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Build padding position ids with the same rank/prefix shape as the source sample."""
    position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
    if not isinstance(source_position_ids, torch.Tensor):
        return position_ids

    position_ids = position_ids.to(device=source_position_ids.device, dtype=source_position_ids.dtype)
    if source_position_ids.dim() <= 1:
        return position_ids

    view_shape = (1,) * (source_position_ids.dim() - 1) + (position_ids.size(-1),)
    return position_ids.reshape(view_shape).expand(*source_position_ids.shape[:-1], -1).clone()


def build_padding_routed_experts(source_routed_experts: Any, seq_len: int) -> torch.Tensor | None:
    """Build a zero routed-experts tensor matching the source per-token expert shape."""
    if not isinstance(source_routed_experts, torch.Tensor):
        return None
    if source_routed_experts.dim() == 0:
        return torch.zeros_like(source_routed_experts)
    return torch.zeros(
        (seq_len, *source_routed_experts.shape[1:]),
        dtype=source_routed_experts.dtype,
        device=source_routed_experts.device,
    )


def construct_minimal_padding_template(
    source_td: dict,
    source_tag: dict,
    eos_token_id: int,
) -> tuple[dict, dict]:
    """Construct a minimal text-only padding template of one prompt token and one response token.

    Args:
        source_td: A single sample dict retrieved from TransferQueue.
        source_tag: The corresponding tag dict for that sample.
        eos_token_id: The EOS token id from the tokenizer.

    Returns:
        A tuple of (template_sample, template_tag) ready for padding.
    """
    # Copy the sample template from an existing sample.
    template_sample = {}
    for key in source_td.keys():
        value = source_td[key]
        template_sample[key] = value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)

    # Deep copy the template tag from an existing sample.
    template_tag = copy.deepcopy(source_tag)

    # Build minimal sequence
    prompts = torch.full((1,), eos_token_id, dtype=torch.int64)
    input_ids = prompts.repeat(2)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    response_mask = torch.zeros_like(prompts)
    position_ids = build_padding_position_ids(template_sample.get("position_ids"), attention_mask)
    routed_experts = build_padding_routed_experts(template_sample.get("routed_experts"), input_ids.size(0))

    # Update the fields and remove redundant parts
    template_sample.update(
        prompts=prompts,
        responses=prompts.clone(),
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        num_turns=0,
        response_mask=response_mask,
        loss_mask=response_mask,
        rm_scores=torch.zeros_like(response_mask, dtype=torch.float32),
        rollout_log_probs=torch.zeros_like(response_mask, dtype=torch.float32),
    )
    if "multi_modal_inputs" in template_sample:
        template_sample["multi_modal_inputs"] = {}
    if routed_experts is not None:
        template_sample["routed_experts"] = routed_experts
    else:
        template_sample.pop("routed_experts", None)

    # Padding flag is deployed to protect metrics calculation (e.g. response length, score, reward).
    template_tag.update(is_padding=True, prompt_len=1, response_len=1, seq_len=2)
    return template_sample, template_tag


def upsample_batch_to_divisible_size(
    batch: KVBatchMeta,
    batch_multiple: int,
    eos_token_id: int,
) -> KVBatchMeta:
    """Append synthetic no-op samples so the batch size becomes divisible by *batch_multiple*.

    The synthetic samples reuse the first real sample as a metadata template,
    but manually construct a minimal ``prompt_len=1 / response_len=1`` sequence
    and zero out reward-related fields so they do not contribute to PPO,
    entropy, or KL losses.  An ``is_padding`` flag is added in the tag for
    downstream metrics filtering.

    Args:
        batch: The current KVBatchMeta from TransferQueue.
        batch_multiple: The required divisor (e.g. lcm of dp_size and mini-batch sizes).
        eos_token_id: The EOS token id from the tokenizer.

    Returns:
        The (possibly enlarged) KVBatchMeta.
    """
    remainder = len(batch) % batch_multiple
    if remainder == 0:
        return batch

    # Take the first trajectory as the metadata template for padding data.
    source_idx = 0
    source_key = batch.keys[source_idx]
    source_td = tq.kv_batch_get(keys=[source_key], partition_id=batch.partition_id)[0]

    # Construct the minimal padding template of one prompt token and one response token
    template_sample, template_tag = construct_minimal_padding_template(source_td, batch.tags[source_idx], eos_token_id)

    # All padding data use the same uid (also the same trajectory_id 0 but with ascending session_ids)
    # This uid is not identical to any of the actual data, so it won't affect the grpo advantage value.
    pad_uid = f"pad{uuid.uuid4().hex}"
    template_sample["uid"] = pad_uid

    # Construct the padding samples in a for-loop
    pad_keys = []
    pad_tags = []
    pad_fields = []
    pad_size = batch_multiple - remainder
    for local_idx in range(pad_size):
        sample = copy.deepcopy(template_sample)
        # Use incremental local_idx as different session_ids
        pad_keys.append(f"{pad_uid}_{local_idx}_0")
        if "session_id" in sample:
            sample["session_id"] = local_idx
        pad_fields.append(sample)
        pad_tags.append(copy.deepcopy(template_tag))

    tq.kv_batch_put(
        keys=pad_keys,
        partition_id=batch.partition_id,
        fields=list_of_dict_to_tensordict(pad_fields),
        tags=pad_tags,
    )
    logger.info(
        "Upsampled batch from %d to %d with %d synthetic padding samples for required_multiple=%d",
        len(batch),
        len(batch) + pad_size,
        pad_size,
        batch_multiple,
    )
    return KVBatchMeta(
        keys=batch.keys + pad_keys,
        tags=batch.tags + pad_tags,
        partition_id=batch.partition_id,
        fields=batch.fields,
        extra_info=batch.extra_info,
    )
