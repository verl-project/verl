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
Contains utilities/classes for on-policy distillation
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings
from verl.trainer.distillation.common import DistillationLossInputs, Stage
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationConfig
from verl.workers.utils.padding import _slice_response_from_unpad_output


def topk_logprobs_from_logits(
    logits: torch.Tensor, k: int, compute_topk: bool, topk_indices: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute and/or gather top-k log probabilities from logits.

    This function supports two modes that can be used independently or together:
    1. Gathering log probabilities at pre-specified indices (topk_indices)
    2. Computing new top-k log probabilities from logits

    When both modes are active, the results are concatenated and deduplicated
    to handle overlap between teacher and student top-k sets.

    Args:
        logits (torch.Tensor):
            Logits from model forward pass, shape (total_tokens, vocab_size).
        k (int):
            Number of top log probabilities to compute or gather.
        compute_topk (bool):
            Whether to compute top-k log probabilities from the logits.
        topk_indices (torch.Tensor, optional):
            Pre-computed indices for gathering log probabilities, shape (total_tokens, k) or
            (total_tokens, 2*k). Required when gathering from existing indices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - topk_logprobs: Top-k log probabilities, shape (total_tokens, k) or (total_tokens, 2*k).
            - topk_indices: Indices for the top-k log probabilities, same shape as topk_logprobs.
              Duplicate indices (from merging teacher/student top-k) have their log probs set to -inf.
    """
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from megatron.core.tensor_parallel.utils import VocabUtility


    partition_vocab_size = logits.shape[-1]

    # Get the partition's vocab indices
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, rank, world_size
    )

    vocab_shards = [None for _ in range(world_size)]
    local_shard = (vocab_start_index, logits)
    torch.distributed.all_gather_object(vocab_shards, local_shard, group=get_tensor_model_parallel_group())
    logits = torch.cat([v for start, v in sorted(vocab_shards)], dim=-1)
    logprobs = logits.values().log_softmax(dim=-1)

    raise NotADirectoryError
    return topk_logprobs, topk_indices