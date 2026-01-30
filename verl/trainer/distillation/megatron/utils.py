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

from verl.workers.config import DistillationConfig
from verl.workers.config import DistillationConfig

def compute_topk_log_probs(logits: torch.Tensor, config: DistillationConfig) -> dict[str, torch.Tensor]:
    """Compute top-k log probabilities."""
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
    log_probs = logits.log_softmax(dim=-1)

    topk_log_probs, topk_indices = log_probs.topk(k=config.topk, dim=-1)
    return topk_log_probs, topk_indices