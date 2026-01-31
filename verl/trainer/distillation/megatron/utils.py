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
from typing import Tuple

def vocab_parallel_softmax(
    vp_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1. Converts logits to float (in calculate_logits_max)
    2. Finds max logit across all partitions
    3. Shifts logits by the max for stability
    4. Exponentiates the shifted logits
    5. Computes the sum of exponentiated shifted logits across all partitions
    """
    from megatron.core.fusions.fused_cross_entropy import calculate_logits_max
    from megatron.core.parallel_state import get_tensor_model_parallel_group

    # seq_len, batch_size, top_k = target_topk_logps.size()
    vp_logits, logits_max = calculate_logits_max(vp_logits)

    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group())

    vp_logits = vp_logits - logits_max.unsqueeze(dim=-1)
    exp_logits = vp_logits.exp()
    sum_exp_logits = exp_logits.sum(dim=-1)

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
    )
    return exp_logits, sum_exp_logits


def compute_topk_log_probs(logits: torch.Tensor, config: DistillationConfig, eps: float = 1e-20) -> dict[str, torch.Tensor]:
    """Compute top-k log probabilities."""
    from megatron.core.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from megatron.core.tensor_parallel.utils import VocabUtility



    # Compute log probs
    exp_logits, sum_exp_logits = vocab_parallel_softmax(logits)
    log_probs = exp_logits.clamp_min(eps).log() - sum_exp_logits.unsqueeze(dim=-1).clamp_min(eps).log()

    # Get the partition's vocab indices
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    partition_vocab_size = logits.shape[-1]
    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(partition_vocab_size, rank, world_size)

    # Compute local top-k
    local_topk_log_probs, local_topk_indices = log_probs.topk(k=config.topk, dim=-1)
    local_topk_indices += vocab_start_index

    # Gather all top-k from all partitions
    vocab_shards = [None for _ in range(world_size)]
    local_shard = (local_topk_log_probs, local_topk_indices)
    torch.distributed.all_gather_object(vocab_shards, local_shard, group=get_tensor_model_parallel_group()) 

    # Compute top-k over all partitions
    gathered_topk_log_probs = torch.cat([v[0] for v in vocab_shards], dim=-1)
    gathered_topk_indices = torch.cat([v[1] for v in vocab_shards], dim=-1)
    global_topk_log_probs, topk_indices_indices = gathered_topk_log_probs.topk(k=config.topk, dim=-1)
    global_topk_indices = torch.gather(gathered_topk_indices, dim=-1, index=topk_indices_indices)

    return global_topk_log_probs, global_topk_indices