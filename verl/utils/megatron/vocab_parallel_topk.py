# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Export the teacher's top-k next-token distribution from a training forward.

The distillation loss consumes ``teacher_topk_token_ids`` / ``teacher_topk_logprobs``
per position. When the teacher is the training model itself (on-policy
self-distillation: same weights, privileged context) those tensors have to come
out of the trainer, and the only place holding the distribution is the forward's
logits — sharded across the tensor-parallel group.

Two facts make the sharded reduction cheap and exact:

* the global log-sum-exp needs only each shard's max and sum-exp (two small
  collectives), after which every shard can express its entries in global
  log-prob space;
* the global top-k is contained in the union of the per-shard top-k, because an
  entry that is not in its own shard's top-k is already beaten by k entries of
  that same shard.

``topk_logprobs_from_logits`` is the runtime entry point; ``merge_topk_shards``
and ``shard_logprobs`` are split out so the reduction is testable without a
process group.
"""

from __future__ import annotations

import torch


def shard_logprobs(
    logits: torch.Tensor,
    global_max: torch.Tensor,
    global_sumexp: torch.Tensor,
) -> torch.Tensor:
    """This shard's logits expressed as log-probs over the FULL vocabulary."""
    return (logits - global_max) - global_sumexp.log()


def merge_topk_shards(
    shard_logps: list[torch.Tensor],
    shard_ids: list[torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-shard top-k candidates to the global top-k.

    Args:
        shard_logps: one ``(..., k_local)`` tensor per shard, already in GLOBAL
            log-prob space.
        shard_ids: the matching ``(..., k_local)`` global vocabulary ids.
        k: how many entries the caller wants.

    Returns:
        ``(logps, ids)`` of shape ``(..., k)``, sorted by descending log-prob.
    """
    if len(shard_logps) != len(shard_ids):
        raise ValueError(f"{len(shard_logps)} log-prob shards vs {len(shard_ids)} id shards")
    if not shard_logps:
        raise ValueError("no shards to merge")
    logps = torch.cat(shard_logps, dim=-1)
    ids = torch.cat(shard_ids, dim=-1)
    if logps.shape != ids.shape:
        raise ValueError(f"logps {tuple(logps.shape)} and ids {tuple(ids.shape)} must match")
    if k > logps.shape[-1]:
        raise ValueError(f"cannot take top-{k} from {logps.shape[-1]} candidates")
    top_logps, top_idx = torch.topk(logps, k, dim=-1)
    # topk indexes the candidate axis; the vocabulary ids ride along the same axis.
    return top_logps, ids.gather(-1, top_idx)


def topk_logprobs_from_logits(
    logits: torch.Tensor,
    k: int,
    *,
    process_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global top-k log-probs and ids from a vocab-parallel logits shard.

    Args:
        logits: ``(..., vocab_shard)`` — this rank's slice of the vocabulary.
        k: number of entries to return. Values above the global vocabulary size
            are clamped to it.
        process_group: tensor-parallel group; ``None`` means this rank owns the
            whole vocabulary.

    Returns:
        ``(topk_logps, topk_ids)`` of shape ``(..., min(k, global_vocab))``. Ids
        are GLOBAL vocab ids; logps are normalised over the FULL vocabulary.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if logits.dim() < 1:
        raise ValueError("logits need at least one dimension")

    distributed = process_group is not None
    world_size = torch.distributed.get_world_size(process_group) if distributed else 1
    rank = torch.distributed.get_rank(process_group) if distributed else 0
    local_vocab = logits.shape[-1]
    vocab_start = rank * local_vocab
    k = min(k, local_vocab * world_size)
    # Accumulating exp/sum/log in BF16 can shift exported log-probs by ~1e-2.
    # Match the training loss's FP32 vocabulary normalizer.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    # 1. Global normaliser over the full vocabulary.
    global_max = logits.max(dim=-1, keepdim=True).values
    if distributed:
        torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=process_group)
    global_sumexp = (logits - global_max).exp().sum(dim=-1, keepdim=True)
    if distributed:
        torch.distributed.all_reduce(global_sumexp, op=torch.distributed.ReduceOp.SUM, group=process_group)

    # 2. This shard's candidates in global log-prob space, then the global top-k.
    logps = shard_logprobs(logits, global_max, global_sumexp)
    k_local = min(k, local_vocab)
    local_top_logps, local_top_idx = torch.topk(logps, k_local, dim=-1)
    local_top_ids = local_top_idx + vocab_start

    if not distributed:
        return local_top_logps, local_top_ids

    gathered_logps = [torch.empty_like(local_top_logps) for _ in range(world_size)]
    gathered_ids = [torch.empty_like(local_top_ids) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_logps, local_top_logps.contiguous(), group=process_group)
    torch.distributed.all_gather(gathered_ids, local_top_ids.contiguous(), group=process_group)
    return merge_topk_shards(gathered_logps, gathered_ids, k)
