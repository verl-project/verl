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
"""Segment-level GRPO for compacted agent rollouts.

A rollout that was compacted into segments carries one combined reward per
segment, ``R̃_i = α R + (1-α) r_i``. Advantages are normalized across every
segment that shares a ``uid``, then broadcast onto that segment's response
tokens. The policy loss uses the same boundaries so each segment contributes
equally inside its rollout and each rollout contributes equally in the batch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss, compute_grpo_outcome_advantage, register_adv_est


@register_adv_est("grpo_segment")
def compute_grpo_segment_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize combined segment rewards inside each uid group.

    ``non_tensor_batch`` must provide ``segment_boundaries`` and
    ``combined_rewards`` (one list per row). Missing segment data falls back
    to outcome GRPO. A uid group with a single segment has std 0, so its
    advantage is 0.
    """
    non_tensor_batch = kwargs.get("non_tensor_batch")
    if non_tensor_batch is None:
        return compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    segment_boundaries = _as_list(non_tensor_batch.get("segment_boundaries"))
    combined_rewards = _as_list(non_tensor_batch.get("combined_rewards"))
    if segment_boundaries is None or combined_rewards is None:
        return compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    with torch.no_grad():
        bsz = response_mask.shape[0]
        device = response_mask.device
        seg_rewards_list: list[float] = []
        seg_to_uid: list[Any] = []
        for bs_idx in range(bsz):
            boundaries = _row(segment_boundaries, bs_idx)
            rewards = _row(combined_rewards, bs_idx)
            for seg_i, _boundary in enumerate(boundaries):
                reward = float(rewards[seg_i]) if seg_i < len(rewards) else 0.0
                seg_rewards_list.append(reward)
                seg_to_uid.append(index[bs_idx])

        if not seg_rewards_list:
            return compute_grpo_outcome_advantage(
                token_level_rewards=token_level_rewards,
                response_mask=response_mask,
                index=index,
                epsilon=epsilon,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )

        seg_rewards = torch.tensor(seg_rewards_list, dtype=torch.float32, device=device)
        id2scores: dict[Any, list[int]] = defaultdict(list)
        for seg_idx, uid_val in enumerate(seg_to_uid):
            id2scores[uid_val].append(seg_idx)

        id2mean = {}
        id2std = {}
        for uid_val, seg_indices in id2scores.items():
            scores_tensor = torch.stack([seg_rewards[i] for i in seg_indices])
            id2mean[uid_val] = torch.mean(scores_tensor)
            if len(seg_indices) == 1:
                id2std[uid_val] = torch.tensor(0.0, device=device)
            else:
                id2std[uid_val] = torch.std(scores_tensor)

        seg_advantages = torch.zeros(len(seg_rewards_list), dtype=torch.float32, device=device)
        for seg_idx, uid_val in enumerate(seg_to_uid):
            centered = seg_rewards[seg_idx] - id2mean[uid_val]
            if norm_adv_by_std_in_grpo:
                seg_advantages[seg_idx] = centered / (id2std[uid_val] + epsilon)
            else:
                seg_advantages[seg_idx] = centered

        advantages = torch.zeros_like(response_mask, dtype=torch.float32)
        seg_idx = 0
        for bs_idx in range(bsz):
            for boundary in _row(segment_boundaries, bs_idx):
                if seg_idx < len(seg_advantages):
                    start = int(boundary.get("response_start", 0))
                    end = int(boundary.get("response_end", 0))
                    end = min(max(end, 0), response_mask.shape[1])
                    start = min(max(start, 0), end)
                    if end > start:
                        advantages[bs_idx, start:end] = seg_advantages[seg_idx]
                seg_idx += 1
        advantages = advantages * response_mask

    return advantages, advantages


def agg_segment_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    segment_boundaries: list,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
    n_rollouts: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate loss as ``1/G · Σ_j 1/(I_j+1) · Σ_i mean_token(segment_i)``.

    Without boundaries this falls back to sequence-mean token-mean. ``n_rollouts``
    is G for the whole collection batch; when it is omitted the micro-batch mean
    is used instead.
    """
    if segment_boundaries is None or len(segment_boundaries) == 0:
        return agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode="seq-mean-token-mean",
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
            loss_scale_factor=loss_scale_factor,
        )

    bsz = loss_mat.shape[0]
    rollout_losses = []
    for bs_idx in range(bsz):
        boundaries = _row(segment_boundaries, bs_idx)
        if not boundaries:
            if n_rollouts is not None:
                continue
            seq_mask = loss_mask[bs_idx].sum()
            if seq_mask > 0:
                rollout_losses.append((loss_mat[bs_idx] * loss_mask[bs_idx]).sum() / seq_mask)
            continue

        seg_losses = []
        for boundary in boundaries:
            start = int(boundary.get("response_start", 0))
            end = int(boundary.get("response_end", 0))
            if end <= start:
                continue
            end = min(end, loss_mat.shape[1])
            start = min(start, end)
            seg_mask = loss_mask[bs_idx, start:end]
            seg_token_count = seg_mask.sum().clamp(min=1)
            seg_losses.append((loss_mat[bs_idx, start:end] * seg_mask).sum() / seg_token_count)
        if not seg_losses:
            continue
        num_segments = int(boundaries[0].get("num_segments_in_rollout", len(boundaries)) or 1)
        num_segments = max(num_segments, 1)
        rollout_losses.append(torch.stack(seg_losses).sum() / num_segments)

    if not rollout_losses:
        batch_loss = torch.zeros((), device=loss_mat.device, dtype=loss_mat.dtype)
    elif n_rollouts is not None and n_rollouts > 0:
        batch_loss = torch.stack(rollout_losses).sum() / float(n_rollouts)
    else:
        batch_loss = torch.stack(rollout_losses).mean()
    return batch_loss * dp_size


def _as_list(value: Any) -> list | None:
    if value is None:
        return None
    if hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    return list(value)


def _row(rows: list, index: int) -> list:
    if index >= len(rows) or rows[index] is None:
        return []
    row = rows[index]
    if hasattr(row, "tolist") and not isinstance(row, list):
        row = row.tolist()
    return list(row)
