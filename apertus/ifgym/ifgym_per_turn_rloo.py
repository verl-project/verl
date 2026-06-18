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
"""Per-turn RLOO (REINFORCE Leave-One-Out) advantage estimator.

For each assistant turn k in each rollout i within group g:

    A_{i,k} = r_{i,k} - mean_{j != i, in g, has turn k}(r_{j,k})

The scalar advantage is broadcast uniformly to every token of turn k in
rollout i. Tokens outside any assistant turn keep advantage 0.

Sibling of ifgym_advantage.py (GRPO variant): same per-turn segmentation
and group structure, but the baseline is leave-one-out and there is no
std normalization.

Registers the ``ifgym_per_turn_rloo`` advantage estimator on import.
Set IFGYM_PERTURN_RLOO_DEBUG=1 for per-group/per-turn debug printing.
"""

import os
from collections import defaultdict

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est

_DEBUG = os.environ.get("IFGYM_PERTURN_RLOO_DEBUG", "0") == "1"


def _find_turn_spans(mask_row: torch.Tensor) -> list:
    """Return [(start, end), ...] inclusive indices for each contiguous run of 1s."""
    spans = []
    in_run = False
    start = 0
    arr = mask_row.to(torch.int64).cpu().tolist()
    for t, v in enumerate(arr):
        if v == 1 and not in_run:
            start = t
            in_run = True
        elif v == 0 and in_run:
            spans.append((start, t - 1))
            in_run = False
    if in_run:
        spans.append((start, len(arr) - 1))
    return spans


@register_adv_est("ifgym_per_turn_rloo")
def compute_ifgym_per_turn_rloo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    **kwargs,
):
    """Per-turn leave-one-out advantage.

    Args:
        token_level_rewards: (B, T) per-turn rewards placed at each turn's last token.
        response_mask: (B, T) of {0, 1}; each contiguous run of 1s is one assistant turn.
        index: (B,) group IDs; rollouts with equal index share a prompt.

    Returns:
        (advantages, returns), both (B, T). returns == advantages (no critic).
    """
    B, _T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    per_rollout_spans = []
    per_rollout_turn_rewards = []
    for i in range(B):
        spans = _find_turn_spans(response_mask[i])
        per_rollout_spans.append(spans)
        per_rollout_turn_rewards.append([float(token_level_rewards[i, s : e + 1].sum().item()) for s, e in spans])

    group_to_rollouts = defaultdict(list)
    for i in range(B):
        group_to_rollouts[index[i]].append(i)

    advantages = torch.zeros_like(token_level_rewards, dtype=dtype, device=device)

    for group_id, rollout_idxs in group_to_rollouts.items():
        if len(rollout_idxs) < 2:
            if _DEBUG:
                print(f"[per-turn-rloo] group={group_id} only {len(rollout_idxs)} rollout(s); skip")
            continue

        max_turns = max(len(per_rollout_turn_rewards[i]) for i in rollout_idxs)
        for k in range(max_turns):
            peers = [i for i in rollout_idxs if k < len(per_rollout_turn_rewards[i])]
            if len(peers) < 2:
                continue

            rewards_k = [per_rollout_turn_rewards[i][k] for i in peers]
            total = float(np.sum(rewards_k))
            n = len(peers)
            adv_per_rollout = {}
            for i, r_i in zip(peers, rewards_k, strict=False):
                loo_mean = (total - r_i) / (n - 1)
                adv_per_rollout[i] = r_i - loo_mean

            if _DEBUG:
                pretty_r = ", ".join(f"{r:.3f}" for r in rewards_k)
                pretty_a = ", ".join(f"{adv_per_rollout[i]:+.3f}" for i in peers)
                print(f"[per-turn-rloo] group={group_id} turn={k} rewards=[{pretty_r}] adv=[{pretty_a}]")

            for i in peers:
                s, e = per_rollout_spans[i][k]
                advantages[i, s : e + 1] = adv_per_rollout[i]

    returns = advantages.clone()
    return advantages, returns
