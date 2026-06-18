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
"""Per-turn GRPO advantage estimator for IFGym multi-turn rollouts.

For each assistant turn k in each rollout i within group g:

    A_{i,k} = (r_{i,k} - mu_{group,k}) / (sigma_{group,k} + eps)

The scalar advantage is broadcast uniformly to every token of turn k in
rollout i. Turn boundaries come from contiguous runs of 1s in
response_mask. Each turn's reward is the sum of token_level_rewards
within that run (the IFGym agent loop places one per-turn score at each
turn's last token, see ifgym_agent_loop.py and the per-turn rm_scores
placement in agent_loop._postprocess).

Falls back to advantage=0 for any (group, turn) with <2 peers.

Registers the ``ifgym_per_turn_grpo`` advantage estimator on import.
"""

import os
from collections import defaultdict

import torch

from verl.trainer.ppo.core_algos import register_adv_est

_DEBUG = os.environ.get("IFGYM_PERTURN_GRPO_DEBUG") == "1"


def _split_turns(mask_row, rewards_row):
    """Return [(start, end, turn_reward), ...] for each contiguous run of 1s in mask_row."""
    T = mask_row.shape[-1]
    turns = []
    in_run = False
    start = None
    for t in range(T):
        m = int(mask_row[t].item()) if hasattr(mask_row[t], "item") else int(mask_row[t])
        if m == 1 and not in_run:
            in_run = True
            start = t
        elif m == 0 and in_run:
            end = t - 1
            turns.append((start, end, rewards_row[start : end + 1].sum().item()))
            in_run = False
    if in_run:
        end = T - 1
        turns.append((start, end, rewards_row[start : end + 1].sum().item()))
    return turns


@register_adv_est("ifgym_per_turn_grpo")
def compute_ifgym_per_turn_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    bsz, T = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards)

    sample_turns = [_split_turns(response_mask[i], token_level_rewards[i]) for i in range(bsz)]

    group_turns = defaultdict(list)
    group_ids = defaultdict(list)
    for i in range(bsz):
        group_turns[index[i]].append(sample_turns[i])
        group_ids[index[i]].append(i)

    with torch.no_grad():
        for idx, rollout_lists in group_turns.items():
            rollout_indices = group_ids[idx]
            max_turns = max((len(t) for t in rollout_lists), default=0)
            for k in range(max_turns):
                rewards_k, idx_k = [], []
                for ri, turns in zip(rollout_indices, rollout_lists, strict=False):
                    if k < len(turns):
                        rewards_k.append(turns[k][2])
                        idx_k.append(ri)
                if len(rewards_k) < 2:
                    continue
                rt = torch.tensor(rewards_k, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
                mu = rt.mean()
                sigma = rt.std()
                normalized = (rt - mu) / (sigma + epsilon)
                for j, ri in enumerate(idx_k):
                    s, e, _ = sample_turns[ri][k]
                    advantages[ri, s : e + 1] = normalized[j].item()
                if _DEBUG:
                    print(
                        f"[per-turn-grpo] group={idx} turn={k} n={len(rewards_k)} "
                        f"rewards={[round(r, 3) for r in rewards_k]} "
                        f"mu={mu.item():.3f} sigma={sigma.item():.3f} "
                        f"adv={[round(a, 3) for a in normalized.tolist()]}",
                        flush=True,
                    )
    advantages = advantages * response_mask
    return advantages, advantages
