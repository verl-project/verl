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

from typing import Any

import numpy as np
import torch

from verl import DataProto


def get_reward_variance_filter_mask(
    rewards: torch.Tensor,
    uids: np.ndarray,
    live_mask: torch.Tensor | None = None,
    *,
    strategy: str = "top_p",
    top_p: float = 0.9,
    top_k: int = 1,
    include_zero: bool = False,
    variance_ddof: int = 1,
    selection_eps: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Select prompt groups by cumulative variance mass or a fixed highest-variance count."""
    if rewards.ndim != 1 or len(rewards) != len(uids):
        raise ValueError(
            f"rewards and uids must be one-dimensional and equal length, got {rewards.shape} and {uids.shape}"
        )
    if live_mask is not None and live_mask.shape != rewards.shape:
        raise ValueError(f"live_mask must have shape {rewards.shape}, got {live_mask.shape}")
    if strategy not in ("top_p", "top_k"):
        raise ValueError(f"strategy must be 'top_p' or 'top_k', got {strategy!r}")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if variance_ddof < 0:
        raise ValueError(f"variance_ddof must be non-negative, got {variance_ddof}")
    if selection_eps < 0.0:
        raise ValueError(f"selection_eps must be non-negative, got {selection_eps}")

    uid_list = uids.tolist()
    ordered_uids = list(dict.fromkeys(uid_list))
    if live_mask is None:
        live_mask = torch.ones_like(rewards, dtype=torch.bool)
    else:
        live_mask = live_mask.bool()

    variances: dict[Any, float] = {}
    for uid in ordered_uids:
        indices = torch.tensor([value == uid for value in uid_list], device=rewards.device) & live_mask
        values = rewards[indices]
        values = values[torch.isfinite(values)]
        variances[uid] = (
            float(torch.var(values, correction=variance_ddof).item()) if values.numel() > variance_ddof else 0.0
        )

    total_variance = sum(variances.values())
    selected_uids: set[Any] = set()
    if total_variance <= 0.0:
        if include_zero:
            selected_uids.update(ordered_uids[:top_k] if strategy == "top_k" else ordered_uids)
    elif strategy == "top_p" and include_zero and top_p == 1.0:
        selected_uids.update(ordered_uids)
    else:
        candidates = [uid for uid in ordered_uids if include_zero or variances[uid] > 0.0]
        candidates.sort(key=lambda uid: variances[uid], reverse=True)
        if strategy == "top_k":
            selected_uids.update(candidates[:top_k])
        else:
            target = top_p * total_variance - selection_eps
            cumulative = 0.0
            # RAGEN uses this slack to skip near-zero-signal batches entirely.
            if target > 0.0:
                for uid in candidates:
                    selected_uids.add(uid)
                    cumulative += variances[uid]
                    if cumulative >= target:
                        break

    keep_mask = torch.tensor([uid in selected_uids for uid in uid_list], device=rewards.device)
    selected_variance = sum(variances[uid] for uid in selected_uids)
    num_groups = len(ordered_uids)
    metrics = {
        "reward_variance_filtering/num_groups": float(num_groups),
        "reward_variance_filtering/num_kept_groups": float(len(selected_uids)),
        "reward_variance_filtering/kept_group_ratio": len(selected_uids) / num_groups if num_groups else 0.0,
        "reward_variance_filtering/total_variance": total_variance,
        "reward_variance_filtering/selected_variance_ratio": (
            selected_variance / total_variance if total_variance > 0.0 else 0.0
        ),
        "reward_variance_filtering/num_zero_variance_groups": float(sum(value == 0.0 for value in variances.values())),
    }
    return keep_mask, metrics


def apply_reward_variance_filter(data: DataProto, config: Any) -> tuple[DataProto, dict[str, float]]:
    """Apply reward-variance filtering to a DataProto by masking complete response sequences."""
    response_mask = data.batch["response_mask"]
    rewards = (data.batch["token_level_scores"] * response_mask).sum(dim=-1)
    live_mask = response_mask.sum(dim=-1) > 0
    keep_mask, metrics = get_reward_variance_filter_mask(
        rewards,
        data.non_tensor_batch["uid"],
        live_mask,
        strategy=config.get("strategy", "top_p"),
        top_p=config.get("top_p", 0.9),
        top_k=config.get("top_k", 1),
        include_zero=config.get("include_zero", False),
        variance_ddof=config.get("variance_ddof", 1),
        selection_eps=config.get("selection_eps", 0.01),
    )
    data.batch["response_mask"] = response_mask * keep_mask.unsqueeze(-1)
    return data, metrics
