from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .types import Trajectory


def _resolve_trajectory_value(value: Any, index: int, count: int) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, list)):
        # Keep tuple-like values broadcastable only when they are not trajectory-aligned containers.
        return value
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(f"reward_info sequence length must match trajectories: {len(value)} != {count}")
        return value[index]
    return value


def normalize_trajectory_rewards(
    trajectories: Sequence[Trajectory],
    reward_info: Mapping[str, Any] | None = None,
) -> list[Trajectory]:
    normalized: list[Trajectory] = []
    count = len(trajectories)

    for index, trajectory in enumerate(trajectories):
        merged_reward_info = dict(trajectory.reward_info)
        if reward_info is not None:
            for key, value in reward_info.items():
                merged_reward_info[key] = _resolve_trajectory_value(value, index=index, count=count)

        if trajectory.reward_score is None:
            raise ValueError(f"Trajectory {trajectory.uid} has no reward_score. reward_fn must return a score for every trajectory.")

        normalized.append(replace(trajectory, reward_info=merged_reward_info))

    return normalized


def validate_trajectory(trajectory: Trajectory) -> Trajectory:
    if len(trajectory.response_ids) != len(trajectory.response_mask):
        raise ValueError("response_mask length must match response_ids length")

    if trajectory.response_logprobs is not None and len(trajectory.response_logprobs) != len(trajectory.response_ids):
        raise ValueError("response_logprobs length must match response_ids length")

    if trajectory.num_turns < 0:
        raise ValueError("num_turns must be non-negative")
    # TODO: check if the checks for routed_experts are consistent with existent practices
    if trajectory.routed_experts is not None:
        if isinstance(trajectory.routed_experts, np.ndarray):
            routed_experts = trajectory.routed_experts
        elif isinstance(trajectory.routed_experts, torch.Tensor):
            routed_experts = trajectory.routed_experts
        else:
            raise TypeError(f"Unsupported routed_experts type: {type(trajectory.routed_experts)}")

        if routed_experts.ndim != 3:
            raise ValueError("routed_experts must have shape [total_tokens, num_layers, topk]")
        expected_length = len(trajectory.prompt_ids) + len(trajectory.response_ids)
        if routed_experts.shape[0] != expected_length:
            raise ValueError("routed_experts token dimension must match prompt_ids + response_ids")

    return trajectory
