from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

from .helpers import normalize_trajectory_rewards, validate_trajectory
from .types import Trajectory


def _to_long_tensor(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.long)


def _to_float_tensor(values: Sequence[float]) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)

# TODO: check if this helper function is uncessarily complex.
class TrajectoryAssembler:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def assemble(self, trajectories: Sequence[Trajectory]) -> DataProto:
        if not trajectories:
            raise ValueError("trajectories must be non-empty")

        normalized = [validate_trajectory(trajectory) for trajectory in trajectories]
        normalized = normalize_trajectory_rewards(normalized)

        prompt_width = max(len(trajectory.prompt_ids) for trajectory in normalized)
        response_width = max(len(trajectory.response_ids) for trajectory in normalized)
        batch_size = len(normalized)

        prompts = torch.full((batch_size, prompt_width), self.pad_token_id, dtype=torch.long)
        responses = torch.full((batch_size, response_width), self.pad_token_id, dtype=torch.long)
        response_mask = torch.zeros((batch_size, response_width), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, prompt_width + response_width), dtype=torch.long)

        rollout_log_probs = None
        if any(trajectory.response_logprobs is not None for trajectory in normalized):
            rollout_log_probs = torch.zeros((batch_size, response_width), dtype=torch.float32)

        routed_experts = None
        first_routed_experts = next((trajectory.routed_experts for trajectory in normalized if trajectory.routed_experts is not None), None)
        if first_routed_experts is not None:
            if isinstance(first_routed_experts, np.ndarray):
                routed_shape = first_routed_experts.shape[1:]
                routed_dtype = torch.from_numpy(first_routed_experts).dtype
            else:
                routed_shape = tuple(first_routed_experts.shape[1:])
                routed_dtype = first_routed_experts.dtype
            routed_experts = torch.zeros((batch_size, prompt_width + response_width, *routed_shape), dtype=routed_dtype)

        for index, trajectory in enumerate(normalized):
            prompt_ids = _to_long_tensor(trajectory.prompt_ids)
            response_ids = _to_long_tensor(trajectory.response_ids)
            response_mask_ids = _to_long_tensor(trajectory.response_mask)

            prompt_offset = prompt_width - len(trajectory.prompt_ids)
            prompts[index, prompt_offset:] = prompt_ids
            responses[index, : len(trajectory.response_ids)] = response_ids
            response_mask[index, : len(trajectory.response_mask)] = response_mask_ids

            attention_mask[index, prompt_offset:prompt_width] = 1
            attention_mask[index, prompt_width : prompt_width + len(trajectory.response_ids)] = 1

            if rollout_log_probs is not None and trajectory.response_logprobs is not None:
                rollout_log_probs[index, : len(trajectory.response_logprobs)] = _to_float_tensor(trajectory.response_logprobs)

            if routed_experts is not None and trajectory.routed_experts is not None:
                experts_tensor = (
                    torch.from_numpy(trajectory.routed_experts.copy())
                    if isinstance(trajectory.routed_experts, np.ndarray)
                    else trajectory.routed_experts
                )
                start = prompt_offset
                end = start + experts_tensor.shape[0]
                routed_experts[index, start:end] = experts_tensor

        input_ids = torch.cat([prompts, responses], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_tensors: dict[str, torch.Tensor] = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if rollout_log_probs is not None:
            batch_tensors["rollout_log_probs"] = rollout_log_probs
        if routed_experts is not None:
            batch_tensors["routed_experts"] = routed_experts

        if all(trajectory.reward_score is not None for trajectory in normalized):
            rm_scores = torch.zeros((batch_size, response_width), dtype=torch.float32)
            for index, trajectory in enumerate(normalized):
                if trajectory.response_ids:
                    rm_scores[index, len(trajectory.response_ids) - 1] = float(trajectory.reward_score)
            batch_tensors["rm_scores"] = rm_scores

        reward_extra_keys = list(
            dict.fromkeys(
                key
                for trajectory in normalized
                for key in trajectory.reward_info
            )
        )

        non_tensor_batch: dict[str, np.ndarray] = {
            "__num_turns__": np.array([trajectory.num_turns for trajectory in normalized], dtype=np.int32),
        }
        for key in reward_extra_keys:
            values = np.empty(batch_size, dtype=object)
            values[:] = [trajectory.reward_info.get(key) for trajectory in normalized]
            non_tensor_batch[key] = values

        meta_info: dict[str, object] = {}
        if "rm_scores" in batch_tensors:
            meta_info["reward_extra_keys"] = reward_extra_keys

        return DataProto(
            batch=TensorDict(batch_tensors, batch_size=batch_size),
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )
