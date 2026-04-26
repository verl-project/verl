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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto


@dataclass(frozen=True)
class DynamicSamplingResult:
    """Cumulative status after adding one candidate generation batch."""

    ready: bool
    metrics: dict[str, float]


class DynamicSamplingAccumulator:
    """Accumulate informative prompt groups for DAPO-style dynamic sampling.

    The accumulator is intentionally trainer-agnostic. It only knows how to
    filter `DataProto` rows by prompt `uid`, keep reward payloads aligned with
    the filtered rows, and stop when enough prompt groups have been retained.
    """

    def __init__(self, filter_config: Any, train_prompt_bsz: int, rollout_n: int):
        self.filter_config = filter_config
        self.train_prompt_bsz = int(train_prompt_bsz)
        self.rollout_n = int(rollout_n)
        self.metric_name = _get_config_value(filter_config, "metric")
        self.max_num_gen_batches = int(_get_config_value(filter_config, "max_num_gen_batches", 0) or 0)

        if self.train_prompt_bsz <= 0:
            raise ValueError(f"train_prompt_bsz must be positive, got {self.train_prompt_bsz}")
        if self.rollout_n <= 0:
            raise ValueError(f"rollout_n must be positive, got {self.rollout_n}")
        if not self.metric_name:
            raise ValueError("algorithm.filter_groups.metric must be set when filter_groups is enabled")

        self.num_gen_batches = 0
        self.num_prompt_in_batch = 0
        self._kept_batches: list[DataProto] = []
        self._kept_reward_tensors: list[torch.Tensor] = []
        self._kept_reward_extra_infos: list[dict[str, np.ndarray]] = []
        self._reward_extra_keys: tuple[str, ...] | None = None

        self._kept_prompts = 0
        self._dropped_prompts = 0
        self._kept_trajectories = 0
        self._dropped_trajectories = 0

    def add_candidate(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict[str, Any],
    ) -> DynamicSamplingResult:
        """Filter one candidate generation batch and append its kept groups."""

        self.num_gen_batches += 1
        metric_values = self._extract_metric(batch, reward_tensor, reward_extra_infos_dict)
        uids = _require_1d_array(batch.non_tensor_batch.get("uid"), "uid")
        if len(uids) != len(batch):
            raise ValueError(f"uid length {len(uids)} must match batch length {len(batch)}")
        if len(metric_values) != len(batch):
            raise ValueError(
                f"filter metric {self.metric_name!r} length {len(metric_values)} must match batch length {len(batch)}"
            )
        if reward_tensor.shape[0] != len(batch):
            raise ValueError(f"reward tensor batch size {reward_tensor.shape[0]} must match batch length {len(batch)}")

        prompt_uid2metric_vals: dict[Any, list[float]] = defaultdict(list)
        prompt_uids_in_order = []
        seen_uids = set()
        for uid, metric_value in zip(uids, metric_values, strict=True):
            if uid not in seen_uids:
                prompt_uids_in_order.append(uid)
                seen_uids.add(uid)
            prompt_uid2metric_vals[uid].append(float(metric_value))

        invalid_group_sizes = {
            uid: len(prompt_uid2metric_vals[uid])
            for uid in prompt_uids_in_order
            if len(prompt_uid2metric_vals[uid]) != self.rollout_n
        }
        if invalid_group_sizes:
            raise ValueError(
                f"Each prompt group must contain rollout_n={self.rollout_n} trajectories for dynamic sampling. "
                f"Invalid group sizes: {invalid_group_sizes}"
            )

        kept_prompt_uids = [
            uid
            for uid in prompt_uids_in_order
            if np.std(prompt_uid2metric_vals[uid]) > 0 or len(prompt_uid2metric_vals[uid]) == 1
        ]
        kept_prompt_uid_set = set(kept_prompt_uids)
        kept_traj_idxs = np.asarray(
            [idx for idx, uid in enumerate(uids) if uid in kept_prompt_uid_set],
            dtype=np.int64,
        )

        num_prompts = len(prompt_uids_in_order)
        num_kept_prompts = len(kept_prompt_uids)
        num_kept_trajectories = int(kept_traj_idxs.shape[0])
        self.num_prompt_in_batch += num_kept_prompts
        self._kept_prompts += num_kept_prompts
        self._dropped_prompts += num_prompts - num_kept_prompts
        self._kept_trajectories += num_kept_trajectories
        self._dropped_trajectories += len(batch) - num_kept_trajectories

        if num_kept_trajectories > 0:
            kept_batch = batch[kept_traj_idxs]
            kept_batch.meta_info = _stable_meta_info_for_concat(kept_batch.meta_info)
            self._kept_batches.append(kept_batch)
            tensor_indices = torch.as_tensor(kept_traj_idxs, dtype=torch.long, device=reward_tensor.device)
            self._kept_reward_tensors.append(reward_tensor[tensor_indices])
            self._kept_reward_extra_infos.append(
                self._slice_reward_extra_infos(reward_extra_infos_dict, kept_traj_idxs, batch_len=len(batch))
            )

        return DynamicSamplingResult(ready=self.is_ready(), metrics=self.metrics)

    def is_ready(self) -> bool:
        return self.num_prompt_in_batch >= self.train_prompt_bsz

    def should_continue(self) -> bool:
        return not self.is_ready() and (
            self.max_num_gen_batches <= 0 or self.num_gen_batches < self.max_num_gen_batches
        )

    def finalize(self) -> tuple[DataProto, torch.Tensor, dict[str, np.ndarray], dict[str, float]]:
        if not self.is_ready():
            raise ValueError(
                f"num_prompt_in_batch={self.num_prompt_in_batch} < train_prompt_bsz={self.train_prompt_bsz}. "
                f"Generated {self.num_gen_batches} batch(es), max_num_gen_batches={self.max_num_gen_batches}."
            )
        if not self._kept_batches:
            raise ValueError("No trajectories were kept by dynamic sampling")

        batch = self._kept_batches[0] if len(self._kept_batches) == 1 else DataProto.concat(self._kept_batches)
        reward_tensor = (
            self._kept_reward_tensors[0]
            if len(self._kept_reward_tensors) == 1
            else torch.cat(self._kept_reward_tensors, dim=0)
        )
        reward_extra_infos = _concat_reward_extra_infos(self._kept_reward_extra_infos, self._reward_extra_keys or ())

        traj_bsz = self.train_prompt_bsz * self.rollout_n
        batch = batch[:traj_bsz]
        reward_tensor = reward_tensor[:traj_bsz]
        reward_extra_infos = {key: value[:traj_bsz] for key, value in reward_extra_infos.items()}
        return batch, reward_tensor, reward_extra_infos, self.metrics

    @property
    def metrics(self) -> dict[str, float]:
        total_prompts = self._kept_prompts + self._dropped_prompts
        kept_ratio = self._kept_prompts / total_prompts if total_prompts > 0 else 0.0
        reached_max = (
            self.max_num_gen_batches > 0 and self.num_gen_batches >= self.max_num_gen_batches and not self.is_ready()
        )
        return {
            "dynamic_sampling/num_gen_batches": float(self.num_gen_batches),
            "dynamic_sampling/num_prompt_in_batch": float(self.num_prompt_in_batch),
            "dynamic_sampling/kept_prompts": float(self._kept_prompts),
            "dynamic_sampling/dropped_prompts": float(self._dropped_prompts),
            "dynamic_sampling/kept_trajectories": float(self._kept_trajectories),
            "dynamic_sampling/dropped_trajectories": float(self._dropped_trajectories),
            "dynamic_sampling/kept_prompt_ratio": float(kept_ratio),
            "dynamic_sampling/reached_max_num_gen_batches": float(reached_max),
        }

    def _extract_metric(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict[str, Any],
    ) -> np.ndarray:
        metric_name = self.metric_name
        if metric_name == "seq_reward":
            scores = batch.batch["token_level_scores"] if "token_level_scores" in batch.batch.keys() else reward_tensor
            return _tensor_to_numpy_1d(scores.sum(dim=-1), metric_name)

        if metric_name == "seq_final_reward":
            if "token_level_rewards" not in batch.batch.keys():
                raise ValueError(
                    "algorithm.filter_groups.metric='seq_final_reward' requires token_level_rewards, "
                    "which are not available before one-step-off advantage/KL computation. "
                    "Use metric='seq_reward', 'acc', or another reward extra key instead."
                )
            return _tensor_to_numpy_1d(batch.batch["token_level_rewards"].sum(dim=-1), metric_name)

        if metric_name in reward_extra_infos_dict:
            return _values_to_numpy_1d(reward_extra_infos_dict[metric_name], metric_name)
        if metric_name in batch.non_tensor_batch:
            return _values_to_numpy_1d(batch.non_tensor_batch[metric_name], metric_name)
        if metric_name in batch.batch.keys():
            value = batch.batch[metric_name]
            if value.ndim > 1:
                value = value.sum(dim=-1)
            return _tensor_to_numpy_1d(value, metric_name)

        available = sorted(
            set(batch.non_tensor_batch.keys()) | set(batch.batch.keys()) | set(reward_extra_infos_dict.keys())
        )
        raise ValueError(f"Could not find filter metric {metric_name!r}. Available keys: {available}")

    def _slice_reward_extra_infos(
        self,
        reward_extra_infos_dict: dict[str, Any],
        indices: np.ndarray,
        batch_len: int,
    ) -> dict[str, np.ndarray]:
        normalized = {
            key: _require_1d_array(value, f"reward_extra_infos_dict[{key!r}]")
            for key, value in reward_extra_infos_dict.items()
        }
        for key, value in normalized.items():
            if len(value) != batch_len:
                raise ValueError(
                    f"reward_extra_infos_dict[{key!r}] length {len(value)} must match batch length {batch_len}"
                )
        reward_extra_keys = tuple(sorted(normalized.keys()))
        if self._reward_extra_keys is None:
            self._reward_extra_keys = reward_extra_keys
        elif reward_extra_keys != self._reward_extra_keys:
            raise ValueError(
                f"Reward extra info keys must be stable across dynamic sampling batches. "
                f"Expected {self._reward_extra_keys}, got {reward_extra_keys}."
            )
        return {key: value[indices] for key, value in normalized.items()}


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _stable_meta_info_for_concat(meta_info: dict[str, Any]) -> dict[str, Any]:
    stable_meta_info = {}
    if "reward_extra_keys" in meta_info:
        stable_meta_info["reward_extra_keys"] = sorted(meta_info["reward_extra_keys"])
    return stable_meta_info


def _tensor_to_numpy_1d(value: torch.Tensor, name: str) -> np.ndarray:
    if value.ndim != 1:
        raise ValueError(f"{name} must be 1-D after reduction, got shape {tuple(value.shape)}")
    return value.detach().cpu().numpy()


def _values_to_numpy_1d(value: Any, name: str) -> np.ndarray:
    array = _require_1d_array(value, name)
    try:
        return array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values for dynamic sampling") from exc


def _require_1d_array(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required for dynamic sampling")
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        array = np.asarray(value, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D sequence, got shape {array.shape}")
    return array


def _concat_reward_extra_infos(
    reward_extra_infos: list[dict[str, np.ndarray]],
    reward_extra_keys: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if not reward_extra_keys:
        return {}
    return {
        key: np.concatenate([extra_info[key] for extra_info in reward_extra_infos], axis=0) for key in reward_extra_keys
    }
