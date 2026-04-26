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

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from verl import DataProto

FILTER_TYPE_SAME_VALUE = "same_value"
FILTER_TYPE_BAND_PASS = "band_pass"
SUPPORTED_FILTER_TYPES = {FILTER_TYPE_SAME_VALUE, FILTER_TYPE_BAND_PASS}
BOUND_TOLERANCE = 1e-7


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    get_method = getattr(config, "get", None)
    if callable(get_method):
        return get_method(key, default)
    return getattr(config, key, default)


def is_filter_groups_enabled(filter_groups_config: Any) -> bool:
    return bool(filter_groups_config and _get_config_value(filter_groups_config, "enable", False))


def validate_filter_groups_config(filter_groups_config: Any, adv_estimator: Any) -> None:
    if not is_filter_groups_enabled(filter_groups_config):
        return

    metric = _get_config_value(filter_groups_config, "metric")
    filter_type = _get_config_value(filter_groups_config, "filter_type", FILTER_TYPE_SAME_VALUE)
    lower_bound = _get_config_value(filter_groups_config, "lower_bound")
    upper_bound = _get_config_value(filter_groups_config, "upper_bound")

    if not metric:
        raise ValueError("algorithm.filter_groups.metric must be set when filter_groups is enabled.")
    if filter_type not in SUPPORTED_FILTER_TYPES:
        raise ValueError(
            f"algorithm.filter_groups.filter_type must be one of {sorted(SUPPORTED_FILTER_TYPES)}, got {filter_type!r}."
        )
    if filter_type == FILTER_TYPE_BAND_PASS:
        if lower_bound is None and upper_bound is None:
            raise ValueError("band-pass filtering requires at least one of lower_bound or upper_bound.")
        if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
            raise ValueError(
                "algorithm.filter_groups.lower_bound must be less than or equal to upper_bound, "
                f"got {lower_bound=} and {upper_bound=}."
            )
    if str(adv_estimator).lower().endswith("remax"):
        raise ValueError("algorithm.filter_groups is not supported with REMAX in the first implementation.")


def profile_state_after_dynamic_filter_skip(
    *,
    completed_step_profile: bool,
    next_step_profile: bool,
    retry_step_profile: bool,
    profile_continuous_steps: bool,
) -> tuple[bool, bool]:
    """Return profiler state for retrying the same global step after a backfill skip."""
    profile_still_active = profile_continuous_steps and completed_step_profile and next_step_profile
    return profile_still_active, retry_step_profile


@dataclass
class DynamicFilterResult:
    batch: Optional[DataProto]
    should_generate_more: bool
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DynamicFilterState:
    num_gen_batches: int = 0
    num_prompt_groups_seen: int = 0
    num_prompt_groups_kept: int = 0
    num_prompt_groups_rejected: int = 0
    num_prompt_groups_too_low: int = 0
    num_prompt_groups_too_high: int = 0
    kept_group_metric_values: list[float] = field(default_factory=list)
    all_group_metric_values: list[float] = field(default_factory=list)
    accumulated_batch: Optional[DataProto] = None
    timing_raw: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def clear(self) -> None:
        self.num_gen_batches = 0
        self.num_prompt_groups_seen = 0
        self.num_prompt_groups_kept = 0
        self.num_prompt_groups_rejected = 0
        self.num_prompt_groups_too_low = 0
        self.num_prompt_groups_too_high = 0
        self.kept_group_metric_values = []
        self.all_group_metric_values = []
        self.accumulated_batch = None
        self.timing_raw = defaultdict(float)

    def add_timing(self, timing_raw: dict[str, float]) -> None:
        for name, value in timing_raw.items():
            self.timing_raw[name] += value

    def pop_timing(self, timing_raw: dict[str, float]) -> dict[str, float]:
        self.add_timing(timing_raw)
        total_timing = defaultdict(float)
        total_timing.update(self.timing_raw)
        self.timing_raw = defaultdict(float)
        return total_timing

    @property
    def accepted_prompt_groups(self) -> int:
        if self.accumulated_batch is None or "uid" not in self.accumulated_batch.non_tensor_batch:
            return 0
        return len(OrderedDict.fromkeys(self.accumulated_batch.non_tensor_batch["uid"]))

    def accumulate(self, batch: DataProto) -> None:
        if len(batch) == 0:
            return
        self.accumulated_batch = (
            batch if self.accumulated_batch is None else type(batch).concat([self.accumulated_batch, batch])
        )

    def metrics(self, filter_groups_config: Any) -> dict[str, float]:
        filter_type = _get_config_value(filter_groups_config, "filter_type", FILTER_TYPE_SAME_VALUE)
        metrics = {
            "dynamic_filter/gen_batches": float(self.num_gen_batches),
            "dynamic_filter/accepted_prompt_groups": float(self.num_prompt_groups_kept),
            "dynamic_filter/rejected_prompt_groups": float(self.num_prompt_groups_rejected),
        }
        if self.num_prompt_groups_seen > 0:
            metrics["dynamic_filter/accept_ratio"] = self.num_prompt_groups_kept / self.num_prompt_groups_seen
            metrics["dynamic_filter/reject_ratio"] = self.num_prompt_groups_rejected / self.num_prompt_groups_seen
        if self.all_group_metric_values:
            all_values = np.asarray(self.all_group_metric_values, dtype=np.float32)
            metrics.update(
                {
                    "dynamic_filter/metric_mean_before": float(np.mean(all_values)),
                    "dynamic_filter/metric_min_before": float(np.min(all_values)),
                    "dynamic_filter/metric_max_before": float(np.max(all_values)),
                }
            )
        if self.kept_group_metric_values:
            kept_values = np.asarray(self.kept_group_metric_values, dtype=np.float32)
            metrics.update(
                {
                    "dynamic_filter/metric_mean_after": float(np.mean(kept_values)),
                    "dynamic_filter/metric_min_after": float(np.min(kept_values)),
                    "dynamic_filter/metric_max_after": float(np.max(kept_values)),
                }
            )
        if filter_type == FILTER_TYPE_SAME_VALUE and self.num_prompt_groups_seen > 0:
            metrics["dynamic_filter/same_value_reject_ratio"] = (
                self.num_prompt_groups_rejected / self.num_prompt_groups_seen
            )
        if filter_type == FILTER_TYPE_BAND_PASS:
            lower_bound = _get_config_value(filter_groups_config, "lower_bound")
            upper_bound = _get_config_value(filter_groups_config, "upper_bound")
            if lower_bound is not None:
                metrics["dynamic_filter/band_lower"] = float(lower_bound)
                metrics["dynamic_filter/too_low_ratio"] = self.num_prompt_groups_too_low / max(
                    self.num_prompt_groups_seen, 1
                )
            if upper_bound is not None:
                metrics["dynamic_filter/band_upper"] = float(upper_bound)
                metrics["dynamic_filter/too_high_ratio"] = self.num_prompt_groups_too_high / max(
                    self.num_prompt_groups_seen, 1
                )
        return metrics


def resolve_filter_metric(batch: DataProto, metric: str) -> np.ndarray:
    if metric == "seq_reward":
        if "token_level_scores" not in batch.batch:
            raise ValueError("metric='seq_reward' requires token_level_scores in batch.batch.")
        return batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()

    if metric in batch.non_tensor_batch:
        metric_values = batch.non_tensor_batch[metric]
    elif metric in batch.batch:
        metric_tensor = batch.batch[metric]
        if metric_tensor.ndim != 1:
            raise ValueError(
                f"algorithm.filter_groups.metric={metric!r} resolved to tensor with shape "
                f"{tuple(metric_tensor.shape)}; expected one scalar per trajectory."
            )
        metric_values = metric_tensor.detach().cpu().numpy()
    else:
        available_batch_keys = list(batch.batch.keys()) if batch.batch is not None else []
        available_non_tensor_keys = list(batch.non_tensor_batch.keys())
        raise ValueError(
            f"algorithm.filter_groups.metric={metric!r} was not found. "
            f"Available batch keys: {available_batch_keys}. "
            f"Available non_tensor_batch keys: {available_non_tensor_keys}. "
            "For reward-function metrics, return a dict from compute_score containing the requested key. "
            "For sequence reward filtering, use metric='seq_reward'."
        )

    metric_values = np.asarray(metric_values)
    if metric_values.ndim != 1:
        raise ValueError(
            f"algorithm.filter_groups.metric={metric!r} must provide one scalar per trajectory, "
            f"got shape {metric_values.shape}."
        )
    if metric_values.shape[0] != len(batch):
        raise ValueError(
            f"algorithm.filter_groups.metric={metric!r} has length {metric_values.shape[0]}, "
            f"but batch has length {len(batch)}."
        )
    return metric_values.astype(np.float32)


def group_indices_by_uid(uids: np.ndarray) -> OrderedDict[Any, list[int]]:
    grouped_indices: OrderedDict[Any, list[int]] = OrderedDict()
    for idx, uid in enumerate(uids):
        grouped_indices.setdefault(uid, []).append(idx)
    return grouped_indices


def _should_keep_group(group_values: np.ndarray, filter_groups_config: Any) -> tuple[bool, str, float]:
    filter_type = _get_config_value(filter_groups_config, "filter_type", FILTER_TYPE_SAME_VALUE)
    group_metric = float(np.mean(group_values))

    if filter_type == FILTER_TYPE_SAME_VALUE:
        return len(group_values) == 1 or float(np.std(group_values)) > 0.0, "kept", group_metric

    lower_bound = _get_config_value(filter_groups_config, "lower_bound")
    upper_bound = _get_config_value(filter_groups_config, "upper_bound")
    if lower_bound is not None and group_metric < float(lower_bound) - BOUND_TOLERANCE:
        return False, "too_low", group_metric
    if upper_bound is not None and group_metric > float(upper_bound) + BOUND_TOLERANCE:
        return False, "too_high", group_metric
    return True, "kept", group_metric


def compute_group_keep_mask(
    metric_values: np.ndarray, uids: np.ndarray, filter_groups_config: Any, state: DynamicFilterState
) -> np.ndarray:
    grouped_indices = group_indices_by_uid(uids)
    keep_mask = np.zeros(metric_values.shape[0], dtype=bool)

    for indices in grouped_indices.values():
        group_values = metric_values[indices]
        should_keep, reason, group_metric = _should_keep_group(group_values, filter_groups_config)
        state.num_prompt_groups_seen += 1
        state.all_group_metric_values.append(group_metric)
        if should_keep:
            state.num_prompt_groups_kept += 1
            state.kept_group_metric_values.append(group_metric)
            keep_mask[indices] = True
        else:
            state.num_prompt_groups_rejected += 1
            if reason == "too_low":
                state.num_prompt_groups_too_low += 1
            elif reason == "too_high":
                state.num_prompt_groups_too_high += 1

    return keep_mask


def apply_dynamic_filter(batch: DataProto, config: Any, state: DynamicFilterState) -> DynamicFilterResult:
    filter_groups_config = config.algorithm.filter_groups
    state.num_gen_batches += 1

    metric = _get_config_value(filter_groups_config, "metric")
    metric_values = resolve_filter_metric(batch, metric)

    if "uid" not in batch.non_tensor_batch:
        raise ValueError("algorithm.filter_groups requires batch.non_tensor_batch['uid'] for prompt grouping.")
    keep_mask = compute_group_keep_mask(metric_values, batch.non_tensor_batch["uid"], filter_groups_config, state)
    state.accumulate(batch[keep_mask])

    prompt_bsz = config.data.train_batch_size
    max_num_gen_batches = _get_config_value(filter_groups_config, "max_num_gen_batches", 0)
    if state.accepted_prompt_groups < prompt_bsz and (
        max_num_gen_batches <= 0 or state.num_gen_batches < max_num_gen_batches
    ):
        return DynamicFilterResult(batch=None, should_generate_more=True)

    if state.accepted_prompt_groups < prompt_bsz:
        filter_type = _get_config_value(filter_groups_config, "filter_type", FILTER_TYPE_SAME_VALUE)
        raise ValueError(
            f"Dynamic filtering accepted {state.accepted_prompt_groups} prompt groups, "
            f"but requires {prompt_bsz} after {state.num_gen_batches} generation batches. "
            f"metric={metric!r}, filter_type={filter_type!r}. "
            "Please loosen the filter thresholds or increase max_num_gen_batches."
        )

    rollout_n = config.actor_rollout_ref.rollout.n
    num_trajs_target = prompt_bsz * rollout_n
    assert state.accumulated_batch is not None
    final_batch = state.accumulated_batch[:num_trajs_target]
    return DynamicFilterResult(
        batch=final_batch,
        should_generate_more=False,
        metrics=state.metrics(filter_groups_config),
    )


def get_reward_extra_infos_from_batch(batch: DataProto) -> dict[str, np.ndarray]:
    reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
    return {key: batch.non_tensor_batch[key] for key in reward_extra_keys if key in batch.non_tensor_batch}
