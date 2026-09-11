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
"""Sample-level dispatch over ReplayBufferAsync.

Keeps an in-flight sample cap and dispatches the next prompt once newly finished
samples reach that prompt's GRPO group size, regardless of which groups those
completions came from. The default async path is batch-level: ``prepare_step``
dumps a full batch every iteration.

This sampler owns dispatch. The trainer must not add warmup / per-step batches
(``trainer_owns_dispatch = False``).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync, _accumulate_eviction_metrics

logger = logging.getLogger(__name__)


def _kw_int(kwargs: Any, key: str, default: int) -> int:
    value = default
    if kwargs is not None and hasattr(kwargs, "get"):
        raw = kwargs.get(key, default)
        if raw is not None:
            value = raw
    return int(value)


def count_inflight_samples(
    pending_uids: set[str],
    running_uids: set[str],
    trajectory_keys: set[str],
    group_size: int,
) -> int:
    """Sessions still outstanding: pending groups count as ``group_size`` each."""
    written = Counter(key.split("_")[0] for key in trajectory_keys)
    total = group_size * len(pending_uids)
    for uid in running_uids:
        total += max(0, group_size - written.get(uid, 0))
    return total


def compute_sample_level_dispatch(
    *,
    group_size: int,
    max_inflight_samples: int,
    inflight_samples: int,
    sample_credit: int,
    sampleable_groups: int,
    target_groups: int,
) -> tuple[int, int]:
    """Return ``(dispatch_prompts, remaining_credit)``.

    Empty pipeline seeds up to the in-flight cap. Otherwise one new prompt is
    worth ``group_size`` completed samples, taken from any mix of groups.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size}")
    if max_inflight_samples < group_size:
        raise ValueError(f"max_inflight_samples ({max_inflight_samples}) must be >= group_size ({group_size})")
    if sampleable_groups >= target_groups:
        return 0, sample_credit
    if inflight_samples <= 0:
        return max_inflight_samples // group_size, sample_credit
    available = max(0, max_inflight_samples - inflight_samples)
    dispatch = min(available // group_size, sample_credit // group_size)
    return dispatch, sample_credit - dispatch * group_size


class SampleLevelReplayBuffer(ReplayBufferAsync):
    """ReplayBufferAsync that refills by completed-sample credit, not by batch."""

    trainer_owns_dispatch = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = _kw_int(self.sampler_kwargs, "group_size", 8)
        self.max_inflight_samples = _kw_int(self.sampler_kwargs, "max_inflight_samples", self.group_size * 8)
        if self.group_size <= 0:
            raise ValueError(f"group_size must be a positive integer, got {self.group_size}")
        if self.max_inflight_samples < self.group_size:
            raise ValueError(
                f"max_inflight_samples ({self.max_inflight_samples}) must be >= group_size ({self.group_size})"
            )
        self._sample_credit = 0
        self._seen_traj_keys: set[str] = set()
        self._dispatched_prompts = 0
        logger.info(
            "SampleLevelReplayBuffer: group_size=%s max_inflight_samples=%s",
            self.group_size,
            self.max_inflight_samples,
        )

    def _update_sample_credit(self, partition_id: str) -> int:
        current = set(self.partitions[partition_id].keys())
        new_keys = current - self._seen_traj_keys
        self._sample_credit += len(new_keys)
        self._seen_traj_keys = current
        return len(new_keys)

    def _inflight_samples(self, partition_id: str) -> int:
        return count_inflight_samples(
            self.pending_keys[partition_id],
            self.running_keys[partition_id],
            set(self.partitions[partition_id].keys()),
            self.group_size,
        )

    def _maybe_dispatch(self, partition_id: str, sampleable_keys: set[str], target_count: int) -> int:
        if partition_id == "val" or self.refill_fn is None:
            return 0
        self._update_sample_credit(partition_id)
        inflight = self._inflight_samples(partition_id)
        dispatch, remaining = compute_sample_level_dispatch(
            group_size=self.group_size,
            max_inflight_samples=self.max_inflight_samples,
            inflight_samples=inflight,
            sample_credit=self._sample_credit,
            sampleable_groups=len(sampleable_keys),
            target_groups=target_count,
        )
        if dispatch <= 0:
            return 0
        self.refill_fn(dispatch)
        self._sample_credit = remaining
        self._dispatched_prompts += dispatch
        logger.info(
            "sample-level dispatch=%s inflight=%s credit=%s sampleable=%s/%s",
            dispatch,
            inflight,
            self._sample_credit,
            len(sampleable_keys),
            target_count,
        )
        return dispatch

    def wait_for_sampleable(self, global_steps: int, partition_id: str, target_count: int) -> tuple[set[str], dict]:
        last_debug_time = time.time()
        eviction_metrics: dict = {}

        while True:
            self._sync_metadata_from_transfer_queue()

            eviction_reasons = self._terminal_eviction_reasons(global_steps, partition_id)
            evicted_uids, stale_count, _dapo_count, metrics = self._evict_terminal_groups(
                global_steps, partition_id, eviction_reasons
            )
            if evicted_uids:
                _accumulate_eviction_metrics(eviction_metrics, metrics, stale_count)

            sampleable_keys = self._sampleable_terminal_keys(partition_id, eviction_reasons)
            if self._has_enough_samples(global_steps, partition_id, target_count, sampleable_keys):
                inflight = self._inflight_samples(partition_id)
                eviction_metrics["training/sample_level/dispatched_prompts"] = self._dispatched_prompts
                eviction_metrics["training/sample_level/inflight_samples"] = inflight
                eviction_metrics["training/sample_level/sample_credit"] = self._sample_credit
                return sampleable_keys, eviction_metrics

            if self._maybe_dispatch(partition_id, sampleable_keys, target_count):
                continue

            last_debug_time = self._wait_for_next_poll(partition_id, last_debug_time)
