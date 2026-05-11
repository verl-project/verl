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

"""Helpers for KV cache handling around SGLang weight sync."""

from __future__ import annotations

from typing import Any


def should_flush_kv_cache_after_weight_sync(rollout_cfg: Any) -> bool:
    """Return whether VERL should call ``flush_cache`` after ``sgl_update_weights`` buckets.

    When ``allow_stale_kv_cache_after_weight_sync`` is True (PipelineRL-style async rollout),
    returns False so KV from prior forwards may remain while weights are already updated;
    logits can be inconsistent by design.

    Args:
        rollout_cfg: ``RolloutConfig`` or structured config node (e.g. OmegaConf) for
            ``actor_rollout_ref.rollout``.
    """
    if rollout_cfg is None:
        return True
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(rollout_cfg):
            stale = OmegaConf.select(rollout_cfg, "allow_stale_kv_cache_after_weight_sync", default=False)
            return not bool(stale)
    except Exception:
        pass
    stale = getattr(rollout_cfg, "allow_stale_kv_cache_after_weight_sync", False)
    return not bool(stale)
