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
"""
Unified helpers for DeepSpeed TP/DP/SP layout and batch normalization.

DeepSpeed workers historically fetched SP/DP/TP settings from multiple
locations (top-level config, deepspeed_config block, rollout config)
and reimplemented batch-size normalization per role. This module provides
one entry point to:
  - resolve and synchronize the Ulysses SP size for a role
  - build a parallel layout (dp_rank, sp_rank, collect mask)
  - normalize per-rank batch sizes consistently
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import torch.distributed as dist
from omegaconf import DictConfig


@dataclass
class ParallelLayout:
    world_size: int
    rank: int
    sp_size: int
    dp_size: int
    dp_rank: int
    sp_rank: int
    tp_size: int = 1  # for rollout engines (vLLM/sglang)

    @property
    def collect(self) -> bool:
        """Whether this rank should participate in collect for DP mesh."""
        return self.sp_rank == 0


def _get_attr(cfg: Any, name: str, default: int | None = None) -> int | None:
    """Read attribute/key from either dataclass-like or dict-like configs."""
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if isinstance(cfg, DictConfig) and name in cfg:
        return cfg.get(name, default)
    if isinstance(cfg, dict) and name in cfg:
        return cfg.get(name, default)
    if name == "deepspeed_config":
        alt = "deepspeed"
        if hasattr(cfg, alt):
            return getattr(cfg, alt)
        if isinstance(cfg, DictConfig) and alt in cfg:
            return cfg.get(alt, default)
        if isinstance(cfg, dict) and alt in cfg:
            return cfg.get(alt, default)
    return default


def resolve_and_sync_sp_size(role_cfg: Any) -> int:
    """
    Resolve SP size from the role config and its deepspeed_config block.
    Make sure the two places are kept in sync to avoid silent mismatches.
    """
    top_sp = _get_attr(role_cfg, "ulysses_sequence_parallel_size", None)
    ds_cfg = _get_attr(role_cfg, "deepspeed_config", None)
    ds_sp = _get_attr(ds_cfg, "ulysses_sequence_parallel_size", None) if ds_cfg is not None else None

    # Pick the first non-None value, default to 1
    sp_candidates = [x for x in (top_sp, ds_sp) if x is not None]
    sp_size = sp_candidates[0] if len(sp_candidates) > 0 else 1

    # Enforce consistency
    if top_sp is not None and ds_sp is not None and top_sp != ds_sp:
        raise ValueError(
            f"ulysses_sequence_parallel_size mismatch: top-level={top_sp}, deepspeed_config={ds_sp}. "
            "Please set them to the same value."
        )

    # Sync both views
    if hasattr(role_cfg, "ulysses_sequence_parallel_size"):
        if _get_attr(role_cfg, "ulysses_sequence_parallel_size") != sp_size:
            try:
                role_cfg.ulysses_sequence_parallel_size = sp_size
            except FrozenInstanceError:
                pass
    if ds_cfg is not None and hasattr(ds_cfg, "ulysses_sequence_parallel_size"):
        if _get_attr(ds_cfg, "ulysses_sequence_parallel_size") != sp_size:
            try:
                ds_cfg.ulysses_sequence_parallel_size = sp_size
            except FrozenInstanceError:
                pass

    return int(sp_size)


def build_parallel_layout(role_cfg: Any, tp_size: int = 1) -> ParallelLayout:
    """Construct a ParallelLayout for a role using its SP size and global world info."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before building ParallelLayout.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    sp_size = resolve_and_sync_sp_size(role_cfg)
    if sp_size < 1 or world_size % sp_size != 0:
        raise ValueError(f"world_size {world_size} must be divisible by ulysses sp_size {sp_size}")

    dp_size = world_size // sp_size
    dp_rank = rank // sp_size
    sp_rank = rank % sp_size

    return ParallelLayout(
        world_size=world_size,
        rank=rank,
        sp_size=sp_size,
        dp_size=dp_size,
        dp_rank=dp_rank,
        sp_rank=sp_rank,
        tp_size=tp_size,
    )


def normalize_actor_batches(actor_cfg: Any, rollout_n: int, dp_size: int, sp_size: int = 1):
    """
    Normalize actor batch config to per-DP-rank values.
    """
    actor_cfg.ppo_mini_batch_size *= rollout_n
    actor_cfg.ppo_mini_batch_size //= dp_size
    if actor_cfg.ppo_mini_batch_size <= 0:
        raise ValueError(f"Normalized actor ppo_mini_batch_size {actor_cfg.ppo_mini_batch_size} must be > 0")

    derived_from_mbs = False
    if actor_cfg.ppo_micro_batch_size is not None:
        micro = actor_cfg.ppo_micro_batch_size // dp_size
        if micro <= 0:
            raise ValueError(
                f"actor.ppo_micro_batch_size becomes {micro} after normalization (dp={dp_size})"
            )
        actor_cfg.ppo_micro_batch_size = micro
        actor_cfg.ppo_micro_batch_size_per_gpu = micro
        derived_from_mbs = True

    if actor_cfg.ppo_micro_batch_size_per_gpu is not None and not derived_from_mbs:
        micro = actor_cfg.ppo_micro_batch_size_per_gpu

    if actor_cfg.ppo_micro_batch_size_per_gpu is not None:
        assert actor_cfg.ppo_mini_batch_size % actor_cfg.ppo_micro_batch_size_per_gpu == 0, (
            f"normalized ppo_mini_batch_size {actor_cfg.ppo_mini_batch_size} must be divisible by "
            f"ppo_micro_batch_size_per_gpu {actor_cfg.ppo_micro_batch_size_per_gpu}"
        )


def normalize_critic_batches(critic_cfg: Any, dp_size: int, sp_size: int = 1):
    """
    Normalize critic batch config to per-DP-rank values.
    """
    critic_cfg.ppo_mini_batch_size //= dp_size
    if critic_cfg.ppo_mini_batch_size <= 0:
        raise ValueError(f"Normalized critic ppo_mini_batch_size {critic_cfg.ppo_mini_batch_size} must be > 0")

    derived_from_mbs = False
    if getattr(critic_cfg, "ppo_micro_batch_size", None) is not None:
        micro = critic_cfg.ppo_micro_batch_size // dp_size
        if micro <= 0:
            raise ValueError(
                f"critic.ppo_micro_batch_size becomes {micro} after normalization (dp={dp_size})"
            )
        critic_cfg.ppo_micro_batch_size = micro
        critic_cfg.ppo_micro_batch_size_per_gpu = micro
        derived_from_mbs = True

    if critic_cfg.ppo_micro_batch_size_per_gpu is not None and not derived_from_mbs:
        micro = critic_cfg.ppo_micro_batch_size_per_gpu

    if critic_cfg.ppo_micro_batch_size_per_gpu is not None:
        assert critic_cfg.ppo_mini_batch_size % critic_cfg.ppo_micro_batch_size_per_gpu == 0, (
            f"normalized ppo_mini_batch_size {critic_cfg.ppo_mini_batch_size} must be divisible by "
            f"ppo_micro_batch_size_per_gpu {critic_cfg.ppo_micro_batch_size_per_gpu}"
        )
