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
Shared DeepSpeed parallel-layout helpers.

Design note:
- DeepSpeed workers in this branch run with SP disabled.
- Layout information is still exposed through `ParallelLayout` so call sites can
  keep a uniform interface with other backends.
- Batch normalization is handled once here to avoid role-specific drift.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import torch.distributed as dist
from omegaconf import DictConfig, open_dict


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
        """All ranks collect when SP is disabled."""
        return True


def _get_attr(cfg: Any, name: str, default: Any = None) -> Any:
    """Read one field from dataclass-like, DictConfig, or dict configs."""
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


def _set_attr(cfg: Any, name: str, value: Any) -> None:
    """Set one field on common config containers (best effort)."""
    if cfg is None:
        return
    if isinstance(cfg, DictConfig):
        try:
            with open_dict(cfg):
                cfg[name] = value
        except Exception:
            if name in cfg:
                cfg[name] = value
        return
    if isinstance(cfg, dict):
        cfg[name] = value
        return
    if hasattr(cfg, name):
        try:
            setattr(cfg, name, value)
        except FrozenInstanceError:
            pass


def resolve_and_sync_sp_size(role_cfg: Any) -> int:
    """
    Force SP to `1` for DeepSpeed workers.

    We also write back the normalized value to config views so later reads are
    consistent (`role_cfg` and its `deepspeed_config` section).
    """
    ds_cfg = _get_attr(role_cfg, "deepspeed_config", None)
    sp_size = 1

    # Keep both config views aligned with the enforced value.
    if _get_attr(role_cfg, "ulysses_sequence_parallel_size", None) != sp_size:
        _set_attr(role_cfg, "ulysses_sequence_parallel_size", sp_size)
    # DeepSpeedEngineConfig does not define ulysses_sequence_parallel_size.
    # Only write to DS sub-config when the field already exists.
    ds_sp = _get_attr(ds_cfg, "ulysses_sequence_parallel_size", None)
    if ds_sp is not None and ds_sp != sp_size:
        _set_attr(ds_cfg, "ulysses_sequence_parallel_size", sp_size)

    return int(sp_size)


def build_parallel_layout(role_cfg: Any, tp_size: int = 1) -> ParallelLayout:
    """Build per-rank layout metadata for DeepSpeed workers."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before building ParallelLayout.")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    _ = resolve_and_sync_sp_size(role_cfg)
    sp_size = 1
    dp_size = world_size
    dp_rank = rank
    sp_rank = 0

    return ParallelLayout(
        world_size=world_size,
        rank=rank,
        sp_size=sp_size,
        dp_size=dp_size,
        dp_rank=dp_rank,
        sp_rank=sp_rank,
        tp_size=tp_size,
    )


def normalize_actor_batches(actor_cfg: Any, rollout_n: int, dp_size: int):
    """Normalize actor PPO batch sizes to per-rank values."""
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


def normalize_critic_batches(critic_cfg: Any, dp_size: int):
    """Normalize critic PPO batch sizes to per-rank values."""
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
