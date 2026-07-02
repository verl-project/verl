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
"""FSDP-teacher counterpart to ``MultiTeacherModelManager``.

When ``distillation.teacher_backend == 'fsdp'``, the trainer hosts each teacher
as a Ray worker group of :class:`TeacherFSDPWorker` instead of a vLLM rollout
replica. This module provides a thin manager that holds the spawned worker group
handle(s) and dispatches ``compute_logprobs_at_ids`` calls to them.

Worker spawning happens in ``ray_trainer.py`` (which registers the worker on the
teacher resource pool, spawns, then calls :meth:`set_worker_group`). Single-teacher
is wired; multi-teacher FSDP raises ``NotImplementedError``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from omegaconf import DictConfig
from tensordict import TensorDict

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPTeacherManager:
    """Per-teacher wrapper around a Ray worker group of :class:`TeacherFSDPWorker`.

    Mirrors the per-teacher surface of the vLLM ``TeacherModelManager`` so the
    trainer's backend branching stays symmetric, but does *not* spawn its own
    workers — the trainer spawns the worker group and calls :meth:`set_worker_group`.
    """

    def __init__(self, key: str, distillation_config: DistillationConfig):
        self.key = key
        self.distillation_config = distillation_config
        self.worker_group: Any = None

    def set_worker_group(self, worker_group: Any) -> None:
        """Install the spawned :class:`TeacherFSDPWorker` Ray worker group handle."""
        self.worker_group = worker_group

    def compute_logprobs_at_ids(self, data: TensorDict) -> Optional[TensorDict]:
        """Forward a teacher-side ``compute_logprobs_at_ids`` call to the worker group.

        See :meth:`verl.workers.teacher.fsdp_teacher_worker.TeacherFSDPWorker.compute_logprobs_at_ids`
        for the data contract.
        """
        if self.worker_group is None:
            raise RuntimeError(
                f"FSDPTeacherManager[{self.key!r}].worker_group has not been installed yet — "
                "the trainer must call set_worker_group(...) after spawning the unified "
                "Ray worker dict in init_workers."
            )
        return self.worker_group.compute_logprobs_at_ids(data)


class MultiTeacherFSDPManager:
    """Manages one :class:`FSDPTeacherManager` per teacher routing key.

    Worker-group spawn is driven by the trainer (so the teacher worker joins the
    unified ``RayWorkerGroup`` spawn); the trainer then calls :meth:`set_worker_group`
    to install the handle.
    """

    def __init__(self, config: DictConfig, resource_pool: Any):
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        if self.distillation_config.teacher_backend != "fsdp":
            raise ValueError(
                f"MultiTeacherFSDPManager requires distillation.teacher_backend='fsdp', "
                f"got {self.distillation_config.teacher_backend!r}."
            )
        if self.distillation_config.teacher_fsdp_config is None:
            raise ValueError("distillation.teacher_fsdp_config must be set when teacher_backend='fsdp'.")

        # Resource pool is held for symmetry with MultiTeacherModelManager (the
        # vLLM equivalent), but the FSDP path drives spawn from the trainer.
        self.resource_pool = resource_pool

        teacher_models = self.distillation_config.teacher_models
        if len(teacher_models) != 1:
            raise NotImplementedError(
                "Multi-teacher FSDP backend is not yet supported; configure exactly one teacher "
                "(teacher_models has 1 entry) when teacher_backend='fsdp'."
            )

        self.teacher_managers: dict[str, FSDPTeacherManager] = {
            key: FSDPTeacherManager(key=key, distillation_config=self.distillation_config) for key in teacher_models
        }

    @property
    def teacher_keys(self) -> list[str]:
        return list(self.teacher_managers)

    def set_worker_group(self, worker_group: Any, key: Optional[str] = None) -> None:
        """Install the spawned worker group handle for the (single) teacher.

        Args:
            worker_group: spawned Ray worker group handle returned by
                ``wg_dict.spawn(...)``.
            key: routing key of the teacher to attach the worker group to. May
                be omitted in single-teacher setups.
        """
        if key is None:
            if len(self.teacher_managers) != 1:
                raise ValueError(
                    "MultiTeacherFSDPManager.set_worker_group requires an explicit `key` when "
                    "multiple teachers are configured."
                )
            key = next(iter(self.teacher_managers))
        if key not in self.teacher_managers:
            raise ValueError(f"Teacher key {key!r} not configured; configured: {self.teacher_keys}.")
        self.teacher_managers[key].set_worker_group(worker_group)

    def get(self, routing_key: Optional[str] = None) -> FSDPTeacherManager:
        if len(self.teacher_managers) == 1:
            return next(iter(self.teacher_managers.values()))
        if routing_key is None:
            raise ValueError("Routing key is required when multiple FSDP teachers are configured.")
        if routing_key not in self.teacher_managers:
            raise ValueError(
                f"No FSDP teacher configured for routing key {routing_key!r}; "
                f"configured teachers: {sorted(self.teacher_managers)}."
            )
        return self.teacher_managers[routing_key]

    def compute_logprobs_at_ids(self, data: TensorDict, routing_key: Optional[str] = None) -> Optional[TensorDict]:
        return self.get(routing_key).compute_logprobs_at_ids(data)


__all__ = ["FSDPTeacherManager", "MultiTeacherFSDPManager"]
