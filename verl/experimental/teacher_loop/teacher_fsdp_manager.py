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
replica. This module exposes the matching multi-teacher manager
(``MultiTeacherFSDPManager``) and a per-teacher wrapper (``FSDPTeacherManager``).

Worker-group spawning is finalized alongside the trainer wiring in
``verl/trainer/ppo/ray_trainer.py``; the not-yet-wired entrypoints raise
``NotImplementedError``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from omegaconf import DictConfig
from tensordict import TensorDict

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import DistillationConfig, DistillationTeacherModelConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FSDPTeacherManager:
    """Per-teacher wrapper around a Ray worker group of :class:`TeacherFSDPWorker`.

    Mirrors the public surface of the vLLM equivalent ``TeacherModelManager`` so the
    trainer's resource-pool plumbing can switch on ``teacher_backend`` transparently.

    Args:
        distillation_config: Resolved distillation config (must have
            ``teacher_backend == 'fsdp'`` and a non-None ``teacher_fsdp_config``).
        teacher_model_config: Per-teacher entry from
            ``distillation_config.teacher_models``.
        resource_pool: Dedicated Ray resource pool for this teacher.
    """

    def __init__(
        self,
        distillation_config: DistillationConfig,
        teacher_model_config: DistillationTeacherModelConfig,
        resource_pool: RayResourcePool,
    ):
        self.distillation_config = distillation_config
        self.teacher_model_config = teacher_model_config
        self.resource_pool = resource_pool
        self.worker_group: Any = None  # populated by _spawn_worker_group

    def _spawn_worker_group(self) -> Any:
        """Build a Ray worker group of :class:`TeacherFSDPWorker` over this
        manager's resource pool.

        Deferred to the trainer integration, which reuses the existing
        ``RayWorkerGroup`` plumbing in ``ray_trainer.py``.
        """
        raise NotImplementedError(
            "FSDPTeacherManager._spawn_worker_group is wired alongside the "
            "trainer's resource-pool dispatch refactor."
        )

    def compute_logprobs_at_ids(self, data: TensorDict) -> Optional[TensorDict]:
        """Dispatch a forward + chunked gather to the teacher worker group.

        Returns the trainer-facing tensor dict with ``teacher_on_student_logp``
        populated as a nested tensor of shape ``(bsz, seq_len_i, K)``.
        """
        if self.worker_group is None:
            raise RuntimeError(
                "FSDPTeacherManager.compute_logprobs_at_ids called before _spawn_worker_group. "
                "Wire this manager into the trainer's main loop before invoking."
            )
        # Will become ``return self.worker_group.compute_logprobs_at_ids(data)``
        # once the worker group dispatch decorators are aligned with the trainer.
        raise NotImplementedError(
            "FSDPTeacherManager.compute_logprobs_at_ids is wired alongside the "
            "trainer's main-loop refactor."
        )


class MultiTeacherFSDPManager:
    """Manages one :class:`FSDPTeacherManager` per teacher routing key.

    Mirrors the vLLM ``MultiTeacherModelManager`` so the trainer can branch on
    ``distillation.teacher_backend`` while keeping the rest of the glue identical.
    """

    def __init__(self, config: DictConfig, resource_pool: RayResourcePool):
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        if self.distillation_config.teacher_backend != "fsdp":
            raise ValueError(
                f"MultiTeacherFSDPManager requires distillation.teacher_backend='fsdp', "
                f"got {self.distillation_config.teacher_backend!r}."
            )
        if self.distillation_config.teacher_fsdp_config is None:
            raise ValueError(
                "distillation.teacher_fsdp_config must be set when teacher_backend='fsdp'."
            )

        self.resource_pool = resource_pool
        self.teacher_managers: dict[str, FSDPTeacherManager] = {}
        self._initialize_teacher_managers()

    def _initialize_teacher_managers(self) -> None:
        teacher_models = self.distillation_config.teacher_models
        # FSDP teachers ignore vllm/sglang inference parallelism; num_replicas=1 is
        # pinned in config post-init. Multi-teacher splits the pool per teacher.
        split_sizes = [self.resource_pool.world_size] if len(teacher_models) == 1 else None
        if split_sizes is None:
            raise NotImplementedError(
                "Multi-teacher FSDP pool sizing is wired once the trainer-side "
                "teacher resource pool layout is finalized."
            )
        split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)
        for (key, teacher_model_config), teacher_pool in zip(
            teacher_models.items(), split_pools, strict=True
        ):
            self.teacher_managers[key] = FSDPTeacherManager(
                distillation_config=self.distillation_config,
                teacher_model_config=teacher_model_config,
                resource_pool=teacher_pool,
            )

    def get(self, routing_key: Optional[str] = None) -> FSDPTeacherManager:
        if len(self.teacher_managers) == 1:
            return next(iter(self.teacher_managers.values()))
        if routing_key is None:
            raise ValueError(
                "Routing key is required when multiple FSDP teachers are configured."
            )
        if routing_key not in self.teacher_managers:
            raise ValueError(
                f"No FSDP teacher configured for routing key {routing_key!r}; "
                f"configured teachers: {sorted(self.teacher_managers)}."
            )
        return self.teacher_managers[routing_key]


__all__ = ["FSDPTeacherManager", "MultiTeacherFSDPManager"]
