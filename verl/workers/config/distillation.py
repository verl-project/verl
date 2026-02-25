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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

from .rollout import RolloutConfig

__all__ = ["DistillationLossConfig", "DistillationTeacherModelConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_policy_loss (bool):
        Whether to include policy gradient loss alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with policy loss.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_policy_loss: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    enable_resource_pool (bool):
        Whether to enable separate resource pool for teacher model(s).
    n_gpus_per_node (int):
        Number of GPUs per node to use for distillation teacher model(s).
    nnodes (int):
        Number of nodes to use for distillation teacher model(s).
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    """

    _mutable_fields = BaseConfig._mutable_fields

    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    num_workers (int):
        Number of teacher model replicas.
    teacher_model (TeacherModelConfig):
        Configuration for the teacher model used for distillation.
    distillation_loss (DistillationLossConfig):
        Configuration for distillation loss settings.
    """

    _mutable_fields = BaseConfig._mutable_fields

    enabled: bool = False
    num_workers: int = 8
    teacher_model: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        engine_name = self.teacher_model.inference.name
        engine_kwargs = self.teacher_model.inference.engine_kwargs
        if self.distillation_loss.topk is None or not self.enabled:
            return
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = self.distillation_loss.topk
                    max_logprobs = self.distillation_loss.topk
                if max_logprobs < self.distillation_loss.topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({self.distillation_loss.topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )
