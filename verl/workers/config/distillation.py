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
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    task (str, optional):
        Task identifier to route examples to the teacher model for multi-teacher support.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    """

    _mutable_fields = BaseConfig._mutable_fields

    task: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)

    def is_configured(self, is_multi: bool) -> bool:
        configured = self.model_path is not None
        if self.task is not None and not configured:
            raise ValueError(f"{self.task=} is set but model_path is not set for this teacher model config.")
        if is_multi and configured and self.task is None:
            raise ValueError("task must be specified for multi-teacher setups.")
        return configured

    def validate_and_prepare_for_distillation(self, use_topk: bool, topk: Optional[int]) -> None:
        # Prompt + Response from student are fed into teacher as context
        max_model_len = self.inference.max_model_len
        max_num_batched_tokens = self.inference.max_num_batched_tokens
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token, but got {student_prompt_length=}, "
                f"{student_response_length=}, {required_context_len=}, {max_model_len=}."
            )
        if max_num_batched_tokens is not None and required_context_len > max_num_batched_tokens:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the full student "
                f"response, and one generated token within the engine batching budget, but got "
                f"{student_prompt_length=}, {student_response_length=}, {required_context_len=}, "
                f"{max_num_batched_tokens=}."
            )

        self.inference.prompt_length = self.inference.prompt_length + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk or topk is None:
            return

        # Ensure max log probs is aligned with top-k
        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


MAX_NUM_TEACHERS = 4


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    enable_resource_pool (bool):
        Whether to enable a separate resource pool for distillation teacher model(s).
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_model (TeacherModelConfig):
        Configuration for the teacher model used for distillation.
    teacher_model0/1/2/3 (TeacherModelConfig):
        Configuration for the teacher model used for distillation with `multi_teachers`.
    distillation_loss (DistillationLossConfig):
        Configuration for distillation loss settings.
    teacher_models (dict[str, TeacherModelConfig]):
        Runtime-populated mapping of teacher model keys to their configurations.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models"}

    enabled: bool = False
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_model: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    teacher_model0: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    teacher_model1: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    teacher_model2: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    teacher_model3: DistillationTeacherModelConfig = field(default_factory=DistillationTeacherModelConfig)
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self.enabled:
            return

        self.teacher_models.clear()
        self.teacher_models.update(self._resolve_distillation_models())
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
            )
        if len(self.teacher_models) != 1:
            raise NotImplementedError("multiple teachers are not supported yet in the runtime path.")

    def get_multi_teachers(self) -> dict[str, DistillationTeacherModelConfig]:
        models = {}
        for idx in range(MAX_NUM_TEACHERS):
            key = f"teacher_model{idx}"
            teacher_model = self.get_teacher_cfg(idx)
            if teacher_model.is_configured(is_multi=True):
                models[key] = teacher_model
        return models

    def get_teacher_cfg(self, idx: int) -> DistillationTeacherModelConfig:
        if idx < 0 or idx >= MAX_NUM_TEACHERS:
            raise IndexError(
                f"teacher_model{idx} is out of range. Add it to the distillation YAML and DistillationConfig "
                f"dataclass if you need to support more than {MAX_NUM_TEACHERS} teachers."
            )
        return getattr(self, f"teacher_model{idx}")

    def get_single_teacher_model(self) -> DistillationTeacherModelConfig:
        if len(self.teacher_models) != 1:
            raise ValueError(
                f"Expected exactly one active distillation teacher config, but got {len(self.teacher_models)}."
            )
        return next(iter(self.teacher_models.values()))

    def _resolve_distillation_models(self) -> dict[str, DistillationTeacherModelConfig]:
        multi_teachers = self.get_multi_teachers()
        if self.teacher_model.is_configured(is_multi=False) and multi_teachers:
            raise ValueError("Specify either distillation.teacher_model or distillation.teacher_model{k}, not both.")
        if multi_teachers:
            return multi_teachers
        if not self.teacher_model.is_configured(is_multi=False):
            raise ValueError(
                "No distillation teacher model configured. Please configure at least one "
                "teacher model in the distillation config."
            )
        return {"teacher": self.teacher_model}
