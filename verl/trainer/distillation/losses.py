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

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

import verl.trainer.distillation.fsdp.losses as fsdp_losses
from verl.base_config import BaseConfig
from verl.trainer.distillation.types import DistillationLossInputs
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import DistillationConfig

DistillationLossFn = Callable[
    [
        DistillationLossInputs,  # inputs
        torch.Tensor,  # response_mask
        DistillationConfig,  # config
        str,  # loss_agg_mode
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


@dataclass
class DistillationLossSettings(BaseConfig):
    """Settings for a distillation loss function to be registered."""

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    use_full: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_full, self.use_topk, self.use_estimator]) > 1:
            raise ValueError(
                f"Expected only one of use_full, use_estimator, use_topk, but got "
                f"{self.use_full=}, {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    distillation_losses_response = distillation_losses[response_mask]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    inputs: DistillationLossInputs,
    response_mask: torch.Tensor,
    config: DistillationConfig,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Args:
        inputs (DistillationLossInputs):
            Inputs containing top-k log probabilities and indices of the tokens corresponding to the
            top-k log probabilities for teacher policy and logits for the student policy.
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        config (DistillationConfig):
            Distillation configuration.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]: A tuple containing:
            - distillation_loss: Aggregated distillation loss scalar.
            - distillation_metrics: Dictionary of metrics.
    """
    assert config is not None
    distillation_metrics = {}

    teacher_topk_log_probs = inputs.teacher_log_probs
    teacher_topk_indices = inputs.teacher_topk_indices
    student_logits = inputs.student_logits
    if teacher_topk_log_probs is None or teacher_topk_indices is None or student_logits is None:
        raise ValueError(
            f"Expected teacher_topk_log_probs ({teacher_topk_log_probs is None}), "
            f"teacher_topk_indices {(teacher_topk_indices is None)}, "
            f"and student_logits {(student_logits is None)} to be provided in inputs."
        )

    match config.strategy:
        case "fsdp":
            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")
    distillation_losses, student_mass, teacher_mass = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=teacher_topk_log_probs,
        teacher_topk_indices=teacher_topk_indices,
        config=config,
    )

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask]
    teacher_mass = teacher_mass[response_mask]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }
    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if config.loss_max_clamp is not None:
        distillation_losses = distillation_losses.clamp_max(config.loss_max_clamp)

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)
    distillation_loss = agg_loss(
        loss_mat=distillation_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    return distillation_loss, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    inputs: DistillationLossInputs,
    response_mask: torch.Tensor,
    config: DistillationConfig,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Args:
        inputs (DistillationLossInputs):
            Inputs containing log-probabilities of the sampled tokens under the teacher and
            student policies, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        config (DistillationConfig):
            Distillation configuration containing loss_mode and loss_clamp.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]: A tuple containing:
            - distillation_loss: Aggregated distillation loss scalar.
            - distillation_metrics: Dictionary of metrics.
    """
    assert config is not None
    student_log_probs = inputs.student_log_probs
    teacher_log_probs = inputs.teacher_log_probs
    if student_log_probs is None or teacher_log_probs is None:
        raise ValueError("Expected student_log_probs and teacher_log_probs to be provided in inputs.")
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=config.loss_mode
    )
    distillation_losses_response = distillation_losses[response_mask]
    distillation_metrics = compute_distillation_loss_range(
        distillation_losses=distillation_losses_response, response_mask=response_mask
    )
    if config.loss_clamp is not None:
        distillation_losses = distillation_losses.clamp_max(config.loss_clamp)

    distillation_loss = agg_loss(
        loss_mat=distillation_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return distillation_loss, distillation_metrics
