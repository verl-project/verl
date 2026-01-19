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

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
import torch.nn.functional as F

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import DistillationConfig


class DistillationLossFn(Protocol):
    """Protocol for distillation loss functions."""

    def __call__(
        self,
        *,
        teacher_log_probs: torch.Tensor,
        student_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        config: DistillationConfig,
        loss_agg_mode: str = "token-mean",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...


@dataclass
class DistillationLossSettings(BaseConfig):
    """Settings for a distillation loss function to be registered."""

    names: str | list[str] = field(default_factory=list)
    use_student_topk: bool = False
    use_teacher_topk: bool = False
    use_estimator: bool = False
    use_full: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        self.use_topk = self.use_student_topk or self.use_teacher_topk
        if sum([self.use_full, self.use_topk, self.use_estimator]) > 1:
            raise ValueError(
                f"Expected only one of use_full, use_estimator, use_student_topk/use_teacher_topk, but got {self.use_full=}, {self.use_estimator=}, {self.use_student_topk=}, {self.use_teacher_topk=}."
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


def clamp_log_probs(log_p: torch.Tensor, log_q: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp log probabilities to avoid numerical instability and handle inf minus inf for masked top-k probs."""
    min_log_prob = math.log(eps)
    log_p_clamped = torch.clamp(log_p, min=min_log_prob)
    log_q_clamped = torch.clamp(log_q, min=min_log_prob)
    return log_p_clamped, log_q_clamped


def jensen_shannon_divergence(log_q: torch.Tensor, log_p: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence between two distributions given their log probabilities.

    JSD(β) = β * KL(p || m) + (1 - β) * KL(q || m), where m = beta * p + (1 - beta) * q

    The gradients of JSD(β) behave similarly to forward KL and reverse KL when β is close
    to 0 and 1 respectively. See https://arxiv.org/abs/2306.13649

    Args:
        log_q (torch.Tensor):
            Student log probabilities, shape (batch_size, response_length, vocab_size) or
            (batch_size, response_length, topk).
        log_p (torch.Tensor):
            Teacher log probabilities, same shape as log_q.
        beta (float):
            JSD interpolation weight. When beta=0, behaves like forward KL.
            When beta=1, behaves like reverse KL.

    Returns:
        torch.Tensor: JSD loss per token, shape (batch_size, response_length).
    """
    log_p, log_q = clamp_log_probs(log_p, log_q)
    q = log_q.exp()
    p = log_p.exp()
    m = beta * p + (1 - beta) * q
    log_m = m.log()
    kl1 = F.kl_div(input=log_m, target=log_p, reduction="none", log_target=True).sum(dim=-1)
    kl2 = F.kl_div(input=log_m, target=log_q, reduction="none", log_target=True).sum(dim=-1)
    loss = beta * kl1 + (1 - beta) * kl2
    return loss


def kullback_leibler_divergence(log_q: torch.Tensor, log_p: torch.Tensor, loss_mode: str = "forward") -> torch.Tensor:
    """
    Compute forward or reverse KL divergence between two distributions given their log probabilities.

    forward KL: KL(p || q) = sum(p * (log_p - log_q))
    reverse KL: KL(q || p) = sum(q * (log_q - log_p))

    Args:
        log_q (torch.Tensor):
            Student log probabilities, shape (batch_size, response_length, vocab_size) or
            (batch_size, response_length, topk).
        log_p (torch.Tensor):
            Teacher log probabilities, same shape as log_q.
        loss_mode (str, optional):
            KL divergence direction: "forward" or "reverse".

    Returns:
        torch.Tensor: KL divergence loss per token, shape (batch_size, response_length).
    """
    log_p, log_q = clamp_log_probs(log_p, log_q)
    match loss_mode:
        case "forward":
            return F.kl_div(input=log_q, target=log_p, reduction="none", log_target=True).sum(dim=-1)
        case "reverse":
            return F.kl_div(input=log_p, target=log_q, reduction="none", log_target=True).sum(dim=-1)
        case _:
            raise ValueError(f"Unsupported loss mode: {loss_mode}. Supported modes are: ['forward', 'reverse']")


@register_distillation_loss(DistillationLossSettings(names="forward_kl_topk", use_teacher_topk=True))  # type: ignore[arg-type]
@register_distillation_loss(DistillationLossSettings(names="reverse_kl_topk", use_student_topk=True))  # type: ignore[arg-type]
@register_distillation_loss(DistillationLossSettings(names="jsd_topk", use_student_topk=True, use_teacher_topk=True))  # type: ignore[arg-type]
def compute_distillation_loss_topk(
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    student_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
    config: DistillationConfig,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using top-k log probabilities.

    Supports forward KL, reverse KL, and JSD loss modes. The teacher and student top-k
    indices must match (i.e., loss is computed over the same vocabulary subset).

    Args:
        teacher_log_probs (torch.Tensor):
            Log-probabilities under the teacher policy, shape (batch_size, response_length).
        student_log_probs (torch.Tensor):
            Log-probabilities under the student policy, shape (batch_size, response_length).
        teacher_topk_logprobs (torch.Tensor):
            Top-k log-probabilities under the teacher policy,
            shape (batch_size, response_length, topk) or (batch_size, response_length, 2*topk).
        student_topk_logprobs (torch.Tensor):
            Top-k log-probabilities under the student policy, same shape as teacher_topk_logprobs.
        teacher_topk_indices (torch.Tensor):
            Top-k token indices under the teacher policy, same shape as teacher_topk_logprobs.
        student_topk_indices (torch.Tensor):
            Top-k token indices under the student policy, same shape as student_topk_logprobs.
            Must be equal to teacher_topk_indices.
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
    loss_settings: DistillationLossSettings = config.loss_settings
    distillation_metrics = {}
    if not teacher_topk_indices.equal(student_topk_indices):
        raise ValueError(
            f"Expected teacher and student topk indices to be the same, but got {teacher_topk_indices} and {student_topk_indices}."
        )
    topk = config.topk
    if loss_settings.use_student_topk and loss_settings.use_teacher_topk:
        expected_num_logprobs = 2 * topk
    elif loss_settings.use_student_topk or loss_settings.use_teacher_topk:
        expected_num_logprobs = topk
    else:
        raise ValueError(
            f"Expected at least one of student or teacher topk logprobs to be used, but got {loss_settings.use_student_topk} and {loss_settings.use_teacher_topk}."
        )
    if (
        teacher_topk_logprobs.shape[-1] != expected_num_logprobs
        or student_topk_logprobs.shape[-1] != expected_num_logprobs
    ):
        raise ValueError(
            f"Expected topk logprobs to have shape (batch_size, response_length, {expected_num_logprobs}), but got {teacher_topk_logprobs.shape=} and {student_topk_logprobs.shape=}."
        )

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_topk_logprobs.exp().sum(dim=-1)[response_mask]
    teacher_mass = teacher_topk_logprobs.exp().sum(dim=-1)[response_mask]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
    }
    match config.loss_mode:
        case "forward_kl_topk" | "forward_kl_full":
            distillation_losses = kullback_leibler_divergence(
                log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, loss_mode="forward"
            )
        case "reverse_kl_topk" | "reverse_kl_full":
            distillation_losses = kullback_leibler_divergence(
                log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, loss_mode="reverse"
            )
        case "jsd_topk" | "jsd_full":
            distillation_losses = jensen_shannon_divergence(
                log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, beta=config.jsd_beta
            )
        case _:
            raise NotImplementedError(f"Unsupported distillation loss mode: {config.loss_mode}")
    distillation_metrics.update(
        {
            "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses.min()),
            "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses.max()),
        }
    )
    if config.loss_clamp is not None:
        distillation_losses = distillation_losses.clamp_max(config.loss_clamp)
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
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    config: DistillationConfig,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Args:
        teacher_log_probs (torch.Tensor):
            Log-probabilities of the sampled actions under the teacher policy,
            shape (batch_size, response_length).
        student_log_probs (torch.Tensor):
            Log-probabilities of the sampled actions under the student policy,
            shape (batch_size, response_length).
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
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=config.loss_mode
    )
    distillation_metrics = {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses.max()),
    }
    if config.loss_clamp is not None:
        distillation_losses = distillation_losses.clamp_max(config.loss_clamp)

    distillation_loss = agg_loss(
        loss_mat=distillation_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return distillation_loss, distillation_metrics
