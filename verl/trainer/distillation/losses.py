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

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Any, Union
from omegaconf import DictConfig
from verl.workers.config import ActorConfig, DistillationConfig
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.metric import Metric, AggregationType 
from dataclasses import dataclass

DistillationLossFn = Callable[
    [
        torch.Tensor,  # teacher_log_probs
        torch.Tensor,  # student_log_probs
        torch.Tensor,  # teacher_topk_logprobs
        torch.Tensor,  # student_topk_logprobs
        torch.Tensor,  # teacher_topk_indices
        torch.Tensor,  # student_topk_indices
        torch.Tensor,  # response_mask
        DistillationConfig,   # config
        str,           # loss_agg_mode
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
USE_STUDENT_TOPK_REGISTRY: set[str] = set()
USE_TEACHER_TOPK_REGISTRY: set[str] = set()
USE_FULL_REGISTRY: set[str] = set()

@dataclass
class DistillationLossInfo:
    """Information about a distillation loss function to be registered."""
    names: Union[list[str], str]
    use_student_topk: bool = False
    use_teacher_topk: bool = False
    use_full: bool = False

    def __post_init__(self):
        self.use_topk = self.use_student_topk or self.use_teacher_topk
        if sum([self.use_full, self.use_topk]) > 1:
            raise ValueError(
                f"Expected only one of use_full, use_student_topk/use_teacher_topk, but got {self.use_full=}, {self.use_student_topk=}, {self.use_teacher_topk=}."
            )

    @classmethod
    def from_loss_name(cls, loss_name: str) -> "DistillationLossInfo":
        """Create a DistillationLossInfo object from a loss name.

        Args:
            loss_name (str): The name of the distillation loss function.
        
        Returns:
            DistillationLossInfo: An object containing information about the distillation loss function.
        """
        if loss_name not in DISTILLATION_LOSS_REGISTRY:
            raise ValueError(
                f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
            )
        return cls(names=loss_name, use_student_topk=use_student_topk_logprobs(loss_name), use_teacher_topk=use_teacher_topk_logprobs(loss_name), use_full=use_full_logprobs(loss_name))
             

def register_distillation_loss(loss_info: DistillationLossInfo) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name.

    Args:
        loss_info (DistillationLossInfo): Information about the distillation loss to register.

    Returns:
        function: Decorator function that registers the distillation loss function.
    """

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        names = loss_info.names if isinstance(loss_info.names, list) else [loss_info.names]
        for name in names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            if loss_info.use_student_topk:
                USE_STUDENT_TOPK_REGISTRY.add(name)
            if loss_info.use_teacher_topk:
                USE_TEACHER_TOPK_REGISTRY.add(name)
            if loss_info.use_full:
                USE_FULL_REGISTRY.add(name)
        return func

    return decorator

def use_student_topk_logprobs(loss_name: str) -> bool:
    """Check if the distillation loss function with the given name uses student top-k log probabilities.

    Args:
        loss_name (str): The name of the distillation loss function.

    Returns:
        bool: True if the distillation loss function uses student top-k log probabilities, False otherwise.
    """
    return loss_name in USE_STUDENT_TOPK_REGISTRY

def use_teacher_topk_logprobs(loss_name: str) -> bool:
    """Check if the distillation loss function with the given name uses teacher top-k log probabilities.

    Args:
        loss_name (str): The name of the distillation loss function.

    Returns:
        bool: True if the distillation loss function uses teacher top-k log probabilities, False otherwise.
    """
    return loss_name in USE_TEACHER_TOPK_REGISTRY

def use_full_logprobs(loss_name: str) -> bool:
    """Check if the distillation loss function with the given name uses full log probabilities.

    Args:
        loss_name (str): The name of the distillation loss function.

    Returns:
        bool: True if the distillation loss function uses full log probabilities, False otherwise.
    """
    return loss_name in USE_FULL_REGISTRY

def get_distillation_loss_fn(name):
    """Get the distillation loss with a given name.

    Args:
        name: `(str)`
            The name of the distillation loss.

    Returns:
        `(callable)`: The distillation loss function.
    """
    loss_name = name
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]

def clamp_log_probs(log_p: torch.Tensor, log_q: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp log probabilities to avoid numerical instability.

    Args:
        log_p (torch.Tensor): Log probabilities of the teacher distribution.
        log_q (torch.Tensor): Log probabilities of the student distribution.
        eps (float): Small constant to avoid numerical instability when computing log probabilities.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Clamped log probabilities of the teacher and student distributions.
    """
    min_log_prob = torch.log(torch.tensor(eps)).item()
    log_p_clamped = torch.clamp(log_p, min=min_log_prob)
    log_q_clamped = torch.clamp(log_q, min=min_log_prob)
    return log_p_clamped, log_q_clamped

def jensen_shannon_divergence(log_q: torch.Tensor, log_p: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Computes Jensen-Shannon Divergence between two distributions given their log probabilities.
    
    JSD(β) = β * KL(p || m) + (1 - β) * KL(q || m), where m = beta * p + (1 - beta) * q
    forward KL: KL(p || q) 
    reverse KL: KL(q || p) 

    "gradients of JSD(β) behave similarly to forward KL and reverse KL when β is close to 0 and 1 respectively."
    https://arxiv.org/abs/2306.13649

    Args:
        log_q (torch.Tensor): 
            student log probabilities
        log_p (torch.Tensor): 
            teacher log probabilities
        beta (float): 
            JSD weight
    
    Returns:
        torch.Tensor: JSD loss
    """
    log_p, log_q = clamp_log_probs(log_p, log_q)
    q = log_q.exp()
    p = log_p.exp()
    m = beta * p + (1 - beta) * q
    log_m = m.log()    
    kl1 = F.kl_div(input=log_m, target=log_p, reduction='none', log_target=True).sum(dim=-1)
    kl2 = F.kl_div(input=log_m, target=log_q, reduction='none', log_target=True).sum(dim=-1)
    loss = beta * kl1 + (1 - beta) * kl2
    return loss

def kullback_leibler_divergence(log_q: torch.Tensor, log_p: torch.Tensor, loss_mode: str = "forward"):
    """
    Computes forward and reverse KL divergence between two distributions given their log probabilities.

    forward KL: KL(p || q) = sum(p * (log_p - log_q))
    reverse KL: KL(q || p) = sum(q * (log_q - log_p))

    Args:
        log_q (torch.Tensor): 
            student log probabilities
        log_p (torch.Tensor):
            teacher log probabilities
        loss_mode (str): 
            "forward" or "reverse"
    
    Returns:
        torch.Tensor: KL divergence loss
    """
    log_p, log_q = clamp_log_probs(log_p, log_q)
    match loss_mode:
        case "forward":
            return F.kl_div(input=log_q, target=log_p, reduction='none', log_target=True).sum(dim=-1)
        case "reverse":
            return F.kl_div(input=log_p, target=log_q, reduction='none', log_target=True).sum(dim=-1)
        case _:
            raise ValueError(f"Unsupported loss mode: {loss_mode}. Supported modes are: ['forward', 'reverse']")

@register_distillation_loss(DistillationLossInfo(names=["forward_kl_full", "reverse_kl_full", "jsd_full"], use_full=True))  # type: ignore[arg-type]
@register_distillation_loss(DistillationLossInfo(names="forward_kl_topk", use_teacher_topk=True))  # type: ignore[arg-type]
@register_distillation_loss(DistillationLossInfo(names="reverse_kl_topk", use_student_topk=True))  # type: ignore[arg-type]
@register_distillation_loss(DistillationLossInfo(names="jsd_topk", use_student_topk=True, use_teacher_topk=True))  # type: ignore[arg-type]
def compute_distillation_loss_jsd(
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
    Compute the distillation loss and related metrics for JSD and variants using top-k/full log probs.

    Args:
        teacher_log_probs (torch.Tensor):
            Log-probabilities of actions under the teacher policy, shape (batch_size, response_length).
        student_log_probs (torch.Tensor):
            Log-probabilities of actions under the student policy, shape (batch_size, response_length).
        teacher_topk_logprobs (torch.Tensor):
            Top-k log-probabilities of actions under the teacher policy, shape (batch_size, response_length, topk).
        student_topk_logprobs (torch.Tensor):
            Top-k log-probabilities of actions under the student policy, shape (batch_size, response_length, topk).
        teacher_topk_indices (torch.Tensor):
            Top-k action indices under the teacher policy, shape (batch_size, response_length, topk).
        student_topk_indices (torch.Tensor):
            Top-k action indices under the student policy, shape (batch_size, response_length, topk).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        config: `(verl.trainer.config.DistillationConfig)`:
            distillation config.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert config is not None
    loss_mode = config.loss_mode
    loss_info = DistillationLossInfo.from_loss_name(loss_mode)
    distillation_metrics = {}
    if loss_info.use_full:
        # TODO (JacobHelwig)
        raise NotImplementedError
    else:
        if not teacher_topk_indices.equal(student_topk_indices):
            raise ValueError(
                f"Expected teacher and student topk indices to be the same, but got {teacher_topk_indices} and {student_topk_indices}."
            )
        topk = config.topk
        if loss_info.use_student_topk and loss_info.use_teacher_topk:
            expected_num_logprobs = 2 * topk
        elif loss_info.use_student_topk or loss_info.use_teacher_topk:
            expected_num_logprobs = topk
        else:
            raise ValueError(
                f"Expected at least one of student or teacher topk logprobs to be used, but got {loss_info.use_student_topk} and {loss_info.use_teacher_topk}."
            )
        if teacher_topk_logprobs.shape[-1] != expected_num_logprobs or student_topk_logprobs.shape[-1] != expected_num_logprobs:
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
            distillation_losses = kullback_leibler_divergence(log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, loss_mode="forward")
        case "reverse_kl_topk" | "reverse_kl_full":
            distillation_losses = kullback_leibler_divergence(log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, loss_mode="reverse")
        case "jsd_topk" | "jsd_full":
            distillation_losses = jensen_shannon_divergence(log_q=student_topk_logprobs, log_p=teacher_topk_logprobs, beta=config.jsd_beta)
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
    distillation_loss = agg_loss(
        loss_mat=distillation_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossInfo(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"]))
def compute_distillation_loss_kl_estimator(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    student_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
    config: DistillationConfig,
    loss_agg_mode: str = "token-mean",
):
    """Compute the distillation loss and related metrics using KL estimator"""
    assert config is not None
    log_p, log_q = clamp_log_probs(log_p, log_q)
    distillation_losses = kl_penalty(logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=config.loss_mode)
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
    
