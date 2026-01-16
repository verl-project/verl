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
from typing import Callable, Optional, Any
from omegaconf import DictConfig
from verl.workers.config import DistillationConfig

# TODO: Update args
DistillationLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | DistillationConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}

def register_distillation_loss(name: str) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name.

    Args:
        name (str): The name to register the distillation loss function under.

    Returns:
        function: Decorator function that registers the distillation loss function.
    """

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        DISTILLATION_LOSS_REGISTRY[name] = func
        return func

    return decorator


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

@register_distillation_loss("student_kl_topk")  # type: ignore[arg-type]
def compute_distillation_loss_student_kl_topk(
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
    Compute the distillation loss and related metrics for KL div wrt the student using top-k log probs.

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
            config for the actor.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    breakpoint()
    assert config is not None
    topk = config.topk
    if teacher_topk_logprobs.shape[-1] != topk or student_topk_logprobs.shape[-1] != topk:
        raise ValueError(
            f"Expected topk logprobs to have shape (batch_size, response_length, {topk}), but got {teacher_topk_logprobs.shape} and {student_topk_logprobs.shape}."
        )
    distillation_loss = None
    distillation_metrics = {}
    # distillation_metrics = {
    #     "actor/pg_clipfrac": pg_clipfrac.detach().item(),
    #     "actor/ppo_kl": ppo_kl.detach().item(),
    #     "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    # }
    return distillation_loss, distillation_metrics


