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
"""
Contains utilities/classes for on-policy distillation 
"""

from typing import Union, Optional, Callable, Any
from enum import Enum
from omegaconf import DictConfig
from verl.workers.config import ActorConfig
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu



class Stage(Enum):
    """
    Stages for PPO training
    """
    OLD_LOG_PROB = "old_log_prob"
    REF_LOG_PROB = "ref_log_prob"
    ACTOR_UPDATE = "actor_update"

    @classmethod
    def get_topk_keys(cls, stage: Union[str, "Stage"]):
        if isinstance(stage, str):
            stage = cls(stage)
        return f"{stage.value}_topk_log_probs", f"{stage.value}_topk_indices"


def topk_logprobs_from_logits(logits: torch.Tensor, k: int, compute_both: bool, topk_indices: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:    
    logprobs = F.log_softmax(logits, dim=-1)

    needs_dedupe = False
    if compute_both:
        if topk_indices is None or topk_indices.shape[-1] == k:
            should_compute_topk = True
        elif topk_indices.shape[-1] == 2 * k:
            should_compute_topk = False
        else:
            raise ValueError(f"{topk_indices.shape=} is not expected with {k=}")
    else:
        if topk_indices is None:
            should_compute_topk = True
        elif topk_indices.shape[-1] == k:
            should_compute_topk = False
        else:
            raise ValueError(f"{topk_indices.shape=} is not expected with {k=}")
            

    topk_logprobs_ls = []
    topk_logprobs_indices_ls = []
    
    # Gather logits for provided indices.
    if topk_indices is not None:
        topk_logprobs = torch.gather(logprobs, dim=-1, index=topk_indices)
        topk_logprobs_ls.append(topk_logprobs)
        topk_logprobs_indices_ls.append(topk_indices)

    # Compute top-k logprobs.
    if should_compute_topk:
        topk_logprobs, topk_indices = torch.topk(logprobs, k=k, dim=-1)
        topk_logprobs_ls.append(topk_logprobs)
        topk_logprobs_indices_ls.append(topk_indices)

    topk_logprobs = torch.cat(topk_logprobs_ls, dim=-1)
    topk_indices = torch.cat(topk_logprobs_indices_ls, dim=-1)

    # If top-k have been provided AND new top-k have been computed, we need to deduplicate the indices and logprobs. 
    if needs_dedupe:

        # Make sure indices are sorted so that we can identify duplicates.
        topk_indices_diff = topk_indices.diff(dim=-1)
        if topk_indices_diff.lt(0).any():
            topk_indices, sort_indices = topk_indices.sort(dim=-1)
            topk_logprobs = torch.gather(topk_logprobs, dim=-1, index=sort_indices)
            topk_indices_diff = topk_indices.diff(dim=-1)

        # Find duplicate indices and set their prob to ~0.
        if topk_indices_diff.eq(0).any():
            index_diffs = torch.nn.functional.pad(topk_indices_diff, (0, 1), value=1)
            dupe_mask = index_diffs.eq(0)
            topk_logprobs[dupe_mask] = -torch.inf

    return topk_logprobs, topk_indices

def compute_topk_outputs(logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor):
    """
    TODO: Docstring for compute_topk_outputs
    """
    stage = batch["stage"]
    topk_logprobs, topk_indices = topk_logprobs_from_logits(logits=logits, k=2, compute_both=True, topk_indices=batch.get("topk_indices", None))
    topk_logprobs_key, topk_indices_key = Stage.get_topk_keys(stage)
    output = {
        topk_logprobs_key: torch.nested.nested_tensor_from_jagged(topk_logprobs.squeeze(0), cu_seqlens),
        topk_indices_key: torch.nested.nested_tensor_from_jagged(topk_indices.squeeze(0), cu_seqlens),
    }
    return output

def gather_topk_outputs(stage: Stage, output: TensorDict):
    """
    TODO: Docstring for gather_topk_outputs
    """
    topk_logprobs_key, topk_indices_key = Stage.get_topk_keys(stage)
    topk_logprobs = tu.get(output, topk_logprobs_key)
    if topk_logprobs is not None:
        return {
            topk_logprobs_key: topk_logprobs.float(),
            topk_indices_key: tu.get(output, topk_indices_key),
        }
    else:
        return {}

# TODO: Update args
DistillationLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
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

from verl.workers.config import DistillationConfig

@register_distillation_loss("student_kl_topk")  # type: ignore[arg-type]
def compute_distillation_loss_student_kl_topk(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
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
        config: `(verl.trainer.config.DistillationConfig)`:
            config for the actor.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
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