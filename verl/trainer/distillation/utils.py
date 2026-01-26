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

from enum import Enum
from typing import Optional

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings
from verl.trainer.distillation.types import DistillationLossInputs
from verl.workers.config import DistillationConfig
from verl.workers.utils.padding import _slice_response_from_unpad_output


class Stage(Enum):
    """Stages for on-policy distillation training."""

    OLD_LOG_PROB = "old_log_prob"
    REF_LOG_PROB = "ref_log_prob"
    ACTOR_UPDATE = "actor_update"


TEACHER_TOPK_LOGITS_KEY = "teacher_topk_logits"
TEACHER_TOPK_INDICES_KEY = "teacher_topk_indices"
STUDENT_LOGITS_KEY = "student_logits"


def compute_topk_distillation_inputs(
    logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    """Compute distillation inputs using top-k log probabilities of teacher."""
    # Gather inputs for top-k distillation losses.
    topk = config.topk
    stage = batch["stage"]

    match stage:
        case Stage.OLD_LOG_PROB:
            return {}
        case Stage.REF_LOG_PROB:
            # Teacher model
            teacher_topk_logits, teacher_topk_indices = logits.topk(k=topk, dim=-1)
            nested_logits = torch.nested.nested_tensor_from_jagged(teacher_topk_logits, cu_seqlens)
            nested_indices = torch.nested.nested_tensor_from_jagged(teacher_topk_indices, cu_seqlens)
            return {TEACHER_TOPK_LOGITS_KEY: nested_logits, TEACHER_TOPK_INDICES_KEY: nested_indices}
        case Stage.ACTOR_UPDATE:
            # Student model
            nested_logits = torch.nested.nested_tensor_from_jagged(logits, cu_seqlens)
            return {STUDENT_LOGITS_KEY: nested_logits}
        case _:
            raise ValueError(f"Unexpected stage: {stage}")


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


def compute_distillation_inputs(
    logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor, config: Optional[DistillationConfig]
) -> dict[str, torch.Tensor]:
    """Compute the distillation inputs for a given stage of training."""
    if not is_distillation_enabled(config):
        return {}
    distillation_settings: DistillationLossSettings = config.loss_settings
    if distillation_settings.use_full:
        return NotImplementedError  # TODO: JacobHelwig
    elif distillation_settings.use_estimator:
        return {}
    elif distillation_settings.use_topk:
        return compute_topk_distillation_inputs(logits=logits, batch=batch, cu_seqlens=cu_seqlens, config=config)
    else:
        raise ValueError


def extract_distillation_inputs(
    stage: Stage, output: TensorDict, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    """Extract distillation loss inputs from model output for a given stage. Used in trainer."""
    distillation_settings = get_distillation_loss_settings(config.loss_mode)
    if distillation_settings.use_full:
        raise NotImplementedError(
            "Full logprobs are not currently supported for distillation loss. Please use top-k logprobs instead."
        )
    elif distillation_settings.use_estimator:
        return {}
    elif distillation_settings.use_topk:
        if isinstance(stage, Stage):
            stage = stage.value
        if stage == Stage.REF_LOG_PROB.value:
            return {
                TEACHER_TOPK_INDICES_KEY: output[TEACHER_TOPK_INDICES_KEY],
                TEACHER_TOPK_LOGITS_KEY: output[TEACHER_TOPK_LOGITS_KEY],
            }
        else:
            raise ValueError(f"Unexpected stage: {stage}")
    else:
        raise ValueError


def prepare_distillation_inputs(
    log_prob: torch.Tensor, data: TensorDict, model_output: dict[str, torch.Tensor], config: DistillationConfig
) -> DistillationLossInputs:
    """Prepare distillation loss inputs for loss computation. Called in ppo_loss before computing distillation loss."""
    distillation_settings: DistillationLossSettings = config.loss_settings
    if distillation_settings.use_full:
        raise NotImplementedError(
            "Full logprobs are not currently supported for distillation loss. Please use top-k logprobs instead."
        )
    elif distillation_settings.use_estimator:
        return DistillationLossInputs(student_log_probs=log_prob, teacher_log_probs=data["ref_log_prob"])
    elif distillation_settings.use_topk:
        teacher_topk_logits = _slice_response_from_unpad_output(data[TEACHER_TOPK_LOGITS_KEY], data)
        teacher_topk_indices = _slice_response_from_unpad_output(data[TEACHER_TOPK_INDICES_KEY], data)
        student_logits = _slice_response_from_unpad_output(model_output[STUDENT_LOGITS_KEY], data)
        return DistillationLossInputs(
            student_logits=student_logits, teacher_logits=teacher_topk_logits, teacher_topk_indices=teacher_topk_indices
        )
    else:
        raise ValueError
