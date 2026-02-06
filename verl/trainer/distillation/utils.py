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

from typing import Optional

import torch
from tensordict import TensorDict

from verl.trainer.distillation.fsdp import utils as fsdp_utils
from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings
from verl.trainer.distillation.megatron import utils as megatron_utils
from verl.trainer.distillation.types import DistillationLossInputs
from verl.utils.stages import Stage
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.utils.padding import no_padding_2_padding

# Estimator distillation key
TEACHER_LOG_PROBS_KEY = "teacher_log_probs"

# Top-k distillation keys
TEACHER_TOPK_LOG_PROBS_KEY = "teacher_topk_log_probs"
TEACHER_TOPK_INDICES_KEY = "teacher_topk_indices"


STUDENT_LOGITS_KEY = "student_logits"


def compute_topk_distillation_inputs(
    logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    """Compute distillation inputs using top-k log probabilities of teacher."""
    # Gather inputs for top-k distillation losses.
    stage = batch["stage"]

    match stage:
        case Stage.OLD_LOG_PROB | Stage.REF_LOG_PROB:
            return {}
        case Stage.ACQUIRE_TEACHER_KNOWLEDGE:
            # Teacher model
            match config.strategy:
                case "fsdp":
                    compute_topk_log_probs = fsdp_utils.compute_topk_log_probs
                case "megatron":
                    compute_topk_log_probs = megatron_utils.compute_topk_log_probs
                case _:
                    raise ValueError(f"Unsupported strategy: {config.strategy}")
            teacher_topk_log_probs, teacher_topk_indices = compute_topk_log_probs(logits=logits, config=config)
            nested_log_probs = torch.nested.nested_tensor_from_jagged(teacher_topk_log_probs, cu_seqlens)
            nested_indices = torch.nested.nested_tensor_from_jagged(teacher_topk_indices, cu_seqlens)
            return {TEACHER_TOPK_LOG_PROBS_KEY: nested_log_probs, TEACHER_TOPK_INDICES_KEY: nested_indices}
        case Stage.ACTOR_UPDATE:
            # Student model
            nested_log_probs = torch.nested.nested_tensor_from_jagged(logits, cu_seqlens)
            return {STUDENT_LOGITS_KEY: nested_log_probs}
        case _:
            raise ValueError(f"Unexpected stage: {stage}")


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


def distillation_requires_logits(config: DistillationConfig) -> bool:
    """Check if distillation loss requires logits based on the provided configuration."""
    loss_config: DistillationLossConfig = config.distillation_loss
    distillation_settings: DistillationLossSettings = loss_config.loss_settings
    return distillation_settings.use_topk or distillation_settings.use_full


def compute_distillation_inputs(
    logits: Optional[torch.Tensor],
    batch: TensorDict,
    cu_seqlens: Optional[torch.Tensor],
    config: Optional[DistillationConfig],
) -> dict[str, torch.Tensor]:
    """Compute the distillation inputs for a given stage of training."""
    if not is_distillation_enabled(config):
        return {}
    loss_config: DistillationLossConfig = config.distillation_loss
    distillation_settings: DistillationLossSettings = loss_config.loss_settings
    if distillation_settings.use_estimator:
        return {}
    if logits is None:
        raise ValueError(f"logits must be provided for distillation loss computation with {loss_config.loss_mode=}.")
    if cu_seqlens is None:
        if not logits.is_nested:
            raise ValueError("cu_seqlens must be provided if logits is not a nested tensor.")
        cu_seqlens = logits.offsets()
        logits = logits.values()
    if distillation_settings.use_full:
        return NotImplementedError  # TODO: JacobHelwig
    elif distillation_settings.use_topk:
        return compute_topk_distillation_inputs(logits=logits, batch=batch, cu_seqlens=cu_seqlens, config=config)
    else:
        raise ValueError


def extract_distillation_inputs(
    stage: Stage, output: TensorDict, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    """Extract distillation loss inputs from model output for a given stage. Used in trainer."""
    loss_config: DistillationLossConfig = config.distillation_loss
    distillation_settings = get_distillation_loss_settings(loss_config.loss_mode)
    if isinstance(stage, Stage):
        stage = stage.value
    if distillation_settings.use_full:
        raise NotImplementedError(
            "Full log probs are not currently supported for distillation loss. Please use top-k log probs instead."
        )
    elif distillation_settings.use_estimator:
        return {TEACHER_LOG_PROBS_KEY: output["log_probs"]}
    elif distillation_settings.use_topk:
        if stage == Stage.ACQUIRE_TEACHER_KNOWLEDGE.value:
            return {
                TEACHER_TOPK_INDICES_KEY: output[TEACHER_TOPK_INDICES_KEY],
                TEACHER_TOPK_LOG_PROBS_KEY: output[TEACHER_TOPK_LOG_PROBS_KEY],
            }
        else:
            raise ValueError(f"Unexpected stage: {stage}")
    else:
        raise ValueError(f"Unexpected distillation settings: {distillation_settings}")


def prepare_distillation_inputs(
    log_prob: torch.Tensor, data: TensorDict, model_output: dict[str, torch.Tensor], config: DistillationConfig
) -> DistillationLossInputs:
    """Prepare distillation loss inputs for loss computation. Called in ppo_loss before computing distillation loss."""
    loss_config: DistillationLossConfig = config.distillation_loss
    distillation_settings: DistillationLossSettings = loss_config.loss_settings
    if distillation_settings.use_full:
        raise NotImplementedError(
            "Full log probs are not currently supported for distillation loss. Please use top-k log probs instead."
        )
    elif distillation_settings.use_estimator:
        return DistillationLossInputs(student_log_probs=log_prob, teacher_log_probs=data[TEACHER_LOG_PROBS_KEY])
    elif distillation_settings.use_topk:
        teacher_topk_log_probs = no_padding_2_padding(data[TEACHER_TOPK_LOG_PROBS_KEY], data)
        teacher_topk_indices = no_padding_2_padding(data[TEACHER_TOPK_INDICES_KEY], data)
        student_logits = no_padding_2_padding(model_output[STUDENT_LOGITS_KEY], data)
        return DistillationLossInputs(
            student_logits=student_logits,
            teacher_topk_log_probs=teacher_topk_log_probs,
            teacher_topk_indices=teacher_topk_indices,
        )
    else:
        raise ValueError
