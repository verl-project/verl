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

from verl.trainer.distillation.losses import DistillationLossSettings
from verl.trainer.distillation.types import DistillationLossInputs
from verl.utils.stages import Stage
from verl.workers.config import DistillationConfig, DistillationLossConfig
from verl.workers.utils.padding import no_padding_2_padding


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


def prepare_student_distillation_inputs(
    logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor, config: Optional[DistillationConfig]
) -> dict[str, torch.Tensor]:
    """Prepare student distillation inputs."""
    stage = batch["stage"]
    if not is_distillation_enabled(config) or stage in {Stage.OLD_LOG_PROB, Stage.REF_LOG_PROB}:
        return {}
    assert stage == Stage.ACTOR_UPDATE, f"Unexpected stage: {stage}"
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
    if distillation_settings.use_topk:
        nested_logits = torch.nested.nested_tensor_from_jagged(logits, cu_seqlens)
        return {"student_logits": nested_logits}
    else:
        raise ValueError


def prepare_distillation_inputs(
    log_prob: torch.Tensor, data: TensorDict, model_output: dict[str, torch.Tensor], config: DistillationConfig
) -> DistillationLossInputs:
    """Prepare distillation loss inputs for loss computation. Called in ppo_loss before computing distillation loss."""
    loss_config: DistillationLossConfig = config.distillation_loss
    distillation_settings: DistillationLossSettings = loss_config.loss_settings
    if distillation_settings.use_estimator:
        return DistillationLossInputs(
            student_log_probs=log_prob, teacher_log_probs=data["teacher_logprobs"].squeeze(-1)
        )
    elif distillation_settings.use_topk:
        teacher_topk_log_probs = data["teacher_logprobs"]
        teacher_topk_ids = data["teacher_ids"]
        student_logits = no_padding_2_padding(model_output["student_logits"], data)
        return DistillationLossInputs(
            student_logits=student_logits,
            teacher_topk_log_probs=teacher_topk_log_probs,
            teacher_topk_ids=teacher_topk_ids,
        )
    else:
        raise ValueError
