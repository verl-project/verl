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
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings
from verl.trainer.distillation.types import DistillationLossInputs
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationConfig
from verl.workers.utils.padding import _slice_response_from_unpad_output


class Stage(Enum):
    """Stages for on-policy distillation training."""

    OLD_LOG_PROB = "old_log_prob"
    REF_LOG_PROB = "ref_log_prob"
    ACTOR_UPDATE = "actor_update"


def get_topk_keys(stage: str | Stage) -> tuple[str, str]:
    """Get the TensorDict keys for storing top-k log probabilities and indices for a given stage."""
    if isinstance(stage, Stage):
        stage = stage.value
    return f"{stage}_topk_log_probs", f"{stage}_topk_indices"


def topk_logprobs_from_logits(
    logits: torch.Tensor, k: int, compute_topk: bool, topk_indices: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute and/or gather top-k log probabilities from logits.

    This function supports two modes that can be used independently or together:
    1. Gathering log probabilities at pre-specified indices (topk_indices)
    2. Computing new top-k log probabilities from logits

    When both modes are active, the results are concatenated and deduplicated
    to handle overlap between teacher and student top-k sets.

    Args:
        logits (torch.Tensor):
            Logits from model forward pass, shape (total_tokens, vocab_size).
        k (int):
            Number of top log probabilities to compute or gather.
        compute_topk (bool):
            Whether to compute top-k log probabilities from the logits.
        topk_indices (torch.Tensor, optional):
            Pre-computed indices for gathering log probabilities, shape (total_tokens, k) or
            (total_tokens, 2*k). Required when gathering from existing indices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - topk_logprobs: Top-k log probabilities, shape (total_tokens, k) or (total_tokens, 2*k).
            - topk_indices: Indices for the top-k log probabilities, same shape as topk_logprobs.
              Duplicate indices (from merging teacher/student top-k) have their log probs set to -inf.
    """
    logprobs = F.log_softmax(logits, dim=-1)
    topk_logprobs_ls = []
    topk_logprobs_indices_ls = []

    # Gather logits for provided indices.
    if topk_indices is not None:
        if topk_indices.is_nested:
            topk_indices = topk_indices.values()
        if topk_indices.shape[-1] not in [k, 2 * k]:
            raise ValueError(
                f"Expected topk_indices to have shape [-1, {k}] or [-1, {2 * k}], but got {topk_indices.shape}."
            )
        topk_logprobs = torch.gather(logprobs, dim=-1, index=topk_indices)
        topk_logprobs_ls.append(topk_logprobs)
        topk_logprobs_indices_ls.append(topk_indices)

    # Compute top-k logprobs.
    if compute_topk:
        topk_logprobs, topk_indices = torch.topk(logprobs, k=k, dim=-1)
        topk_logprobs_ls.append(topk_logprobs)
        topk_logprobs_indices_ls.append(topk_indices)

    topk_logprobs = torch.cat(topk_logprobs_ls, dim=-1)
    topk_indices = torch.cat(topk_logprobs_indices_ls, dim=-1)
    if topk_logprobs.shape != topk_indices.shape:
        raise ValueError(
            f"Expected topk_logprobs and topk_indices to have the same shape, "
            f"but got {topk_logprobs.shape} and {topk_indices.shape}."
        )
    if topk_logprobs.shape[-1] not in [k, 2 * k]:
        raise ValueError(
            f"Expected topk_logprobs to have shape [-1, {k}] or [-1, {2 * k}], but got {topk_logprobs.shape}."
        )

    # If we have 2 * k logprobs, we are gathering top-k logprobs from both teacher and student.
    # We need to de-duplicate to handle overlap between teacher and student top-k logprobs.
    if topk_logprobs.shape[-1] == 2 * k:
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
        return compute_full_distillation_inputs(logits=logits, batch=batch, cu_seqlens=cu_seqlens, config=config)
    elif distillation_settings.use_estimator:
        return {}
    elif distillation_settings.use_topk:
        return compute_topk_distillation_inputs(logits=logits, batch=batch, cu_seqlens=cu_seqlens, config=config)
    else:
        raise ValueError


def compute_full_distillation_inputs(
    logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    """Compute distillation inputs using full vocabulary log probabilities."""
    raise NotImplementedError(
        "Full logprobs are not currently supported for distillation loss. Please use top-k logprobs instead."
    )


def compute_topk_distillation_inputs(
    logits: torch.Tensor, batch: TensorDict, cu_seqlens: torch.Tensor, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    """
    Compute distillation inputs using top-k log probabilities.

    This function handles different stages of distillation training:
    - OLD_LOG_PROB: Student computes its own top-k indices
    - REF_LOG_PROB: Teacher gathers log probs at student indices, computes own top-k
    - ACTOR_UPDATE: Student gathers log probs at teacher indices

    Args:
        logits (torch.Tensor):
            Model output logits, shape (total_tokens, vocab_size).
        batch (TensorDict):
            Batch data containing "stage" key and potentially previous top-k indices.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths for creating nested tensors, shape (batch_size + 1,).
        config (DistillationConfig):
            Distillation configuration with topk, loss_mode, and loss_settings.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing:
            - {stage}_topk_log_probs: Nested tensor of top-k log probabilities.
            - {stage}_topk_indices: Nested tensor of top-k token indices.
    """
    # Gather inputs for top-k distillation losses.
    topk = config.topk
    loss_mode = config.loss_mode
    distillation_settings: DistillationLossSettings = config.loss_settings
    stage = batch["stage"]

    use_student_topk = distillation_settings.use_student_topk
    use_teacher_topk = distillation_settings.use_teacher_topk
    should_compute_topk = False
    topk_indices = None
    match stage:
        case Stage.OLD_LOG_PROB:
            # 1. First pass with student model
            if use_student_topk:
                should_compute_topk = True
        case Stage.REF_LOG_PROB:
            # 2. Teacher model
            if use_teacher_topk:
                should_compute_topk = True
            if use_student_topk:
                _, student_topk_indices_key = get_topk_keys(Stage.OLD_LOG_PROB)
                topk_indices = tu.get(batch, student_topk_indices_key)
                if topk_indices is None:
                    raise ValueError(
                        f"Expected student topk indices for teacher log prob stage, got None with {loss_mode=}."
                    )
                if topk_indices.shape[-1] != topk:
                    raise ValueError(
                        f"Expected student topk indices shape [-1, {topk}], got {topk_indices.shape} with {loss_mode=}."
                    )
        case Stage.ACTOR_UPDATE:
            # 3. Second pass with student model
            if use_student_topk or use_teacher_topk:
                _, teacher_topk_indices_key = get_topk_keys(Stage.REF_LOG_PROB)
                topk_indices = tu.get(batch, teacher_topk_indices_key)
                if topk_indices is None:
                    raise ValueError(
                        f"Expected teacher topk indices for student update stage, got None with {loss_mode=}."
                    )
                if use_student_topk and use_teacher_topk:
                    if topk_indices.shape[-1] != 2 * topk:
                        raise ValueError(
                            f"Expected teacher topk indices shape [-1, {2 * topk}], "
                            f"got {topk_indices.shape} with {loss_mode=}."
                        )
                elif topk_indices.shape[-1] != topk:
                    raise ValueError(
                        f"Expected teacher topk indices shape [-1, {topk}], got {topk_indices.shape} with {loss_mode=}."
                    )
        case _:
            raise ValueError(f"Unexpected stage: {stage}")
    topk_logprobs, topk_indices = topk_logprobs_from_logits(
        logits=logits,
        k=topk,
        compute_topk=should_compute_topk,
        topk_indices=topk_indices,
    )
    topk_logprobs_key, topk_indices_key = get_topk_keys(stage)
    return {
        topk_logprobs_key: torch.nested.nested_tensor_from_jagged(topk_logprobs, cu_seqlens),
        topk_indices_key: torch.nested.nested_tensor_from_jagged(topk_indices, cu_seqlens),
    }


def extract_distillation_inputs(
    stage: Stage, output: TensorDict, config: DistillationConfig
) -> dict[str, torch.Tensor]:
    """Extract distillation loss inputs from model output for a given stage. Used in trainer"""
    distillation_settings = get_distillation_loss_settings(config.loss_mode)
    if distillation_settings.use_full:
        raise NotImplementedError(
            "Full logprobs are not currently supported for distillation loss. Please use top-k logprobs instead."
        )
    elif distillation_settings.use_estimator:
        return {}
    elif distillation_settings.use_topk:
        topk_logprobs_key, topk_indices_key = get_topk_keys(stage)
        topk_logprobs = tu.get(output, topk_logprobs_key)
        if topk_logprobs is not None:
            return {
                topk_logprobs_key: topk_logprobs.float(),
                topk_indices_key: tu.get(output, topk_indices_key),
            }
        else:
            return {}
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
        teacher_topk_logprobs, teacher_topk_indices = unpad_distillation_logprobs(
            outputs=data, data=data, stage=Stage.REF_LOG_PROB, distillation_settings=distillation_settings
        )
        student_topk_logprobs, student_topk_indices = unpad_distillation_logprobs(
            outputs=model_output, data=data, stage=Stage.ACTOR_UPDATE, distillation_settings=distillation_settings
        )
        return DistillationLossInputs(
            student_topk_logprobs=student_topk_logprobs,
            teacher_topk_logprobs=teacher_topk_logprobs,
            student_topk_indices=student_topk_indices,
            teacher_topk_indices=teacher_topk_indices,
        )
    else:
        raise ValueError


def unpad_distillation_logprobs(
    outputs: TensorDict | dict[str, torch.Tensor],
    data: TensorDict,
    stage: Stage,
    distillation_settings: DistillationLossSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and unpad distillation log probabilities from model outputs."""
    if distillation_settings.use_full:
        raise NotImplementedError(
            "Full logprobs are not currently supported for distillation loss. Please use top-k logprobs instead."
        )
    elif distillation_settings.use_topk:
        topk_logprobs_key, topk_indices_key = get_topk_keys(stage)
        topk_logprobs, topk_indices = outputs[topk_logprobs_key], outputs[topk_indices_key]
        topk_logprobs_unpad = _slice_response_from_unpad_output(topk_logprobs, data)
        topk_indices_unpad = _slice_response_from_unpad_output(topk_indices, data)
        return topk_logprobs_unpad, topk_indices_unpad
    else:
        raise ValueError
