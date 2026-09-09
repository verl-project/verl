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
from typing import Any, Optional
from uuid import uuid4

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_MAX_PROMPT_LOGPROB_TOKEN_IDS = 16384


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
    prompt_length: Optional[int] = None,
    prompt_logprob_token_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    # Temperature has no effect on prompt_logprobs: the teacher performs a forward pass over
    # existing tokens (no sampling). Always use temperature=1.0 regardless of the config value.
    # The default distillation.yaml copies the student rollout temperature via Hydra interpolation
    # (temperature: ${oc.select:actor_rollout_ref.rollout.temperature}), which causes a spurious
    # crash when rollout.temperature != 1.0.
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    if distillation_loss_config.loss_mode == "reverse_kl_topk":
        if prompt_length is None or prompt_length <= 0:
            raise ValueError("reverse_kl_topk requires a positive prompt length for causal suffix scoring.")
        if not prompt_logprob_token_ids:
            raise ValueError("reverse_kl_topk requires non-empty student top-k token IDs.")
        return {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprob_token_ids": prompt_logprob_token_ids,
            "prompt_logprob_start": prompt_length - 1,
        }

    if not distillation_loss_config.loss_settings.use_topk:
        num_logprobs = 0
    else:
        num_logprobs = distillation_loss_config.topk
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
    }


def _prepare_student_topk_ids(
    student_topk_ids: list[list[int]], response_length: int, topk: int
) -> tuple[torch.Tensor, list[int]]:
    support_ids = torch.tensor(student_topk_ids, dtype=torch.int32)
    expected_shape = (response_length, topk)
    if tuple(support_ids.shape) != expected_shape:
        raise ValueError(
            f"Expected response-aligned student top-k IDs with shape {expected_shape}, got {support_ids.shape}."
        )

    # Preserve first-seen order while satisfying vLLM's no-duplicates contract.
    union_ids = list(dict.fromkeys(int(token_id) for token_id in support_ids.flatten().tolist()))
    if len(union_ids) > _MAX_PROMPT_LOGPROB_TOKEN_IDS:
        raise ValueError(
            f"Student top-k union has {len(union_ids)} IDs, exceeding vLLM PR #54335's "
            f"{_MAX_PROMPT_LOGPROB_TOKEN_IDS}-ID request limit. Reduce topk or response length."
        )
    return support_ids, union_ids


def _gather_teacher_logprobs(prompt_token_id_logprobs: dict[str, Any], support_ids: torch.Tensor) -> torch.Tensor:
    fixed_token_ids = [int(token_id) for token_id in prompt_token_id_logprobs["token_ids"]]
    fixed_logprobs = torch.as_tensor(prompt_token_id_logprobs["logprobs"], dtype=torch.float32)
    expected_shape = (support_ids.shape[0], len(fixed_token_ids))
    if tuple(fixed_logprobs.shape) != expected_shape:
        raise RuntimeError(f"Expected vLLM fixed-token scores with shape {expected_shape}, got {fixed_logprobs.shape}.")

    token_to_column = {token_id: column for column, token_id in enumerate(fixed_token_ids)}
    try:
        support_columns = torch.tensor(
            [[token_to_column[int(token_id)] for token_id in row] for row in support_ids.tolist()],
            dtype=torch.int64,
        )
    except KeyError as exc:
        raise RuntimeError(f"vLLM fixed-token result is missing requested token ID {exc.args[0]}.") from exc
    return torch.gather(fixed_logprobs, dim=1, index=support_columns)


def _align_response_teacher_outputs(
    prompt_length: int,
    support_ids: torch.Tensor,
    response_logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place response scores on the causal source rows used by the training loss."""
    if prompt_length <= 0:
        raise ValueError("Teacher output alignment requires a positive prompt length.")
    if support_ids.shape != response_logprobs.shape:
        raise ValueError(
            f"Teacher support IDs and logprobs must have the same shape, got "
            f"{support_ids.shape} and {response_logprobs.shape}."
        )

    # no_padding_2_padding left-shifts model outputs: source row i predicts token i + 1.
    # The first response token is therefore scored at prompt_length - 1. Keep a final
    # dummy row so the tensors still span the complete prompt + response sequence.
    prefix_shape = (prompt_length - 1, support_ids.shape[1])
    dummy_shape = (1, support_ids.shape[1])
    teacher_ids = torch.cat(
        (
            torch.zeros(prefix_shape, dtype=support_ids.dtype),
            support_ids,
            torch.zeros(dummy_shape, dtype=support_ids.dtype),
        ),
        dim=0,
    )
    teacher_logprobs = torch.cat(
        (
            torch.zeros(prefix_shape, dtype=response_logprobs.dtype),
            response_logprobs,
            torch.zeros(dummy_shape, dtype=response_logprobs.dtype),
        ),
        dim=0,
    )
    return teacher_ids, teacher_logprobs


def _pad_teacher_outputs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_width: int,
    response_width: int,
    prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO(wuxibin): remove padding and use tensordict.
    left_pad_size = prompt_width - prompt_length
    right_pad_size = response_width - response_length
    padding = (0, 0, left_pad_size, right_pad_size)
    return (
        F.pad(teacher_ids, padding, value=pad_token_id).unsqueeze(0),
        F.pad(teacher_logprobs, padding, value=0.0).unsqueeze(0),
    )


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.teacher_key: str = self.distillation_config.teacher_key

        self.teacher_model_configs: dict[str, DistillationTeacherModelConfig] = self.distillation_config.teacher_models
        expected = set(self.teacher_model_configs)
        if set(teacher_client.keys()) != expected:
            raise ValueError(
                f"teacher client keys {sorted(teacher_client.keys())} "
                f"do not match teacher routing keys {sorted(expected)}."
            )
        self.teacher_client: dict[str, LLMServerClient] = teacher_client

    def _resolve_teacher_key(self, routing_key: Optional[str]) -> str:
        if len(self.teacher_model_configs) == 1:
            # Single-teacher path: route everything to the one teacher regardless of the sample's key.
            return next(iter(self.teacher_model_configs))
        if routing_key is None:
            raise ValueError(
                f"Routing key is required for multi-teacher distillation "
                f"(configured via distillation.teacher_key={self.teacher_key!r})."
            )
        if routing_key not in self.teacher_model_configs:
            raise ValueError(
                f"No teacher configured for routing key {routing_key!r}. "
                f"Configured teachers: {sorted(self.teacher_model_configs)}."
            )
        return routing_key

    async def compute_teacher_logprobs_single(
        self,
        sequence_ids: list[int],
        prompt_length: Optional[int] = None,
        student_topk_ids: Optional[list[list[int]]] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        support_ids = None
        prompt_logprob_token_ids = None
        if self.distillation_loss_config.loss_mode == "reverse_kl_topk":
            if teacher_model_config.inference.name != "vllm":
                raise NotImplementedError("reverse_kl_topk fixed-token scoring is currently implemented for vLLM only.")
            if prompt_length is None or student_topk_ids is None:
                raise ValueError("reverse_kl_topk requires prompt_length and response-aligned student_topk_ids.")
            response_length = len(sequence_ids) - prompt_length
            topk = self.distillation_loss_config.topk
            if topk is None:
                raise ValueError("reverse_kl_topk requires distillation_loss.topk.")
            support_ids, prompt_logprob_token_ids = _prepare_student_topk_ids(
                student_topk_ids=student_topk_ids,
                response_length=response_length,
                topk=topk,
            )

        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(
                teacher_model_config,
                self.distillation_loss_config,
                prompt_length=prompt_length,
                prompt_logprob_token_ids=prompt_logprob_token_ids,
            ),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if self.distillation_loss_config.loss_mode == "reverse_kl_topk":
            prompt_token_id_logprobs = teacher_output.extra_fields.get("prompt_token_id_logprobs")
            if prompt_token_id_logprobs is None:
                raise NotImplementedError(
                    "reverse_kl_topk fixed-token scoring requires the current vLLM PR #54335 API "
                    "and RequestOutput.prompt_token_id_logprobs."
                )
            assert support_ids is not None and prompt_length is not None
            response_logprobs = _gather_teacher_logprobs(prompt_token_id_logprobs, support_ids)
            return _align_response_teacher_outputs(prompt_length, support_ids, response_logprobs)

        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs
