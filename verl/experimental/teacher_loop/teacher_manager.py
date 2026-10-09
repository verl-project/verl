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


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
) -> dict[str, Any]:
    """Get sampling parameters for teacher model when computing log probabilities for distillation."""
    if teacher_model_config.inference.temperature != 1.0:
        logger.warning(
            "Teacher inference temperature is set to %.1f, but temperature has no effect "
            "on prompt_logprobs (forward pass only). Using temperature=1.0.",
            teacher_model_config.inference.temperature,
        )
    if distillation_loss_config.loss_settings.use_full_vocab:
        # Full-vocab KL: the teacher exports pre-lm_head hidden states instead of
        # logprobs. prompt_logprobs=0 is enough to make vLLM compute logits for
        # every prompt position, which triggers the compute_logits capture.
        return {
            "max_tokens": 1,
            "temperature": 1.0,
            "prompt_logprobs": 0,
        }

    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": teacher_model_config.inference.temperature,
        "prompt_logprobs": num_logprobs,
    }


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
        # Fallback step source for full-vocab export when the caller cannot
        # provide the trainer's global_steps (see compute_teacher_full_vocab_single).
        self._full_vocab_step_counter = 0

    def _next_full_vocab_step(self) -> int:
        self._full_vocab_step_counter += 1
        return self._full_vocab_step_counter

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
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute teacher log probabilities for a single unpadded sequence."""
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
        )
        # Shapes: # S, (1 or K), where S is the response length, K is either 1 or topk depending on
        # the distillation loss settings.
        teacher_ids = torch.tensor(teacher_output.extra_fields["prompt_ids"], dtype=torch.int32)
        teacher_logprobs = torch.tensor(teacher_output.extra_fields["prompt_logprobs"])
        assert teacher_ids.shape[0] == teacher_logprobs.shape[0] == len(sequence_ids)
        return teacher_ids, teacher_logprobs

    async def compute_teacher_full_vocab_single(
        self,
        sequence_ids: list[int],
        *,
        step: Optional[int] = None,
        uid: Optional[str] = None,
        routing_key: Optional[str] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Export the teacher's pre-lm_head hidden states for one unpadded sequence.

        Runs a prefill-only teacher forward (max_tokens=1, prompt_logprobs=0); the
        teacher server captures the hidden states, writes them to TransferQueue and
        returns only the artifact metadata dict, which this method returns.

        ``step``/``uid`` key the TQ entry (``{teacher}/step={step}/sample={uid}`` in
        partition ``..._step_{step}``). When ``step`` is None (callers without access
        to the trainer's global_steps, e.g. the v0 agent-loop path) a monotonic
        per-manager counter is used instead: every call then lands in its own
        partition, which keeps keys unique but fragments per-step cleanup. When
        ``uid`` is None a random uuid is used.

        Fail-loud: raises if the teacher output carries no artifact.
        """
        multi_modal_data = multi_modal_data or {}
        teacher_key = self._resolve_teacher_key(routing_key)
        teacher_model_config = self.teacher_model_configs[teacher_key]
        client = self.teacher_client[teacher_key]
        if step is None:
            step = self._next_full_vocab_step()
        if uid is None:
            uid = uuid4().hex
        else:
            # Guarantee per-export key uniqueness inside the per-step partition even
            # when the caller's uid repeats (rollout.n>1 repeats of one prompt, a
            # duplicated uid column in the dataset, retries). A second put to the same
            # key silently overwrites the first entry, after which the first sample's
            # artifact points at a tensor with a different shape (reshape error at
            # consumption). The caller uid stays as a readable prefix.
            uid = f"{uid}_{uuid4().hex[:8]}"
        teacher_output = await client.generate(
            request_id=uuid4().hex,
            prompt_ids=sequence_ids,
            sampling_params=_get_teacher_sampling_params(teacher_model_config, self.distillation_loss_config),
            image_data=multi_modal_data.get("images"),
            video_data=multi_modal_data.get("videos"),
            audio_data=multi_modal_data.get("audios"),
            mm_processor_kwargs=mm_processor_kwargs,
            full_vocab={"teacher_name": teacher_key, "step": step, "uid": uid},
        )
        artifact = teacher_output.extra_fields.get("teacher_full_vocab_artifact")
        if artifact is None:
            raise RuntimeError(
                f"Teacher {teacher_key!r} returned no 'teacher_full_vocab_artifact' in extra_fields "
                f"(step={step}, uid={uid!r}, seq_len={len(sequence_ids)}). The teacher server must be "
                "started with full_vocab_export_config enabled; a missing artifact would silently "
                "disable full-vocab distillation for this sample."
            )
        return artifact
