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


def resolve_teacher_system_prompt(teacher_model_config: DistillationTeacherModelConfig) -> Optional[str]:
    """Return the teacher-only system prompt text, or None when unset.

    ``system_prompt`` wins over ``system_prompt_path``. Empty / whitespace-only
    values are treated as unset so Hydra ``""`` overrides stay no-ops.
    """
    text = teacher_model_config.system_prompt
    if text is not None and str(text).strip():
        return str(text)
    path = teacher_model_config.system_prompt_path
    if path is None or not str(path).strip():
        return None
    return _load_system_prompt_spec(str(path))


def _load_system_prompt_spec(spec: str) -> str:
    path = os.path.expanduser(spec)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            raise ValueError(f"distillation teacher system prompt file is empty: {path!r}")
        return text
    if "/" in spec or spec.endswith(".txt"):
        raise FileNotFoundError(f"distillation teacher system_prompt_path not found: {path!r}")
    if not spec.strip():
        raise ValueError("distillation teacher system prompt is empty")
    return spec


def resolve_teacher_system_prompt_for_sample(
    teacher_model_config: DistillationTeacherModelConfig,
    routing_key: Optional[str] = None,
) -> Optional[str]:
    """System prompt for one sample. Honors ``system_prompt_by_key`` when set."""
    by_key = teacher_model_config.system_prompt_by_key
    if by_key:
        mapping = dict(by_key)
        if routing_key is None:
            raise ValueError(
                "system_prompt_by_key is set but the sample has no routing key "
                f"(distillation.teacher_key). Configured keys: {sorted(mapping)}"
            )
        if routing_key not in mapping:
            raise ValueError(
                f"No system_prompt_by_key entry for routing key {routing_key!r}. "
                f"Configured keys: {sorted(mapping)}"
            )
        spec = mapping[routing_key]
        if spec is None or not str(spec).strip():
            return None
        return _load_system_prompt_spec(str(spec))
    return resolve_teacher_system_prompt(teacher_model_config)


def inject_system_message(messages: list[dict], system_prompt: str) -> list[dict]:
    """Return a copy of ``messages`` with ``system_prompt`` as the leading system turn.

    If the conversation already starts with a system/developer turn, that turn's
    content is replaced (teacher-only context should not stack on a student system
    message that the teacher never saw during student rollout).
    """
    if not system_prompt:
        raise ValueError("system_prompt must be non-empty")
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") in ("system", "developer"):
        msgs[0] = {"role": "system", "content": system_prompt}
    else:
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


def align_teacher_outputs_to_student(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    student_prompt_len: int,
    student_response_len: int,
    teacher_prompt_len: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap teacher prompt_logprobs onto the student sequence layout.

    Native OPD feeds student ``prompt_ids + response_ids`` to the teacher. When the
    teacher is scored on a *different* prompt prefix (e.g. an injected system turn)
    of length ``teacher_prompt_len``, the response token ids are still shared, but
    the full teacher tensor is longer/shorter than the student sequence.

    Downstream ``no_padding_2_padding`` left-shifts by one and slices the response
    using *student* prompt/response lengths. Combined with vLLM
    ``extract_prompt_logprobs`` (drop first, append trailing zeros), response token
    ``j`` is read from index ``prompt_len - 1 + j``. Copy that window from the
    teacher tensor onto a student-length tensor so loss alignment is unchanged.
    """
    if student_prompt_len < 1:
        raise ValueError(f"student_prompt_len must be >= 1, got {student_prompt_len}")
    if student_response_len < 1:
        raise ValueError(f"student_response_len must be >= 1, got {student_response_len}")
    if teacher_prompt_len < 1:
        raise ValueError(f"teacher_prompt_len must be >= 1, got {teacher_prompt_len}")

    expected_teacher_len = teacher_prompt_len + student_response_len
    if teacher_ids.shape[0] != expected_teacher_len or teacher_logprobs.shape[0] != expected_teacher_len:
        raise ValueError(
            "teacher output length must equal teacher_prompt_len + student_response_len: "
            f"got ids={tuple(teacher_ids.shape)} logprobs={tuple(teacher_logprobs.shape)} "
            f"expected_len={expected_teacher_len} "
            f"({teacher_prompt_len=}, {student_response_len=})"
        )

    # Identity layout: keep the original tensors (including prompt-side logprobs).
    if teacher_prompt_len == student_prompt_len:
        return teacher_ids, teacher_logprobs

    student_len = student_prompt_len + student_response_len
    # Window of length student_response_len that survives left-shift response slicing.
    src = slice(teacher_prompt_len - 1, teacher_prompt_len + student_response_len - 1)
    dst = slice(student_prompt_len - 1, student_prompt_len + student_response_len - 1)

    if teacher_ids.ndim == 1:
        aligned_ids = torch.full((student_len,), pad_token_id, dtype=teacher_ids.dtype)
        aligned_logprobs = torch.zeros(student_len, dtype=teacher_logprobs.dtype)
    else:
        aligned_ids = torch.full(
            (student_len, *teacher_ids.shape[1:]),
            pad_token_id,
            dtype=teacher_ids.dtype,
        )
        aligned_logprobs = torch.zeros(
            (student_len, *teacher_logprobs.shape[1:]),
            dtype=teacher_logprobs.dtype,
        )

    aligned_ids[dst] = teacher_ids[src]
    aligned_logprobs[dst] = teacher_logprobs[src]
    return aligned_ids, aligned_logprobs


def _get_teacher_sampling_params(
    teacher_model_config: DistillationTeacherModelConfig,
    distillation_loss_config: DistillationLossConfig,
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
    num_logprobs = distillation_loss_config.topk if distillation_loss_config.loss_settings.use_topk else 0
    return {
        "max_tokens": 1,
        "temperature": 1.0,
        "prompt_logprobs": num_logprobs,
        "detokenize": False,
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
