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
    SelfDistillationConfig,
)
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


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


def build_privileged_sequence(
    response_ids: list[int],
    sample_kwargs: dict[str, Any],
    self_distillation_config: "SelfDistillationConfig",
    tokenizer: Any,
) -> tuple[list[int], int]:
    """(OPSD S1) Build the privileged teacher sequence [teacher_prompt, ŷ].

    Matching the reference impl (siyan-zhao/OPSD), the teacher prompt is a *proper chat
    turn* rebuilt from the raw problem x and reference solution y* via the chat template
    — NOT y* appended after the student's already-templated prompt (which would place the
    solution after the assistant generation marker and break the chat structure).

    Resolves the problem and solution from ``sample_kwargs`` via the (dotted) ``problem_key``
    and ``reference_key``. The solution is truncated to ``max_reference_length`` tokens so the
    teacher context stays within the window sized in DistillationTeacherModelConfig.

    Returns (priv_sequence_ids, priv_prefix_len) where priv_prefix_len == len(teacher_prompt).
    """
    cfg = self_distillation_config

    def _resolve(key: str) -> str:
        val: Any = sample_kwargs
        for part in key.split("."):
            val = val[part] if isinstance(val, dict) else getattr(val, part)
        return val.item() if hasattr(val, "item") else str(val)

    problem = _resolve(cfg.problem_key)
    solution = _resolve(cfg.reference_key)
    # Bound the reference solution to its token budget (the teacher context is sized for this).
    solution = tokenizer.decode(tokenizer.encode(solution, add_special_tokens=False)[: cfg.max_reference_length])
    teacher_user_message = cfg.teacher_template.format(problem=problem, solution=solution)
    teacher_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": teacher_user_message}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=cfg.teacher_thinking,
    )
    return list(teacher_prompt_ids) + list(response_ids), len(teacher_prompt_ids)


def remap_privileged_to_student_layout(
    priv_ids: torch.Tensor,
    priv_logprobs: torch.Tensor,
    prompt_len: int,
    resp_len: int,
    priv_prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(OPSD S2) Re-map a privileged-sequence teacher tensor onto the student [x, ŷ] layout.

    The privileged forward runs over [teacher_prompt, ŷ] (priv_prefix_len + resp_len rows).
    We emit a (prompt_len + resp_len)-row tensor in the SAME [prompt-rows ; response-rows]
    convention as AsyncTeacherLLMServerManager.compute_teacher_logprobs_single: the prompt
    rows are filler (taken from the privileged-prefix head; the teacher prompt differs from
    the student prompt, but these positions are masked out by response_mask downstream and
    never enter the loss), and the response rows are the teacher's ŷ log-probs under the
    privileged context, i.e. priv[priv_prefix_len : priv_prefix_len + resp_len]. So
    _pad_teacher_outputs and the downstream causal alignment stay byte-for-byte identical to
    OPD — no off-by-one to track. Works for both 1-D (topk-disabled) and 2-D (topk) tensors.
    """
    x_ids, x_lp = priv_ids[:prompt_len], priv_logprobs[:prompt_len]
    resp_slice = slice(priv_prefix_len, priv_prefix_len + resp_len)
    y_ids, y_lp = priv_ids[resp_slice], priv_logprobs[resp_slice]
    return torch.cat([x_ids, y_ids], dim=0), torch.cat([x_lp, y_lp], dim=0)


class AsyncTeacherLLMServerManager:
    """Teacher-specific async client used for distillation logprob computation."""

    def __init__(
        self,
        config: DictConfig,
        teacher_client: dict[str, LLMServerClient],
        tokenizer: Any = None,
    ):
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        self.distillation_loss_config: DistillationLossConfig = self.distillation_config.distillation_loss
        self.self_distillation_config = self.distillation_config.self_distillation
        # tokenizer is only needed for OPSD (encoding the y*+bridge privileged context).
        self.tokenizer = tokenizer
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

    async def compute_self_distill_logprobs_single(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        sample_kwargs: dict[str, Any],
        multi_modal_data: Optional[dict[str, Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OPSD frozen self-distillation: the teacher (theta_0 on the OPD pool) evaluates the
        privileged context [x, y*, bridge, ŷ], and the result is remapped onto the student
        [x, ŷ] layout. Reuses the plain-OPD forward verbatim; only the input sequence and the
        output layout differ. Returns the same (teacher_ids, teacher_logprobs) contract as
        ``compute_teacher_logprobs_single``.
        """
        assert self.tokenizer is not None, "OPSD self-distillation requires a tokenizer."
        priv_sequence_ids, priv_prefix_len = build_privileged_sequence(
            response_ids, sample_kwargs, self.self_distillation_config, self.tokenizer
        )
        priv_ids, priv_logprobs = await self.compute_teacher_logprobs_single(
            sequence_ids=priv_sequence_ids,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            routing_key=routing_key,
        )
        return remap_privileged_to_student_layout(
            priv_ids,
            priv_logprobs,
            prompt_len=len(prompt_ids),
            resp_len=len(response_ids),
            priv_prefix_len=priv_prefix_len,
        )
