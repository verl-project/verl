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
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Any, Callable, ClassVar

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


RawRewardFn = Callable[..., Any] | None

# Heavy per-token / multimodal trajectory fields omitted from streaming reward RPC by default
# so multi-megabyte training payloads (e.g. 32K top-k log-prob heads) are not serialized to CPU
# reward workers unless requested via ``RewardManagerBase.extra_input_fields``.
OPTIONAL_REWARD_INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "prompt_ids",
        "prompt_logprobs",
        "response_logprobs",
        "response_mask",
        "response_topk_ids",
        "response_topk_log_probs",
        "response_support_complete",
        "routed_experts",
        "multi_modal_data",
    }
)


class RewardManagerBase(ABC):
    _class_initialized = False
    # Optional trajectory fields (from OPTIONAL_REWARD_INPUT_FIELDS) to include in extra_info
    # on top of the default reward inputs.
    # TODO: Add ``default_input_fields`` so default DataProto columns and ``extra_input_fields``
    # share one declarative input contract across streaming and colocated reward paths.
    extra_input_fields: ClassVar[Collection[str]] = ()

    @classmethod
    def get_extra_input_fields(cls, config: DictConfig) -> frozenset[str]:
        """Return optional trajectory fields required by this reward manager beyond default inputs."""
        fields = frozenset(cls.extra_input_fields)
        unsupported = fields - OPTIONAL_REWARD_INPUT_FIELDS
        if unsupported:
            raise ValueError(
                f"{cls.__name__} declared unsupported extra_input_fields {sorted(unsupported)}; "
                f"supported optional fields are {sorted(OPTIONAL_REWARD_INPUT_FIELDS)}."
            )
        return fields

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, compute_score: RawRewardFn):
        """Initialize reward manager.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.loop = get_event_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: DataProto):
        raise NotImplementedError

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        """Assemble per-sample reward scores into the ``rm_scores`` tensor for a batch.

        Args:
            data: The concatenated batch passed to :meth:`run_single`.
                ``data.batch["prompts"]``, ``data.batch["responses"]`` and
                ``data.batch["attention_mask"]`` are expected to be present
                for the default implementation.
            scores: List of scalar reward scores, one per sample in ``data``.

        Returns:
            torch.Tensor: The ``rm_scores`` tensor with leading dimension equal to
            ``len(data)``.
        """
        prompt_length = data.batch["prompts"].size(1)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        rm_scores[torch.arange(rm_scores.size(0), device=rm_scores.device), valid_response_length - 1] = (
            rm_scores.new_tensor(scores)
        )
        return rm_scores
