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
from typing import Any, Callable

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


RawRewardFn = Callable[..., Any] | None


class RewardManagerBase(ABC):
    _class_initialized = False

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

    def _require_data_source(self, data_item: Any) -> Any:
        non_tensor_batch = data_item.non_tensor_batch or {}
        if "data_source" not in non_tensor_batch:
            self._raise_data_source_error("missing", data_item)

        data_source = non_tensor_batch["data_source"]
        if data_source is None or (isinstance(data_source, str) and data_source.strip() == ""):
            self._raise_data_source_error("empty", data_item)
        return data_source

    def _raise_data_source_error(self, reason: str, data_item: Any):
        non_tensor_batch = data_item.non_tensor_batch or {}
        extra_info = non_tensor_batch.get("extra_info", {})
        context = []
        if isinstance(extra_info, dict):
            for key in ("split", "index"):
                if key in extra_info:
                    context.append(f"{key}={extra_info[key]!r}")

        message = (
            f"Reward data source is {reason}. The non-tensor batch field `data_source` selects the rule-based "
            "reward function. For the GSM8K quickstart, regenerate the parquet files with "
            "`python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k` and keep "
            "`data_source` as `openai/gsm8k`. For custom datasets, configure "
            "`reward.custom_reward_function.path` and `reward.custom_reward_function.name`."
        )
        if context:
            message += f" Sample context: {', '.join(context)}."
        raise ValueError(message)

    @abstractmethod
    async def run_single(self, data: DataProto):
        raise NotImplementedError
