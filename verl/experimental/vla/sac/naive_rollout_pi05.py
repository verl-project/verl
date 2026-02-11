# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
In single GPU rollout, the sequences are generated directly by sampling from the model.
The output will contain
1. output_ids
2. attention_masks (left padding)
3. eos_masks
4. log_probs
"""

import logging
from typing import Any

import torch

from verl import DataProto
from verl.experimental.vla.naive_rollout_rob import NaiveRolloutRob
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)

__all__ = ["PI0RolloutRob"]


class PI0RolloutRob(NaiveRolloutRob):
    def __init__(
        self,
        model_config: dict,
        module: torch.nn.Module,
        tokenizer: Any,
    ):
        self.model_config = model_config
        self.module = module
        self.tokenizer = tokenizer

        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(self.module, "sample_actions")

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences"""

        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            prompts.to(get_device_id())
            output, s, a = self.module.sample_actions(prompts, tokenizer=self.tokenizer)

        ret = DataProto.from_dict(
            {
                "action": output.action,
                "full_action": a["full_action"],
                "images": s["images"],
                "image_masks": s["image_masks"],
                "lang_tokens": s["lang_tokens"],
                "lang_masks": s["lang_masks"],
                "states": s["states"],
            }
        )

        return ret
