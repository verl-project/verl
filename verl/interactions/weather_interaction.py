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

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class WeatherInteraction(BaseInteraction):
    """A demo interaction for handling weather-related queries.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the assistant.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        content = "no tool call"
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "tool":
                content = item.get("content")
                break
        self._instance_dict[instance_id]["response"] = content

        reward = await self.calculate_score(instance_id)
        if reward == 1.0:
            response = "Thank you for your weather query!"
            should_terminate_sequence = True
        else:
            response = "Please use the weather tool to get the weather information."
            should_terminate_sequence = True
        return should_terminate_sequence, response, reward, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # For weather interaction, we can implement a more complex scoring logic
        # For now, we'll just return a default score of 1.0
        if self._instance_dict[instance_id]["response"] == "no tool call":
            return 0.0
        return 1.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
