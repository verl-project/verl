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

from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.workers.config import DistillationConfig

from .teacher_model import TeacherModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TeacherLoopManager:
    """
    TeacherLoopManager run in single controller.
    This class will create teacher loop workers and manage them.
    """

    def __init__(self, config: DictConfig, teacher_resource_pool: RayResourcePool = None):
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(
            self.config.distillation
        )  # to dataclass for the post init to handle top-k and engine kwargs and get distillation_loss_settings
        self.teacher_model_manager = TeacherModelManager(self.distillation_config.teacher_model, teacher_resource_pool)
        self.teacher_router_address = self.teacher_model_manager.get_router_address()

    @auto_await
    async def wake_up(self):
        """Wake up all rollout replica instances."""
        await self.teacher_model_manager.wake_up()

    @auto_await
    async def sleep(self):
        """Sleep all rollout replica instances."""
        await self.teacher_model_manager.sleep()
