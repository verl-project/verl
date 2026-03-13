# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from abc import ABC, abstractmethod
from typing import Optional

from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl.workers.config import DiffusersModelConfig


class DiffusionModelBase(ABC):
    """
    Helper class to define the commonly used methods for diffusion model training.

    Since the forward and sampling process of different diffusion models can be quite different,
    we define an abstract base class for diffusion models to implement their own forward and sampling process.
    Users can check the implementation of QwenImage for reference.
    """

    @classmethod
    @abstractmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusersModelConfig, device: str):
        """
        Abstract method for setting timesteps and sigmas for diffusion model schedulers during model init,
        and move the timesteps and sigmas to the correct device.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusersModelConfig,
        micro_batch: TensorDict,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        step: int,
    ):
        """Abstract method for forwarding the model and sampling previous step.
        It is usually used for RL-algorithms based on reversed-sampling process."""
        raise NotImplementedError
