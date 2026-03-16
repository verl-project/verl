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

from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl.utils.device import get_device_name
from verl.workers.config import DiffusersModelConfig


def set_timesteps(scheduler: SchedulerMixin, model_config: DiffusersModelConfig):
    """Set correct timesteps and sigmas for diffusion model schedulers."""
    if model_config.path.endswith("Qwen-Image"):
        from .qwen_image import QwenImage

        QwenImage.set_timesteps(scheduler, model_config, get_device_name())
    else:
        raise NotImplementedError("unsupported model for setting timesteps.")


def forward_and_sample_previous_step(
    module: ModelMixin,
    scheduler: SchedulerMixin,
    model_config: DiffusersModelConfig,
    model_inputs: dict,
    negative_model_inputs: Optional[dict],
    scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
    step: int,
):
    """Forward the model and sample previous step.
    This method is usually used for RL-algorithms based on reversed-sampling process.
    Such as FlowGRPO, DanceGRPO, etc.
    """
    if model_config.path.endswith("Qwen-Image"):
        from .qwen_image import QwenImage

        return QwenImage.forward_and_sample_previous_step(
            module, scheduler, model_config, model_inputs, negative_model_inputs, scheduler_inputs, step
        )
    else:
        raise NotImplementedError("unsupported model for sampling previous step.")
