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

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl.workers.config import DiffusionModelConfig


class DiffusionModelBase(ABC):
    """
    Helper class to define the commonly used methods for diffusion model training.

    Since the forward and sampling process of different diffusion models can be quite different,
    we define an abstract base class for diffusion models to implement their own forward and sampling process.
    Users can check the implementation of QwenImage for reference.

    To register a new model, decorate the subclass with ``@DiffusionModelBase.register("your-model-name")``.
    The name must match the pipeline ``_class_name`` from ``model_index.json`` (i.e.
    the ``architecture`` field of ``DiffusionModelConfig``, which is auto-detected by default).
    """

    _registry: dict[str, type["DiffusionModelBase"]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers a subclass under *name*.

        Example::

            @DiffusionModelBase.register("MyModel")
            class MyModel(DiffusionModelBase):
                ...
        """

        def decorator(subclass: type["DiffusionModelBase"]) -> type["DiffusionModelBase"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, model_config: DiffusionModelConfig) -> type["DiffusionModelBase"]:
        """Return the registered subclass for *model_config.architecture*.

        Raises:
            NotImplementedError: if ``model_config.architecture`` is not in the registry.
        """
        try:
            return cls._registry[model_config.architecture]
        except KeyError:
            registered = list(cls._registry)
            raise NotImplementedError(
                f"No diffusion model registered for architecture={model_config.architecture!r}. "
                f"Registered: {registered}"
            ) from None

    @classmethod
    @abstractmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        """
        Abstract method for setting timesteps and sigmas for diffusion model schedulers during model init,
        and move the timesteps and sigmas to the correct device.

        Args:
            scheduler (SchedulerMixin): the scheduler used for the diffusion process.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            device (str): the device to move the timesteps and sigmas to.
        """
        pass

    @classmethod
    @abstractmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Abstract method for forwarding the model and sampling previous step.
        It is usually used for RL-algorithms based on reversed-sampling process.

        Args:
            module (ModelMixin): the diffusion model to be forwarded.
            scheduler (SchedulerMixin): the scheduler used for the diffusion process.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            model_inputs (dict[str, torch.Tensor]): the inputs to the diffusion model.
            negative_model_inputs (Optional[dict[str, torch.Tensor]]): the negative inputs for guidance.
            scheduler_inputs (Optional[TensorDict | dict[str, torch.Tensor]]): the extra inputs for the scheduler,
                which may contain the latents and timesteps.
            step (int): the current step in the diffusion process.
        """
        pass
