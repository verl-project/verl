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
from verl.workers.config import DiffusionModelConfig

from .base import DiffusionModelBase


def prepare_model_inputs(
    module: ModelMixin,
    model_config: DiffusionModelConfig,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_embeds_mask: torch.Tensor,
    micro_batch: TensorDict,
    step: int,
) -> tuple[dict, dict]:
    """Build architecture-specific model inputs for the forward pass.
    Dispatches to the registered DiffusionModelBase subclass for the current architecture.

    Args:
        module (ModelMixin): the diffusion transformer module.
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
        latents (torch.Tensor): full latent tensor from the micro-batch, shape (B, T, ...).
        timesteps (torch.Tensor): full timestep tensor from the micro-batch, shape (B, T).
        prompt_embeds (torch.Tensor): dense positive prompt embeddings, shape (B, L, D).
        prompt_embeds_mask (torch.Tensor): attention mask for prompt_embeds, shape (B, L).
        negative_prompt_embeds (torch.Tensor): dense negative prompt embeddings, shape (B, L, D).
        negative_prompt_embeds_mask (torch.Tensor): attention mask for negative_prompt_embeds.
        micro_batch (TensorDict): the full micro-batch, available for architecture-specific
            metadata (e.g. height, width, vae_scale_factor).
        step (int): the current denoising step index.
    """
    return DiffusionModelBase.get_class(model_config).prepare_model_inputs(
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    )


def build_scheduler(model_config: DiffusionModelConfig) -> SchedulerMixin:
    """Build and configure the scheduler for the diffusion model.
    The returned scheduler has timesteps and sigmas already set.

    Args:
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
    """
    return DiffusionModelBase.get_class(model_config).build_scheduler(model_config)


def set_timesteps(scheduler: SchedulerMixin, model_config: DiffusionModelConfig):
    """Set correct timesteps and sigmas for diffusion model schedulers.

    Args:
        scheduler (SchedulerMixin): the scheduler used for the diffusion process.
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
    """
    DiffusionModelBase.get_class(model_config).set_timesteps(scheduler, model_config, get_device_name())


def forward_and_sample_previous_step(
    module: ModelMixin,
    scheduler: SchedulerMixin,
    model_config: DiffusionModelConfig,
    model_inputs: dict,
    negative_model_inputs: Optional[dict],
    scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
    step: int,
):
    """Forward the model and sample previous step.
    This method is usually used for RL-algorithms based on reversed-sampling process.
    Such as FlowGRPO, DanceGRPO, etc.

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
    return DiffusionModelBase.get_class(model_config).forward_and_sample_previous_step(
        module, scheduler, model_config, model_inputs, negative_model_inputs, scheduler_inputs, step
    )
