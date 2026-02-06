# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class FlowMatchSDEDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        log_prob (`torch.FloatTensor` of shape `(batch_size,)`):
            The log probability of the previous sample.
        prev_sample_mean (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The mean of the computed sample of previous timestep.
        std_dev_t (`torch.FloatTensor` of shape `(batch_size, 1, 1, 1)` for images):
            The standard deviation used to compute `prev_sample`.
    """

    prev_sample: torch.FloatTensor
    log_prob: torch.FloatTensor
    prev_sample_mean: torch.FloatTensor
    std_dev_t: torch.FloatTensor


class FlowMatchSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        noise_level: float = 0.7,
        prev_sample: Optional[torch.FloatTensor] = None,
        sde_type: Literal["sde", "cps"] = "sde",
    ) -> FlowMatchSDEDiscreteSchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Modified from https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchSDEDiscreteSchedulerOutput`] or tuple.
            noise_level (`float`, *optional*, defaults to 0.7):
                The noise level used in the SDE.
            prev_sample (`torch.FloatTensor`, *optional*):
                The sample from the previous timestep. If not provided, it will be sampled inside the function.
            sde_type (`str`, *optional*, defaults to "sde"):
                The type of SDE to use. Choose between "sde" and "cps".

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchSDEDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchSDEDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        if prev_sample is not None:
            prev_sample = prev_sample.to(torch.float32)

        prev_sample, log_prob, prev_sample_mean, std_dev_t = self.sample_previous_step(
            sample=sample,
            model_output=model_output,
            generator=generator,
            per_token_timesteps=per_token_timesteps,
            noise_level=noise_level,
            prev_sample=prev_sample,
            sde_type=sde_type,
        )

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample, log_prob, prev_sample_mean, std_dev_t)
        return FlowMatchSDEDiscreteSchedulerOutput(
            prev_sample=prev_sample, log_prob=log_prob, prev_sample_mean=prev_sample_mean, std_dev_t=std_dev_t
        )

    def sample_previous_step(
        self,
        sample: torch.Tensor,
        model_output: torch.Tensor,
        timestep: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        noise_level: float = 0.7,
        prev_sample: Optional[torch.Tensor] = None,
        sde_type: Literal["cps", "sde"] = "sde",
    ):
        # check inputs
        assert sample.dtype == torch.float32
        if prev_sample is not None:
            assert prev_sample.dtype == torch.float32

        if per_token_timesteps is not None:
            raise NotImplementedError("per_token_timesteps is not supported yet for FlowMatchSDEDiscreteScheduler.")
        else:
            if timestep is None:
                sigma_idx = self.step_index
                sigma = self.sigmas[sigma_idx]
                sigma_next = self.sigmas[sigma_idx + 1]
            else:
                sigma_idx = torch.tensor([self.index_for_timestep(t) for t in timestep])
                sigma = self.sigmas[sigma_idx].view(-1, *([1] * (len(sample.shape) - 1)))
                sigma_next = self.sigmas[sigma_idx + 1].view(-1, *([1] * (len(sample.shape) - 1)))

            sigma_max = self.sigmas[1]
            dt = sigma_next - sigma

        if sde_type == "sde":
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level

            # our sde
            prev_sample_mean = (
                sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

            log_prob = (
                -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-1 * dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )

        elif sde_type == "cps":
            std_dev_t = sigma_next * math.sin(noise_level * math.pi / 2)  # sigma_t in paper
            pred_original_sample = sample - sigma * model_output  # predicted x_0 in paper
            noise_estimate = sample + model_output * (1 - sigma)  # predicted x_1 in paper
            prev_sample_mean = pred_original_sample * (1 - sigma_next) + noise_estimate * torch.sqrt(
                sigma_next**2 - std_dev_t**2
            )

            if prev_sample is None:
                variance_noise = randn_tensor(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                prev_sample = prev_sample_mean + std_dev_t * variance_noise

            # remove all constants
            log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)

        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, log_prob, prev_sample_mean, std_dev_t
