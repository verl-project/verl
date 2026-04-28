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
"""Custom vllm-omni pipeline for BAGEL RL rollouts with VeRL.

Extends :class:`BagelPipeline` to:
* Replace the scheduler with an SDE scheduler for stochastic denoising
  with log-probability recording.
* Always enable trajectory recording.
* Read SDE kwargs from ``sampling_params.extra_args``.
* Return RL artifacts in ``DiffusionOutput.custom_output``.

Loaded via ``custom_pipeline_args``:

.. code-block:: python

    custom_pipeline_args={
        "pipeline_class": "examples.flowgrpo_trainer.vllm_omni.pipeline_bagel.BagelPipelineWithLogProb"
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from ..scheduler import FlowMatchSDEDiscreteScheduler

logger = logging.getLogger(__name__)


def _maybe_to_cpu(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, list):
        return [_maybe_to_cpu(x) for x in v]
    return v


@dataclass
class _AdapterStepOutput:
    """Adapter output matching what bagel_transformer.generate_image expects."""

    prev_sample: torch.Tensor
    log_prob: torch.Tensor | None


class _BagelSchedulerAdapter:
    """Wraps the diffusers-based FlowMatchSDEDiscreteScheduler to match
    BAGEL's calling convention: ``step(v_t, sigma, x_t, dt, **kwargs)``.

    BAGEL's transformer calls ``scheduler.step(model_output, timesteps[i],
    sample, dts[i], **scheduler_kwargs)`` with 4 positional args, while the
    diffusers scheduler takes ``step(model_output, timestep, sample, **kwargs)``
    and computes dt internally.  This adapter bridges the gap.
    """

    def __init__(self, inner: FlowMatchSDEDiscreteScheduler):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def step(
        self,
        model_output: torch.Tensor,
        sigma: float | torch.Tensor,
        sample: torch.Tensor,
        dt: float | torch.Tensor,  # noqa: ARG002 — not used, inner computes from timestep schedule
        **kwargs,
    ) -> _AdapterStepOutput:
        out = self._inner.step(
            model_output=model_output,
            timestep=sigma,
            sample=sample,
            return_dict=False,
            **kwargs,
        )
        # step() with return_dict=False returns (prev_sample, log_prob, prev_sample_mean, std_dev_t)
        prev_sample, log_prob = out[0], out[1]
        return _AdapterStepOutput(prev_sample=prev_sample, log_prob=log_prob)


class BagelPipelineWithLogProb(BagelPipeline):
    """BAGEL pipeline variant for RL rollouts with VeRL."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        inner = FlowMatchSDEDiscreteScheduler()
        self.scheduler = _BagelSchedulerAdapter(inner)
        logger.info("BagelPipelineWithLogProb: SDE scheduler enabled for RL rollouts")

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # Force trajectory recording on for RL
        req.sampling_params.return_trajectory_latents = True

        # Read SDE scheduler kwargs from extra_args
        extra_args = req.sampling_params.extra_args
        logprobs = extra_args.get("logprobs", True)
        self.scheduler_kwargs = {k: extra_args[k] for k in ("noise_level", "sde_type", "generator") if k in extra_args}
        self.scheduler_kwargs["return_logprobs"] = logprobs

        # Per-request scheduler setup: compute BAGEL's shifted sigmas so
        # the inner SDE scheduler's sigma schedule matches what
        # generate_image() computes internally.
        assert req.sampling_params.num_inference_steps is not None, "num_inference_steps must be set for RL rollouts"
        num_timesteps = req.sampling_params.num_inference_steps
        timestep_shift = 3.0  # must match BagelPipeline.forward() hardcoded value

        t = np.linspace(1, 0, num_timesteps)
        t_shifted = timestep_shift * t / (1 + (timestep_shift - 1) * t)
        sigmas = t_shifted[:-1].tolist()  # drop terminal 0; set_timesteps appends it

        inner = self.scheduler._inner
        inner.set_shift(1.0)  # identity — sigmas already shifted
        inner.set_timesteps(sigmas=sigmas)
        inner.set_begin_index(0)

        output = super().forward(req)

        # Enrich custom_output with RL-specific fields
        custom = output.custom_output or {}
        if output.trajectory_latents is not None:
            custom["all_latents"] = _maybe_to_cpu(output.trajectory_latents)
        if output.trajectory_timesteps is not None:
            custom["all_timesteps"] = _maybe_to_cpu(output.trajectory_timesteps)
        if output.trajectory_log_probs is not None:
            custom["all_log_probs"] = _maybe_to_cpu(output.trajectory_log_probs)
        output.custom_output = custom

        return output
