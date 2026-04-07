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
import asyncio
import random
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AsyncLLMServerManager,
    DictConfigWrap,
    _agent_loop_registry,
    _get_rollout_and_model_config,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.workers.config import DiffusionModelConfig, DiffusionRolloutConfig


class DiffusionAgentLoopOutput(BaseModel):
    """Agent loop output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    """Prompt token ids."""
    response_diffusion_output: Any
    """Response diffusion output (torch.Tensor): image tensor (CHW) / video tensor (TCHW)."""
    response_logprobs: Optional[Any] = None
    """Log probabilities for the response tokens. (torch.Tensor)"""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalDiffusionAgentLoopOutput(DiffusionAgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_diffusion_output: torch.Tensor
    """Response diffusion output: image (NCHW format) / video (NTCHW format)."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids)."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Log probabilities for the response tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g. pixel_values, image_grid_thw, video_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DiffusionAgentLoopWorker:
    """Diffusion Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        servers (list[tuple[str, ray.actor.ActorHandle]]): (address, handle) pairs for each LLM server.
        load_balancer_handle (ray.actor.ActorHandle): shared global load balancer actor.
        teacher_servers (list[tuple[str, ray.actor.ActorHandle]]): Not used.
        teacher_load_balancer_handle (ray.actor.ActorHandle): Not used.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        teacher_servers: list[tuple[str, ray.actor.ActorHandle]] = None,
        teacher_load_balancer_handle: ray.actor.ActorHandle = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        rollout_config, model_config = _get_rollout_and_model_config(config)
        self.rollout_config: DiffusionRolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: DiffusionModelConfig = omega_conf_to_dataclass(model_config)

        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(
                config,
                servers,
                load_balancer_handle=load_balancer_handle,
            )

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_loop_worker_handles = reward_loop_worker_handles

        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        self.max_prompt_embed_length = self.model_config.extra_configs.get(
            "max_sequence_length", self.rollout_config.prompt_length
        )

        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: diffusion output, typically [bsz, C, H, W] (image) or [bsz, T, C, H, W] (video).
            - rm_scores (optional): [bsz, 1], reward model scores.
            - meta_info:
              - metrics: List[dict], per-sample agent loop metrics.
              - reward_extra_keys (optional): List[str], keys for reward extra info for logging/validation.
            ...
        """
        config = self.rollout_config

        sampling_params = dict(self.model_config.extra_configs)
        sampling_params.update(
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["num_inference_steps"] = config.val_kwargs.num_inference_steps
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["noise_level"] = config.val_kwargs.noise_level

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, **kwargs)))
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)

        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalDiffusionAgentLoopOutput:
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )
        output: DiffusionAgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalDiffusionAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        # handling extra tensor ouputs from vllm-omni, like prompt embedding, etc.
        extra_fields = {}
        for k, v in output.extra_fields.items():
            if isinstance(v, torch.Tensor):
                # handle prompt embedding padding
                # TODO (andy): reduce padding length for more effiency
                if k in ["prompt_embeds", "negative_prompt_embeds"]:
                    pad_tuple = (0, 0, 0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                elif k in ["prompt_embeds_mask", "negative_prompt_embeds_mask"]:
                    pad_tuple = (0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                extra_fields[k] = v.unsqueeze(0)
            else:
                extra_fields[k] = v

        extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        response_diffusion_output = output.response_diffusion_output.unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            response_logprobs = output.response_logprobs.unsqueeze(0)

        attention_mask = prompt_output["attention_mask"]
        input_ids = prompt_output["input_ids"]

        await self._compute_score(
            output,
            prompts=input_ids,
            responses=response_diffusion_output,
            attention_mask=attention_mask,
            input_ids=input_ids,
            kwargs=kwargs,
        )

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]

        return _InternalDiffusionAgentLoopOutput(
            prompt_ids=input_ids,
            response_diffusion_output=response_diffusion_output,
            input_ids=input_ids,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        if output.reward_score is None and enable_async_reward:
            batch = TensorDict(
                {
                    "prompts": prompts,  # [1, prompt_length]
                    "responses": responses,  # [1, C, H, W] or [1, T, C, H, W]
                    "attention_mask": attention_mask,  # [1, prompt_length]
                    "input_ids": input_ids,  # [1, prompt_length]
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
            result = await selected_reward_loop_worker_handle.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

    def _postprocess(
        self,
        inputs: list[_InternalDiffusionAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_diffusion_output = torch.cat([input.response_diffusion_output for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        # Handle extra fields that are tensors
        extra_keys = [k for k, v in inputs[0].extra_fields.items() if isinstance(v, torch.Tensor)]
        for key in extra_keys:
            optional_outputs[key] = torch.cat([input.extra_fields[key] for input in inputs], dim=0)
            for input in inputs:
                del input.extra_fields[key]

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_diffusion_output,  # [bsz, C, H, W] or [bsz, T, C, H, W]
                "input_ids": input_ids,  # [bsz, prompt_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length]
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            rm_scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )
