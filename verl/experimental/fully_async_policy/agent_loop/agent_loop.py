# Copyright 2025 Meituan Ltd. and/or its affiliates
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
import logging
import os
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    AsyncLLMServerManager,
    TokenOutput,
)
from verl.experimental.teacher_loop import TeacherModelManager
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import (
    rollout_trace_op,
)
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    """FullyAsyncLLMServerManager supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.

        Returns:
            TokenOutput: token output
        """
        prompt_ids = normalize_token_ids(prompt_ids)

        limit_key = None
        if "max_tokens" in sampling_params:
            limit_key = "max_tokens"
        elif "max_new_tokens" in sampling_params:
            limit_key = "max_new_tokens"
        original_max_tokens = sampling_params.get(limit_key) if limit_key else None

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None

        while True:
            # 1. generate tokens
            output = await super().generate(
                request_id=request_id,
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
            )

            # 2. merge output into final_output
            final_output.token_ids.extend(output.token_ids)
            if output.log_probs is not None:
                final_output.log_probs.extend(output.log_probs)
            # On partial rollout resume the model version may differ, so keep
            # existing routing and only append routing for newly generated tokens.
            if output.routed_experts is not None and len(output.token_ids) > 0:
                if final_output.routed_experts is None:
                    final_output.routed_experts = output.routed_experts
                else:
                    final_output.routed_experts = torch.cat(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]],
                        dim=0,
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            final_output.stop_reason = output.stop_reason

            # update model weights version
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # 3. update max_new_tokens
            if original_max_tokens is not None:
                sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)
                if len(final_output.token_ids) >= original_max_tokens:
                    final_output.stop_reason = "length"
                    break

            # 4. check stop reason
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break
        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorker):
    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        teacher_servers: list[tuple[str, ray.actor.ActorHandle]] = None,
        teacher_load_balancer_handle: ray.actor.ActorHandle = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.server_manager = FullyAsyncLLMServerManager(config, servers, load_balancer_handle)
        super().__init__(
            config,
            servers,
            load_balancer_handle,
            teacher_servers,
            teacher_load_balancer_handle,
            reward_loop_worker_handles,
        )


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        teacher_model_manager: TeacherModelManager = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker
        super().__init__(config, worker_group, rollout_resource_pool, teacher_model_manager, reward_loop_worker_handles)
        if self.distillation_enabled:
            raise NotImplementedError("Distillation is not implemented in FullyAsyncAgentLoopManager yet.")

    @auto_await
    async def generate_sequences_single(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch. Single sample data
        Returns:
            DataProto: Output batch.
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(prompts)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker
