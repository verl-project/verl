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
from typing import Any, Optional, Sequence

import hydra
import numpy as np
import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    AsyncLLMServerManager,
    DictConfigWrap,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.utils.rollout_trace import (
    rollout_trace_attr,
    rollout_trace_op,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    @rollout_trace_op
    async def generate_for_partial(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> tuple[list[Any], list[Any], Any] | tuple[Sequence[int], list[float], bool]:
        """Generate tokens from prompt ids, used for async partial.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            output: A tuple representing the generation output.
            - Element 0 (Sequence[int]): Generated response token IDs.
            - Element 1 (list[float]): Log probabilities for the response token IDs.
            - Element 2 (bool): A flag or status indicating cancellation.
        """
        server = self._choose_server(request_id)
        output = await server.generate_for_partial.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )
        return output


@ray.remote
class FullyAsyncAgentLoopWorker(AgentLoopWorker):
    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.server_manager = FullyAsyncLLMServerManager(config, server_handles)
        super().__init__(config, server_handles, reward_loop_worker_handles)
        # A shared cancellation event for all agent loops running on this worker.
        self.cancellation_event = asyncio.Event()

    async def generate_sequences_no_post(
        self, batch: DataProto, partial_output_list: Optional[list[AgentLoopOutput]]
    ) -> tuple[list[AgentLoopOutput], bool] | tuple[DataProto, bool]:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: List of agent loop outputs, one per sample in the batch.
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        if not partial_output_list:
            partial_output_list = [None] * len(batch)
        try:
            tasks = []
            for i in range(len(batch)):
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
                kwargs["output"] = partial_output_list[i]
                tasks.append(
                    asyncio.create_task(self._partial_run_agent_loop(sampling_params, trajectory_info[i], **kwargs))
                )
            outputs = await asyncio.gather(*tasks)
        except Exception:
            logger.exception("_partial_run_agent_loop failed")
            raise

        is_cancel = any(output.extra_fields.get("is_cancel", False) for output in outputs)
        if not is_cancel:
            output = self._postprocess(outputs)
            output = self._addition_process(output)
            return output, is_cancel
        return outputs, is_cancel

    def _addition_process(self, output: DataProto):
        """collect metirics"""
        metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
        processing_times_list = [item["generate_sequences"] for item in metrics]
        tool_calls_times_list = [item["tool_calls"] for item in metrics]
        output.non_tensor_batch["processing_times"] = processing_times_list
        output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
        return output

    async def _partial_run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> AgentLoopOutput:
        # Completed, return directly
        if kwargs["output"] is not None and not kwargs["output"].extra_fields.get("is_cancel", False):
            logger.info("In _partial_run_agent_loop, already completed, return derictly!")
            return kwargs["output"]
        try:
            with rollout_trace_attr(
                step=trajectory["step"],
                sample_index=trajectory["sample_index"],
                rollout_n=trajectory["rollout_n"],
                validate=trajectory["validate"],
                name="agent_loop",
            ):
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
                    dataset_config=DictConfigWrap(config=self.config.data),
                )
                output: AgentLoopOutput = await agent_loop.run(
                    sampling_params, cancellation_event=self.cancellation_event, **kwargs
                )
                if not output.extra_fields.get("is_cancel", False):
                    kwargs.pop("output", None)
                    output = await self._agent_loop_postprocess(output, **kwargs)

                return output
        except Exception:
            logger.exception("Agent_loop run failed")
            raise

    async def cancel_agent_loops(self):
        """Set the shared cancellation event to stop all agent loops."""
        self.cancellation_event.set()

    async def resume_agent_loops(self):
        """Clear the shared cancellation event."""
        self.cancellation_event.clear()


class FullyAsyncAgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        self.worker_group = worker_group
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.agent_loop_workers_class = FullyAsyncAgentLoopWorker

        # Select rollout replica class based on rollout name
        rollout_name = config.actor_rollout_ref.rollout.name
        if rollout_name == "sglang":
            from verl.experimental.fully_async_policy.sglang_rollout.sglang_async_server import FullyAsyncSGLangReplica

            self.rollout_replica_class = FullyAsyncSGLangReplica
            print("[FullyAsyncAgentLoopManager] SGLang replica class selected")
        elif rollout_name == "vllm":
            from verl.experimental.fully_async_policy.vllm_rollout.vllm_async_server import FullyAsyncvLLMReplica

            self.rollout_replica_class = FullyAsyncvLLMReplica
            print("[FullyAsyncAgentLoopManager] vLLM replica class selected")
        else:
            raise ValueError(f"Unsupported rollout name: {rollout_name}. Supported values are 'sglang' and 'vllm'.")

        self.rollout_replicas = None
        self.server_handles = None
        self.server_addresses = None
        self.agent_loop_workers = None

    @classmethod
    async def create(
        cls,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        instance = cls(config, worker_group, reward_loop_worker_handles)
        await instance._async_init()
        return instance

    async def _async_init(self):
        await self._initialize_llm_servers_async()
        self._init_agent_loop_workers()

    async def _initialize_llm_servers_async(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group:
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")
        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            await asyncio.to_thread(
                update_prometheus_config, rollout_config.prometheus, self.server_addresses, rollout_config.name
            )

    async def generate_single_sample_async(
        self,
        sample: DataProto,
        partial_output_list: Optional[list[AgentLoopOutput]],
    ) -> tuple[list[AgentLoopOutput], bool] | tuple[DataProto, bool]:
        """
        Asynchronously process a single sample

        Args:
            sample: Single sample data
            partial_output_list: Optional[List[AgentLoopOutput]]: already rollout result.

        Returns:
            list[AgentLoopOutput]: Processing results
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences_no_post.remote(sample, partial_output_list)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker

    async def cancel(self):
        worker_cancel_tasks = [worker.cancel_agent_loops.remote() for worker in self.agent_loop_workers]
        rollout_cancel_tasks = [replica.cancel() for replica in self.rollout_replicas]
        await asyncio.gather(*rollout_cancel_tasks, *worker_cancel_tasks)

    async def resume(self):
        rollout_resume_tasks = [replica.resume() for replica in self.rollout_replicas]
        worker_resume_tasks = [worker.resume_agent_loops.remote() for worker in self.agent_loop_workers]
        await asyncio.gather(*rollout_resume_tasks, *worker_resume_tasks)

    async def wake_up(self):
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_replicas])

    async def sleep(self):
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_replicas])

    async def clear_kv_cache(self):
        await asyncio.gather(*[replica.clear_kv_cache() for replica in self.rollout_replicas])
