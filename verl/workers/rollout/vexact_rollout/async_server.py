#!/usr/bin/env python3
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

"""
Ray-based async server wrapping VExact for RL inference.

This provides a server interface using VExact as the underlying inference engine,
following the vLLM async server interface pattern.
"""

import asyncio
import logging
import os
from typing import Any, Optional

import ray
from ray.actor import ActorHandle
from transformers import GenerationConfig

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings
from verl.workers.rollout.vexact_rollout.vexact_rollout import ServerAdapter

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VExactServer:
    """VExact server in single node, following vLLMHttpServer interface pattern.

    This is equivalent to launching a VExact inference server with specified configuration.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        """
        Args:
            config: Full rollout configuration.
            model_config: Model configuration.
            rollout_mode: Rollout mode (HYBRID, COLOCATED, or STANDALONE).
            workers: List of Ray actor handles for workers.
            replica_rank: Replica rank for multi-replica setup.
            node_rank: Node rank for multi-node setup.
            gpus_per_node: Number of GPUs per node.
            nnodes: Number of nodes.
            cuda_visible_devices: CUDA visible devices string.
        """
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)

        # Validate max_model_len
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        elif self.config.max_model_len > max_position_embeddings:
            raise ValueError(
                f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                f"max_position_embeddings ({max_position_embeddings})"
            )

        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # Server state
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # Engine will be initialized in launch_server
        self.engine = None
        self.tokenizer = None

        logger.info(
            f"VExactServer initialized (replica_rank={replica_rank}, node_rank={node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"gpus_per_node={gpus_per_node}, nnodes={nnodes})"
        )

    def get_server_address(self):
        """Get server address and port."""
        assert self._server_port is not None, "server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        """Launch the VExact engine.

        Args:
            master_address: Master address for multi-node setup (unused for single-node).
            master_port: Master port for multi-node setup (unused for single-node).
        """
        from vexact.config import (
            CacheConfig,
            DriverConfig,
            ModelConfig,
            ParallelConfig,
            ProfilerConfig,
            SchedulerConfig,
            VeXactConfig,
        )
        from vexact.engine import VExact

        engine_kwargs = self.config.engine_kwargs.pop("vexact", {})
        logger.info(f"Extra {engine_kwargs=}")

        attn_impl = engine_kwargs.pop("attn_impl", os.environ.get("INFER_FA_IMPL", "fa-invariant"))
        vexact_config = VeXactConfig(
            model=ModelConfig(
                model_path=self.model_config.local_path,
                attn_impl=attn_impl,
                enable_batch_invariant=True,
                enable_memory_saver=self.config.free_cache_engine,
                enforce_eager=self.config.enforce_eager,
                use_fp32_logits=self.model_config.use_fused_kernels,
            ),
            parallel=ParallelConfig(
                pipeline_parallel_size=self.config.pipeline_model_parallel_size,
            ),
            scheduler=SchedulerConfig(
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                enable_chunked_prefill=self.config.enable_chunked_prefill,
            ),
            driver=DriverConfig(
                is_worker_proc_managed=True,
                driver_id=f"verl_rollout_replica_{self.replica_rank}",
            ),
            cache=CacheConfig(max_cache_blocks=engine_kwargs.get("max_cache_blocks", 1024)),
            profiler=ProfilerConfig(
                backend="torch" if self.config.profiler.enable else None,
                delay_iterations=10000,  # delay some iterations to skip prefill stage
                max_iterations=200,
                output_path=self.config.profiler.save_path,
                profile_all_ranks=True,
            ),
        )

        self.engine = VExact(vexact_config)
        self.tokenizer = self.engine.tokenizer

        # Resolve eos_token_id from the model config (config.json), NOT the tokenizer.
        # Some models (e.g. Moonlight Instruct) have tokenizer.eos_token_id != config.json eos_token_id.
        from transformers import AutoConfig

        model_hf_config = AutoConfig.from_pretrained(
            self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
        )
        self._eos_token_id = getattr(model_hf_config, "eos_token_id", self.tokenizer.eos_token_id)
        if self._eos_token_id != self.tokenizer.eos_token_id:
            logger.warning(
                f"Model config eos_token_id ({self._eos_token_id}) differs from "
                f"tokenizer eos_token_id ({self.tokenizer.eos_token_id}). "
                f"Using model config value for generation stop condition."
            )

        # VExact doesn't use network ports, set to 0
        self._server_port = 0

        logger.info(f"VExact server ready (replica_rank={self.replica_rank}, node_rank={self.node_rank})")

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,  # noqa: ARG002
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        from vexact.core.request import DriverRequest

        if image_data is not None or video_data is not None:
            logger.warning("image_data and video_data not supported by VExact, ignoring")

        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        assert "max_tokens" not in sampling_params, "sampling_params should not contain 'max_tokens'"
        assert "max_new_tokens" not in sampling_params, "sampling_params should not contain 'max_new_tokens'"
        max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        temperature = sampling_params.pop("temperature", self.config.temperature)
        top_p = sampling_params.pop("top_p", self.config.top_p)
        top_k = sampling_params.pop("top_k", self.config.top_k)
        repetition_penalty = sampling_params.pop("repetition_penalty", self.config.repetition_penalty)
        do_sample = sampling_params.pop("do_sample", self.config.do_sample)

        logprobs_requested = sampling_params.pop("logprobs", False)
        if isinstance(logprobs_requested, int):
            logprobs_requested = logprobs_requested > 0
        if sampling_params:
            logger.warning(f"Remaining sampling_params not supported: {sampling_params}")

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            max_length=len(prompt_ids) + max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if (do_sample and top_k > 0) else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self._eos_token_id,
            output_scores=True,
            output_logits=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        request = DriverRequest(
            request_id=request_id,
            generation_config=gen_config,
            input_ids_list=prompt_ids,
        )

        result = await self.engine.generate(request)

        token_ids = result.new_token_ids
        log_probs = result.new_logprobs if logprobs_requested else None

        finish_reason = getattr(result, "reason", None)
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length", None):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=None,  # VExact doesn't support routing replay
            stop_reason=stop_reason,
            num_preempted=None,
        )

    async def wake_up(self, tags: Optional[list[str]] = None):
        """Restore GPU memory from CPU."""
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            loop = asyncio.get_event_loop()
            tag = tags[0] if tags else None
            await loop.run_in_executor(None, self.engine.wake_up, tag)
        elif self.rollout_mode == RolloutMode.COLOCATED:
            loop = asyncio.get_event_loop()
            tag = tags[0] if tags else None
            await loop.run_in_executor(None, self.engine.wake_up, tag)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        """Offload GPU memory to CPU."""
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.engine.sleep)

    async def clear_kv_cache(self):
        """Clear KV cache. VExact manages KV cache internally via scheduler."""
        pass

    async def wait_for_requests_to_drain(self):
        """Wait for all pending requests to complete. VExact processes requests synchronously."""
        pass

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Abort all ongoing generation requests."""
        return {"aborted_count": 0, "request_ids": []}

    async def resume_generation(self):
        """Resume generation after abort_all_requests. No-op for VExact."""
        pass

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:  # noqa: ARG002
        """Abort a specific generation request."""
        return {"aborted": False, "request_id": request_id, "error": "VExact doesn't support request abortion"}

    async def receive_weights(self):
        """Receive model weights via IPC on all workers."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.engine.driver_client.receive_weights)

    async def start_profile(self, **kwargs):  # noqa: ARG002
        """Start profiling on the server."""
        pass

    async def stop_profile(self):
        """Stop profiling on the server."""
        pass


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class VExactReplica(RolloutReplica):
    """VExact rollout replica extending RolloutReplica base class.

    This class manages VExact servers across multiple nodes, following the vLLMReplica pattern.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(VExactServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        return RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )

    async def launch_servers(self):
        """Launch VExact server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [info[1] for info in worker_infos]
        worker_node_ids = [info[0] for info in worker_infos]

        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node

        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            name = (
                f"vexact_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"vexact_server_reward_{self.replica_rank}_{node_rank}"
            )

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        await asyncio.gather(*[server.launch_server.remote() for server in self.servers])

        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = f"{server_address}:{server_port}"

    async def sleep(self):
        """Sleep each rollout server."""
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing generation requests across all servers."""
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def resume_generation(self):
        """Resume generation on all servers after abort_all_requests."""
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])

    async def resume_all_requests(self):
        """Resume all requests on all servers."""
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it."""
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}
