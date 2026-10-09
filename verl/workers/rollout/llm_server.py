# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Utility classes for manage and request LLM servers:
- LLMServerManager: manage life-cycle of LLM servers, including launch, tear-down replicas.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
- GlobalRequestLoadBalancer: global load balancer for LLMServerClient.
"""

import asyncio
import logging
import os
from typing import Any, cast
from uuid import uuid4

import torch
from cachetools import LRUCache
from omegaconf import DictConfig

from verl.runtime import (
    ClassWithInitArgs,
    RemoteCall,
    RemoteWorkerGroup,
    ResourcePool,
    Worker,
    WorkerGroup,
    current_runtime,
)
from verl.utils import normalize_token_ids
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ROUTING_CACHE_SIZE = 10000


class GlobalRequestLoadBalancer(Worker):
    """Global sticky-session + in-flight load balancer shared by all AgentLoopWorkers.

    When a sticky session points to a removed server, the cache entry is
    automatically invalidated and a new server is selected.

    The manager runs this class in a one-rank Runtime WorkerGroup. The class
    remains directly usable so existing subclasses and Ray-side tests continue
    to work while their construction migrates to Runtime.

    Key features:
    - **Atomic acquire**: ``acquire_server()`` returns ``(server_id, handle)``
    - **Sticky Session**: Uses LRUCache to map request_id → server_id, ensuring
      multi-turn conversations route to the same server.
    - **Least-loaded Selection**: When no sticky session exists, selects the
      server with the fewest in-flight requests.
    - **Deterministic Routing**: When ``full_determinism=True``, routes every
      request by ``hash(request_id) % len(servers)`` over the full pool so the
      same request always routes to the same replica across runs.
    - **Dynamic Server Management**: Supports add/remove servers at runtime
      for hybrid scaling.
    """

    def __init__(
        self,
        servers: dict[str, RemoteWorkerGroup],
        max_cache_size: int = DEFAULT_ROUTING_CACHE_SIZE,
        full_determinism: bool = False,
    ):
        if "WORLD_SIZE" in os.environ:
            super().__init__()
        if not servers:
            raise ValueError("servers must be non-empty")

        self._servers: dict[str, RemoteWorkerGroup] = dict(servers)
        self._inflight_requests: dict[str, int] = {sid: 0 for sid in servers}
        self._request_id_to_server: LRUCache = LRUCache(maxsize=max_cache_size)
        self._full_determinism = full_determinism

    async def acquire_server(self, request_id: str) -> tuple[str, RemoteWorkerGroup]:
        """Acquire a server for the given request (sticky + least-loaded).

        Returns:
            A tuple of ``(server_id, actor_handle)`` in a single atomic call.
        """
        # Try sticky session first
        if request_id in self._request_id_to_server:
            server_id = self._request_id_to_server[request_id]
            # Check if server is still in the active pool
            if server_id in self._inflight_requests:
                self._inflight_requests[server_id] += 1
                return server_id, self._servers[server_id]
            # Server was removed, clear stale cache entry and re-select
            del self._request_id_to_server[request_id]

        # Select new server (least-loaded among available)
        if not self._inflight_requests:
            raise RuntimeError("No available servers in load balancer")

        if self._full_determinism:
            # Full-hash routing: same request_id always lands on the same replica
            # across runs. Least-loaded selection depends on async arrival timing,
            # which varies run-to-run, so it is bypassed entirely here.
            server_id = list(self._servers)[hash(request_id) % len(self._servers)]
        else:
            min_count = min(self._inflight_requests.values())
            candidates = [sid for sid, count in self._inflight_requests.items() if count == min_count]
            server_id = candidates[0]
        self._request_id_to_server[request_id] = server_id
        self._inflight_requests[server_id] += 1
        return server_id, self._servers[server_id]

    async def release_server(self, server_id: str) -> None:
        """Release a server after a request completes."""
        if server_id not in self._inflight_requests:
            return
        if self._inflight_requests[server_id] > 0:
            self._inflight_requests[server_id] -= 1

    async def add_servers(self, servers: dict[str, RemoteWorkerGroup]) -> None:
        """Atomically add multiple servers to the load balancer pool.

        This is more efficient than calling :meth:`add_server` in a loop
        because it performs a single bulk update on the internal state.

        Args:
            servers: Dict mapping server_id → actor_handle for all servers
                to register.
        """
        for sid, handle in servers.items():
            self._inflight_requests[sid] = 0
            self._servers[sid] = handle
        logger.info(f"[GlobalLoadBalancer] added {len(servers)} servers")

    async def remove_servers(self, server_ids: list[str]) -> None:
        """Atomically remove multiple servers from the load balancer pool.

        More efficient than calling :meth:`remove_server` in a loop.

        Args:
            server_ids: List of server identifiers to remove.
        """
        for sid in server_ids:
            self._inflight_requests.pop(sid, None)
            self._servers.pop(sid, None)
        logger.info(f"[GlobalLoadBalancer] removed {len(server_ids)} servers")

    async def get_inflight_count(self, server_id: str) -> int:
        """Get number of in-flight requests for a server."""
        return self._inflight_requests.get(server_id, 0)

    async def get_all_servers(self) -> list[str]:
        """Get list of all active server IDs."""
        return list(self._inflight_requests.keys())

    async def get_status(self) -> dict:
        """Return current load balancer state for debugging."""
        return {
            "servers": dict(self._inflight_requests),
            "total_inflight": sum(self._inflight_requests.values()),
            "active_servers": len(self._inflight_requests),
            "registered_handles": list(self._servers.keys()),
        }


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        rollout_config: DictConfig | None = None,
        load_balancer_handle: RemoteWorkerGroup | None = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            rollout_config (DictConfig): selected rollout config; defaults to the legacy config path.
            load_balancer_handle: shared global load balancer WorkerGroup projection.
                that also holds the server-handle registry. Optional; subclasses that
                manage server routing externally can pass None.
        """
        self.config = config
        self.rollout_config = rollout_config if rollout_config is not None else config.actor_rollout_ref.rollout
        self._load_balancer = load_balancer_handle

    async def _acquire_server(self, request_id: str) -> tuple[str, RemoteWorkerGroup]:
        if self._load_balancer is None:
            raise RuntimeError("load balancer is not configured")
        call = cast(
            RemoteCall[tuple[str, RemoteWorkerGroup]],
            self._load_balancer.submit("acquire_server", kwargs={"request_id": request_id}),
        )
        return await call

    async def _release_server(self, server_id: str) -> None:
        if self._load_balancer is None:
            raise RuntimeError("load balancer is not configured")
        call = cast(
            RemoteCall[None],
            self._load_balancer.submit("release_server", kwargs={"server_id": server_id}),
        )
        await call

    def _vllm_request_id(self, request_id: str) -> str:
        # request_id passed to vLLM. Default: a fresh uuid per turn so each turn
        # is an independent vLLM request. Under full_determinism the caller's
        # request_id is passed straight through so vLLM sees a stable id across runs.
        if getattr(self.rollout_config, "full_determinism", False):
            return request_id
        return uuid4().hex

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput | DiffusionOutput: token or diffusion output
        """
        server_id, server = await self._acquire_server(request_id)
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = {"priority": priority} if priority != 0 and self.rollout_config.name == "vllm" else {}
            output: TokenOutput = await server.submit(
                "generate",
                kwargs={
                    "request_id": self._vllm_request_id(request_id),
                    "prompt_ids": prompt_ids,
                    "sampling_params": sampling_params,
                    "image_data": image_data,
                    "video_data": video_data,
                    **multimodal_kwargs,
                    **priority_kwargs,
                    **kwargs,
                },
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            await self._release_server(server_id)


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            image_data (Optional[List[Any]]): Image data for the chat completion.
            video_data (Optional[List[Any]]): Video data for the chat completion.
            audio_data (Optional[List[Any]]): Audio data for the chat completion.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Multimodal processor kwargs.

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
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                **kwargs,
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
            # If partial rollout not enable, aborted samples will be dropped.
            # For v1 trainer, should_retry is always True. Since self.config.async_training is not exist.
            should_retry = True
            if hasattr(self.config, "async_training") and not self.config.async_training.partial_rollout:
                should_retry = False
            if output.stop_reason not in ("aborted", "abort") or not should_retry:
                break

            await asyncio.sleep(1)

        final_output.extra_fields["global_steps"] = global_steps
        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Launch global load balancer
    - Elastic launch/tear-down new replicas

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        rollout_config (DictConfig): rollout config selected by topology ``config_key``.
        model_config (DictConfig): model config selected by topology ``config_key``.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
        start_rank (int): First ``replica_rank`` to assign.  Defaults to 0.
        load_balancer_cls: Optional subclass of
            :class:`GlobalRequestLoadBalancer` to use as the routing actor
            (created in a Runtime WorkerGroup). Defaults to
            :class:`GlobalRequestLoadBalancer`, whose routing honors the
            ``full_determinism`` flag. Pass a subclass that overrides
            :meth:`acquire_server` to take full control of routing.
    """

    def __init__(
        self,
        config: DictConfig,
        rollout_config: DictConfig | None = None,
        model_config: DictConfig | None = None,
        worker_group: WorkerGroup[Worker] | None = None,
        rollout_resource_pool: ResourcePool | None = None,
        start_rank: int = 0,
        load_balancer_cls: type | None = None,
    ):
        self.config = config
        self.rollout_config = rollout_config if rollout_config is not None else config.actor_rollout_ref.rollout
        self.model_config = model_config if model_config is not None else config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.start_rank = start_rank
        self._load_balancer_cls = load_balancer_cls or GlobalRequestLoadBalancer

        if worker_group is None and rollout_resource_pool is None and self.rollout_config.nnodes <= 0:
            raise ValueError("standalone mode requires a declared ResourcePool or rollout.nnodes > 0")

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager."""
        instance = cls(*args, **kwargs)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        return instance

    async def _initialize_llm_servers(self, start_rank: int = None):
        """Initialize the LLM server replicas.

        Args:
            start_rank: First ``replica_rank`` to assign.  Defaults to ``self.start_rank``
                so standalone replicas can avoid Ray named-actor collisions with hybrid
                replicas (which start at 0) when both coexist (e.g. separate async).
        """
        if start_rank is None:
            start_rank = self.start_rank
        rollout_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )
        # PD inflates per-replica footprint; miss this and init_hybrid slices
        # past worker_group → empty workers on replica_rank>=1.
        disagg = getattr(self.rollout_config, "disaggregation", None)
        if disagg is not None and getattr(disagg, "enabled", False):
            prefill_tp = self.rollout_config.tensor_model_parallel_size
            # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
            decode_tp = (
                disagg.decode_tensor_model_parallel_size
                if disagg.decode_tensor_model_parallel_size is not None
                else prefill_tp
            )
            rollout_world_size = (
                (prefill_tp * disagg.prefill_replicas + decode_tp * disagg.decode_replicas)
                * self.rollout_config.data_parallel_size
                * self.rollout_config.pipeline_model_parallel_size
            )
        if self.worker_group:
            world_size = self.worker_group.world_size
        elif self.rollout_resource_pool is not None:
            world_size = self.rollout_resource_pool.world_size
        else:
            world_size = self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        if world_size <= 0 or world_size % rollout_world_size:
            raise ValueError(
                f"rollout world size {world_size} must be a positive multiple of "
                f"replica parallelism {rollout_world_size}"
            )
        num_replicas = world_size // rollout_world_size
        gpus_per_node = (
            self.rollout_resource_pool.processes_per_node
            if self.rollout_resource_pool is not None
            else self.rollout_config.n_gpus_per_node
        )

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(
                *[server.init_hybrid(self.worker_group, self.rollout_resource_pool) for server in self.rollout_replicas]
            )
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            declared_pool = self.rollout_resource_pool
            if declared_pool is None:
                await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])
            else:
                replica_pools = (
                    [declared_pool]
                    if len(self.rollout_replicas) == 1
                    else [
                        declared_pool.slice(slice(start, start + rollout_world_size))
                        for start in range(0, world_size, rollout_world_size)
                    ]
                )
                await asyncio.gather(
                    *[
                        server.init_standalone(resource_pool)
                        for server, resource_pool in zip(self.rollout_replicas, replica_pools, strict=True)
                    ]
                )

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus configuration with server addresses
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)

    async def _init_global_load_balancer(self) -> None:
        runtime = current_runtime()
        load_balancer_pool = runtime.create_resource_pool(
            nnodes=1,
            processes_per_node=1,
            device_type="cpu",
            on="controller",
        )
        load_balancer_cls = self._load_balancer_cls
        kwargs = dict(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            max_cache_size=DEFAULT_ROUTING_CACHE_SIZE,
        )
        if load_balancer_cls is GlobalRequestLoadBalancer:
            kwargs["full_determinism"] = getattr(self.rollout_config, "full_determinism", False)
        load_balancer_group = cast(
            WorkerGroup[GlobalRequestLoadBalancer],
            await runtime.create_worker_group_async(
                ClassWithInitArgs(
                    load_balancer_cls,
                    **kwargs,
                ),
                on=load_balancer_pool,
            ),
        )
        self._load_balancer_group = load_balancer_group
        self.global_load_balancer = load_balancer_group.remote()

    def close(self) -> None:
        """Close routing and rollout resources owned by this manager."""
        from verl.runtime import ExceptionGroup

        errors: list[Exception] = []
        load_balancer_group = getattr(self, "_load_balancer_group", None)
        if load_balancer_group is not None:
            try:
                load_balancer_group.close()
            except Exception as exc:  # noqa: BLE001 - finish owned cleanup
                errors.append(exc)
            else:
                self._load_balancer_group = None
                self.global_load_balancer = None

        replicas = list(getattr(self, "rollout_replicas", ()))
        server_handles = list(getattr(self, "server_handles", ()))
        server_addresses = list(getattr(self, "server_addresses", ()))
        failed_replica_indices: list[int] = []
        for index in range(len(replicas) - 1, -1, -1):
            try:
                replicas[index].close()
            except Exception as exc:  # noqa: BLE001 - finish owned cleanup
                errors.append(exc)
                failed_replica_indices.append(index)
        failed_replica_indices.reverse()
        self.rollout_replicas = [replicas[index] for index in failed_replica_indices]
        if len(server_handles) == len(replicas):
            self.server_handles = [server_handles[index] for index in failed_replica_indices]
        elif not failed_replica_indices:
            self.server_handles = []
        if len(server_addresses) == len(replicas):
            self.server_addresses = [server_addresses[index] for index in failed_replica_indices]
        elif not failed_replica_indices:
            self.server_addresses = []

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup("runtime failures", errors)

    def get_client(self, client_cls: type[LLMServerClient] | None = None, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        Args:
            client_cls: The client class to instantiate (default: ``LLMServerClient``).
            **kwargs: Additional client arguments. The manager always supplies its selected rollout config.
                Pass ``FullyAsyncLLMServerClient`` for abort-resume support.
        """
        client_cls = client_cls or LLMServerClient
        return client_cls(
            config=self.config,
            rollout_config=self.rollout_config,
            load_balancer_handle=self.global_load_balancer,
            **kwargs,
        )

    def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas."""
        return self.rollout_replicas

    @auto_await
    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    @auto_await
    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])
