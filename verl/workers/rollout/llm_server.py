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
- LLMServerManager: rollout control plane as a single dedicated Ray actor
  (zero CPUs): replica lifecycle + the embedded global load balancer.
- LLMServerClient: proxy client to request LLM servers, used by AgentLoopWorker.
"""

import asyncio
import inspect
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import numpy as np
import ray
from omegaconf import DictConfig

from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import normalize_token_ids
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tracking import RLInsightLogger
from verl.workers.rollout.replica import RolloutReplica, TokenOutput, get_rollout_replica_class
from verl.workers.rollout.router import (  # noqa: F401
    GlobalRequestLoadBalancer,
    RouterSpec,
    resolve_router_spec,
)
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _as_actor(cls: type["LLMServerManager"]) -> "ray.actor.ActorClass":
    """Wrap a manager class as a zero-CPU Ray actor (``ray.remote(cls)`` picks up subclass overrides)."""
    return ray.remote(cls).options(num_cpus=0)


class LLMServerClient:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least in-flight requests load balancing via global coordination
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        **kwargs,
    ):
        """Initialize the LLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): the LLMServerManager
                actor handle (hosts the load balancer). Optional; subclasses
                that manage server routing externally can pass None.
        """
        self.config = config
        self._load_balancer = load_balancer_handle
        # Balancer field declarations, queried lazily once.
        self._lb_require_acquire_fields: list[str] | None = None
        self._lb_require_release_fields: list[str] | None = None

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # Only the declared fields are serialized.
        if self._lb_require_acquire_fields is None:
            acquire_fields, release_fields = await asyncio.gather(
                self._load_balancer.require_acquire_fields.remote(),
                self._load_balancer.require_release_fields.remote(),
            )
            self._lb_require_acquire_fields = list(acquire_fields)
            self._lb_require_release_fields = list(release_fields)
        fields = {name: extra[name] for name in self._lb_require_acquire_fields if name in extra}
        return await self._load_balancer.acquire_server.remote(request_id=request_id, **fields)

    def _release_server(self, server_id: str, request_id: str | None = None) -> None:
        # Fire-and-forget: release is just a counter decrement, no need to await.
        # Awaiting here risks blocking the finally clause if the LB actor is unresponsive.
        pool = {"request_id": request_id}
        fields = {name: pool[name] for name in self._lb_require_release_fields if name in pool}
        self._load_balancer.release_server.remote(server_id=server_id, **fields)

    def _vllm_request_id(self, request_id: str) -> str:
        # request_id passed to vLLM. Default: a fresh uuid per turn so each turn
        # is an independent vLLM request. Under full_determinism the caller's
        # request_id is passed straight through so vLLM sees a stable id across runs.
        if getattr(self.config.actor_rollout_ref.rollout, "full_determinism", False):
            return request_id
        return uuid4().hex

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
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
        server_id, server = await self._acquire_server(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            **kwargs,
        )
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # SGLang cannot take raw frames: its video_data accepts only a path/url/base64 or a
            # processor_output dict. Always hand it the pre-computed payload (None when the model's
            # processor produced no pixel_values_videos, e.g. a non-Qwen-VL family) and never the raw
            # frames -- dropping video beats sending SGLang tensors it cannot parse. vLLM keeps the
            # frames and never enters this branch. Neither server signature accepts **kwargs, so pop.
            mm_processor_output = kwargs.pop("mm_processor_output", None)
            if self.config.actor_rollout_ref.rollout.name == "sglang":
                video_data = mm_processor_output
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority} if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm" else {}
            )
            output: TokenOutput = await server.generate.remote(
                request_id=self._vllm_request_id(request_id),  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                session_id=request_id,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            return output
        finally:
            self._release_server(
                server_id,
                request_id=request_id,
            )


class FullyAsyncLLMServerClient(LLMServerClient):
    """FullyLLMServerClient supports resume generation on partial rollout, making rollout interruption
    invisible to the AgentLoop.
    """

    def __init__(
        self,
        config: DictConfig,
        load_balancer_handle: ray.actor.ActorHandle = None,
        only_hybrid: bool = False,
        **kwargs,
    ):
        """Initialize the FullyAsyncLLMServerClient.

        Args:
            config (DictConfig): whole config for main entrypoint.
            load_balancer_handle (ray.actor.ActorHandle): the LLMServerManager
                actor handle (hosts the load balancer).
            only_hybrid (bool): When ``True``, hybrid replicas are the *only* rollout
                resource.  If the load balancer is temporarily empty (e.g. during
                weight synchronisation) :meth:`_acquire_server` will keep retrying
                every 1 second instead of raising immediately.
        """
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._only_hybrid = only_hybrid

    async def _acquire_server(self, request_id: str, **extra) -> tuple[str, ray.actor.ActorHandle]:
        # Atomic acquire: returns (server_id, handle) in one Ray RPC.
        # When only_hybrid is True, hybrid replicas are the sole rollout resource and
        # the LB may be temporarily empty during weight sync / scaling transitions.
        # In that case keep retrying every 1 s until a server becomes available.
        # Otherwise raise immediately so callers see the error right away.
        while True:
            try:
                return await super()._acquire_server(request_id, **extra)
            except RuntimeError as e:
                if "No available servers in load balancer" in str(e) and self._only_hybrid:
                    await asyncio.sleep(1)
                else:
                    raise

    def _configured_response_length(self) -> Optional[int]:
        """Per-response token budget from the rollout config, or ``None`` when unavailable.

        Tests and lightweight callers may pass a config stub without the rollout section; in that
        case the resume loop keeps its previous behaviour of deferring to the server default.
        """
        rollout_config = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        response_length = getattr(rollout_config, "response_length", None)
        if isinstance(response_length, int) and response_length > 0:
            return response_length
        return None

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
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

        # The budget below is rewritten on every attempt, and the caller reuses its dict across
        # turns, so never mutate the caller's copy.
        sampling_params = dict(sampling_params)

        if original_max_tokens is None:
            # Without an explicit limit each attempt falls back to the server-side default, which is
            # derived from len(prompt_ids) and is only correct on the first attempt: a resume passes
            # prompt + tokens generated so far, so the default charges generated tokens against the
            # *prompt* budget and re-permits close to a full response_length every time. Pin the
            # cumulative budget here instead, which also makes the bookkeeping in step 3 effective.
            response_length = self._configured_response_length()
            if response_length is not None:
                limit_key = "max_tokens"
                original_max_tokens = response_length
                sampling_params[limit_key] = response_length

        final_output = TokenOutput(
            token_ids=[],
            log_probs=[],
            num_preempted=0,
        )
        min_global_steps, max_global_steps = None, None
        # Prefix-cache hits are reported per prefill. The base client returns the
        # server's TokenOutput directly (so sync mode surfaces num_cached_tokens),
        # but here we rebuild a fresh TokenOutput across resume iterations, so we
        # must carry it forward explicitly or the consumer sees 0. Take the first
        # (initial-prompt) prefill's hit count, matching single-prefill semantics.
        num_cached_tokens = None

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
                    final_output.routed_experts = np.concatenate(
                        [final_output.routed_experts, output.routed_experts[-len(output.token_ids) :]]
                    )
            if output.num_preempted is not None:
                final_output.num_preempted += output.num_preempted
            # sampler top-k heads for score centering, one row per newly generated token
            for key in ("response_topk_ids", "response_topk_log_probs"):
                if key in output.extra_fields:
                    previous = final_output.extra_fields.get(key)
                    final_output.extra_fields[key] = (
                        output.extra_fields[key]
                        if previous is None
                        else np.concatenate([previous, output.extra_fields[key]])
                    )
            if "response_topk_ids" in output.extra_fields:
                current = output.extra_fields.get("response_support_complete")
                if current is None:
                    current = np.zeros(len(output.extra_fields["response_topk_ids"]), dtype=np.bool_)
                previous = final_output.extra_fields.get("response_support_complete")
                if previous is None:
                    previous = np.zeros(len(final_output.token_ids) - len(output.token_ids), dtype=np.bool_)
                final_output.extra_fields["response_support_complete"] = np.concatenate([previous, current])
            final_output.stop_reason = output.stop_reason

            # carry the initial prefill's prefix-cache hit count forward
            if num_cached_tokens is None:
                num_cached_tokens = output.extra_fields.get("num_cached_tokens")

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
        final_output.extra_fields["num_cached_tokens"] = num_cached_tokens
        return final_output


class LLMServerManager:
    """LLMServerManager is responsible for:
    - Launch server replicas
    - Host the global load balancer as an internal component and expose its
      surface as this actor's RPC methods
    - Elastic launch/tear-down new replicas

    Created via :meth:`create` / :meth:`create_empty` it lives as a zero-CPU
    Ray actor (call methods via ``handle.method.remote()``); the class stays
    plain Python so unit tests can construct it directly (``__new__``).

    Args:
        config (DictConfig): Config for the trainer entrypoint.
        worker_group (RayWorkerGroup): Worker group for the server replicas. If not none, init hybrid server,
            else init standalone server with a new resource pool.
        rollout_resource_pool (RayResourcePool): Resource pool for the server replicas, only needed for TensorRT-LLM.
        start_rank (int): First ``replica_rank`` to assign.  Defaults to 0.
        load_balancer_cls: Optional subclass of the default router strategy's
            load balancer, embedded as a plain object inside this actor. When
            given, it takes precedence over ``rollout.router_config_path``.
            Pass a subclass that overrides :meth:`acquire_server` to take
            full control of routing.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        start_rank: int = 0,
        load_balancer_cls: type | None = None,
    ):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout
        self.model_config = config.actor_rollout_ref.model
        self.worker_group = worker_group
        self.rollout_resource_pool = rollout_resource_pool
        self.start_rank = start_rank
        self._load_balancer_cls = load_balancer_cls

        assert worker_group is not None or self.rollout_config.nnodes > 0, "nnodes must be > 0 in standalone mode"

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(
                self.rollout_config.name,
                disaggregation_enabled=self.rollout_config.disaggregation.enabled,
            )

    @classmethod
    def _router_spec_from_config(cls, config, load_balancer_cls: type | None = None) -> RouterSpec:
        """Resolve the router spec from the manager config on the caller's node.

        ``router_config_path`` is read defensively: downstream config schemas
        (e.g. verl-omni's DiffusionRolloutConfig) may omit it entirely.
        """
        rollout_config = getattr(getattr(config, "actor_rollout_ref", None), "rollout", None)
        return resolve_router_spec(
            router_config_path=getattr(rollout_config, "router_config_path", None),
            load_balancer_cls=load_balancer_cls,
        )

    @classmethod
    def _launch_manager_actor(cls, args: tuple, kwargs: dict) -> tuple["ray.actor.ActorHandle", RouterSpec]:
        """Resolve the router spec on the caller's node, then launch the actor.

        The router YAML read and plugin import must happen before launch: the
        zero-CPU manager actor can be scheduled on any node and must not need
        the router config file locally, and a bad config fails fast here,
        before replicas are initialized inside the actor.

        Subclass contract: ``config`` and ``load_balancer_cls`` are read from the
        bound factory arguments, so a manager subclass that overrides
        ``__init__`` must declare them explicitly. A forwarding signature such
        as ``__init__(self, *args, **kwargs)`` hides both and silently resolves
        the default strategy. Subclasses that do not override ``__init__`` (or
        declare the base signature) are unaffected.
        """
        bound = inspect.signature(cls).bind(*args, **kwargs)
        router_spec = cls._router_spec_from_config(
            config=bound.arguments.get("config"),
            load_balancer_cls=bound.arguments.get("load_balancer_cls"),
        )
        # An embedded balancer is constructed in-process inside this actor, so a
        # Ray actor class cannot be hosted. Reject it here rather than inside
        # ``_init_global_load_balancer``: by then ``create`` has already launched
        # the replicas.
        if isinstance(router_spec.cls, ray.actor.ActorClass):
            raise TypeError(
                f"{router_spec.cls!r} is a Ray actor class; an embedded load balancer must be "
                "a plain class — the LLMServerManager actor is the host process."
            )
        return _as_actor(cls).remote(*args, **kwargs), router_spec

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the LLMServerManager as a Ray actor and fully initialize it.

        The factory signature is unchanged (``auto_await`` keeps sync/async
        callers working); initialization runs inside the actor. The router
        strategy is resolved on the caller's node and shipped into the actor.
        """
        handle, router_spec = cls._launch_manager_actor(args, kwargs)
        await handle._initialize_llm_servers.remote()
        await handle._init_global_load_balancer.remote(router_spec)
        return handle

    @classmethod
    @auto_await
    async def create_empty(cls, *args, **kwargs):
        """Create an empty LLMServerManager: no replicas are launched and the
        load balancer starts empty.

        Used by trainers that serve rollout exclusively from standalone
        resources (e.g. v1 separate_async without ``hybrid_rollout.enable_switch``)
        so that no inference engine is initialized on the training GPUs.
        """
        handle, router_spec = cls._launch_manager_actor(args, kwargs)
        await handle._init_empty_replicas.remote()
        await handle._init_global_load_balancer.remote(router_spec)
        return handle

    async def _init_empty_replicas(self) -> None:
        """Empty the replica tracking lists (actor-side step of ``create_empty``)."""
        self.rollout_replicas = []
        self.server_handles = []
        self.server_addresses = []

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
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.rollout_config.n_gpus_per_node * self.rollout_config.nnodes
        )
        num_replicas = world_size // rollout_world_size

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + replica_rank,
                config=self.rollout_config,
                model_config=self.model_config,
                gpus_per_node=self.rollout_config.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and self.rollout_config.name != "trtllm":
            await asyncio.gather(*[server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # TODO: unify trtllm to init_hybrid
        elif self.worker_group and self.rollout_config.name == "trtllm":
            await asyncio.gather(
                *[
                    server.init_hybrid_colocated(self.worker_group, self.rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            await asyncio.gather(*[server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")

        # Update Prometheus / rl-insight metrics with server addresses
        needs_metrics = self.rollout_config.prometheus.enable or RLInsightLogger.enabled()
        if self.rollout_config.disable_log_stats:
            if needs_metrics:
                raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")
        if not self.rollout_config.disable_log_stats:
            if self.rollout_config.prometheus.enable:
                update_prometheus_config(
                    self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name
                )
            if RLInsightLogger.enabled():
                RLInsightLogger.register_rollout_metrics(
                    self.server_addresses,
                    self.rollout_config.name,
                    labels=[{"replica": server.replica_rank} for server in self.rollout_replicas],
                )

    async def _init_global_load_balancer(self, router_spec: RouterSpec) -> None:
        """Embed the load balancer as a plain component of this actor.

        ``router_spec`` was resolved on the caller's node (the YAML read and
        plugin import stay out of the actor); only the ``servers`` pool is
        host-local.
        """
        from verl.workers.rollout.router import create_embedded_load_balancer

        self.global_load_balancer = create_embedded_load_balancer(
            servers=dict(zip(self.server_addresses, self.server_handles, strict=True)),
            router_spec=router_spec,
            full_determinism=getattr(self.rollout_config, "full_determinism", False),
        )

    # ── load-balancing plane: the embedded balancer's surface, exposed as
    # this actor's RPC methods ─────────────────────────────────────────────

    async def acquire_server(self, request_id: str, **extra) -> tuple[str, "ray.actor.ActorHandle"]:
        """Sticky-session + least-inflight acquire: returns ``(server_id, handle)``."""
        return self.global_load_balancer.acquire_server(request_id, **extra)

    async def release_server(self, server_id: str, request_id: str | None = None) -> None:
        """Release a server after a request completes."""
        # Forward request_id only when the strategy declares it: legacy
        # release_server(self, server_id) signatures must keep working.
        if "request_id" in self.global_load_balancer.require_release_fields():
            return self.global_load_balancer.release_server(server_id, request_id=request_id)
        return self.global_load_balancer.release_server(server_id)

    async def require_acquire_fields(self) -> list[str]:
        """Acquire-time ``generate()`` kwargs this router consumes."""
        return self.global_load_balancer.require_acquire_fields()

    async def require_release_fields(self) -> list[str]:
        """Release-time identity fields this router consumes."""
        return self.global_load_balancer.require_release_fields()

    async def add_servers(self, servers: dict[str, "ray.actor.ActorHandle"]) -> None:
        """Bulk-add servers to the embedded load balancer pool."""
        return self.global_load_balancer.add_servers(servers)

    async def remove_servers(self, server_ids: list[str]) -> None:
        """Bulk-remove servers from the embedded load balancer pool."""
        return self.global_load_balancer.remove_servers(server_ids)

    async def clear_sticky_cache(self) -> dict:
        """Clear sticky-session state to force request redistribution."""
        return self.global_load_balancer.clear_sticky_cache()

    async def get_total_inflight(self) -> int:
        """Total in-flight requests across registered servers."""
        return self.global_load_balancer.get_total_inflight()

    async def get_inflight_count(self, server_id: str) -> int:
        """In-flight request count for one server."""
        return self.global_load_balancer.get_inflight_count(server_id)

    async def get_all_servers(self) -> list[str]:
        """List all active server IDs."""
        return self.global_load_balancer.get_all_servers()

    async def get_status(self) -> dict:
        """Current load balancer state for debugging."""
        return self.global_load_balancer.get_status()

    async def get_client(self, client_cls: type[LLMServerClient] | None = None, **kwargs) -> LLMServerClient:
        """Get the LLMServerClient to request LLM server replicas.

        The client is constructed inside the actor and returned by value; its
        routing handle is this actor's own handle
        (``ray.get_runtime_context().current_actor``).

        Args:
            client_cls: The client class to instantiate. Defaults to
                :class:`LLMServerClient`. Pass a subclass to customize
                request-id handling (e.g. a deterministic client that forwards
                the caller's ``request_id`` straight to vLLM), or
                :class:`FullyAsyncLLMServerClient` for abort-resume support.
            **kwargs: Forwarded to the client constructor.
        """
        client_cls = client_cls or LLMServerClient
        return client_cls(
            config=self.config,
            load_balancer_handle=ray.get_runtime_context().current_actor,
            **kwargs,
        )

    async def get_addresses(self) -> list[str]:
        """Get the OpenAI chat completion API http addresses of the LLM server replicas."""
        return self.server_addresses

    async def get_server_registry(self) -> dict[str, "ray.actor.ActorHandle"]:
        """Get the ``{server_address: server actor handle}`` mapping."""
        return dict(zip(self.server_addresses, self.server_handles, strict=True))

    async def get_replicas(self) -> list[RolloutReplica]:
        """Get the LLM server replicas (by value — each RPC returns a fresh snapshot)."""
        return self.rollout_replicas

    async def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        await asyncio.gather(*[replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    async def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        await asyncio.gather(*[replica.stop_profile() for replica in self.rollout_replicas])
