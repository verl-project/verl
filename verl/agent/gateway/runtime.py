from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import ray

from verl.workers.rollout.replica import DiffusionOutput, TokenOutput


async def _await_ray_ref(object_ref):
    return await asyncio.wrap_future(object_ref.future())


class GatewayServingRuntime:
    """Standalone serving runtime that owns gateway actors and backend routing."""

    def __init__(
        self,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle | None,
        *,
        gateway_manager=None,
        gateway_count: int = 0,
        gateway_actor_kwargs: dict[str, Any] | None = None,
    ):
        self._load_balancer = load_balancer_handle
        self._server_id_to_handle: dict[str, ray.actor.ActorHandle] = dict(servers or [])
        self.owned_gateway_actors: list[ray.actor.ActorHandle] = []
        self.gateway_manager = gateway_manager

        if self.gateway_manager is None and gateway_count > 0:
            self._initialize_gateway_runtime(
                gateway_count=gateway_count,
                gateway_actor_kwargs=gateway_actor_kwargs,
            )

    def _initialize_gateway_runtime(
        self,
        *,
        gateway_count: int,
        gateway_actor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from verl.agent.gateway.gateway import GatewayActor
        from verl.agent.gateway.manager import GatewayManager

        gateway_actor_kwargs = dict(gateway_actor_kwargs or {})
        if "backend" not in gateway_actor_kwargs:
            gateway_actor_kwargs["backend"] = self

        self.owned_gateway_actors = [GatewayActor.remote(**gateway_actor_kwargs) for _ in range(gateway_count)]
        ray.get([gateway.start.remote() for gateway in self.owned_gateway_actors])
        self.gateway_manager = GatewayManager(self.owned_gateway_actors)

    def _require_session_runtime(self):
        if self.gateway_manager is None:
            raise RuntimeError("Session runtime is disabled because gateway_count=0")
        return self.gateway_manager

    async def create_session(self, session_id: str, **kwargs):
        gateway_manager = self._require_session_runtime()
        return await gateway_manager.create_session(session_id=session_id, **kwargs)

    async def finalize_session(self, session_id: str):
        gateway_manager = self._require_session_runtime()
        return await gateway_manager.finalize_session(session_id=session_id)

    async def abort_session(self, session_id: str) -> None:
        gateway_manager = self._require_session_runtime()
        await gateway_manager.abort_session(session_id=session_id)

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        gateway_manager = self._require_session_runtime()
        await gateway_manager.wait_for_completion(session_id=session_id, timeout=timeout)

    async def shutdown(self) -> None:
        if self.owned_gateway_actors:
            await asyncio.gather(*[_await_ray_ref(gateway.shutdown.remote()) for gateway in self.owned_gateway_actors])
        self.owned_gateway_actors = []
        self.gateway_manager = None

    async def _acquire_server(self, request_id: str) -> tuple[str, ray.actor.ActorHandle]:
        if self._load_balancer is None:
            raise RuntimeError("GatewayServingRuntime has no configured load balancer")
        server_id = await self._load_balancer.acquire_server.remote(request_id=request_id)
        handle = self._server_id_to_handle.get(server_id)
        if handle is None:
            raise RuntimeError(f"Unknown server_id returned by load balancer: {server_id}")
        return server_id, handle

    def _release_server(self, server_id: str) -> None:
        self._load_balancer.release_server.remote(server_id=server_id)

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        **kwargs: Any,
    ) -> TokenOutput | DiffusionOutput:
        server_id, server = await self._acquire_server(request_id)
        try:
            output: TokenOutput | DiffusionOutput = await server.generate.remote(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **kwargs,
            )
            return output
        finally:
            self._release_server(server_id)
