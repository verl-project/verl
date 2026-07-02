from __future__ import annotations

import asyncio
from typing import Any

import ray

from verl.workers.rollout.llm_server import LLMServerClient


async def _await_ray_ref(object_ref):
    return await asyncio.wrap_future(object_ref.future())


class GatewayServingRuntime:
    """Standalone serving runtime that owns gateway actors and delegates backend routing."""

    def __init__(
        self,
        llm_client: LLMServerClient,
        *,
        gateway_manager=None,
        gateway_count: int = 0,
        gateway_actor_kwargs: dict[str, Any] | None = None,
    ):
        self._llm_client = llm_client
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

        # Round-robin across alive CPU nodes so gateway actors do not all pack onto
        # the driver node under Ray's default PACK scheduling. Mirrors
        # AgentLoopWorker placement (verl/experimental/agent_loop/agent_loop.py).
        node_ids = [
            node["NodeID"]
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("CPU", 0) > 0
        ]
        if not node_ids:
            raise RuntimeError("No alive CPU nodes available for GatewayActor placement")

        self.owned_gateway_actors = [
            GatewayActor.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_ids[i % len(node_ids)], soft=True,
                ),
            ).remote(**gateway_actor_kwargs)
            for i in range(gateway_count)
        ]
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

    async def complete_session(self, session_id: str) -> None:
        gateway_manager = self._require_session_runtime()
        await gateway_manager.complete_session(session_id=session_id)

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

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._llm_client.generate(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            **kwargs,
        )
