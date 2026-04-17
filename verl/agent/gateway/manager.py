from __future__ import annotations

import asyncio


async def _await_object_ref(object_ref):
    return await asyncio.wrap_future(object_ref.future())


class GatewayManager:
    """Session-routing component owned by the serving runtime."""

    def __init__(self, gateways: list):
        self.gateways = gateways
        self.gateway_count = len(gateways)
        self.active_sessions_per_gateway = [0 for _ in gateways]
        self._session_to_gateway_index: dict[str, int] = {}

    def _select_gateway_index(self) -> int:
        if not self.gateways:
            raise RuntimeError("No gateway actors configured")
        return min(range(len(self.gateways)), key=lambda index: self.active_sessions_per_gateway[index])

    def _get_gateway_index(self, session_id: str) -> int:
        gateway_index = self._session_to_gateway_index.get(session_id)
        if gateway_index is None:
            raise KeyError(session_id)
        return gateway_index

    def _get_gateway(self, session_id: str):
        gateway_index = self._get_gateway_index(session_id)
        return self.gateways[gateway_index], gateway_index

    async def create_session(self, session_id: str, **kwargs):
        gateway_index = self._select_gateway_index()
        gateway = self.gateways[gateway_index]
        handle = await _await_object_ref(gateway.create_session.remote(session_id=session_id, **kwargs))
        self._session_to_gateway_index[session_id] = gateway_index
        self.active_sessions_per_gateway[gateway_index] += 1
        return handle

    async def finalize_session(self, session_id: str):
        gateway, gateway_index = self._get_gateway(session_id)
        trajectories = await _await_object_ref(gateway.finalize_session.remote(session_id=session_id))
        self._session_to_gateway_index.pop(session_id, None)
        self.active_sessions_per_gateway[gateway_index] -= 1
        return trajectories

    async def abort_session(self, session_id: str) -> None:
        gateway, gateway_index = self._get_gateway(session_id)
        await _await_object_ref(gateway.abort_session.remote(session_id=session_id))
        self._session_to_gateway_index.pop(session_id, None)
        self.active_sessions_per_gateway[gateway_index] -= 1

    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None:
        gateway, _ = self._get_gateway(session_id)
        await _await_object_ref(gateway.wait_for_completion.remote(session_id=session_id, timeout=timeout))
