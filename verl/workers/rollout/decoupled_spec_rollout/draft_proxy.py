from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional

from .protocol import DraftRequest, DraftResult, DraftRoute, DraftServerEndpoint, SessionKey, VerifyResult


@dataclass
class DraftProxyRuntime:
    proxy: "DraftProxy"
    process: mp.Process
    stop_event: mp.synchronize.Event
    ready_event: mp.synchronize.Event


@dataclass
class DraftProxy:
    verify_replica_rank: int
    num_speculative_steps: int
    draft_endpoints: list[DraftServerEndpoint] = field(default_factory=list)
    session_routes: dict[str, DraftRoute] = field(default_factory=dict)
    inflight_per_replica: dict[int, int] = field(default_factory=dict)
    pending_requests: dict[str, DraftRequest] = field(default_factory=dict)
    pending_results: dict[str, DraftResult] = field(default_factory=dict)

    def __post_init__(self):
        self.register_drafters(self.draft_endpoints)

    def register_drafters(self, draft_endpoints: list[DraftServerEndpoint]) -> None:
        self.draft_endpoints = list(draft_endpoints)
        self.inflight_per_replica = {endpoint.replica_rank: 0 for endpoint in self.draft_endpoints}

    def acquire_route(self, session_key: SessionKey) -> DraftRoute:
        routing_key = session_key.routing_key
        route = self.session_routes.get(routing_key)
        if route is not None:
            return route
        if not self.draft_endpoints:
            raise ValueError("DraftProxy has no registered draft endpoints")

        endpoint = min(
            self.draft_endpoints,
            key=lambda item: (self.inflight_per_replica.get(item.replica_rank, 0), item.replica_rank),
        )
        route = DraftRoute(
            session_key=session_key,
            draft_replica_rank=endpoint.replica_rank,
            draft_server_address=endpoint.server_address,
        )
        self.session_routes[routing_key] = route
        return route

    def submit_prefill(self, request: DraftRequest) -> DraftRoute:
        route = self.acquire_route(request.session_key)
        request.draft_replica_rank = route.draft_replica_rank
        self.pending_requests[request.request_id] = request
        self.inflight_per_replica[route.draft_replica_rank] = self.inflight_per_replica.get(route.draft_replica_rank, 0) + 1
        return route

    def submit_decode(self, request: DraftRequest) -> DraftRoute:
        return self.submit_prefill(request)

    def await_result(self, request_id: str) -> Optional[DraftResult]:
        return self.pending_results.get(request_id)

    def notify_verify_result(self, result: VerifyResult) -> None:
        request = self.pending_requests.pop(result.request_id, None)
        if request is None:
            return
        if request.draft_replica_rank is not None:
            self.inflight_per_replica[request.draft_replica_rank] = max(
                0, self.inflight_per_replica.get(request.draft_replica_rank, 0) - 1
            )

    def release_session(self, session_key: SessionKey) -> None:
        self.session_routes.pop(session_key.routing_key, None)


def launch_draftproxy_subprocess(
    *,
    verify_replica_rank: int,
    num_speculative_steps: int,
    draft_endpoints: list[DraftServerEndpoint],
) -> DraftProxyRuntime:
    from verl.workers.rollout.decoupled_spec_rollout.sglang_patch.draftproxy_subprocess import (
        run_draftproxy_subprocess,
    )

    proxy = DraftProxy(
        verify_replica_rank=verify_replica_rank,
        num_speculative_steps=num_speculative_steps,
        draft_endpoints=draft_endpoints,
    )
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    ready_event = ctx.Event()
    process = ctx.Process(
        target=run_draftproxy_subprocess,
        kwargs={
            "verify_replica_rank": verify_replica_rank,
            "num_speculative_steps": num_speculative_steps,
            "draft_endpoints": [endpoint.to_metadata() for endpoint in draft_endpoints],
            "stop_event": stop_event,
            "ready_event": ready_event,
        },
        daemon=True,
    )
    process.start()
    ready_event.wait(timeout=5)
    return DraftProxyRuntime(proxy=proxy, process=process, stop_event=stop_event, ready_event=ready_event)
