from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .protocol import (
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftRequest,
    DraftRequestKind,
    DraftResult,
    DraftRoute,
    SessionKey,
    VerifyResult,
)


@dataclass
class DraftProxy:
    verify_replica_rank: int
    num_speculative_steps: int
    draft_actor_handles: list[Any] = field(default_factory=list)
    session_routes: dict[str, DraftRoute] = field(default_factory=dict)
    inflight_per_index: list[int] = field(default_factory=list)
    pending_requests: dict[str, DraftRequest] = field(default_factory=dict)
    pending_results: dict[str, DraftResult] = field(default_factory=dict)

    def __post_init__(self):
        self.register_draft_handles(self.draft_actor_handles)

    def register_draft_handles(self, handles: list[Any]) -> None:
        self.draft_actor_handles = list(handles)
        self.inflight_per_index = [0] * len(self.draft_actor_handles)

    def acquire_route(self, session_key: SessionKey) -> DraftRoute:
        routing_key = session_key.routing_key
        route = self.session_routes.get(routing_key)
        if route is not None:
            return route
        if not self.draft_actor_handles:
            raise ValueError("DraftProxy has no registered draft actor handles")

        best_idx = min(
            range(len(self.draft_actor_handles)),
            key=lambda i: (self.inflight_per_index[i], i),
        )
        route = DraftRoute(session_key=session_key, draft_index=best_idx)
        self.session_routes[routing_key] = route
        return route

    def submit_prefill(self, request: DraftRequest) -> DraftRoute:
        route = self.acquire_route(request.session_key)
        request.draft_index = route.draft_index
        self.pending_requests[request.request_id] = request
        self.inflight_per_index[route.draft_index] += 1
        return route

    def submit_decode(self, request: DraftRequest) -> DraftRoute:
        return self.submit_prefill(request)

    def await_result(self, request_id: str) -> Optional[DraftResult]:
        return self.pending_results.get(request_id)

    def notify_verify_result(self, result: VerifyResult) -> None:
        request = self.pending_requests.pop(result.request_id, None)
        if request is None:
            if result.finished:
                self.pending_results.pop(result.request_id, None)
                self.release_session(SessionKey(request_id=result.request_id, session_id=result.session_id))
            return
        if request.draft_index is not None:
            self.inflight_per_index[request.draft_index] = max(
                0, self.inflight_per_index[request.draft_index] - 1
            )
        if result.finished:
            self.pending_results.pop(result.request_id, None)
            self.release_session(SessionKey(request_id=result.request_id, session_id=result.session_id))

    def release_session(self, session_key: SessionKey) -> None:
        self.session_routes.pop(session_key.routing_key, None)

    def submit_request(self, request: DraftRequest) -> DraftRoute:
        if request.request_kind == DraftRequestKind.PREFILL:
            return self.submit_prefill(request)
        return self.submit_decode(request)

    def complete_request(self, request_id: str, result: DraftResult) -> DraftResult:
        request = self.pending_requests.pop(request_id, None)
        self.pending_results[request_id] = result
        draft_index = result.metadata.get("draft_index")
        if draft_index is None and request is not None:
            draft_index = request.draft_index
        if draft_index is not None:
            idx = int(draft_index)
            if 0 <= idx < len(self.inflight_per_index):
                self.inflight_per_index[idx] = max(0, self.inflight_per_index[idx] - 1)
        return result

    def handle_message(self, message: DraftProxyMessage) -> Optional[DraftProxyMessage]:
        if message.message_type == DraftProxyMessageType.VERIFY_RESULT and message.verify_result is not None:
            self.notify_verify_result(message.verify_result)
            return None
        if message.message_type == DraftProxyMessageType.SHUTDOWN:
            return DraftProxyMessage.shutdown()
        return None
