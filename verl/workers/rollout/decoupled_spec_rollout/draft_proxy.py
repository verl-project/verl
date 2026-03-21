from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .protocol import (
    DraftProxyMessage,
    DraftProxyMessageType,
    DraftRequest,
    DraftResult,
    DraftRoute,
    RequestTerminateMessage,
)


@dataclass
class DraftProxy:
    verify_replica_rank: int
    num_speculative_steps: int
    draft_actor_handles: list[Any] = field(default_factory=list)
    request_routes: dict[str, DraftRoute] = field(default_factory=dict)
    inflight_requests: dict[str, int] = field(default_factory=dict)
    inflight_per_index: list[int] = field(default_factory=list)
    pending_results: dict[str, DraftResult] = field(default_factory=dict)

    def __post_init__(self):
        self.register_draft_handles(self.draft_actor_handles)

    def register_draft_handles(self, handles: list[Any]) -> None:
        self.draft_actor_handles = list(handles)
        self.inflight_per_index = [0] * len(self.draft_actor_handles)

    def acquire_route(self, request_id: str) -> DraftRoute:
        route = self.request_routes.get(request_id)
        if route is not None:
            return route
        if not self.draft_actor_handles:
            raise ValueError("DraftProxy has no registered draft actor handles")

        best_idx = min(
            range(len(self.draft_actor_handles)),
            key=lambda i: (self.inflight_per_index[i], i),
        )
        route = DraftRoute(request_id=request_id, draft_index=best_idx)
        self.request_routes[request_id] = route
        return route

    def submit_request(self, request: DraftRequest) -> DraftRoute:
        route = self.acquire_route(request.request_id)
        self.inflight_requests[request.request_id] = route.draft_index
        self.inflight_per_index[route.draft_index] += 1
        return route

    def await_result(self, request_id: str) -> Optional[DraftResult]:
        return self.pending_results.get(request_id)

    def notify_verify_request(self, request: DraftRequest) -> None:
        self.pending_results.pop(request.request_id, None)

    def release_request(self, request_id: str) -> None:
        self.request_routes.pop(request_id, None)
        self.pending_results.pop(request_id, None)

    def terminate_request(self, message: RequestTerminateMessage) -> None:
        draft_index = self.inflight_requests.pop(message.request_id, None)
        if draft_index is not None and 0 <= draft_index < len(self.inflight_per_index):
            self.inflight_per_index[draft_index] = max(0, self.inflight_per_index[draft_index] - 1)
        self.release_request(message.request_id)

    def complete_request(self, request_id: str, result: DraftResult) -> DraftResult:
        self.pending_results[request_id] = result
        draft_index = self.inflight_requests.pop(request_id, None)
        if draft_index is not None:
            idx = int(draft_index)
            if 0 <= idx < len(self.inflight_per_index):
                self.inflight_per_index[idx] = max(0, self.inflight_per_index[idx] - 1)
        return result

    def handle_message(self, message: DraftProxyMessage) -> Optional[DraftProxyMessage]:
        if message.message_type == DraftProxyMessageType.VERIFY_RESULT and message.request is not None:
            self.notify_verify_request(message.request)
            return None
        if message.message_type == DraftProxyMessageType.REQUEST_TERMINATE and message.terminate is not None:
            self.terminate_request(message.terminate)
            return None
        if message.message_type == DraftProxyMessageType.SHUTDOWN:
            return DraftProxyMessage.shutdown()
        return None
