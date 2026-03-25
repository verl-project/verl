from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .protocol import (
    DraftLookupKey,
    DraftRequest,
    DraftResult,
    DraftRoute,
    RequestTerminateMessage,
)


@dataclass
class InflightDraft:
    draft_index: int
    object_ref: Any


@dataclass
class DraftProxy:
    verify_replica_rank: int
    num_speculative_steps: int
    draft_actor_handles: list[Any] = field(default_factory=list)
    request_routes: dict[str, DraftRoute] = field(default_factory=dict) # rid -> draft index
    inflight_requests: dict[DraftLookupKey, InflightDraft] = field(default_factory=dict) # (request_id, round_id) -> (draft_index, object_ref)
    inflight_per_index: list[int] = field(default_factory=list) # 记录每个 drafter 正在生成的 DraftRequest 个数(同一个 request_id 可能同时有多个 in-flight 的 DraftRequest)
    ready_results: dict[DraftLookupKey, DraftResult] = field(default_factory=dict) # (request_id, round_id) -> DraftResult ，存放已经poll过来但还未被Scheduler poll走的DraftResult

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

    def _release_inflight(self, key: DraftLookupKey) -> None:
        inflight = self.inflight_requests.pop(key, None)
        if inflight is None:
            return
        draft_index = int(inflight.draft_index)
        if 0 <= draft_index < len(self.inflight_per_index):
            self.inflight_per_index[draft_index] = max(0, self.inflight_per_index[draft_index] - 1)

    def submit_request(self, request: DraftRequest, object_ref: Any) -> DraftRoute:
        route = self.acquire_route(request.request_id)
        self.inflight_requests[request.key] = InflightDraft(
            draft_index=route.draft_index,
            object_ref=object_ref,
        )
        self.inflight_per_index[route.draft_index] += 1
        return route

    def peek_ready_results(
        self,
        keys: list[DraftLookupKey],
    ) -> tuple[list[DraftResult], list[DraftLookupKey]]:
        ready_results = []
        missing_keys = []
        for key in keys:
            result = self.ready_results.get(key)
            if result is None:
                missing_keys.append(key)
            else:
                ready_results.append(result)
        return ready_results, missing_keys

    def pop_ready_results(self, keys: list[DraftLookupKey]) -> list[DraftResult]:
        popped_results = []
        for key in keys:
            result = self.ready_results.pop(key, None)
            if result is not None:
                popped_results.append(result)
        return popped_results

    def release_request(self, request_id: str) -> None:
        self.request_routes.pop(request_id, None)
        for key in list(self.ready_results):
            if key.request_id == request_id:
                self.ready_results.pop(key, None)

    def terminate_request(self, message: RequestTerminateMessage) -> None:
        upper_bound = message.draft_round_id_upper_bound
        for key in list(self.inflight_requests):
            if key.request_id != message.request_id:
                continue
            if upper_bound is not None and key.draft_round_id > upper_bound:
                continue
            self._release_inflight(key)

        for key in list(self.ready_results):
            if key.request_id != message.request_id:
                continue
            if upper_bound is not None and key.draft_round_id > upper_bound:
                continue
            self.ready_results.pop(key, None)

        has_newer_inflight = any(
            key.request_id == message.request_id and key.draft_round_id > upper_bound
            for key in self.inflight_requests
        ) if upper_bound is not None else False
        has_newer_ready = any(
            key.request_id == message.request_id and key.draft_round_id > upper_bound
            for key in self.ready_results
        ) if upper_bound is not None else False
        if upper_bound is None or (not has_newer_inflight and not has_newer_ready):
            self.release_request(message.request_id)

    def complete_request(self, key: DraftLookupKey, result: DraftResult) -> Optional[DraftResult]:
        self._release_inflight(key)
        self.ready_results[key] = result
        return result
