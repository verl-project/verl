from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DraftProxyMessageType(str, Enum):
    SUBMIT_DRAFT = "submit_draft"
    POLL_DRAFT_RESULTS = "poll_draft_results"
    POLL_RESPONSE = "poll_response"
    REQUEST_TERMINATE = "request_terminate"


class DraftPollWaitMode(str, Enum):
    ALL = "all"


_SCHEDULER_TO_PROXY_IPC_ENV = "VERL_DRAFT_PROXY_SCHEDULER_TO_PROXY_IPC"
_PROXY_TO_SCHEDULER_IPC_ENV = "VERL_DRAFT_PROXY_PROXY_TO_SCHEDULER_IPC"
_DP_IPC_CONFIG_ENV = "VERL_DRAFT_PROXY_DP_IPC_CONFIG"
_VERIFY_REPLICA_RANK_ENV = "VERL_DRAFT_PROXY_VERIFY_REPLICA_RANK"
_NUM_SPECULATIVE_STEPS_ENV = "VERL_DRAFT_PROXY_NUM_SPECULATIVE_STEPS"


@dataclass(frozen=True)
class DraftLookupKey:
    # 唯一标识：某一请求某一轮次的draft
    request_id: str
    draft_round_id: int


@dataclass
class DraftRoute:
    request_id: str
    draft_index: int


@dataclass
class DraftRequest:
    request_id: str
    verify_replica_rank: int
    draft_round_id: int = 0 # 该请求 draft 的轮次
    scheduler_dp_rank: int = 0
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_token_ids: list[int] = field(default_factory=list)
    num_speculative_steps: int = 0
    sampling_params: dict[str, Any] = field(default_factory=dict)

    @property
    def full_token_ids(self) -> list[int]:
        return list(self.prompt_token_ids) + list(self.committed_token_ids)

    @property
    def key(self) -> DraftLookupKey:
        return DraftLookupKey(request_id=self.request_id, draft_round_id=self.draft_round_id)


@dataclass
class DraftResult:
    request_id: str
    draft_round_id: int = 0
    draft_token_ids: list[int] = field(default_factory=list)

    @property
    def key(self) -> DraftLookupKey:
        return DraftLookupKey(request_id=self.request_id, draft_round_id=self.draft_round_id)


class RequestTerminateReason(str, Enum):
    FINISHED = "finished"
    ABORT = "abort"


@dataclass
class RequestTerminateMessage:
    request_id: str
    reason: RequestTerminateReason
    draft_round_id_upper_bound: Optional[int] = None


@dataclass
class PollDraftResultsRequest:
    poll_id: str
    scheduler_dp_rank: int = 0
    keys: list[DraftLookupKey] = field(default_factory=list)
    timeout_ms: Optional[int] = None
    wait_mode: DraftPollWaitMode = DraftPollWaitMode.ALL


@dataclass
class PollDraftResultsResponse:
    poll_id: str
    results: list[DraftResult] = field(default_factory=list)
    missing_keys: list[DraftLookupKey] = field(default_factory=list)
    timed_out: bool = False


@dataclass(frozen=True)
class DraftProxyDpIpcEndpoints:
    scheduler_to_proxy_ipc_name: str
    proxy_to_scheduler_ipc_name: str

    @staticmethod
    def init_new() -> "DraftProxyDpIpcEndpoints":
        return DraftProxyDpIpcEndpoints(
            scheduler_to_proxy_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            proxy_to_scheduler_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
        )


@dataclass(frozen=True)
class DraftProxyIpcConfig:
    dp_ipc_endpoints: dict[int, DraftProxyDpIpcEndpoints]

    @staticmethod
    def init_new(dp_size: int = 1) -> "DraftProxyIpcConfig":
        normalized_dp_size = max(1, int(dp_size))
        return DraftProxyIpcConfig(
            dp_ipc_endpoints={dp_rank: DraftProxyDpIpcEndpoints.init_new() for dp_rank in range(normalized_dp_size)}
        )

    @staticmethod
    def from_env() -> Optional["DraftProxyIpcConfig"]:
        encoded_dp_config = os.getenv(_DP_IPC_CONFIG_ENV)
        if encoded_dp_config:
            raw_config = json.loads(encoded_dp_config)
            return DraftProxyIpcConfig(
                dp_ipc_endpoints={
                    int(dp_rank): DraftProxyDpIpcEndpoints(
                        scheduler_to_proxy_ipc_name=endpoints["scheduler_to_proxy_ipc_name"],
                        proxy_to_scheduler_ipc_name=endpoints["proxy_to_scheduler_ipc_name"],
                    )
                    for dp_rank, endpoints in raw_config.items()
                }
            )

        scheduler_to_proxy_ipc_name = os.getenv(_SCHEDULER_TO_PROXY_IPC_ENV)
        proxy_to_scheduler_ipc_name = os.getenv(_PROXY_TO_SCHEDULER_IPC_ENV)
        if not scheduler_to_proxy_ipc_name or not proxy_to_scheduler_ipc_name:
            return None
        return DraftProxyIpcConfig(
            dp_ipc_endpoints={
                0: DraftProxyDpIpcEndpoints(
                    scheduler_to_proxy_ipc_name=scheduler_to_proxy_ipc_name,
                    proxy_to_scheduler_ipc_name=proxy_to_scheduler_ipc_name,
                )
            }
        )

    def export_env(self) -> dict[str, str]:
        normalized_dp_config = {
            str(dp_rank): {
                "scheduler_to_proxy_ipc_name": endpoints.scheduler_to_proxy_ipc_name,
                "proxy_to_scheduler_ipc_name": endpoints.proxy_to_scheduler_ipc_name,
            }
            for dp_rank, endpoints in sorted(self.dp_ipc_endpoints.items())
        }
        rank0_endpoints = self.get_endpoints(0)
        return {
            _DP_IPC_CONFIG_ENV: json.dumps(normalized_dp_config, sort_keys=True),
            _SCHEDULER_TO_PROXY_IPC_ENV: rank0_endpoints.scheduler_to_proxy_ipc_name,
            _PROXY_TO_SCHEDULER_IPC_ENV: rank0_endpoints.proxy_to_scheduler_ipc_name,
        }

    def get_endpoints(self, dp_rank: Optional[int]) -> DraftProxyDpIpcEndpoints:
        normalized_dp_rank = 0 if dp_rank is None else int(dp_rank)
        endpoints = self.dp_ipc_endpoints.get(normalized_dp_rank)
        if endpoints is not None:
            return endpoints
        if normalized_dp_rank != 0 and 0 in self.dp_ipc_endpoints:
            return self.dp_ipc_endpoints[0]
        raise KeyError(f"DraftProxy IPC config missing endpoints for dp_rank={normalized_dp_rank}")


@dataclass
class DraftProxyMessage:
    message_type: DraftProxyMessageType
    request: Optional[DraftRequest] = None
    poll_request: Optional[PollDraftResultsRequest] = None
    poll_response: Optional[PollDraftResultsResponse] = None
    terminate: Optional[RequestTerminateMessage] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_submit_draft(request: DraftRequest) -> "DraftProxyMessage":
        return DraftProxyMessage(message_type=DraftProxyMessageType.SUBMIT_DRAFT, request=request)

    @staticmethod
    def from_poll_request(poll_request: PollDraftResultsRequest) -> "DraftProxyMessage":
        return DraftProxyMessage(
            message_type=DraftProxyMessageType.POLL_DRAFT_RESULTS,
            poll_request=poll_request,
        )

    @staticmethod
    def from_poll_response(poll_response: PollDraftResultsResponse) -> "DraftProxyMessage":
        return DraftProxyMessage(
            message_type=DraftProxyMessageType.POLL_RESPONSE,
            poll_response=poll_response,
        )

    @staticmethod
    def from_request_terminate(terminate: RequestTerminateMessage) -> "DraftProxyMessage":
        return DraftProxyMessage(message_type=DraftProxyMessageType.REQUEST_TERMINATE, terminate=terminate)


def get_draft_proxy_runtime_env(
    *,
    ipc_config: DraftProxyIpcConfig,
    verify_replica_rank: int,
    num_speculative_steps: int,
) -> dict[str, str]:
    runtime_env = ipc_config.export_env()
    runtime_env[_VERIFY_REPLICA_RANK_ENV] = str(verify_replica_rank)
    runtime_env[_NUM_SPECULATIVE_STEPS_ENV] = str(num_speculative_steps)
    return runtime_env


def get_verify_replica_rank_from_env(default: int = -1) -> int:
    return int(os.getenv(_VERIFY_REPLICA_RANK_ENV, str(default)))


def get_num_speculative_steps_from_env(default: int = 0) -> int:
    return int(os.getenv(_NUM_SPECULATIVE_STEPS_ENV, str(default)))
