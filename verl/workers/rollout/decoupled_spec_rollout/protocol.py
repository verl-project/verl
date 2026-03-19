from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DraftRequestKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class DraftStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DraftProxyMessageType(str, Enum):
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESULT = "draft_result"
    VERIFY_RESULT = "verify_result"
    SHUTDOWN = "shutdown"
    ERROR = "error"


_SCHEDULER_TO_PROXY_IPC_ENV = "VERL_DRAFT_PROXY_SCHEDULER_TO_PROXY_IPC"
_PROXY_TO_SCHEDULER_IPC_ENV = "VERL_DRAFT_PROXY_PROXY_TO_SCHEDULER_IPC"
_VERIFY_REPLICA_RANK_ENV = "VERL_DRAFT_PROXY_VERIFY_REPLICA_RANK"
_NUM_SPECULATIVE_STEPS_ENV = "VERL_DRAFT_PROXY_NUM_SPECULATIVE_STEPS"
_DRAFT_ENDPOINTS_ENV = "VERL_DRAFT_PROXY_DRAFT_ENDPOINTS"


@dataclass(frozen=True)
class SessionKey:
    request_id: str
    session_id: Optional[str] = None

    @property
    def routing_key(self) -> str:
        return self.session_id or self.request_id


@dataclass
class DraftMetrics:
    queued_at: Optional[float] = None
    submitted_at: Optional[float] = None
    completed_at: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftServerEndpoint:
    replica_rank: int
    server_address: str
    actor_handle: Any = None
    node_rank: int = 0
    actor_name: Optional[str] = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "replica_rank": self.replica_rank,
            "server_address": self.server_address,
            "node_rank": self.node_rank,
            "actor_name": self.actor_name,
        }

    @staticmethod
    def from_metadata(metadata: dict[str, Any]) -> "DraftServerEndpoint":
        return DraftServerEndpoint(
            replica_rank=metadata["replica_rank"],
            server_address=metadata["server_address"],
            node_rank=metadata.get("node_rank", 0),
            actor_name=metadata.get("actor_name"),
        )


@dataclass
class DraftRoute:
    session_key: SessionKey
    draft_replica_rank: int
    draft_server_address: str


@dataclass
class DraftRequest:
    request_id: str
    session_id: Optional[str]
    verify_replica_rank: int
    draft_replica_rank: Optional[int] = None
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_token_ids: list[int] = field(default_factory=list)
    target_position: int = 0
    num_speculative_steps: int = 0
    request_kind: DraftRequestKind = DraftRequestKind.DECODE
    sampling_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_key(self) -> SessionKey:
        return SessionKey(request_id=self.request_id, session_id=self.session_id)

    @property
    def full_token_ids(self) -> list[int]:
        return list(self.prompt_token_ids) + list(self.committed_token_ids)


@dataclass
class DraftResult:
    request_id: str
    session_id: Optional[str]
    draft_token_ids: list[int] = field(default_factory=list)
    accepted_prefix_len: int = 0
    finished: bool = False
    status: DraftStatus = DraftStatus.PENDING
    metrics: DraftMetrics = field(default_factory=DraftMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyResult:
    request_id: str
    session_id: Optional[str]
    accepted_token_ids: list[int] = field(default_factory=list)
    rollback_to: Optional[int] = None
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DraftProxyIpcConfig:
    scheduler_to_proxy_ipc_name: str
    proxy_to_scheduler_ipc_name: str

    @staticmethod
    def init_new() -> "DraftProxyIpcConfig":
        return DraftProxyIpcConfig(
            scheduler_to_proxy_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            proxy_to_scheduler_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
        )

    @staticmethod
    def from_env() -> Optional["DraftProxyIpcConfig"]:
        scheduler_to_proxy_ipc_name = os.getenv(_SCHEDULER_TO_PROXY_IPC_ENV)
        proxy_to_scheduler_ipc_name = os.getenv(_PROXY_TO_SCHEDULER_IPC_ENV)
        if not scheduler_to_proxy_ipc_name or not proxy_to_scheduler_ipc_name:
            return None
        return DraftProxyIpcConfig(
            scheduler_to_proxy_ipc_name=scheduler_to_proxy_ipc_name,
            proxy_to_scheduler_ipc_name=proxy_to_scheduler_ipc_name,
        )

    def export_env(self) -> dict[str, str]:
        return {
            _SCHEDULER_TO_PROXY_IPC_ENV: self.scheduler_to_proxy_ipc_name,
            _PROXY_TO_SCHEDULER_IPC_ENV: self.proxy_to_scheduler_ipc_name,
        }


@dataclass
class DraftProxyMessage:
    message_type: DraftProxyMessageType
    request: Optional[DraftRequest] = None
    result: Optional[DraftResult] = None
    verify_result: Optional[VerifyResult] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_draft_request(request: DraftRequest) -> "DraftProxyMessage":
        return DraftProxyMessage(message_type=DraftProxyMessageType.DRAFT_REQUEST, request=request)

    @staticmethod
    def from_draft_result(result: DraftResult) -> "DraftProxyMessage":
        return DraftProxyMessage(message_type=DraftProxyMessageType.DRAFT_RESULT, result=result)

    @staticmethod
    def from_verify_result(verify_result: VerifyResult) -> "DraftProxyMessage":
        return DraftProxyMessage(message_type=DraftProxyMessageType.VERIFY_RESULT, verify_result=verify_result)

    @staticmethod
    def shutdown() -> "DraftProxyMessage":
        return DraftProxyMessage(message_type=DraftProxyMessageType.SHUTDOWN)

    @staticmethod
    def error_message(error: str, *, request: Optional[DraftRequest] = None) -> "DraftProxyMessage":
        return DraftProxyMessage(message_type=DraftProxyMessageType.ERROR, request=request, error=error)


def get_draft_proxy_runtime_env(
    *,
    ipc_config: DraftProxyIpcConfig,
    verify_replica_rank: int,
    num_speculative_steps: int,
    draft_endpoints: Optional[list[DraftServerEndpoint]] = None,
) -> dict[str, str]:
    runtime_env = ipc_config.export_env()
    runtime_env[_VERIFY_REPLICA_RANK_ENV] = str(verify_replica_rank)
    runtime_env[_NUM_SPECULATIVE_STEPS_ENV] = str(num_speculative_steps)
    if draft_endpoints is not None:
        runtime_env[_DRAFT_ENDPOINTS_ENV] = json.dumps([endpoint.to_metadata() for endpoint in draft_endpoints])
    return runtime_env


def get_verify_replica_rank_from_env(default: int = -1) -> int:
    return int(os.getenv(_VERIFY_REPLICA_RANK_ENV, str(default)))


def get_num_speculative_steps_from_env(default: int = 0) -> int:
    return int(os.getenv(_NUM_SPECULATIVE_STEPS_ENV, str(default)))


def get_draft_endpoints_from_env() -> list[DraftServerEndpoint]:
    raw_value = os.getenv(_DRAFT_ENDPOINTS_ENV)
    if not raw_value:
        return []
    return [DraftServerEndpoint.from_metadata(item) for item in json.loads(raw_value)]
