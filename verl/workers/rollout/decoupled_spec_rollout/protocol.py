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

# 一条 DraftResult 处于的状态
class DraftStatus(str, Enum):
    PENDING = "pending" # Draft 正在生成/还未生成完毕
    READY = "ready" # 已生成完毕
    FAILED = "failed" # 错误

# IPC 通道上传输的消息种类，用于标识 DraftProxyMessage 类
class DraftProxyMessageType(str, Enum):
    DRAFT_REQUEST = "draft_request"
    DRAFT_RESULT = "draft_result"
    VERIFY_RESULT = "verify_result"
    SHUTDOWN = "shutdown"
    ERROR = "error"


_SCHEDULER_TO_PROXY_IPC_ENV = "VERL_DRAFT_PROXY_SCHEDULER_TO_PROXY_IPC"
_PROXY_TO_SCHEDULER_IPC_ENV = "VERL_DRAFT_PROXY_PROXY_TO_SCHEDULER_IPC"
_DP_IPC_CONFIG_ENV = "VERL_DRAFT_PROXY_DP_IPC_CONFIG"
_VERIFY_REPLICA_RANK_ENV = "VERL_DRAFT_PROXY_VERIFY_REPLICA_RANK"
_NUM_SPECULATIVE_STEPS_ENV = "VERL_DRAFT_PROXY_NUM_SPECULATIVE_STEPS"


@dataclass
class DraftMetrics:
    queued_at: Optional[float] = None
    submitted_at: Optional[float] = None
    completed_at: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftRoute:
    request_id: str
    draft_index: int


@dataclass
class DraftRequest:
    request_id: str
    session_id: Optional[str]
    verify_replica_rank: int
    scheduler_dp_rank: int = 0
    draft_index: Optional[int] = None
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_token_ids: list[int] = field(default_factory=list)
    target_position: int = 0
    num_speculative_steps: int = 0
    request_kind: DraftRequestKind = DraftRequestKind.DECODE
    sampling_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

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
    scheduler_dp_rank: int = 0
    accepted_token_ids: list[int] = field(default_factory=list)
    rollback_to: Optional[int] = None
    finished: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


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
) -> dict[str, str]:
    runtime_env = ipc_config.export_env()
    runtime_env[_VERIFY_REPLICA_RANK_ENV] = str(verify_replica_rank)
    runtime_env[_NUM_SPECULATIVE_STEPS_ENV] = str(num_speculative_steps)
    return runtime_env


def get_verify_replica_rank_from_env(default: int = -1) -> int:
    return int(os.getenv(_VERIFY_REPLICA_RANK_ENV, str(default)))


def get_num_speculative_steps_from_env(default: int = 0) -> int:
    return int(os.getenv(_NUM_SPECULATIVE_STEPS_ENV, str(default)))
