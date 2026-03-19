from __future__ import annotations

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

    def to_metadata(self) -> dict[str, Any]:
        return {
            "replica_rank": self.replica_rank,
            "server_address": self.server_address,
            "node_rank": self.node_rank,
        }


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
