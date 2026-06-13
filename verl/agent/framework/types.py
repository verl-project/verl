from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch


@dataclass
class SessionHandle:
    session_id: str
    base_url: str | None = None


@dataclass
class Trajectory:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None = None
    reward_info: dict[str, Any] = field(default_factory=dict)
    reward_score: float | None = None
    num_turns: int = 0
    routed_experts: torch.Tensor | np.ndarray | None = None
    multi_modal_data: dict[str, Any] | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


class SessionRuntime(Protocol):
    """Protocol for gateway-backed session lifecycle.

    Used by OpenAICompatibleAgentFramework to decouple the framework from the
    concrete AsyncLLMServerManager / GatewayManager implementation, making it
    testable without a Ray cluster.
    """

    async def create_session(self, session_id: str, **kwargs) -> SessionHandle: ...
    async def complete_session(self, session_id: str) -> None: ...
    async def finalize_session(self, session_id: str) -> list[Trajectory]: ...
    async def abort_session(self, session_id: str) -> None: ...
    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None: ...
