from __future__ import annotations

from collections.abc import Awaitable, Callable
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
    uid: str
    session_id: str
    trajectory_id: int
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


@dataclass
class SessionRewardContext:
    """Context passed to ``reward_fn`` after a session is finalized.

    A single session may produce multiple trajectories (e.g. when the agent
    switches conversation context mid-session). ``reward_fn`` receives all of
    them together so the implementor can choose the session-to-trajectory
    scoring policy, but it must return one score per trajectory.

    ``sample_fields`` carries per-sample dataset fields (``data_source``,
    ``reward_model.ground_truth``, ``extra_info``, ...) — the same dict that
    ``AgentLoopWorker._compute_score`` forwards as ``kwargs`` to the reward
    worker.
    """

    trajectories: list[Trajectory]
    sample_fields: dict[str, Any] = field(default_factory=dict)

RewardFn = Callable[[SessionRewardContext], Awaitable[list[float]] | list[float]]


class SessionRuntime(Protocol):
    """Protocol for gateway-backed session lifecycle.

    Used by OpenAICompatibleAgentFramework to decouple the framework from the
    concrete AsyncLLMServerManager / GatewayManager implementation, making it
    testable without a Ray cluster.
    """

    async def create_session(self, session_id: str, **kwargs) -> SessionHandle: ...
    async def finalize_session(self, session_id: str) -> list[Trajectory]: ...
    async def abort_session(self, session_id: str) -> None: ...
    async def wait_for_completion(self, session_id: str, timeout: float | None = None) -> None: ...
