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
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
# TODO: is this class necessary?
class TrajectoryRewardContext:
    """Per-trajectory context passed to reward_fn.

    trajectory holds the raw token sequences and any reward_info injected by the
    agent via the /complete endpoint (e.g. pass rate, format check results).
    prompt_context holds fields from the original prompts.non_tensor_batch for this
    sample (e.g. data_source, ground_truth) — whatever the dataset provides.
    """

    trajectory: Trajectory
    prompt_context: dict[str, Any] = field(default_factory=dict)


# reward_fn receives all trajectory contexts from one generate_sequences call (across all
# sessions in the batch) and returns one float score per trajectory.  Implementors decide
# whether to score each trajectory independently or apply group-level normalization (e.g.
# GRPO requires all rollouts for the same prompt to be scored together).

# TODO: check if this is consistent with VERL's reward manager and other example implementations.
RewardFn = Callable[[list[TrajectoryRewardContext]], Awaitable[list[float]] | list[float]]


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
