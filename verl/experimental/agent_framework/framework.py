from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import replace
from uuid import uuid4

from verl.protocol import DataProto

from .assembler import TrajectoryAssembler
from .types import RewardFn, SessionRewardContext, SessionRuntime, Trajectory


class AgentFramework(ABC):
    @abstractmethod
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Process a trainer batch and return a training-ready DataProto."""
        ...


class OpenAICompatibleAgentFramework(AgentFramework):
    """Reference AgentFramework implementation for OpenAI-compatible agent loops.

    Each sample in the batch is run as an independent session: the agent
    communicates with the Gateway via standard ``/v1/chat/completions``
    requests, and the Gateway collects token-level trajectories.  After
    finalization, ``reward_fn`` scores the session's trajectories and the
    assembler packs everything into a training-ready ``DataProto``.
    """

    def __init__(
        self,
        session_runtime: SessionRuntime,
        agent_runner,
        reward_fn: RewardFn,
        *,
        assembler: TrajectoryAssembler | None = None,
        pad_token_id: int = 0,
        completion_timeout: float | None = 30.0,
        wait_for_completion_after_agent_run: bool = False,
    ):
        self.session_runtime = session_runtime
        self.agent_runner = agent_runner
        self.reward_fn = reward_fn
        self.assembler = assembler or TrajectoryAssembler(pad_token_id=pad_token_id)
        self.completion_timeout = completion_timeout
        self.wait_for_completion_after_agent_run = wait_for_completion_after_agent_run

    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"

        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError("OpenAICompatibleAgentFramework requires non_tensor_batch['raw_prompt']")

        tasks = [
            self._run_session(prompts=prompts, raw_prompt=raw_prompts[i], sample_index=i)
            for i in range(len(prompts))
        ]
        nested = await asyncio.gather(*tasks)
        all_trajectories: list[Trajectory] = [traj for session_trajs in nested for traj in session_trajs]

        return self.assembler.assemble(all_trajectories)

    async def _run_session(
        self, *, prompts: DataProto, raw_prompt, sample_index: int,
    ) -> list[Trajectory]:
        session_id = self._build_session_id(prompts=prompts, sample_index=sample_index)
        sample_fields = {k: v[sample_index] for k, v in prompts.non_tensor_batch.items()}
        session = await self.session_runtime.create_session(session_id)
        try:
            await self.agent_runner(
                raw_prompt=raw_prompt,
                session=session,
                sample_index=sample_index,
            )
            if self.wait_for_completion_after_agent_run:
                await self.session_runtime.wait_for_completion(session_id, timeout=self.completion_timeout)
            session_trajectories = await self.session_runtime.finalize_session(session_id)
        except Exception:
            await self.session_runtime.abort_session(session_id)
            raise

        # Score the session's trajectories immediately after finalization,
        # consistent with VERL's per-sample reward path.
        ctx = SessionRewardContext(trajectories=session_trajectories, sample_fields=sample_fields)
        scores = self.reward_fn(ctx)
        if inspect.isawaitable(scores):
            scores = await scores
        if len(scores) != len(session_trajectories):
            raise ValueError(
                f"reward_fn returned {len(scores)} scores for {len(session_trajectories)} trajectories"
            )
        return [
            replace(traj, reward_score=float(score))
            for traj, score in zip(session_trajectories, scores, strict=True)
        ]

    def _build_session_id(self, prompts: DataProto, sample_index: int) -> str:
        uid_batch = prompts.non_tensor_batch.get("uid")
        if uid_batch is not None:
            return str(uid_batch[sample_index])
        return f"session-{sample_index}-{uuid4().hex}"
