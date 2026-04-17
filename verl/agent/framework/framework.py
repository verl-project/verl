from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import replace
from uuid import uuid4

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl.utils import tensordict_utils as tu

from .assembler import TrajectoryAssembler
from .types import RewardFn, SessionRewardContext, SessionRuntime, Trajectory


class AgentFramework(ABC):
    @abstractmethod
    async def generate_sequences(self, prompts: TensorDict) -> TensorDict:
        """Process a trainer batch and return a training-ready TensorDict."""
        ...


def _broadcast_sample_non_tensor_fields(sample_fields: dict[str, object], repeat_count: int) -> dict[str, list[object]]:
    broadcasted: dict[str, list[object]] = {}
    for key, value in sample_fields.items():
        broadcasted[key] = [value] * repeat_count
    return broadcasted


class OpenAICompatibleAgentFramework(AgentFramework):
    """Reference AgentFramework implementation for OpenAI-compatible agent loops.

    Each sample in the batch is run as an independent session: the agent
    communicates with the Gateway via standard ``/v1/chat/completions``
    requests, and the Gateway collects token-level trajectories.  After
    finalization, ``reward_fn`` scores the session's trajectories and the
    assembler packs everything into a training-ready ``TensorDict``.
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

    async def generate_sequences(self, prompts: TensorDict) -> TensorDict:
        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"

        raw_prompts = tu.get(prompts, "raw_prompt")
        if raw_prompts is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['raw_prompt']")

        tasks = [
            self._run_session(prompts=prompts, raw_prompt=raw_prompts[i], sample_index=i)
            for i in range(len(prompts))
        ]
        nested = await asyncio.gather(*tasks)
        all_trajectories: list[Trajectory] = []
        expanded_non_tensor_batch: dict[str, list[object]] = {}

        for session_trajectories, sample_fields in nested:
            all_trajectories.extend(session_trajectories)
            if not session_trajectories:
                continue
            for key, values in _broadcast_sample_non_tensor_fields(sample_fields, len(session_trajectories)).items():
                expanded_non_tensor_batch.setdefault(key, []).extend(values)

        assembled = self.assembler.assemble(all_trajectories)
        result = assembled.copy()
        for key, values in expanded_non_tensor_batch.items():
            tu.assign_non_tensor(result, **{key: values})
        return result

    async def _run_session(
        self, *, prompts: TensorDict, raw_prompt, sample_index: int,
    ) -> tuple[list[Trajectory], dict[str, object]]:
        session_id = self._build_session_id(prompts=prompts, sample_index=sample_index)
        sample_fields = {}
        for key, value in prompts.items():
            if isinstance(value, torch.Tensor):
                sample_fields[key] = value[sample_index]
            elif isinstance(value, NonTensorStack):
                sample_fields[key] = tu.get(prompts, key)[sample_index]
            else:
                assert isinstance(value, NonTensorData)
                sample_fields[key] = value.data
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
        normalized_scores: list[float] = []
        for trajectory, score in zip(session_trajectories, scores, strict=True):
            if score is None:
                raise ValueError(
                    f"reward_fn must return a score for every trajectory; got None for trajectory {trajectory.uid}"
                )
            normalized_scores.append(float(score))
        return (
            [
                replace(traj, reward_score=score)
                for traj, score in zip(session_trajectories, normalized_scores, strict=True)
            ],
            sample_fields,
        )

    def _build_session_id(self, prompts: TensorDict, sample_index: int) -> str:
        return f"session-{sample_index}-{uuid4().hex}"
