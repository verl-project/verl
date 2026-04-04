from __future__ import annotations

import inspect
from dataclasses import replace
from uuid import uuid4

from verl.protocol import DataProto

from .assembler import TrajectoryAssembler
from .framework import AgentFramework
from .types import RewardFn, SessionRuntime, Trajectory, TrajectoryRewardContext


class OpenAICompatibleAgentFramework(AgentFramework):
    def __init__(
        self,
        session_runtime: SessionRuntime, # e.g., AsyncLLMServerManager
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

        all_trajectories: list[Trajectory] = []
        # TODO: Trajectory already has reward ids and prompts, is reward_contexts a useful abstractions at all? 
        # I think we only need to show the necessary components in this example implementations.
        reward_contexts: list[TrajectoryRewardContext] = []

        for sample_index in range(len(prompts)):
            session_id = self._build_session_id(prompts=prompts, sample_index=sample_index)
            prompt_ctx = {k: v[sample_index] for k, v in prompts.non_tensor_batch.items()}
            session = await self.session_runtime.create_session(session_id)
            try:
                await self.agent_runner(
                    raw_prompt=raw_prompts[sample_index],
                    session=session,
                    sample_index=sample_index,
                )
                if self.wait_for_completion_after_agent_run:
                    await self.session_runtime.wait_for_completion(session_id, timeout=self.completion_timeout)
                session_trajectories = await self.session_runtime.finalize_session(session_id)
            except Exception:
                await self.session_runtime.abort_session(session_id)
                raise

            for traj in session_trajectories:
                all_trajectories.append(traj)
                reward_contexts.append(TrajectoryRewardContext(
                    trajectory=traj,
                    prompt_context=prompt_ctx,
                ))

        # Compute reward scores. reward_fn is the single authoritative source —
        # it receives all trajectory contexts from this batch and returns one float
        # per trajectory. The implementor decides whether to score independently
        # or apply group-level normalization (e.g. GRPO).
        scores = self.reward_fn(reward_contexts)
        if inspect.isawaitable(scores):
            scores = await scores
        all_trajectories = [
            replace(traj, reward_score=float(score))
            for traj, score in zip(all_trajectories, scores, strict=True)
        ]

        return self.assembler.assemble(all_trajectories)

    def _build_session_id(self, prompts: DataProto, sample_index: int) -> str:
        uid_batch = prompts.non_tensor_batch.get("uid")
        if uid_batch is not None:
            return str(uid_batch[sample_index])
        return f"session-{sample_index}-{uuid4().hex}"
