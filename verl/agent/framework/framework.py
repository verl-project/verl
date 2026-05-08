from __future__ import annotations

import asyncio
import logging
import inspect
from abc import ABC, abstractmethod
from dataclasses import replace
from uuid import uuid4

import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl.utils.transferqueue_utils import tq
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask

from .assembler import TrajectoryAssembler
from .multi_modal_postprocess import compute_multi_modal_inputs, compute_position_ids
from .types import RewardFn, SessionRewardContext, SessionRuntime, Trajectory

logger = logging.getLogger(__name__)


class AgentFramework(ABC):
    @abstractmethod
    async def generate_sequences(
        self,
        prompts: TensorDict,
        *,
        global_steps: int,
        partition_id: str,
        num_sessions: int = 1,
    ) -> dict:
        """Run agent sessions and write finalized trajectories to TransferQueue."""
        ...


def _to_long_tensor(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.long)


def _to_float_tensor(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.float32)


def _short_failure_reason(error: BaseException) -> str:
    message = str(error)
    if not message:
        message = error.__class__.__name__
    return message[:512]


_TQ_NESTED_SEQUENCE_FIELDS = {
    "prompts",
    "responses",
    "response_mask",
    "loss_mask",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "rm_scores",
    "teacher_logprobs",
    "teacher_ids",
}


def _list_of_tq_fields_to_tensordict(fields: list[dict[str, object]]) -> TensorDict:
    td = tu.list_of_dict_to_tensordict(fields)
    for key in _TQ_NESTED_SEQUENCE_FIELDS:
        if key not in fields[0]:
            continue
        values = [field[key] for field in fields]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        ragged_idx = 2 if key == "position_ids" and values[0].dim() == 2 else None
        td[key] = tu.nested_tensor_from_tensor_list(values, ragged_idx=ragged_idx)
    return td


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
        reward_fn: RewardFn | None,
        *,
        processor=None,
        assembler: TrajectoryAssembler | None = None,
        pad_token_id: int = 0,
        completion_timeout: float | None = 30.0,
        wait_for_completion_after_agent_run: bool = False,
    ):
        self.session_runtime = session_runtime
        self.agent_runner = agent_runner
        self.reward_fn = reward_fn
        self._processor = processor
        self.assembler = assembler or TrajectoryAssembler(pad_token_id=pad_token_id)
        self.completion_timeout = completion_timeout
        self.wait_for_completion_after_agent_run = wait_for_completion_after_agent_run

    async def generate_sequences(
        self,
        prompts: TensorDict,
        *,
        global_steps: int,
        partition_id: str,
        num_sessions: int = 1,
    ) -> dict:
        """Run agent sessions and write finalized trajectories to TransferQueue.

        This is the TransferQueue-oriented sibling of ``generate_sequences``.
        It preserves the same session lifecycle, but writes each finalized
        trajectory with the key/tag/field schema consumed by
        ``verl.trainer.main_ppo_sync`` instead of returning a batch.
        """
        assert len(prompts) > 0, "generate_sequences requires a non-empty batch"
        if num_sessions <= 0:
            raise ValueError(f"num_sessions must be positive, got {num_sessions}")

        raw_prompts = tu.get(prompts, "raw_prompt")
        if raw_prompts is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['raw_prompt']")

        tasks = [
            self._run_prompt_to_replay_buffer(
                prompts=prompts,
                raw_prompt=raw_prompts[sample_index],
                sample_index=sample_index,
                global_steps=global_steps,
                partition_id=partition_id,
                num_sessions=num_sessions,
            )
            for sample_index in range(len(prompts))
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        failure_reasons: list[str] = []
        stats = {
            "num_input_prompts": len(prompts),
            "num_success_sessions": 0,
            "num_failed_sessions": 0,
            "num_success_outputs": 0,
            "num_failed_uids": 0,
            "failure_reasons": failure_reasons,
        }
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                stats["num_failed_sessions"] += num_sessions
                stats["num_failed_uids"] += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue
            stats["num_success_sessions"] += outcome["num_success_sessions"]
            stats["num_failed_sessions"] += outcome["num_failed_sessions"]
            stats["num_success_outputs"] += outcome["num_success_outputs"]
            stats["num_failed_uids"] += outcome["num_failed_uids"]
            failure_reasons.extend(outcome["failure_reasons"])
        if failure_reasons:
            logger.warning(
                "generate_sequences completed with failures: num_input_prompts=%s num_success_sessions=%s "
                "num_failed_sessions=%s num_success_outputs=%s num_failed_uids=%s failure_reasons=%s",
                stats["num_input_prompts"],
                stats["num_success_sessions"],
                stats["num_failed_sessions"],
                stats["num_success_outputs"],
                stats["num_failed_uids"],
                failure_reasons[:3],
            )
        return stats

    async def _run_prompt_to_replay_buffer(
        self,
        *,
        prompts: TensorDict,
        raw_prompt,
        sample_index: int,
        global_steps: int,
        partition_id: str,
        num_sessions: int,
    ) -> dict:
        sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
        uid = sample_fields.get("uid")
        if uid is None:
            raise ValueError("OpenAICompatibleAgentFramework requires prompts['uid'] for TransferQueue output")
        uid = str(uid)

        tasks = [
            self._run_session(
                prompts=prompts,
                raw_prompt=raw_prompt,
                sample_index=sample_index,
                session_index=session_index,
                session_id=self._build_session_id_with_index(
                    prompts=prompts,
                    sample_index=sample_index,
                    session_index=session_index,
                ),
                runner_kwargs=self._runner_kwargs_for_sample(sample_fields),
            )
            for session_index in range(num_sessions)
        ]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        success_sessions = 0
        failed_sessions = 0
        success_outputs = 0
        failure_reasons: list[str] = []
        for session_index, outcome in enumerate(outcomes):
            if isinstance(outcome, Exception):
                failed_sessions += 1
                failure_reasons.append(_short_failure_reason(outcome))
                continue

            trajectories, session_sample_fields = outcome
            if not trajectories:
                failed_sessions += 1
                failure_reasons.append(f"empty trajectories for uid={uid} session_id={session_index}")
                continue

            success_sessions += 1
            await self._write_session_trajectories_to_tq(
                uid=uid,
                session_id=session_index,
                trajectories=trajectories,
                sample_fields=session_sample_fields,
                global_steps=global_steps,
                partition_id=partition_id,
            )
            success_outputs += len(trajectories)

        if success_sessions > 0:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
            failed_uids = 0
        else:
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})
            failed_uids = 1

        return {
            "num_success_sessions": success_sessions,
            "num_failed_sessions": failed_sessions,
            "num_success_outputs": success_outputs,
            "num_failed_uids": failed_uids,
            "failure_reasons": failure_reasons,
        }

    async def _run_session(
        self,
        *,
        prompts: TensorDict,
        raw_prompt,
        sample_index: int,
        session_index: int = 0,
        session_id: str | None = None,
        runner_kwargs: dict[str, object] | None = None,
    ) -> tuple[list[Trajectory], dict[str, object]]:
        session_id = session_id or self._build_session_id(prompts=prompts, sample_index=sample_index)
        sample_fields = self._extract_sample_fields(prompts=prompts, sample_index=sample_index)
        session = await self.session_runtime.create_session(session_id)
        try:
            await self.agent_runner(
                raw_prompt=raw_prompt,
                session=session,
                sample_index=sample_index,
                **(runner_kwargs or {}),
            )
            if self.wait_for_completion_after_agent_run:
                await self.session_runtime.wait_for_completion(session_id, timeout=self.completion_timeout)
            session_trajectories = await self.session_runtime.finalize_session(session_id)
        except Exception:
            await self.session_runtime.abort_session(session_id)
            raise

        # Score the session's trajectories immediately after finalization,
        # consistent with VERL's per-sample reward path.
        if self.reward_fn is None:
            return session_trajectories, sample_fields

        normalized_scores = await self._score_trajectories(session_trajectories, sample_fields)
        return (
            [
                replace(traj, reward_score=score)
                for traj, score in zip(session_trajectories, normalized_scores, strict=True)
            ],
            sample_fields,
        )

    async def _score_trajectories(
        self,
        session_trajectories: list[Trajectory],
        sample_fields: dict[str, object],
    ) -> list[float]:
        assert self.reward_fn is not None
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
        return normalized_scores

    def _extract_sample_fields(self, *, prompts: TensorDict, sample_index: int) -> dict[str, object]:
        sample_fields = {}
        for key, value in prompts.items():
            if isinstance(value, torch.Tensor):
                sample_fields[key] = value if value.ndim == 0 else value[sample_index]
            elif isinstance(value, NonTensorStack):
                sample_fields[key] = tu.get(prompts, key)[sample_index]
            else:
                assert isinstance(value, NonTensorData)
                sample_fields[key] = value.data
        return sample_fields

    def _runner_kwargs_for_sample(self, sample_fields: dict[str, object]) -> dict[str, object]:
        runner_kwargs = {}
        if "tools_kwargs" in sample_fields:
            runner_kwargs["tools_kwargs"] = sample_fields["tools_kwargs"]
        return runner_kwargs

    async def _write_session_trajectories_to_tq(
        self,
        *,
        uid: str,
        session_id: int,
        trajectories: list[Trajectory],
        sample_fields: dict[str, object],
        global_steps: int,
        partition_id: str,
    ) -> None:
        keys = []
        fields = []
        tags = []
        for index, trajectory in enumerate(trajectories):
            field, tag = self._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields=sample_fields,
                session_id=session_id,
                global_steps=global_steps,
            )
            keys.append(f"{uid}_{session_id}_{index}")
            fields.append(field)
            tags.append(tag)

        await tq.async_kv_batch_put(
            keys=keys,
            fields=_list_of_tq_fields_to_tensordict(fields),
            tags=tags,
            partition_id=partition_id,
        )

    def _trajectory_to_tq_field_and_tag(
        self,
        *,
        trajectory: Trajectory,
        sample_fields: dict[str, object],
        session_id: int,
        global_steps: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        prompts = _to_long_tensor(trajectory.prompt_ids)
        responses = _to_long_tensor(trajectory.response_ids)
        response_mask = _to_long_tensor(trajectory.response_mask)
        input_ids = torch.cat([prompts, responses], dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        multi_modal_inputs = compute_multi_modal_inputs(
            self._processor,
            input_ids.unsqueeze(0),
            trajectory.multi_modal_data,
        )
        if self._processor is None:
            position_ids = compute_position_id_with_mask(attention_mask.unsqueeze(0)).squeeze(0)
        else:
            position_ids = compute_position_ids(
                self._processor,
                input_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
                multi_modal_inputs,
            ).squeeze(0)

        field: dict[str, object] = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "loss_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_inputs": multi_modal_inputs,
        }
        if trajectory.response_logprobs is not None:
            field["rollout_log_probs"] = _to_float_tensor(trajectory.response_logprobs)
        if trajectory.routed_experts is not None:
            field["routed_experts"] = (
                torch.from_numpy(trajectory.routed_experts.copy())
                if hasattr(trajectory.routed_experts, "copy") and not isinstance(trajectory.routed_experts, torch.Tensor)
                else trajectory.routed_experts
            )
        if trajectory.reward_score is not None:
            rm_scores = torch.zeros_like(responses, dtype=torch.float32)
            if responses.numel() > 0:
                rm_scores[-1] = float(trajectory.reward_score)
            field["rm_scores"] = rm_scores

        field.update(trajectory.extra_fields)
        field.pop("multi_modal_data", None)
        for key in ("uid", "raw_prompt", "data_source", "reward_model", "extra_info", "tools_kwargs", "agent_name"):
            if key in sample_fields:
                field[key] = sample_fields[key]
        field["session_id"] = session_id
        field["global_steps"] = global_steps
        field["num_turns"] = torch.tensor(int(trajectory.num_turns), dtype=torch.long)

        prompt_len = prompts.size(0)
        response_len = responses.size(0)
        tag = {
            "global_steps": global_steps,
            "status": "success",
            "prompt_len": prompt_len,
            "response_len": response_len,
            "seq_len": prompt_len + response_len,
        }
        return field, tag

    def _build_session_id(self, prompts: TensorDict, sample_index: int) -> str:
        return f"session-{sample_index}-{uuid4().hex}"

    def _build_session_id_with_index(self, *, prompts: TensorDict, sample_index: int, session_index: int) -> str:
        try:
            return self._build_session_id(prompts=prompts, sample_index=sample_index, session_index=session_index)
        except TypeError:
            if session_index == 0:
                return self._build_session_id(prompts=prompts, sample_index=sample_index)
            return f"session-{sample_index}-{session_index}-{uuid4().hex}"
