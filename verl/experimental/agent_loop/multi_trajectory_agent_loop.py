# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract base class for agent loops that produce multiple independent trajectories per rollout.

A *multi-trajectory* agent loop is one in which a single rollout (one call to
``run()``) naturally produces several independent ``(prompt, response)`` pairs
that should each be treated as an independent training sample. The canonical
example is :class:`GUIAgentLoop`, where each turn of a multi-turn desktop
interaction has a different prompt (due to screenshot pruning) and therefore
cannot be merged into a single ``AgentLoopOutput``.

Because ``AgentLoopBase.run()`` is contractually required to return a single
``AgentLoopOutput``, this base class uses the following packing protocol:

* The subclass's ``run()`` returns the *last* turn's ``AgentLoopOutput`` as the
  main output.
* All intermediate turns are packed into
  ``final_output.extra_fields["intermediate_trajectories"]`` as a list of
  serialized ``IntermediateTrajectory`` dicts.
* The Trainer side (``expand_intermediate_trajectories``) expands these back
  into independent DataProto rows during batch assembly — so the
  ``AgentLoopWorker``, ``MessageQueue`` and ``FullyAsyncRollouter`` layers all
  see a single ``AgentLoopOutput`` and do not need to be modified.

Subclasses should:

1. Call :meth:`append_intermediate_trajectory` at the end of every intermediate
   turn with the fully-tokenized prompt/response for that turn.
2. At the end of ``run()``, build an :class:`AgentLoopOutput` for the final
   turn and return ``self.build_final_output(final_output, shared_reward)``.
"""

from abc import abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput


# The key under which intermediate trajectories are packed into
# ``AgentLoopOutput.extra_fields`` and later discovered by the
# ``expand_intermediate_trajectories`` expander in
# ``verl.experimental.fully_async_policy.detach_utils``.
INTERMEDIATE_TRAJECTORIES_KEY = "intermediate_trajectories"


class IntermediateTrajectory(BaseModel):
    """Serialized representation of a single intermediate-turn trajectory.

    This schema intentionally mirrors the subset of ``AgentLoopOutput`` fields
    that the Trainer-side expander needs in order to rebuild a standalone
    DataProto row. Keeping this as a concrete Pydantic model (rather than a
    free-form dict) guarantees that all multi-trajectory agent loops agree on
    the field names and types.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    """Token ids of the prompt for this turn (already tokenized by the subclass)."""

    response_ids: list[int]
    """Token ids of the LLM response for this turn."""

    response_mask: list[int]
    """Response mask (1 = LLM generated token, 0 = tool/observation token)."""

    response_logprobs: Optional[list[float]] = None
    """Per-token log probabilities for the response, if available."""

    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal payload (e.g., images/videos) for this turn's prompt."""

    num_turns: int = 0
    """Number of chat turns seen up to and including this intermediate turn."""

    extra_fields: dict[str, Any] = {}
    """Per-trajectory extra fields (e.g., ``min_global_steps``, ``turn_number``)."""


class MultiTrajectoryAgentLoop(AgentLoopBase):
    """Base class for agent loops that emit multiple trajectories per rollout.

    The class provides a small, focused API:

    * :meth:`append_intermediate_trajectory` — call once per intermediate turn.
    * :meth:`build_final_output` — call once at the end of ``run()`` with the
      final-turn ``AgentLoopOutput`` and the shared reward.

    Subclasses must still implement ``run()`` (inherited abstract method from
    :class:`AgentLoopBase`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-instance buffer of registered intermediate trajectories.
        # Cleared automatically by ``build_final_output`` before returning.
        self._intermediate_trajectories: list[IntermediateTrajectory] = []

    def append_intermediate_trajectory(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
        num_turns: int = 0,
        **extra_fields: Any,
    ) -> None:
        """Register a completed intermediate-turn trajectory.

        Args:
            prompt_ids: Tokenized prompt for this turn.
            response_ids: Tokenized LLM response for this turn.
            response_mask: Response mask aligned with ``response_ids``.
            response_logprobs: Optional per-token logprobs.
            multi_modal_data: Optional multi-modal payload.
            num_turns: Number of chat turns up to and including this one.
            **extra_fields: Arbitrary per-trajectory extra fields (will be
                preserved and propagated into the Trainer-side DataProto).
        """
        trajectory = IntermediateTrajectory(
            prompt_ids=list(prompt_ids),
            response_ids=list(response_ids),
            response_mask=list(response_mask),
            response_logprobs=list(response_logprobs) if response_logprobs is not None else None,
            multi_modal_data=multi_modal_data,
            num_turns=num_turns,
            extra_fields=dict(extra_fields),
        )
        self._intermediate_trajectories.append(trajectory)

    def build_final_output(
        self,
        final_output: AgentLoopOutput,
        shared_reward: float,
    ) -> AgentLoopOutput:
        """Pack registered intermediate trajectories into ``final_output`` and
        stamp the shared reward across all trajectories (final + intermediate).

        The buffer of intermediate trajectories is cleared before returning so
        that a single :class:`MultiTrajectoryAgentLoop` instance can be reused
        for multiple ``run()`` invocations without cross-run contamination.

        Args:
            final_output: The ``AgentLoopOutput`` representing the last turn.
            shared_reward: Scalar reward to assign to all trajectories.

        Returns:
            The ``final_output`` object with ``extra_fields`` updated to carry
            the serialized intermediate trajectories list, and
            ``reward_score = shared_reward``.
        """
        # Apply shared reward to the final trajectory.
        final_output.reward_score = shared_reward

        # Serialize intermediate trajectories to plain dicts and stamp reward
        # score into each trajectory's ``extra_fields`` for downstream use.
        serialized: list[dict[str, Any]] = []
        for traj in self._intermediate_trajectories:
            traj.extra_fields["reward_score"] = shared_reward
            serialized.append(traj.model_dump())

        # Write packed trajectories into final_output's extra_fields.
        # The Trainer-side ``expand_intermediate_trajectories`` will pick them
        # up by key ``INTERMEDIATE_TRAJECTORIES_KEY``.
        final_output.extra_fields[INTERMEDIATE_TRAJECTORIES_KEY] = serialized

        # Also expose the final trajectory's role explicitly, so downstream
        # consumers can distinguish "final" from "intermediate" rows.
        final_output.extra_fields.setdefault("trajectory_role", "final")

        # Clear internal buffer to prevent cross-run pollution.
        self._intermediate_trajectories = []

        return final_output

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Subclasses must implement the rollout business logic.

        The implementation is expected to:

        * Call :meth:`append_intermediate_trajectory` once per intermediate turn.
        * Build an :class:`AgentLoopOutput` for the final turn.
        * Return ``self.build_final_output(final_output, shared_reward)``.
        """
        raise NotImplementedError
