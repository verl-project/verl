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
"""Abstract base class for agent loops that emit multiple trajectories per rollout.

A *multi-trajectory* agent loop is one in which a single rollout (one call to
``run()``) naturally produces several independent ``(prompt, response)`` pairs
that each should be treated as an independent training sample. Concrete
scenarios where this occurs include:

* Multi-turn VLM agents whose prompt changes every turn (e.g. screenshot
  pruning in a computer-use agent), so turns cannot be collapsed into a single
  ``AgentLoopOutput``.
* Multi-step reasoning agents that want each reasoning step to be trained as
  an independent sample sharing the episode-level reward.
* Prompt-changing multi-tool agents where intermediate tool calls materially
  change the prompt prefix.

Because ``AgentLoopBase.run()`` is contractually required to return a single
``AgentLoopOutput``, this base class uses the following packing protocol:

* The subclass's ``run()`` returns the *last* turn's ``AgentLoopOutput`` as the
  main output.
* All intermediate turns are represented as standalone :class:`AgentLoopOutput`
  instances (same schema as the final turn) and packed into
  ``final_output.extra_fields["intermediate_trajectories"]`` as a list of
  serialized dicts.
* The Trainer-side
  :func:`verl.experimental.fully_async_policy.intermediate_trajectory_utils.expand_intermediate_trajectories`
  expands these back into independent DataProto rows during batch assembly —
  so the ``AgentLoopWorker``, ``MessageQueue`` and ``FullyAsyncRollouter``
  layers all see a single ``AgentLoopOutput`` and do not need to be modified.

Subclasses should:

1. Call :meth:`append_intermediate_trajectory` at the end of every intermediate
   turn with the fully-tokenized prompt/response for that turn.
2. At the end of ``run()``, build an :class:`AgentLoopOutput` for the final
   turn and return ``self.build_final_output(final_output, shared_reward)``.
"""

import logging
import os
from abc import abstractmethod
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# The key under which intermediate trajectories are packed into
# ``AgentLoopOutput.extra_fields`` and later discovered by the Trainer-side
# ``expand_intermediate_trajectories`` expander.
INTERMEDIATE_TRAJECTORIES_KEY = "intermediate_trajectories"


class MultiTrajectoryAgentLoop(AgentLoopBase):
    """Base class for agent loops that emit multiple trajectories per rollout.

    The class provides a small, focused API:

    * :meth:`append_intermediate_trajectory` — call once per intermediate turn.
    * :meth:`build_final_output` — call once at the end of ``run()`` with the
      final-turn ``AgentLoopOutput`` and the shared reward.

    Intermediate trajectories are stored as full :class:`AgentLoopOutput`
    instances so the schema matches the final trajectory exactly (including
    ``routed_experts`` for MoE models). The ``metrics`` field on intermediate
    trajectories is set to an empty :class:`AgentLoopMetrics` (all defaults)
    because only rollout-level timing aggregates are meaningful, and they are
    already carried by the final trajectory.

    Subclasses must still implement ``run()`` (inherited abstract method from
    :class:`AgentLoopBase`).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-instance buffer of registered intermediate trajectories.
        # Cleared automatically by ``build_final_output`` before returning.
        self._intermediate_trajectories: list[AgentLoopOutput] = []

    def append_intermediate_trajectory(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_mask: list[int],
        response_logprobs: Optional[list[float]] = None,
        routed_experts: Optional[Any] = None,
        multi_modal_data: Optional[dict[str, Any]] = None,
        num_turns: int = 0,
        **extra_fields: Any,
    ) -> None:
        """Register a completed intermediate-turn trajectory.

        The trajectory is stored as a full :class:`AgentLoopOutput` so the
        schema matches the final trajectory emitted by ``build_final_output``.
        ``metrics`` is set to an empty :class:`AgentLoopMetrics` — intermediate
        turn timings do not contribute to rollout-level aggregates.
        ``reward_score`` is left ``None`` here and stamped later by
        :meth:`build_final_output` with the shared episode reward.
        """
        trajectory = AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=list(response_ids),
            response_mask=list(response_mask),
            response_logprobs=list(response_logprobs) if response_logprobs is not None else None,
            routed_experts=routed_experts,
            multi_modal_data=multi_modal_data,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields=dict(extra_fields),
        )
        self._intermediate_trajectories.append(trajectory)
        logger.debug(
            "[MultiTrajAgentLoop] appended intermediate trajectory #%d "
            "(prompt_len=%d, response_len=%d, num_turns=%d, buffered=%d)",
            len(self._intermediate_trajectories),
            len(trajectory.prompt_ids),
            len(trajectory.response_ids),
            num_turns,
            len(self._intermediate_trajectories),
        )

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
        """
        # Apply shared reward to the final trajectory.
        final_output.reward_score = shared_reward

        # Stamp the shared reward onto every intermediate trajectory via the
        # first-class ``reward_score`` field (same path as the final trajectory),
        # then serialize to plain dicts for packing into ``extra_fields``.
        serialized: list[dict[str, Any]] = []
        for traj in self._intermediate_trajectories:
            traj.reward_score = shared_reward
            serialized.append(traj.model_dump())

        final_output.extra_fields[INTERMEDIATE_TRAJECTORIES_KEY] = serialized

        # Expose the final trajectory's role explicitly.
        final_output.extra_fields.setdefault("trajectory_role", "final")

        logger.info(
            "[MultiTrajAgentLoop] built final output: reward=%.4f, "
            "packed %d intermediate trajectories (total trajectories this rollout = %d)",
            shared_reward,
            len(serialized),
            len(serialized) + 1,
        )

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
