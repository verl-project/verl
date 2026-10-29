# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Process scoring and advantage shaping for v1 agent rollouts."""

import inspect
import math
from abc import ABC, abstractmethod
from typing import Any

import torch

from verl.utils.import_utils import load_class_from_fqn


class BaseProcessScorer(ABC):
    """Score a generated assistant turn after the full trajectory is available.

    ``evaluate`` receives the original AgentLoopOutput, the half-open response-token
    span, and the sample's input fields. Return a scalar, broadcast to the turn, or
    one score for each token in the span. Scorers must not mutate their inputs.
    """

    @abstractmethod
    async def evaluate(self, output: Any, span: tuple[int, int], context: dict[str, Any]) -> Any:
        pass


def generated_turn_spans(response_mask: list[int], explicit_spans: list[tuple[int, int]] | None = None):
    """Validate turn boundaries against generated-token positions."""
    if explicit_spans is None:
        explicit_spans = []
        start = None
        for pos, is_generated in enumerate([*response_mask, 0]):
            if is_generated and start is None:
                start = pos
            elif not is_generated and start is not None:
                explicit_spans.append((start, pos))
                start = None

    spans = []
    covered = [False] * len(response_mask)
    previous_end = 0
    for start, end in explicit_spans:
        if not 0 <= start < end <= len(response_mask):
            raise ValueError(f"Invalid generated turn span {(start, end)}")
        if start < previous_end:
            raise ValueError("Generated turn spans must be ordered and non-overlapping")
        if any(not response_mask[pos] or covered[pos] for pos in range(start, end)):
            raise ValueError(f"Generated turn span {(start, end)} overlaps or includes a non-generated token")
        covered[start:end] = [True] * (end - start)
        spans.append((start, end))
        previous_end = end
    if covered != [bool(value) for value in response_mask]:
        raise ValueError("Generated turn spans do not cover all generated tokens")
    return spans


def validate_turn_rewards(turn_rewards: Any, num_turns: int) -> torch.Tensor:
    """Normalize one finite scalar per generated turn, without mutating the input."""
    try:
        values = torch.as_tensor(turn_rewards, dtype=torch.float32).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("turn_rewards must contain one finite scalar per generated turn") from error
    if values.shape != (num_turns,) or not torch.isfinite(values).all():
        raise ValueError(f"turn_rewards must contain {num_turns} finite scalars, one per generated turn")
    return values


class ProcessRewardRunner:
    def __init__(self, scorer_configs: list[dict[str, Any]]):
        self.scorers = []
        for scorer_config in scorer_configs:
            cls = load_class_from_fqn(scorer_config["path"], description="process scorer")
            if not issubclass(cls, BaseProcessScorer):
                raise TypeError(f"{cls} must subclass BaseProcessScorer")
            weight = float(scorer_config.get("weight", 1.0))
            if not math.isfinite(weight):
                raise ValueError("Process scorer weight must be finite")
            self.scorers.append((cls(**scorer_config.get("kwargs", {})), weight))

    async def score(self, output: Any, context: dict[str, Any]) -> torch.Tensor:
        mask = output.response_mask
        if len(mask) != len(output.response_ids):
            raise ValueError("response_mask and response_ids must have the same length")
        spans = generated_turn_spans(mask, output.extra_fields.get("generated_turn_spans"))
        scores = torch.zeros(len(mask), dtype=torch.float32)
        if "turn_rewards" in output.extra_fields:
            rewards = validate_turn_rewards(output.extra_fields["turn_rewards"], len(spans))
            for (start, end), reward in zip(spans, rewards, strict=True):
                scores[start:end] = reward
            return scores
        if not self.scorers:
            raise ValueError("Process rewards require extra_fields['turn_rewards'] or at least one scorer")
        for span in spans:
            start, end = span
            for scorer, weight in self.scorers:
                result = scorer.evaluate(output, span, context)
                if inspect.isawaitable(result):
                    result = await result
                if result is None:
                    raise ValueError("Process scorer returned None")
                values = torch.as_tensor(result, dtype=torch.float32).detach().cpu()
                if values.ndim == 0:
                    values = values.expand(end - start)
                if values.shape != (end - start,) or not torch.isfinite(values).all():
                    raise ValueError(f"Process scorer returned invalid scores for span {span}")
                scores[start:end] += weight * values
        if not torch.isfinite(scores).all():
            raise ValueError("Weighted process scores must be finite")
        return scores


def shape_process_advantages(
    advantages: torch.Tensor,
    process_scores: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    mode: str,
    weight: float = 1.0,
    negative_scale: float = 0.2,
    zero_penalty: float = 0.4,
    zero_eps: float = 1e-6,
) -> torch.Tensor:
    """Shape actor advantages after outcome advantage estimation; leave returns intact."""
    if advantages.shape != process_scores.shape or advantages.shape != response_mask.shape:
        raise ValueError("advantages, process_scores and response_mask must have identical shapes")
    if not torch.isfinite(process_scores).all():
        raise ValueError("process_scores must be finite")
    if not math.isfinite(float(weight)):
        raise ValueError("process reward weight must be finite")
    if torch.any(process_scores.masked_select(response_mask == 0) != 0):
        raise ValueError("process_scores must be zero outside generated tokens")
    reward = process_scores * weight
    if mode == "additive":
        shaped = advantages + reward
    elif mode == "gvpo":
        for name, value in (("negative_scale", negative_scale), ("zero_penalty", zero_penalty), ("zero_eps", zero_eps)):
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"GVPO {name} must be finite and non-negative")
        failed = (reward < 0) & response_mask.bool()
        shaped = torch.where(failed & (advantages < -zero_eps), advantages * (1.0 + negative_scale), advantages)
        shaped = torch.where(failed & (advantages.abs() <= zero_eps), torch.full_like(shaped, -zero_penalty), shaped)
        shaped = shaped.masked_fill(failed & (advantages > zero_eps), 0.0)
    elif mode == "process_weighted":
        if torch.any((reward < 0) | (reward > 1)):
            raise ValueError("process_weighted scores after weighting must be in [0, 1]")
        shaped = torch.where(
            advantages >= 0,
            advantages * reward,
            torch.where(reward > 0, reward, advantages),
        )
    else:
        raise ValueError(f"Unknown process reward mode: {mode}")
    return shaped * response_mask
