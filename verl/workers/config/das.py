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
"""Configuration for DAS (distribution-aware speculative decoding) rollouts.

One flat dataclass is consumed by four kinds of processes: the trainer
driver (service creation, step boundaries), agent-loop workers (problem ids,
pushes), the vLLM server actors (delta pump), and the TP-rank GPU workers
(per-problem drafter). The GPU workers receive it as JSON through an
internal environment variable, hence ``to_json`` / ``from_json``.
"""

import json
from dataclasses import dataclass, fields

from verl.base_config import BaseConfig

__all__ = ["DASConfig"]

# Length classes. Ordering matters: budgets may only be upgraded S -> M -> L.
CLS_SHORT = 0
CLS_MEDIUM = 1
CLS_LONG = 2


@dataclass
class DASConfig(BaseConfig):
    """Distribution-aware speculative decoding (DAS, arXiv 2511.13841) on top of vLLM suffix decoding.

    Requires ``rollout.name=vllm`` with ``engine_kwargs.vllm.speculative_config.method=suffix``;
    the draft length cap, tree depth, spec factor and token-probability floor come from that
    vLLM config. DAS adds per-problem suffix trees shared across engines through a central
    store and length-aware draft budgets per problem.
    """

    enable: bool = False
    # Training iterations retained in the central store (older trajectories are evicted).
    window_iterations: int = 8
    # Most recent iterations whose trajectories also carry a 2x-count shadow copy
    # (approximates DAS recency down-weighting with raw-count trees). 0 disables shadows.
    fresh_iterations: int = 2
    # Engine delta-pump poll cadence in seconds.
    poll_interval_s: float = 2.0
    # Max tokens per delta batch shipped to an engine.
    max_delta_batch_tokens: int = 500_000
    # Draft-token budgets per problem length class (capped by vLLM's num_speculative_tokens).
    budget_long: int = 24
    budget_medium: int = 8
    budget_short: int = 0  # 0 == skip speculation for Short problems; data is still collected
    # Generation-length thresholds (tokens) separating Short / Medium / Long problems.
    short_max_tokens: int = 256
    long_min_tokens: int = 1024
    # Central store caps.
    max_total_tokens: int = 64_000_000
    max_problems: int = 8192
    max_seqs_per_problem: int = 512
    # Agent-loop worker -> store push batching.
    push_batch_size: int = 64
    push_flush_interval_s: float = 0.5
    push_queue_capacity: int = 4096
    # Ray actor name of the store (job-scoped: Ray names are per job namespace).
    service_name: str = "verl_das_suffix_service"

    def __post_init__(self) -> None:
        if self.window_iterations < 1:
            raise ValueError(f"das.window_iterations must be >= 1, got {self.window_iterations}")
        if self.fresh_iterations < 0:
            raise ValueError(f"das.fresh_iterations must be >= 0, got {self.fresh_iterations}")
        if self.poll_interval_s <= 0:
            raise ValueError(f"das.poll_interval_s must be > 0, got {self.poll_interval_s}")
        if min(self.budget_long, self.budget_medium, self.budget_short) < 0:
            raise ValueError("das.budget_* must be >= 0")
        if self.short_max_tokens > self.long_min_tokens:
            raise ValueError(
                f"das.short_max_tokens ({self.short_max_tokens}) must be <= "
                f"das.long_min_tokens ({self.long_min_tokens})"
            )
        for name in ("max_delta_batch_tokens", "max_total_tokens", "max_problems", "max_seqs_per_problem"):
            if getattr(self, name) < 1:
                raise ValueError(f"das.{name} must be >= 1")
        if self.push_batch_size < 1 or self.push_queue_capacity < 1:
            raise ValueError("das.push_batch_size and das.push_queue_capacity must be >= 1")
        if not self.service_name:
            raise ValueError("das.service_name must be non-empty")

    def class_for_length(self, length: float) -> int:
        """Length class of a problem given its (smoothed) generation length in tokens."""
        if length >= self.long_min_tokens:
            return CLS_LONG
        if length >= self.short_max_tokens:
            return CLS_MEDIUM
        return CLS_SHORT

    def budget_for_class(self, cls: int) -> int:
        """Max draft tokens per round for a length class."""
        if cls >= CLS_LONG:
            return self.budget_long
        if cls == CLS_MEDIUM:
            return self.budget_medium
        return self.budget_short

    def to_json(self) -> str:
        """Serialize for the internal env hop into vLLM worker subprocesses."""
        return json.dumps({f.name: getattr(self, f.name) for f in fields(self) if f.name != "_target_"})

    @classmethod
    def from_json(cls, payload: str) -> "DASConfig":
        return cls(**json.loads(payload))
