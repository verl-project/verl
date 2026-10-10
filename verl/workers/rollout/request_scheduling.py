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

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional


class RolloutRequestKind(str, Enum):
    """Semantic class of an agent rollout request."""

    FRESH = "fresh"
    CONTINUATION = "continuation"
    RETRY = "retry"


@dataclass(frozen=True)
class RolloutRequestContext:
    """Backend-neutral scheduling metadata for one rollout request attempt."""

    trajectory_id: str
    request_kind: RolloutRequestKind
    turn_index: int
    attempt_index: int
    prompt_tokens: int
    estimated_uncached_tokens: Optional[int]
    enqueued_at: float
    expected_output_tokens: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        context = asdict(self)
        context["request_kind"] = self.request_kind.value
        return context

    @classmethod
    def from_dict(cls, context: dict[str, Any]) -> "RolloutRequestContext":
        """Build a typed context from its transport representation."""
        values = dict(context)
        values["request_kind"] = RolloutRequestKind(values["request_kind"])
        return cls(**values)
