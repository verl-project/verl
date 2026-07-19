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

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PendingTeacherSample:
    sample: Any
    enqueued_at: float
    sequence: int


class ExclusiveTeacherScheduler:
    """Batch exclusive-route samples while keeping teacher swaps bounded and fair."""

    def __init__(
        self,
        teacher_keys: list[str],
        scoring_batch_size: int,
        max_consecutive_batches: int,
        max_wait_seconds: float,
    ):
        if not teacher_keys:
            raise ValueError("ExclusiveTeacherScheduler requires at least one teacher.")
        if scoring_batch_size <= 0:
            raise ValueError("scoring_batch_size must be greater than zero.")
        if max_consecutive_batches <= 0:
            raise ValueError("max_consecutive_batches must be greater than zero.")
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be greater than zero.")

        self.teacher_keys = tuple(teacher_keys)
        self.scoring_batch_size = scoring_batch_size
        self.max_consecutive_batches = max_consecutive_batches
        self.max_wait_seconds = max_wait_seconds
        self.pending = {key: deque() for key in self.teacher_keys}
        self.ready = deque()
        self.last_scored_teacher: Optional[str] = None
        self.consecutive_batches = 0
        self._sequence = 0

    @property
    def pending_count(self) -> int:
        return sum(len(items) for items in self.pending.values())

    @property
    def ready_count(self) -> int:
        return len(self.ready)

    @property
    def available_count(self) -> int:
        return self.pending_count + self.ready_count

    def add_sample(self, teacher_key: str, sample: Any, now: Optional[float] = None) -> None:
        if teacher_key not in self.pending:
            raise ValueError(
                f"No fused teacher configured for routing key {teacher_key!r}. "
                f"Configured keys: {list(self.teacher_keys)}."
            )
        self.pending[teacher_key].append(
            PendingTeacherSample(
                sample=sample,
                enqueued_at=time.monotonic() if now is None else now,
                sequence=self._sequence,
            )
        )
        self._sequence += 1

    def _oldest_item(self, teacher_key: str) -> PendingTeacherSample:
        return self.pending[teacher_key][0]

    def choose_teacher(
        self,
        resident_teacher: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Optional[str]:
        candidates = [key for key in self.teacher_keys if self.pending[key]]
        if not candidates:
            return None

        now = time.monotonic() if now is None else now
        overdue = [
            key
            for key in candidates
            if now - self._oldest_item(key).enqueued_at >= self.max_wait_seconds
        ]
        if overdue:
            return min(
                overdue,
                key=lambda key: (
                    self._oldest_item(key).enqueued_at,
                    self._oldest_item(key).sequence,
                ),
            )

        if resident_teacher in candidates:
            resident_may_continue = (
                self.last_scored_teacher != resident_teacher
                or self.consecutive_batches < self.max_consecutive_batches
                or len(candidates) == 1
            )
            if resident_may_continue:
                return resident_teacher

        alternatives = [key for key in candidates if key != self.last_scored_teacher]
        if self.consecutive_batches >= self.max_consecutive_batches and alternatives:
            candidates = alternatives

        return min(
            candidates,
            key=lambda key: (
                self._oldest_item(key).enqueued_at,
                self._oldest_item(key).sequence,
            ),
        )

    def pop_scoring_batch(self, teacher_key: str) -> list[PendingTeacherSample]:
        if teacher_key not in self.pending:
            raise ValueError(f"Unknown teacher key {teacher_key!r}.")
        items = self.pending[teacher_key]
        return [items.popleft() for _ in range(min(self.scoring_batch_size, len(items)))]

    def mark_scored(self, teacher_key: str, items: list[PendingTeacherSample]) -> None:
        if self.last_scored_teacher == teacher_key:
            self.consecutive_batches += 1
        else:
            self.last_scored_teacher = teacher_key
            self.consecutive_batches = 1
        self.ready.extend(item.sample for item in items)

    def take_ready(self, count: int) -> list[Any]:
        if count > self.ready_count:
            raise ValueError(f"Requested {count} ready samples, only {self.ready_count} available.")
        return [self.ready.popleft() for _ in range(count)]
