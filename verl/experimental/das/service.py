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
"""Centralized trajectory store: the SuffixDataService actor.

One named Ray actor per job gathers rollout trajectories from every agent-loop
worker and redistributes them to engine replicas as a versioned delta log.
Engines apply deltas to local per-problem suffix trees, so the decode hot
path never crosses the network.

The class is a plain object (Ray-agnostic) so unit tests drive it directly;
``get_or_create_suffix_service`` wraps it as the shared actor.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from verl.experimental.das.contracts import (
    SHADOW_OFFSET,
    DeltaBatch,
    TrajectoryDelta,
    TrajectoryPush,
    serialize_delta_batch,
)
from verl.workers.config.das import DASConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_LOG_CAPACITY = 200_000
_EMA_ALPHA = 0.2

# Log entry kinds.
_ADD = "add"
_RM = "rm"
_DROP = "drop"
_CLS = "cls"


@dataclass
class _SeqRecord:
    token_ids: tuple
    iteration: int
    has_shadow: bool


@dataclass
class _ProblemState:
    seqs: OrderedDict[int, _SeqRecord] = field(default_factory=OrderedDict)
    count: int = 0
    ema_mean: float = 0.0
    cls: int = 0


class SuffixDataService:
    """Versioned, sliding-window, per-problem trajectory store."""

    def __init__(self, config: DASConfig | None = None) -> None:
        self.cfg = config or DASConfig()
        self._iteration = 0
        self._version = 0
        # Per-iteration serving: entries logged during the in-progress
        # iteration are withheld from replicas until begin_iteration advances
        # past it. Applies therefore land engine-side once per training step,
        # right at the rollout boundary, independent of engine sleep internals.
        self._servable_version = 0
        self._next_seq = 0
        self._problems: dict[str, _ProblemState] = {}
        self._problem_lru: OrderedDict[str, None] = OrderedDict()
        # iteration -> {"entries": [(problem_id, seq_id)], "shadows_dropped": bool}
        self._window: OrderedDict[int, dict] = OrderedDict()
        self._log: deque[tuple[int, str, object]] = deque(maxlen=_LOG_CAPACITY)
        self._total_tokens = 0
        self._pushes_received = 0

    # ------------------------------------------------------------------ log

    def _emit(self, kind: str, payload: object) -> None:
        self._version += 1
        self._log.append((self._version, kind, payload))

    def _log_floor(self) -> int:
        """Oldest version still replayable from the log."""
        return self._log[0][0] if self._log else self._version + 1

    # -------------------------------------------------------------- ingest

    def add_trajectories(self, worker_id: str, items: list[TrajectoryPush]) -> None:
        for push in items:
            tokens = tuple(push.token_ids)
            if not tokens:
                continue
            self._pushes_received += 1
            self._add_one(push.problem_id, tokens)
        self._enforce_token_cap()

    def _add_one(self, problem_id: str, tokens: tuple) -> None:
        prob = self._problems.get(problem_id)
        if prob is None:
            prob = _ProblemState()
            self._problems[problem_id] = prob
            self._enforce_problem_cap(keep=problem_id)
        self._problem_lru[problem_id] = None
        self._problem_lru.move_to_end(problem_id)

        while len(prob.seqs) >= self.cfg.max_seqs_per_problem:
            old_seq, old_rec = prob.seqs.popitem(last=False)
            self._remove_seq_entries(problem_id, old_seq, old_rec)

        seq_id = self._next_seq
        self._next_seq = (self._next_seq + 1) % SHADOW_OFFSET
        has_shadow = self.cfg.fresh_iterations > 0
        prob.seqs[seq_id] = _SeqRecord(tokens, self._iteration, has_shadow)
        self._window.setdefault(self._iteration, {"entries": [], "shadows_dropped": False})
        self._window[self._iteration]["entries"].append((problem_id, seq_id))
        self._total_tokens += len(tokens)

        self._emit(_ADD, TrajectoryDelta(problem_id, seq_id, tokens))
        if has_shadow:
            self._emit(_ADD, TrajectoryDelta(problem_id, seq_id + SHADOW_OFFSET, tokens))

        self._update_stats(problem_id, prob, len(tokens))

    def _update_stats(self, problem_id: str, prob: _ProblemState, length: int) -> None:
        prob.count += 1
        prob.ema_mean = float(length) if prob.count == 1 else _EMA_ALPHA * length + (1.0 - _EMA_ALPHA) * prob.ema_mean
        new_cls = self.cfg.class_for_length(prob.ema_mean)
        # Emit on the first sequence too: engines treat unknown problems as Medium, so a Short
        # problem that never changes class must still be announced as Short.
        if prob.count == 1 or new_cls != prob.cls:
            prob.cls = new_cls
            self._emit(_CLS, (problem_id, new_cls))

    # ------------------------------------------------------------ eviction

    def _remove_seq_entries(self, problem_id: str, seq_id: int, rec: _SeqRecord) -> None:
        self._total_tokens -= len(rec.token_ids)
        self._emit(_RM, (problem_id, seq_id))
        if rec.has_shadow:
            self._emit(_RM, (problem_id, seq_id + SHADOW_OFFSET))

    def _drop_problem(self, problem_id: str) -> None:
        prob = self._problems.pop(problem_id, None)
        self._problem_lru.pop(problem_id, None)
        if prob is None:
            return
        for rec in prob.seqs.values():
            self._total_tokens -= len(rec.token_ids)
        # A single drop entry stands in for per-seq removals.
        self._emit(_DROP, problem_id)

    def _enforce_problem_cap(self, keep: str) -> None:
        while len(self._problems) > self.cfg.max_problems:
            for candidate in self._problem_lru:
                if candidate != keep:
                    self._drop_problem(candidate)
                    break
            else:
                return

    def _enforce_token_cap(self) -> None:
        while self._total_tokens > self.cfg.max_total_tokens and self._window:
            oldest_iter = next(iter(self._window))
            if oldest_iter >= self._iteration:
                # Only the current iteration remains; drop LRU problems instead.
                if not self._problem_lru:
                    return
                self._drop_problem(next(iter(self._problem_lru)))
                continue
            self._evict_iteration(oldest_iter)

    def _evict_iteration(self, iteration: int) -> None:
        info = self._window.pop(iteration, None)
        if info is None:
            return
        for problem_id, seq_id in info["entries"]:
            prob = self._problems.get(problem_id)
            if prob is None:
                continue
            rec = prob.seqs.pop(seq_id, None)
            if rec is not None:
                self._remove_seq_entries(problem_id, seq_id, rec)

    # ----------------------------------------------------------- iteration

    def begin_iteration(self, step: int) -> dict:
        """Advance the window to training step *step*. Monotone and idempotent per step."""
        if step > self._iteration:
            self._iteration = step
            self._expire_shadows()
            self._expire_window()
            # Everything logged so far (prior iterations' adds + this
            # boundary's expiries) becomes servable in one step-aligned unit;
            # entries logged from here on wait for the next boundary.
            self._servable_version = self._version
        return {
            "iteration": self._iteration,
            "version": self._version,
            "problems": len(self._problems),
            "total_tokens": self._total_tokens,
        }

    def _expire_shadows(self) -> None:
        cutoff = self._iteration - self.cfg.fresh_iterations
        for iteration, info in self._window.items():
            if iteration > cutoff or info["shadows_dropped"]:
                continue
            info["shadows_dropped"] = True
            for problem_id, seq_id in info["entries"]:
                prob = self._problems.get(problem_id)
                rec = prob.seqs.get(seq_id) if prob else None
                if rec is not None and rec.has_shadow:
                    rec.has_shadow = False
                    self._emit(_RM, (problem_id, seq_id + SHADOW_OFFSET))

    def _expire_window(self) -> None:
        cutoff = self._iteration - self.cfg.window_iterations
        for iteration in [it for it in self._window if it <= cutoff]:
            self._evict_iteration(iteration)

    # -------------------------------------------------------------- deltas

    def get_deltas(self, replica_id: str, since_version: int) -> bytes | None:
        """Serialized ``DeltaBatch`` with everything servable after *since_version*, or None if caught up."""
        # Per-iteration serving: only entries up to the last begin_iteration
        # boundary ship. A fresh replica (-1) and an empty service (version 0)
        # are in sync: entry versions start at 1.
        if max(since_version, 0) >= self._servable_version:
            return None
        # Incremental replay needs every entry in (since_version, servable] to
        # still be in the log; versions are contiguous starting at 1.
        if max(since_version + 1, 1) < self._log_floor():
            return serialize_delta_batch(self._build_snapshot())
        return serialize_delta_batch(self._build_incremental(since_version))

    def _build_incremental(self, since_version: int) -> DeltaBatch:
        """Collect log entries after *since_version*, compacted.

        The wire format carries adds/removals/drops as separate lists, so log
        order is not preserved on the receiver. Compaction makes intra-batch
        order irrelevant: a seq_id is added exactly once and removed at most
        once ever, so an add+removal pair inside the collected range cancels
        outright, and a problem drop purges the range's earlier entries for
        that problem (entries after the drop re-create it legitimately).
        Surviving removals/drops therefore always refer to state from before
        this batch, and the receiver may apply drops, then removals, then
        adds, in any internal order.
        """
        batch = DeltaBatch(to_version=since_version, iteration=self._iteration)
        adds: OrderedDict[tuple[str, int], TrajectoryDelta] = OrderedDict()
        removals: list[tuple[str, int]] = []
        drops: list[str] = []
        budget = self.cfg.max_delta_batch_tokens
        for version, kind, payload in self._log:
            if version <= since_version:
                continue
            if version > self._servable_version:
                break
            if kind == _ADD:
                delta: TrajectoryDelta = payload  # type: ignore[assignment]
                if budget - len(delta.token_ids) < 0 and adds:
                    break
                budget -= len(delta.token_ids)
                adds[delta.problem_id, delta.seq_id] = delta
            elif kind == _RM:
                key = payload  # type: ignore[assignment]
                if key in adds:
                    budget += len(adds.pop(key).token_ids)
                else:
                    removals.append(key)
            elif kind == _DROP:
                problem_id: str = payload  # type: ignore[assignment]
                for key in [k for k in adds if k[0] == problem_id]:
                    budget += len(adds.pop(key).token_ids)
                removals = [r for r in removals if r[0] != problem_id]
                batch.problem_cls.pop(problem_id, None)
                drops.append(problem_id)
            elif kind == _CLS:
                problem_id, cls = payload  # type: ignore[misc]
                batch.problem_cls[problem_id] = cls
            batch.to_version = version
        batch.adds = list(adds.values())
        batch.removals = removals
        batch.dropped_problems = drops
        return batch

    def _build_snapshot(self) -> DeltaBatch:
        # to_version is the servable ceiling: current-iteration sequences are
        # excluded here (their adds are withheld log entries with versions
        # above the ceiling) and ship incrementally at the next boundary.
        batch = DeltaBatch(to_version=self._servable_version, iteration=self._iteration, snapshot=True)
        for problem_id, prob in self._problems.items():
            served_any = False
            for seq_id, rec in prob.seqs.items():
                if rec.iteration >= self._iteration:
                    continue
                served_any = True
                batch.adds.append(TrajectoryDelta(problem_id, seq_id, rec.token_ids))
                if rec.has_shadow:
                    batch.adds.append(TrajectoryDelta(problem_id, seq_id + SHADOW_OFFSET, rec.token_ids))
            if served_any or not prob.seqs:
                batch.problem_cls[problem_id] = prob.cls
        return batch

    # ------------------------------------------------------------- metrics

    def get_metrics(self) -> dict[str, float]:
        return {
            "das_iteration": float(self._iteration),
            "das_log_version": float(self._version),
            "das_servable_version": float(self._servable_version),
            "das_withheld_entries": float(self._version - self._servable_version),
            "das_problem_count": float(len(self._problems)),
            "das_total_tokens": float(self._total_tokens),
            "das_window_iterations": float(len(self._window)),
            "das_pushes_received": float(self._pushes_received),
        }


def get_or_create_suffix_service(config: DASConfig | None = None):
    """Create or fetch the job's shared ``SuffixDataService`` actor.

    The actor is named (``config.service_name``) but not detached: Ray scopes
    the name to the job, so concurrent jobs never collide and the store dies
    with the job. Engines resolve it with ``ray.get_actor(config.service_name)``.
    """
    import ray

    cfg = config or DASConfig()
    actor_cls = ray.remote(num_cpus=1)(SuffixDataService)
    return actor_cls.options(name=cfg.service_name, get_if_exists=True).remote(cfg)
