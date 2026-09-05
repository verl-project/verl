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
"""Select and refine communication phase policies from semantic traces.

The tuner is deliberately offline: it never sleeps in a training process and
never invents a hardware-specific delay.  It compares observed policies using
GPU-realized offset, consumer slack, rank skew, and the global pair critical
path.  Refinement points are midpoints of the measured request intervals and
are rejected when interpolated slack would cross the baseline deadline guard.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
DEFAULT_PAIR_FIELDS = (
    "step",
    "iteration",
    "microbatch",
    "virtual_pipeline_stage",
    "layer",
    "bucket_id",
    "weight_version",
    "rollout_id",
)
BASELINE_POLICIES = {"baseline", "concurrent", "eager"}


class TraceFormatError(ValueError):
    """Raised when trace evidence cannot be compared safely."""


@dataclasses.dataclass(frozen=True)
class RankPair:
    """Two communication operations observed on one rank."""

    run_id: str
    context: tuple[tuple[str, Any], ...]
    ordinal: int
    rank: int
    declared_world_size: int | None
    framework: str
    topology_class: str
    timestamp_domain: str
    gpu_timestamp_semantics: str
    clock_sync_error_bound_us: float
    operation_a: str
    operation_b: str
    message_bytes_a: int
    message_bytes_b: int
    transport_a: str
    transport_b: str
    process_group_id_a: str
    process_group_id_b: str
    sequence_id_a: int | None
    sequence_id_b: int | None
    policy: str
    requested_offset_us: float | None
    a_start_ns: int
    a_end_ns: int
    b_start_ns: int
    b_end_ns: int
    b_consumer_ns: int
    critical_path_duration_us: float | None
    sequence_consistent: bool

    @property
    def workload_base(self) -> tuple[Any, ...]:
        """Return fields that must not be mixed in one recommendation."""

        return (
            self.framework,
            self.topology_class,
            self.timestamp_domain,
            self.gpu_timestamp_semantics,
            self.operation_a,
            self.operation_b,
            self.message_bytes_a,
            self.message_bytes_b,
            self.transport_a,
            self.transport_b,
        )

    @property
    def realized_offset_us(self) -> float:
        """Return operation B's GPU start relative to operation A."""

        return (self.b_start_ns - self.a_start_ns) / 1000

    @property
    def consumer_slack_us(self) -> float:
        """Return positive slack, or negative time spent waiting for B."""

        return (self.b_consumer_ns - self.b_end_ns) / 1000


@dataclasses.dataclass(frozen=True)
class Trial:
    """One globally complete operation pair across all participating ranks."""

    run_id: str
    context: tuple[tuple[str, Any], ...]
    ordinal: int
    ranks: tuple[RankPair, ...]

    @property
    def pair_completion_us(self) -> float:
        """Return the global communication-pair completion window."""

        start_ns = min(min(pair.a_start_ns, pair.b_start_ns) for pair in self.ranks)
        end_ns = max(max(pair.a_end_ns, pair.b_end_ns) for pair in self.ranks)
        return (end_ns - start_ns) / 1000

    @property
    def critical_path_us(self) -> float:
        """Return the slowest-rank step critical path, or pair completion fallback."""

        values = [pair.critical_path_duration_us for pair in self.ranks]
        present = [value for value in values if value is not None]
        if present and len(present) != len(values):
            raise TraceFormatError("critical_path_duration_us is missing on only some ranks")
        return max(present) if present else self.pair_completion_us

    @property
    def critical_path_source(self) -> str:
        """Return which trace signal supplies the optimization objective."""

        return "trace_critical_path" if self.ranks[0].critical_path_duration_us is not None else "pair_completion"

    @property
    def rank_skew_us(self) -> float:
        """Return the largest launch or finish skew of the two operations."""

        timestamp_sets = (
            [pair.a_start_ns for pair in self.ranks],
            [pair.a_end_ns for pair in self.ranks],
            [pair.b_start_ns for pair in self.ranks],
            [pair.b_end_ns for pair in self.ranks],
        )
        return max((max(values) - min(values)) / 1000 for values in timestamp_sets)

    @property
    def sequence_consistent(self) -> bool:
        """Return whether every communicator has one logical sequence ID."""

        if not all(pair.sequence_consistent for pair in self.ranks):
            return False
        for group_field, sequence_field in (
            ("process_group_id_a", "sequence_id_a"),
            ("process_group_id_b", "sequence_id_b"),
        ):
            by_group: defaultdict[str, set[int]] = defaultdict(set)
            for pair in self.ranks:
                sequence_id = getattr(pair, sequence_field)
                if sequence_id is not None:
                    by_group[getattr(pair, group_field)].add(sequence_id)
            if any(len(sequence_ids) > 1 for sequence_ids in by_group.values()):
                return False
        return True


@dataclasses.dataclass(frozen=True)
class Candidate:
    """Aggregate trace evidence for one observed policy setting."""

    policy: str
    requested_offset_us: float | None
    trials: tuple[Trial, ...]

    @property
    def rank_pairs(self) -> tuple[RankPair, ...]:
        """Flatten per-trial rank observations."""

        return tuple(pair for trial in self.trials for pair in trial.ranks)


def percentile(values: Sequence[float], percent: float) -> float:
    """Return a linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _record_value(record: Mapping[str, Any], name: str) -> Any:
    value = record.get(name)
    if value is not None:
        return value
    metadata = record.get("metadata")
    return metadata.get(name) if isinstance(metadata, Mapping) else None


def _number(record: Mapping[str, Any], name: str, *, required: bool = True) -> float | None:
    value = _record_value(record, name)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise TraceFormatError(f"{name} must be a finite number")
    return float(value)


def _integer(record: Mapping[str, Any], name: str) -> int:
    value = _number(record, name)
    if value is None or not value.is_integer():
        raise TraceFormatError(f"{name} must be an integer")
    return int(value)


def _string(record: Mapping[str, Any], name: str, *, default: str | None = None) -> str:
    value = _record_value(record, name)
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value:
        raise TraceFormatError(f"{name} must be a non-empty string")
    return value


def _context(record: Mapping[str, Any], pair_fields: Sequence[str]) -> tuple[tuple[str, Any], ...]:
    values = []
    for field in pair_fields:
        value = _record_value(record, field)
        if isinstance(value, dict | list):
            raise TraceFormatError(f"pair field {field!r} must be scalar")
        values.append((field, value))
    return tuple(values)


def _sort_key(record: Mapping[str, Any]) -> tuple[float, float]:
    start = _number(record, "gpu_start_timestamp_ns")
    sequence = _number(record, "communicator_sequence_id", required=False)
    return float(start), -1 if sequence is None else sequence


def _common_value(records: Sequence[Mapping[str, Any]], name: str) -> Any:
    values = {_record_value(record, name) for record in records if _record_value(record, name) is not None}
    if len(values) > 1:
        raise TraceFormatError(f"paired records disagree on {name}: {sorted(values, key=str)!r}")
    return next(iter(values)) if values else None


def _required_common_value(records: Sequence[Mapping[str, Any]], name: str) -> Any:
    values = [_record_value(record, name) for record in records]
    if any(value is None for value in values):
        raise TraceFormatError(f"{name} must be present on both paired records")
    return _common_value(records, name)


def _optional_common_value(records: Sequence[Mapping[str, Any]], name: str) -> Any:
    values = [_record_value(record, name) for record in records]
    if any(value is None for value in values) and not all(value is None for value in values):
        raise TraceFormatError(f"{name} must be present on both paired records or neither")
    return _common_value(records, name)


def _validate_timestamps(record: Mapping[str, Any]) -> tuple[int, int]:
    start = _integer(record, "gpu_start_timestamp_ns")
    end = _integer(record, "gpu_end_timestamp_ns")
    if end < start:
        raise TraceFormatError("gpu_end_timestamp_ns precedes gpu_start_timestamp_ns")
    completion_observed = _record_value(record, "completion_observed")
    if completion_observed is not True:
        raise TraceFormatError("completion_observed=true is required for every operation")
    return start, end


def _pair_records(
    record_a: Mapping[str, Any],
    record_b: Mapping[str, Any],
    *,
    context: tuple[tuple[str, Any], ...],
    ordinal: int,
    operation_a: str,
    operation_b: str,
) -> RankPair:
    a_start, a_end = _validate_timestamps(record_a)
    b_start, b_end = _validate_timestamps(record_b)
    consumer = _integer(record_b, "consumer_timestamp_ns")
    run_id = _string(record_a, "run_id")
    if _string(record_b, "run_id") != run_id:
        raise TraceFormatError("paired records disagree on run_id")
    rank = _integer(record_a, "rank")
    if _integer(record_b, "rank") != rank:
        raise TraceFormatError("paired records disagree on rank")
    if rank < 0:
        raise TraceFormatError("rank must be non-negative")
    declared_world_size = _required_common_value((record_a, record_b), "world_size")
    if (
        isinstance(declared_world_size, bool)
        or not isinstance(declared_world_size, int | float)
        or not float(declared_world_size).is_integer()
        or declared_world_size < 2
    ):
        raise TraceFormatError("world_size must be an integer of at least two")
    declared_world_size = int(declared_world_size)
    if rank >= declared_world_size:
        raise TraceFormatError("rank must be smaller than world_size")

    framework = _required_common_value((record_a, record_b), "framework")
    if not isinstance(framework, str) or not framework:
        raise TraceFormatError("framework must be a non-empty string")
    topology_class = _required_common_value((record_a, record_b), "topology_class")
    if not isinstance(topology_class, str) or not topology_class:
        raise TraceFormatError("topology_class must be a non-empty string")
    timestamp_domain = _required_common_value((record_a, record_b), "timestamp_domain")
    if not isinstance(timestamp_domain, str) or not timestamp_domain:
        raise TraceFormatError("timestamp_domain must be a non-empty string")
    gpu_timestamp_semantics = _required_common_value((record_a, record_b), "gpu_timestamp_semantics")
    if gpu_timestamp_semantics not in {"kernel-observed", "event-bracket"}:
        raise TraceFormatError("gpu_timestamp_semantics must be 'kernel-observed' or 'event-bracket'")
    clock_sync_error_bound_us = _required_common_value((record_a, record_b), "clock_sync_error_bound_us")
    if (
        isinstance(clock_sync_error_bound_us, bool)
        or not isinstance(clock_sync_error_bound_us, int | float)
        or not math.isfinite(clock_sync_error_bound_us)
        or clock_sync_error_bound_us < 0
    ):
        raise TraceFormatError("clock_sync_error_bound_us must be finite and non-negative")
    message_bytes_a = _integer(record_a, "message_bytes")
    message_bytes_b = _integer(record_b, "message_bytes")
    if message_bytes_a <= 0 or message_bytes_b <= 0:
        raise TraceFormatError("message_bytes must be positive")

    requested_offset = _optional_common_value((record_a, record_b), "requested_offset_us")
    if requested_offset is not None:
        if isinstance(requested_offset, bool) or not isinstance(requested_offset, int | float):
            raise TraceFormatError("requested_offset_us must be numeric or null")
        requested_offset = float(requested_offset)
        if not math.isfinite(requested_offset):
            raise TraceFormatError("requested_offset_us must be finite")
    policy = _optional_common_value((record_a, record_b), "policy")
    if policy is None:
        if requested_offset == 0:
            policy = "eager"
        elif requested_offset is None:
            policy = "unspecified"
        else:
            policy = "phase_shifted"
    if not isinstance(policy, str) or not policy:
        raise TraceFormatError("policy must be a non-empty string")

    critical_path_duration_us = _optional_common_value((record_a, record_b), "critical_path_duration_us")
    if critical_path_duration_us is not None:
        if (
            isinstance(critical_path_duration_us, bool)
            or not isinstance(critical_path_duration_us, int | float)
            or not math.isfinite(critical_path_duration_us)
            or critical_path_duration_us <= 0
        ):
            raise TraceFormatError("critical_path_duration_us must be positive and finite")
        critical_path_duration_us = float(critical_path_duration_us)

    sequence_consistent = True
    for record in (record_a, record_b):
        explicit = _record_value(record, "sequence_consistent")
        if explicit is False:
            sequence_consistent = False

    process_group_id_a = _string(record_a, "process_group_id")
    process_group_id_b = _string(record_b, "process_group_id")
    sequence_id_a = _integer(record_a, "communicator_sequence_id")
    sequence_id_b = _integer(record_b, "communicator_sequence_id")
    if sequence_id_a < 0 or sequence_id_b < 0:
        raise TraceFormatError("communicator_sequence_id must be non-negative")

    return RankPair(
        run_id=run_id,
        context=context,
        ordinal=ordinal,
        rank=rank,
        declared_world_size=declared_world_size,
        framework=framework,
        topology_class=topology_class,
        timestamp_domain=timestamp_domain,
        gpu_timestamp_semantics=gpu_timestamp_semantics,
        clock_sync_error_bound_us=float(clock_sync_error_bound_us),
        operation_a=operation_a,
        operation_b=operation_b,
        message_bytes_a=message_bytes_a,
        message_bytes_b=message_bytes_b,
        transport_a=_string(record_a, "transport"),
        transport_b=_string(record_b, "transport"),
        process_group_id_a=process_group_id_a,
        process_group_id_b=process_group_id_b,
        sequence_id_a=sequence_id_a,
        sequence_id_b=sequence_id_b,
        policy=policy,
        requested_offset_us=requested_offset,
        a_start_ns=a_start,
        a_end_ns=a_end,
        b_start_ns=b_start,
        b_end_ns=b_end,
        b_consumer_ns=consumer,
        critical_path_duration_us=critical_path_duration_us,
        sequence_consistent=sequence_consistent,
    )


def pair_trace_records(
    records: Iterable[Mapping[str, Any]],
    operation_a: str,
    operation_b: str,
    pair_fields: Sequence[str] = DEFAULT_PAIR_FIELDS,
) -> list[RankPair]:
    """Pair A/B trace records by run, rank, semantic context, and launch order."""

    grouped: defaultdict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {operation_a: [], operation_b: []}
    )
    selected = 0
    for record in records:
        operation = _record_value(record, "operation")
        if operation not in (operation_a, operation_b):
            continue
        selected += 1
        run_id = _string(record, "run_id")
        rank = _integer(record, "rank")
        context = _context(record, pair_fields)
        declared_world_size = _record_value(record, "world_size")
        if declared_world_size is None:
            raise TraceFormatError("world_size must be present on every selected record")
        if (
            isinstance(declared_world_size, bool)
            or not isinstance(declared_world_size, int | float)
            or not float(declared_world_size).is_integer()
            or declared_world_size < 2
        ):
            raise TraceFormatError("world_size must be an integer of at least two")
        grouped[(run_id, rank, int(declared_world_size), context)][operation].append(record)
    if not selected:
        raise TraceFormatError(f"no records matched operations {operation_a!r} and {operation_b!r}")

    pairs = []
    for (run_id, rank, _, context), by_operation in grouped.items():
        records_a = sorted(by_operation[operation_a], key=_sort_key)
        records_b = sorted(by_operation[operation_b], key=_sort_key)
        if len(records_a) != len(records_b) or not records_a:
            raise TraceFormatError(
                f"run {run_id!r} rank {rank} context {dict(context)!r} has "
                f"{len(records_a)} {operation_a!r} records and {len(records_b)} {operation_b!r} records"
            )
        for ordinal, (record_a, record_b) in enumerate(zip(records_a, records_b, strict=True)):
            pairs.append(
                _pair_records(
                    record_a,
                    record_b,
                    context=context,
                    ordinal=ordinal,
                    operation_a=operation_a,
                    operation_b=operation_b,
                )
            )
    return pairs


def build_candidates(rank_pairs: Sequence[RankPair]) -> dict[tuple[Any, ...], list[Candidate]]:
    """Build complete global trials and group them by portable workload key."""

    trials_by_key: defaultdict[tuple[Any, ...], list[RankPair]] = defaultdict(list)
    ranks_by_run: defaultdict[tuple[Any, ...], set[int]] = defaultdict(set)
    ranks_by_workload: defaultdict[tuple[Any, ...], set[int]] = defaultdict(set)
    declared_sizes_by_workload: defaultdict[tuple[Any, ...], set[int]] = defaultdict(set)
    for pair in rank_pairs:
        grouping_key = (pair.workload_base, pair.declared_world_size)
        candidate_key = (pair.policy, pair.requested_offset_us)
        run_key = (pair.run_id, grouping_key, candidate_key)
        ranks_by_run[run_key].add(pair.rank)
        ranks_by_workload[grouping_key].add(pair.rank)
        if pair.declared_world_size is not None:
            declared_sizes_by_workload[grouping_key].add(pair.declared_world_size)
        trials_by_key[(run_key, pair.context, pair.ordinal)].append(pair)

    expected_by_workload: dict[tuple[Any, ...], tuple[int, ...]] = {}
    for grouping_key, observed_ranks in ranks_by_workload.items():
        declared_sizes = declared_sizes_by_workload[grouping_key]
        if len(declared_sizes) > 1:
            raise TraceFormatError(f"workload has conflicting declared world sizes: {sorted(declared_sizes)}")
        world_size = next(iter(declared_sizes)) if declared_sizes else max(observed_ranks) + 1
        expected_by_workload[grouping_key] = tuple(range(world_size))

    for run_key, observed_ranks in ranks_by_run.items():
        expected = expected_by_workload[run_key[1]]
        if tuple(sorted(observed_ranks)) != expected:
            raise TraceFormatError(f"run {run_key[0]!r} has ranks {tuple(sorted(observed_ranks))}, expected {expected}")

    candidate_trials: defaultdict[tuple[Any, ...], list[Trial]] = defaultdict(list)
    for (run_key, context, ordinal), pairs in trials_by_key.items():
        run_id, grouping_key, candidate_key = run_key
        workload_base, _ = grouping_key
        expected = expected_by_workload[grouping_key]
        observed = tuple(sorted(pair.rank for pair in pairs))
        if observed != expected:
            raise TraceFormatError(
                f"run {run_id!r} context {dict(context)!r} has ranks {observed}, expected {expected}"
            )
        workload_key = (*workload_base, len(expected))
        candidate_trials[(workload_key, candidate_key)].append(
            Trial(run_id, context, ordinal, tuple(sorted(pairs, key=lambda pair: pair.rank)))
        )

    by_workload: defaultdict[tuple[Any, ...], list[Candidate]] = defaultdict(list)
    for (workload_key, candidate_key), trials in candidate_trials.items():
        policy, requested_offset_us = candidate_key
        by_workload[workload_key].append(
            Candidate(policy, requested_offset_us, tuple(sorted(trials, key=_trial_sort_key)))
        )
    return dict(by_workload)


def _trial_sort_key(trial: Trial) -> tuple[str, str, int]:
    return trial.run_id, json.dumps(trial.context, sort_keys=True), trial.ordinal


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def summarize_candidate(candidate: Candidate) -> dict[str, Any]:
    """Summarize one candidate without discarding tail or safety signals."""

    rank_pairs = candidate.rank_pairs
    critical_path_sources = {trial.critical_path_source for trial in candidate.trials}
    if len(critical_path_sources) != 1:
        raise TraceFormatError("critical_path_duration_us is missing on only some trials")
    critical_paths = [trial.critical_path_us for trial in candidate.trials]
    pair_completions = [trial.pair_completion_us for trial in candidate.trials]
    realized_offsets = [pair.realized_offset_us for pair in rank_pairs]
    consumer_slacks = [pair.consumer_slack_us for pair in rank_pairs]
    consumer_waits = [max(0.0, -slack) for slack in consumer_slacks]
    rank_skews = [trial.rank_skew_us for trial in candidate.trials]
    realized_p50 = percentile(realized_offsets, 50)
    requested = candidate.requested_offset_us
    return {
        "policy": candidate.policy,
        "requested_offset_us": requested,
        "trial_count": len(candidate.trials),
        "rank_observation_count": len(rank_pairs),
        "critical_path_source": next(iter(critical_path_sources)),
        "critical_path_us_p50": _rounded(percentile(critical_paths, 50)),
        "critical_path_us_p95": _rounded(percentile(critical_paths, 95)),
        "critical_path_us_p99": _rounded(percentile(critical_paths, 99)),
        "pair_completion_us_p50": _rounded(percentile(pair_completions, 50)),
        "pair_completion_us_p95": _rounded(percentile(pair_completions, 95)),
        "realized_gpu_offset_us_p50": _rounded(realized_p50),
        "realized_gpu_offset_us_p05": _rounded(percentile(realized_offsets, 5)),
        "realized_gpu_offset_us_p95": _rounded(percentile(realized_offsets, 95)),
        "realized_offset_error_us_p50": (_rounded(abs(realized_p50 - requested)) if requested is not None else None),
        "consumer_slack_us_p05": _rounded(percentile(consumer_slacks, 5)),
        "consumer_slack_us_p50": _rounded(percentile(consumer_slacks, 50)),
        "consumer_wait_us_p95": _rounded(percentile(consumer_waits, 95)),
        "rank_skew_us_p95": _rounded(percentile(rank_skews, 95)),
        "clock_sync_error_bound_us": _rounded(max(pair.clock_sync_error_bound_us for pair in rank_pairs)),
        "sequence_consistent": all(trial.sequence_consistent for trial in candidate.trials),
    }


def _bootstrap_improvement_interval(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed_material: str,
) -> tuple[float, float] | None:
    if len(baseline) < 2 or len(candidate) < 2 or resamples <= 0:
        return None
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    generator = random.Random(seed)
    differences = []
    for _ in range(resamples):
        baseline_sample = [baseline[generator.randrange(len(baseline))] for _ in baseline]
        candidate_sample = [candidate[generator.randrange(len(candidate))] for _ in candidate]
        differences.append(percentile(baseline_sample, 95) - percentile(candidate_sample, 95))
    tail_percent = (1 - confidence) * 50
    return percentile(differences, tail_percent), percentile(differences, 100 - tail_percent)


def _paired_improvement_interval(
    baseline: Sequence[Trial],
    candidate: Sequence[Trial],
    *,
    confidence: float,
    resamples: int,
    seed_material: str,
) -> tuple[tuple[float, float] | None, str]:
    baseline_by_context: defaultdict[tuple[Any, ...], list[Trial]] = defaultdict(list)
    candidate_by_context: defaultdict[tuple[Any, ...], list[Trial]] = defaultdict(list)
    for trial in baseline:
        baseline_by_context[(trial.context, trial.ordinal)].append(trial)
    for trial in candidate:
        candidate_by_context[(trial.context, trial.ordinal)].append(trial)

    contexts = set(baseline_by_context) & set(candidate_by_context)
    unambiguous = (
        contexts
        and contexts == set(baseline_by_context) == set(candidate_by_context)
        and all(len(baseline_by_context[context]) == len(candidate_by_context[context]) == 1 for context in contexts)
    )
    if unambiguous and len(contexts) >= 2 and resamples > 0:
        differences = [
            baseline_by_context[context][0].critical_path_us - candidate_by_context[context][0].critical_path_us
            for context in sorted(contexts, key=repr)
        ]
        seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
        generator = random.Random(seed)
        estimates = []
        for _ in range(resamples):
            sample = [differences[generator.randrange(len(differences))] for _ in differences]
            estimates.append(sum(sample) / len(sample))
        tail_percent = (1 - confidence) * 50
        return (
            percentile(estimates, tail_percent),
            percentile(estimates, 100 - tail_percent),
        ), "paired_mean"

    interval = _bootstrap_improvement_interval(
        [trial.critical_path_us for trial in baseline],
        [trial.critical_path_us for trial in candidate],
        confidence=confidence,
        resamples=resamples,
        seed_material=seed_material,
    )
    return interval, "independent_p95"


def _candidate_sort_key(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["critical_path_us_p95"]),
        float(summary["consumer_wait_us_p95"]),
        float(summary["rank_skew_us_p95"]),
    )


def _baseline_index(summaries: Sequence[Mapping[str, Any]]) -> int:
    explicit = [index for index, item in enumerate(summaries) if item["policy"] in BASELINE_POLICIES]
    candidates = explicit or list(range(len(summaries)))
    return min(
        candidates,
        key=lambda index: (
            abs(float(summaries[index]["realized_gpu_offset_us_p50"])),
            _candidate_sort_key(summaries[index]),
        ),
    )


def _workload_dict(workload: tuple[Any, ...]) -> dict[str, Any]:
    (
        framework,
        topology_class,
        timestamp_domain,
        gpu_timestamp_semantics,
        operation_a,
        operation_b,
        message_bytes_a,
        message_bytes_b,
        transport_a,
        transport_b,
        world_size,
    ) = workload
    return {
        "framework": framework,
        "topology_class": topology_class,
        "timestamp_domain": timestamp_domain,
        "gpu_timestamp_semantics": gpu_timestamp_semantics,
        "operation_a": operation_a,
        "operation_b": operation_b,
        "message_bytes_a": message_bytes_a,
        "message_bytes_b": message_bytes_b,
        "transport_a": transport_a,
        "transport_b": transport_b,
        "world_size": world_size,
    }


def _refinement_points(
    summaries: Sequence[Mapping[str, Any]],
    focus_index: int,
    minimum_slack_us: float,
) -> list[dict[str, float]]:
    observed = sorted(
        (float(item["requested_offset_us"]), index)
        for index, item in enumerate(summaries)
        if item["requested_offset_us"] is not None
    )
    position = next((position for position, (_, index) in enumerate(observed) if index == focus_index), None)
    if position is None:
        return []
    proposals = []
    for neighbor_position in (position - 1, position + 1):
        if not 0 <= neighbor_position < len(observed):
            continue
        request_a, index_a = observed[position]
        request_b, index_b = observed[neighbor_position]
        midpoint = (request_a + request_b) / 2
        if midpoint in {request for request, _ in observed}:
            continue
        slack_a = float(summaries[index_a]["consumer_slack_us_p05"])
        slack_b = float(summaries[index_b]["consumer_slack_us_p05"])
        predicted_slack = (slack_a + slack_b) / 2
        if predicted_slack >= minimum_slack_us:
            realized_a = float(summaries[index_a]["realized_gpu_offset_us_p50"])
            realized_b = float(summaries[index_b]["realized_gpu_offset_us_p50"])
            proposals.append(
                {
                    "requested_offset_us": midpoint,
                    "predicted_realized_gpu_offset_us": (realized_a + realized_b) / 2,
                    "predicted_consumer_slack_us_p05": predicted_slack,
                }
            )
    return sorted(proposals, key=lambda proposal: proposal["requested_offset_us"])


def tune_workload(
    workload: tuple[Any, ...],
    candidates: Sequence[Candidate],
    *,
    confidence: float = 0.95,
    bootstrap_resamples: int = 2000,
    min_trials: int = 2,
    max_clock_sync_error_us: float = 50.0,
) -> dict[str, Any]:
    """Choose a safe policy and data-derived refinement offsets for one workload."""

    summaries = [summarize_candidate(candidate) for candidate in candidates]
    critical_path_sources = {summary["critical_path_source"] for summary in summaries}
    if len(critical_path_sources) != 1:
        raise TraceFormatError("all candidates for a workload must use the same critical-path source")
    baseline_index = _baseline_index(summaries)
    baseline = summaries[baseline_index]
    minimum_slack_us = min(0.0, float(baseline["consumer_slack_us_p05"]))
    baseline_rank_skew_us = float(baseline["rank_skew_us_p95"])

    eligible = []
    for index, (candidate, summary) in enumerate(zip(candidates, summaries, strict=True)):
        reasons = []
        if candidate.rank_pairs[0].gpu_timestamp_semantics != "kernel-observed":
            reasons.append("kernel_observed_gpu_timestamps_required")
        if float(summary["clock_sync_error_bound_us"]) > max_clock_sync_error_us:
            reasons.append("clock_sync_error_bound_exceeded")
        if summary["trial_count"] < min_trials:
            reasons.append("insufficient_trials")
        if not summary["sequence_consistent"]:
            reasons.append("communicator_sequence_divergence")
        if float(summary["consumer_slack_us_p05"]) < minimum_slack_us:
            reasons.append("consumer_deadline_regression")
        rank_skew_growth_us = max(0.0, float(summary["rank_skew_us_p95"]) - baseline_rank_skew_us)
        rank_skew_headroom_us = max(0.0, float(summary["consumer_slack_us_p05"]) - minimum_slack_us)
        summary["rank_skew_growth_us"] = _rounded(rank_skew_growth_us)
        summary["rank_skew_slack_headroom_us"] = _rounded(rank_skew_headroom_us)
        if rank_skew_growth_us > rank_skew_headroom_us:
            reasons.append("rank_skew_exceeds_slack_headroom")
        requested = summary["requested_offset_us"]
        realized = float(summary["realized_gpu_offset_us_p50"])
        if requested not in (None, 0) and math.copysign(1, float(requested)) != math.copysign(1, realized):
            reasons.append("realized_offset_direction_mismatch")
        elif requested is not None and requested > 0 and summary["realized_gpu_offset_us_p05"] <= 0:
            reasons.append("realized_offset_direction_unstable")
        elif requested is not None and requested < 0 and summary["realized_gpu_offset_us_p95"] >= 0:
            reasons.append("realized_offset_direction_unstable")

        interval = (0.0, 0.0)
        confidence_method = "identity"
        if index != baseline_index:
            interval, confidence_method = _paired_improvement_interval(
                candidates[baseline_index].trials,
                candidate.trials,
                confidence=confidence,
                resamples=bootstrap_resamples,
                seed_material=f"{workload!r}/{candidate.policy}/{candidate.requested_offset_us}",
            )
        summary["critical_path_improvement_ci_us"] = (
            [_rounded(interval[0]), _rounded(interval[1])] if interval is not None else None
        )
        summary["confidence_method"] = confidence_method
        summary["eligible"] = not reasons
        summary["rejection_reasons"] = reasons
        if not reasons:
            eligible.append(index)

    improving = []
    for index in eligible:
        interval = summaries[index]["critical_path_improvement_ci_us"]
        if index != baseline_index and interval is not None and interval[0] > 0:
            improving.append(index)
    selected_index = (
        min(improving, key=lambda index: _candidate_sort_key(summaries[index])) if improving else baseline_index
    )

    evidence_ready = baseline["trial_count"] >= min_trials and baseline_index in eligible
    if not evidence_ready:
        decision = "insufficient_evidence"
    elif selected_index == baseline_index:
        decision = "keep_baseline"
    else:
        decision = "switch_policy"

    if evidence_ready:
        focus_index = min(eligible, key=lambda index: _candidate_sort_key(summaries[index]))
        next_candidates = _refinement_points(summaries, focus_index, minimum_slack_us)
    else:
        next_candidates = []
    return {
        "workload_key": _workload_dict(workload),
        "decision": decision,
        "baseline_candidate": {
            "policy": baseline["policy"],
            "requested_offset_us": baseline["requested_offset_us"],
        },
        "recommended_candidate": {
            "policy": summaries[selected_index]["policy"],
            "requested_offset_us": summaries[selected_index]["requested_offset_us"],
            "realized_gpu_offset_us_p50": summaries[selected_index]["realized_gpu_offset_us_p50"],
        },
        "consumer_deadline_guard_us": _rounded(minimum_slack_us),
        "refinement": {
            "next_requested_offsets_us": [candidate["requested_offset_us"] for candidate in next_candidates],
            "next_candidates": next_candidates,
            "method": "measured-interval midpoint bounded by consumer slack; no extrapolation",
        },
        "candidates": sorted(
            summaries,
            key=lambda item: (
                item["requested_offset_us"] is None,
                0 if item["requested_offset_us"] is None else item["requested_offset_us"],
                item["policy"],
            ),
        ),
    }


def tune_traces(
    records: Iterable[Mapping[str, Any]],
    operation_a: str,
    operation_b: str,
    *,
    pair_fields: Sequence[str] = DEFAULT_PAIR_FIELDS,
    confidence: float = 0.95,
    bootstrap_resamples: int = 2000,
    min_trials: int = 2,
    max_clock_sync_error_us: float = 50.0,
) -> dict[str, Any]:
    """Build recommendations from framework-neutral semantic trace records."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples must be non-negative")
    if min_trials <= 0:
        raise ValueError("min_trials must be positive")
    if not math.isfinite(max_clock_sync_error_us) or max_clock_sync_error_us < 0:
        raise ValueError("max_clock_sync_error_us must be finite and non-negative")
    if not pair_fields:
        raise ValueError("at least one pair field is required")
    pairs = pair_trace_records(records, operation_a, operation_b, pair_fields)
    by_workload = build_candidates(pairs)
    recommendations = [
        tune_workload(
            workload,
            candidates,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            min_trials=min_trials,
            max_clock_sync_error_us=max_clock_sync_error_us,
        )
        for workload, candidates in sorted(by_workload.items(), key=lambda item: repr(item[0]))
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "operation_a": operation_a,
        "operation_b": operation_b,
        "pair_fields": list(pair_fields),
        "max_clock_sync_error_us": max_clock_sync_error_us,
        "recommendations": recommendations,
    }


def load_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load JSON objects from one or more JSONL trace shards."""

    records = []
    for path in paths:
        with path.open(encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TraceFormatError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise TraceFormatError(f"{path}:{line_number}: each line must contain a JSON object")
                records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-jsonl", nargs="+", type=Path, required=True, help="all rank shards for all candidates")
    parser.add_argument("--operation-a", required=True)
    parser.add_argument("--operation-b", required=True)
    parser.add_argument(
        "--pair-by",
        nargs="+",
        default=list(DEFAULT_PAIR_FIELDS),
        help="scalar trace fields identifying one semantic A/B pair",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--min-trials", type=int, default=2)
    parser.add_argument(
        "--max-clock-sync-error-us",
        type=float,
        default=50.0,
        help="largest measured cross-rank clock error allowed for a policy recommendation",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = tune_traces(
            load_jsonl(args.trace_jsonl),
            args.operation_a,
            args.operation_b,
            pair_fields=args.pair_by,
            confidence=args.confidence,
            bootstrap_resamples=args.bootstrap_resamples,
            min_trials=args.min_trials,
            max_clock_sync_error_us=args.max_clock_sync_error_us,
        )
    except (OSError, TraceFormatError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload["trace_files"] = [str(path) for path in args.trace_jsonl]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
