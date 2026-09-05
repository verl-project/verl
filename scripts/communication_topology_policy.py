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
"""Fingerprint communication topology and select only exact policy cells.

This module intentionally does not configure NIC traffic classes, rails, or
routing.  It creates a stable compatibility boundary for measured policies so
that a recommendation from one topology is never extrapolated to another.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

TOPOLOGY_SCHEMA_VERSION = 1
_FINGERPRINT_FIELDS = {
    "schema_version",
    "scope",
    "local_fabric",
    "node_count",
    "world_size",
    "rank_groups",
    "ranks_per_node",
    "accelerator_models_by_rank",
    "topology_signature",
}


class TopologyEvidenceError(ValueError):
    """Raised when trace records cannot form a safe topology fingerprint."""


class IncompatibleTopologyError(ValueError):
    """Raised when no measured policy cell exactly matches a target topology."""


@dataclasses.dataclass(frozen=True)
class TopologyCompatibility:
    """Exact compatibility result between two topology fingerprints."""

    compatible: bool
    reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class TopologyFingerprint:
    """Portable topology identity for one distributed run.

    Hostnames are deliberately excluded from the identity. ``rank_groups``
    preserves placement while allowing the same layout on different hosts.
    ``topology_signature`` is an opaque inventory hash, required for multi-node
    traces because a node count alone cannot identify an inter-node topology.
    """

    scope: str
    local_fabric: str
    node_count: int
    world_size: int
    rank_groups: tuple[tuple[int, ...], ...]
    accelerator_models_by_rank: tuple[str, ...]
    topology_signature: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "schema_version": TOPOLOGY_SCHEMA_VERSION,
            "scope": self.scope,
            "local_fabric": self.local_fabric,
            "node_count": self.node_count,
            "world_size": self.world_size,
            "rank_groups": [list(group) for group in self.rank_groups],
            "ranks_per_node": [len(group) for group in self.rank_groups],
            "accelerator_models_by_rank": list(self.accelerator_models_by_rank),
            "topology_signature": self.topology_signature,
        }

    @property
    def canonical_json(self) -> str:
        """Return a stable serialization suitable for hashing and storage."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @property
    def topology_class(self) -> str:
        """Return a coarse human-readable class without weakening identity."""

        if self.scope == "multi_node":
            return "multi-node"
        return f"single-node-{self.local_fabric}"

    @property
    def cell_id(self) -> str:
        """Return the content-addressed topology cell identifier."""

        return f"topology-v{TOPOLOGY_SCHEMA_VERSION}:{hashlib.sha256(self.canonical_json.encode()).hexdigest()}"

    def compare(self, other: TopologyFingerprint) -> TopologyCompatibility:
        """Compare every scheduling-relevant field without fuzzy matching."""

        reasons = []
        for field in (
            "scope",
            "local_fabric",
            "node_count",
            "world_size",
            "rank_groups",
            "accelerator_models_by_rank",
            "topology_signature",
        ):
            if getattr(self, field) != getattr(other, field):
                reasons.append(field)
        return TopologyCompatibility(not reasons, tuple(reasons))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TopologyFingerprint:
        """Validate and deserialize a stored fingerprint."""

        if value.get("schema_version") != TOPOLOGY_SCHEMA_VERSION:
            raise TopologyEvidenceError("unsupported topology fingerprint schema_version")
        unexpected = set(value) - _FINGERPRINT_FIELDS
        if unexpected:
            raise TopologyEvidenceError(f"unexpected topology fingerprint fields: {sorted(unexpected)}")
        try:
            scope = value["scope"]
            local_fabric = value["local_fabric"]
            node_count = value["node_count"]
            world_size = value["world_size"]
            raw_groups = value["rank_groups"]
            raw_models = value["accelerator_models_by_rank"]
            if not isinstance(scope, str) or not isinstance(local_fabric, str):
                raise TypeError
            if isinstance(node_count, bool) or not isinstance(node_count, int):
                raise TypeError
            if isinstance(world_size, bool) or not isinstance(world_size, int):
                raise TypeError
            if not isinstance(raw_groups, list) or not all(isinstance(group, list) for group in raw_groups):
                raise TypeError
            if not all(not isinstance(rank, bool) and isinstance(rank, int) for group in raw_groups for rank in group):
                raise TypeError
            if not isinstance(raw_models, list) or not all(isinstance(model, str) for model in raw_models):
                raise TypeError
            rank_groups = tuple(tuple(group) for group in raw_groups)
            models = tuple(raw_models)
            signature = value.get("topology_signature")
            if signature is not None and not isinstance(signature, str):
                raise TypeError
            fingerprint = cls(
                scope=scope,
                local_fabric=local_fabric,
                node_count=node_count,
                world_size=world_size,
                rank_groups=rank_groups,
                accelerator_models_by_rank=models,
                topology_signature=signature,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TopologyEvidenceError("malformed topology fingerprint") from exc
        fingerprint.validate()
        raw_ranks_per_node = value.get("ranks_per_node")
        if not isinstance(raw_ranks_per_node, list) or not all(
            not isinstance(count, bool) and isinstance(count, int) for count in raw_ranks_per_node
        ):
            raise TopologyEvidenceError("ranks_per_node must contain integer counts")
        if raw_ranks_per_node != [len(group) for group in fingerprint.rank_groups]:
            raise TopologyEvidenceError("ranks_per_node disagrees with rank_groups")
        return fingerprint

    def validate(self) -> None:
        """Validate internal topology invariants."""

        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int):
            raise TopologyEvidenceError("node_count must be an integer")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise TopologyEvidenceError("world_size must be an integer")
        if self.scope not in {"single_node", "multi_node"}:
            raise TopologyEvidenceError("scope must be single_node or multi_node")
        if self.local_fabric not in {"pcie", "nvlink", "unknown"}:
            raise TopologyEvidenceError("local_fabric must be pcie, nvlink, or unknown")
        if self.node_count <= 0 or self.world_size < 2:
            raise TopologyEvidenceError("invalid node_count or world_size")
        if len(self.rank_groups) != self.node_count:
            raise TopologyEvidenceError("rank_groups does not match node_count")
        if any(not group for group in self.rank_groups):
            raise TopologyEvidenceError("rank_groups cannot contain an empty node")
        if any(isinstance(rank, bool) or not isinstance(rank, int) for group in self.rank_groups for rank in group):
            raise TopologyEvidenceError("rank_groups must contain integer ranks")
        flattened = tuple(sorted(rank for group in self.rank_groups for rank in group))
        if flattened != tuple(range(self.world_size)):
            raise TopologyEvidenceError("rank_groups must contain every rank exactly once")
        if len(self.accelerator_models_by_rank) != self.world_size:
            raise TopologyEvidenceError("accelerator model vector does not match world_size")
        if any(not isinstance(model, str) or not model for model in self.accelerator_models_by_rank):
            raise TopologyEvidenceError("accelerator models must be non-empty strings")
        if self.scope == "single_node" and self.node_count != 1:
            raise TopologyEvidenceError("single_node scope must contain exactly one node")
        if self.scope == "multi_node" and self.node_count < 2:
            raise TopologyEvidenceError("multi_node scope must contain at least two nodes")
        if self.scope == "multi_node" and not self.topology_signature:
            raise TopologyEvidenceError("multi-node traces require an opaque topology_signature")
        if self.topology_signature is not None and (
            not isinstance(self.topology_signature, str) or not self.topology_signature.strip()
        ):
            raise TopologyEvidenceError("topology_signature must be a non-empty string")


def _value(record: Mapping[str, Any], name: str) -> Any:
    value = record.get(name)
    if value is not None:
        return value
    metadata = record.get("metadata")
    return metadata.get(name) if isinstance(metadata, Mapping) else None


def topology_run_key(record: Mapping[str, Any]) -> tuple[str, int | None]:
    """Return a run key that keeps explicitly different world sizes separate."""

    run_id = _value(record, "run_id")
    if not isinstance(run_id, str) or not run_id:
        raise TopologyEvidenceError("run_id must be a non-empty string")
    declared_world_size = _value(record, "world_size")
    if declared_world_size is None:
        return run_id, None
    if (
        isinstance(declared_world_size, bool)
        or not isinstance(declared_world_size, int | float)
        or not math.isfinite(declared_world_size)
        or not float(declared_world_size).is_integer()
    ):
        raise TopologyEvidenceError("world_size must be an integer")
    return run_id, int(declared_world_size)


def _rank(record: Mapping[str, Any]) -> int:
    value = _value(record, "rank")
    if isinstance(value, bool) or not isinstance(value, int | float) or not float(value).is_integer():
        raise TopologyEvidenceError("rank must be an integer")
    rank = int(value)
    if rank < 0:
        raise TopologyEvidenceError("rank must be non-negative")
    return rank


def _consistent_scalar(records: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Any:
    values = set()
    for record in records:
        for name in names:
            value = _value(record, name)
            if value is not None:
                if isinstance(value, dict | list):
                    raise TopologyEvidenceError(f"{name} must be scalar")
                values.add(value)
                break
    if len(values) > 1:
        raise TopologyEvidenceError(f"conflicting {names[0]} values: {sorted(values, key=str)!r}")
    return next(iter(values)) if values else None


def _scope_and_fabric(topology_class: str | None, node_count: int) -> tuple[str, str]:
    normalized = (topology_class or "unknown").strip().lower().replace("_", "-")
    declared_scope = None
    if "multi-node" in normalized or "multinode" in normalized:
        declared_scope = "multi_node"
    elif "single-node" in normalized or "singlenode" in normalized:
        declared_scope = "single_node"
    inferred_scope = "single_node" if node_count == 1 else "multi_node"
    if declared_scope is not None and declared_scope != inferred_scope:
        raise TopologyEvidenceError("declared topology class disagrees with observed node placement")
    if "nvlink" in normalized:
        local_fabric = "nvlink"
    elif "pcie" in normalized or "pci-e" in normalized:
        local_fabric = "pcie"
    else:
        local_fabric = "unknown"
    return inferred_scope, local_fabric


def fingerprint_trace_run(records: Sequence[Mapping[str, Any]]) -> TopologyFingerprint:
    """Build one fingerprint from all selected rank records of a run."""

    if not records:
        raise TopologyEvidenceError("cannot fingerprint an empty trace run")
    _consistent_scalar(records, ("run_id",))
    ranks = sorted({_rank(record) for record in records})
    declared_sizes = {key[1] for key in (topology_run_key(record) for record in records) if key[1] is not None}
    if len(declared_sizes) > 1:
        raise TopologyEvidenceError("trace run has conflicting world_size values")
    world_size = next(iter(declared_sizes)) if declared_sizes else max(ranks) + 1
    if world_size < 2 or ranks != list(range(world_size)):
        raise TopologyEvidenceError(f"trace ranks {tuple(ranks)} do not cover world_size {world_size}")

    records_by_rank: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_rank[_rank(record)].append(record)

    topology_class = _consistent_scalar(records, ("topology_class",))
    if topology_class is not None and not isinstance(topology_class, str):
        raise TopologyEvidenceError("topology_class must be a string")

    node_by_rank = []
    for rank in range(world_size):
        node = _consistent_scalar(records_by_rank[rank], ("node_id", "node", "hostname"))
        node_by_rank.append(str(node) if node is not None else "__unspecified_node__")
    explicit_nodes = {node for node in node_by_rank if node != "__unspecified_node__"}
    if explicit_nodes and "__unspecified_node__" in node_by_rank:
        raise TopologyEvidenceError("node identity is missing on only some ranks")
    if not explicit_nodes:
        if isinstance(topology_class, str) and ("multi-node" in topology_class or "multinode" in topology_class):
            raise TopologyEvidenceError("multi-node traces require hostname or node_id on every rank")
        node_by_rank = ["__single_node__"] * world_size

    ranks_by_node: defaultdict[str, list[int]] = defaultdict(list)
    for rank, node in enumerate(node_by_rank):
        ranks_by_node[node].append(rank)
    rank_groups = tuple(sorted((tuple(group) for group in ranks_by_node.values()), key=lambda group: group[0]))
    scope, local_fabric = _scope_and_fabric(topology_class, len(rank_groups))

    models = []
    for rank in range(world_size):
        model = _consistent_scalar(records_by_rank[rank], ("accelerator_model", "device_name", "device"))
        models.append(str(model) if model is not None else "unknown")
    signatures_by_rank = [
        _consistent_scalar(records_by_rank[rank], ("topology_signature",)) for rank in range(world_size)
    ]
    present_signatures = {signature for signature in signatures_by_rank if signature is not None}
    if present_signatures and any(signature is None for signature in signatures_by_rank):
        raise TopologyEvidenceError("topology_signature is missing on only some ranks")
    if len(present_signatures) > 1:
        raise TopologyEvidenceError("conflicting topology_signature values across ranks")
    topology_signature = next(iter(present_signatures)) if present_signatures else None
    if topology_signature is not None and not isinstance(topology_signature, str):
        raise TopologyEvidenceError("topology_signature must be a string")

    fingerprint = TopologyFingerprint(
        scope=scope,
        local_fabric=local_fabric,
        node_count=len(rank_groups),
        world_size=world_size,
        rank_groups=rank_groups,
        accelerator_models_by_rank=tuple(models),
        topology_signature=topology_signature,
    )
    fingerprint.validate()
    return fingerprint


def fingerprint_trace_runs(
    records: Iterable[Mapping[str, Any]], operations: Sequence[str] | None = None
) -> dict[tuple[str, int | None], TopologyFingerprint]:
    """Fingerprint every run represented in a trace collection."""

    selected_operations = set(operations) if operations is not None else None
    by_run: defaultdict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if selected_operations is not None and _value(record, "operation") not in selected_operations:
            continue
        by_run[topology_run_key(record)].append(record)
    if not by_run:
        raise TopologyEvidenceError("no trace records matched the topology request")
    fingerprints = {}
    for run_key, run_records in by_run.items():
        try:
            fingerprints[run_key] = fingerprint_trace_run(run_records)
        except TopologyEvidenceError as exc:
            raise TopologyEvidenceError(f"run {run_key[0]!r}: {exc}") from exc
    return fingerprints


def select_compatible_policy_cells(
    recommendations: Sequence[Mapping[str, Any]], target: TopologyFingerprint
) -> list[Mapping[str, Any]]:
    """Return exact topology matches and reject every cross-topology fallback."""

    target.validate()
    matches = []
    mismatch_counts: Counter[str] = Counter()
    for recommendation in recommendations:
        workload_key = recommendation.get("workload_key")
        if not isinstance(workload_key, Mapping):
            raise TopologyEvidenceError("recommendation is missing workload_key")
        stored = workload_key.get("topology_fingerprint")
        if not isinstance(stored, Mapping):
            raise TopologyEvidenceError("recommendation is missing topology_fingerprint")
        fingerprint = TopologyFingerprint.from_dict(stored)
        if workload_key.get("topology_cell_id") != fingerprint.cell_id:
            raise TopologyEvidenceError("stored topology_cell_id does not match its fingerprint")
        compatibility = fingerprint.compare(target)
        if compatibility.compatible:
            matches.append(recommendation)
        else:
            mismatch_counts.update(compatibility.reasons)
    if matches:
        return matches
    mismatch_summary = ", ".join(f"{field} ({count})" for field, count in sorted(mismatch_counts.items()))
    raise IncompatibleTopologyError(
        f"no exact policy cell for {target.cell_id}; mismatched fields: {mismatch_summary or 'no measured cells'}"
    )


def load_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load topology evidence from JSONL trace shards."""

    records = []
    for path in paths:
        with path.open(encoding="utf-8") as trace_file:
            for line_number, line in enumerate(trace_file, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TopologyEvidenceError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(record, dict):
                    raise TopologyEvidenceError(f"{path}:{line_number}: each line must be a JSON object")
                records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--target-trace-jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy_payload = json.loads(args.policy_json.read_text(encoding="utf-8"))
        if not isinstance(policy_payload, dict) or not isinstance(policy_payload.get("recommendations"), list):
            raise TopologyEvidenceError("policy JSON must contain a recommendations list")
        fingerprints = fingerprint_trace_runs(load_jsonl(args.target_trace_jsonl))
        unique_targets = {fingerprint.cell_id: fingerprint for fingerprint in fingerprints.values()}
        if len(unique_targets) != 1:
            raise TopologyEvidenceError("target traces contain more than one topology cell")
        target = next(iter(unique_targets.values()))
        matches = select_compatible_policy_cells(policy_payload["recommendations"], target)
        output = {
            "schema_version": TOPOLOGY_SCHEMA_VERSION,
            "topology_cell_id": target.cell_id,
            "topology_fingerprint": target.to_dict(),
            "recommendations": matches,
        }
    except (OSError, json.JSONDecodeError, TopologyEvidenceError, IncompatibleTopologyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
