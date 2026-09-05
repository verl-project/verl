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
"""Gate measured communication policies on exact NIC and telemetry evidence.

The module is an offline compatibility and eligibility layer. It never changes
NIC traffic classes, rail routing, host QoS, or transport configuration. A
caller must provide a complete operator-authored capability inventory and
telemetry that observed every selected rank. Unknown or incompatible evidence
fails closed instead of falling back to a nearby rail or traffic class.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts.communication_topology_policy import (
        TopologyEvidenceError,
        TopologyFingerprint,
        fingerprint_trace_run,
        load_jsonl,
        topology_run_key,
    )
else:
    from communication_topology_policy import (
        TopologyEvidenceError,
        TopologyFingerprint,
        fingerprint_trace_run,
        load_jsonl,
        topology_run_key,
    )

NIC_CAPABILITY_SCHEMA_VERSION = 1
NETWORK_TELEMETRY_SCHEMA_VERSION = 1
NETWORK_POLICY_SCHEMA_VERSION = 1
SUPPORTED_NETWORK_FABRICS = frozenset({"infiniband", "roce"})
_UNKNOWN_TOKENS = frozenset({"", "auto", "n/a", "none", "unknown", "unspecified"})
_RAIL_FIELDS = {"rail_id", "traffic_classes"}
_RANK_CAPABILITY_FIELDS = {"rank", "rails"}
_NIC_CAPABILITY_FIELDS = {
    "schema_version",
    "topology_cell_id",
    "topology_fingerprint",
    "network_fabric",
    "inventory_signature",
    "rank_capabilities",
}
_TELEMETRY_FIELDS = {
    "schema_version",
    "topology_cell_id",
    "nic_capability_id",
    "nic_capability_fingerprint",
    "network_fabric",
    "telemetry_source",
    "telemetry_schema_signature",
}
_ASSIGNMENT_FIELDS = {"operation", "rank", "rail_id", "traffic_class"}
_ELIGIBILITY_FIELDS = {
    "schema_version",
    "eligibility_id",
    "network_telemetry_cell_id",
    "network_telemetry_fingerprint",
    "required_assignments",
}


class NetworkEvidenceError(ValueError):
    """Raised when NIC capability or telemetry evidence is incomplete."""


class IneligibleNetworkPolicyError(ValueError):
    """Raised when no measured network policy is eligible for a target."""


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unexpected = set(value) - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise NetworkEvidenceError(f"{label} fields are invalid: {', '.join(details)}")


def _require_schema_version(value: Mapping[str, Any], expected: int, label: str) -> None:
    version = value.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != expected:
        raise NetworkEvidenceError(f"unsupported {label} schema_version")


def _known_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value != value.strip() or value.casefold() in _UNKNOWN_TOKENS:
        raise NetworkEvidenceError(f"{label} must be an explicit, non-unknown string")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NetworkEvidenceError(f"{label} must be an integer")
    return value


@dataclasses.dataclass(frozen=True)
class RailCapability:
    """Traffic classes explicitly supported on one logical rail."""

    rail_id: str
    traffic_classes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "rail_id": self.rail_id,
            "traffic_classes": list(self.traffic_classes),
        }

    def validate(self) -> None:
        """Validate that no rail or traffic-class capability is implicit."""

        _known_string(self.rail_id, "rail_id")
        if not self.traffic_classes:
            raise NetworkEvidenceError(f"rail {self.rail_id!r} has no explicit traffic classes")
        if tuple(sorted(self.traffic_classes)) != self.traffic_classes:
            raise NetworkEvidenceError("traffic_classes must be in canonical sorted order")
        if len(set(self.traffic_classes)) != len(self.traffic_classes):
            raise NetworkEvidenceError(f"rail {self.rail_id!r} contains duplicate traffic classes")
        for traffic_class in self.traffic_classes:
            _known_string(traffic_class, "traffic_class")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RailCapability:
        """Validate and deserialize one rail capability."""

        _require_exact_fields(value, _RAIL_FIELDS, "rail capability")
        rail_id = _known_string(value["rail_id"], "rail_id")
        raw_traffic_classes = value["traffic_classes"]
        if not isinstance(raw_traffic_classes, list):
            raise NetworkEvidenceError("traffic_classes must be a list")
        traffic_classes = tuple(sorted(_known_string(item, "traffic_class") for item in raw_traffic_classes))
        capability = cls(rail_id=rail_id, traffic_classes=traffic_classes)
        capability.validate()
        return capability


@dataclasses.dataclass(frozen=True)
class RankNicCapability:
    """Logical rail inventory visible to one distributed rank."""

    rank: int
    rails: tuple[RailCapability, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "rank": self.rank,
            "rails": [rail.to_dict() for rail in self.rails],
        }

    def validate(self) -> None:
        """Validate one rank's complete logical rail inventory."""

        if _strict_int(self.rank, "rank") < 0:
            raise NetworkEvidenceError("rank must be non-negative")
        if not self.rails:
            raise NetworkEvidenceError(f"rank {self.rank} has no explicit rails")
        if tuple(sorted(self.rails, key=lambda rail: rail.rail_id)) != self.rails:
            raise NetworkEvidenceError("rails must be in canonical sorted order")
        rail_ids = [rail.rail_id for rail in self.rails]
        if len(set(rail_ids)) != len(rail_ids):
            raise NetworkEvidenceError(f"rank {self.rank} contains duplicate rail IDs")
        for rail in self.rails:
            rail.validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RankNicCapability:
        """Validate and deserialize one rank capability."""

        _require_exact_fields(value, _RANK_CAPABILITY_FIELDS, "rank capability")
        rank = _strict_int(value["rank"], "rank")
        raw_rails = value["rails"]
        if not isinstance(raw_rails, list):
            raise NetworkEvidenceError("rails must be a list")
        rails = []
        for raw_rail in raw_rails:
            if not isinstance(raw_rail, Mapping):
                raise NetworkEvidenceError("each rail capability must be an object")
            rails.append(RailCapability.from_dict(raw_rail))
        capability = cls(rank=rank, rails=tuple(sorted(rails, key=lambda rail: rail.rail_id)))
        capability.validate()
        return capability


@dataclasses.dataclass(frozen=True)
class NicCapabilityFingerprint:
    """Content-addressed NIC inventory tied to an exact topology cell."""

    topology: TopologyFingerprint
    network_fabric: str
    inventory_signature: str
    rank_capabilities: tuple[RankNicCapability, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "schema_version": NIC_CAPABILITY_SCHEMA_VERSION,
            "topology_cell_id": self.topology.cell_id,
            "topology_fingerprint": self.topology.to_dict(),
            "network_fabric": self.network_fabric,
            "inventory_signature": self.inventory_signature,
            "rank_capabilities": [capability.to_dict() for capability in self.rank_capabilities],
        }

    @property
    def canonical_json(self) -> str:
        """Return a stable serialization for hashing and storage."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @property
    def capability_id(self) -> str:
        """Return the content-addressed NIC capability identifier."""

        digest = hashlib.sha256(self.canonical_json.encode()).hexdigest()
        return f"nic-capability-v{NIC_CAPABILITY_SCHEMA_VERSION}:{digest}"

    def validate(self) -> None:
        """Validate topology binding and complete per-rank capabilities."""

        try:
            self.topology.validate()
        except TopologyEvidenceError as exc:
            raise NetworkEvidenceError(f"invalid topology fingerprint: {exc}") from exc
        if self.topology.scope != "multi_node":
            raise NetworkEvidenceError("NIC rail policies require a multi-node topology cell")
        if self.network_fabric not in SUPPORTED_NETWORK_FABRICS:
            raise NetworkEvidenceError(f"network_fabric must be one of {sorted(SUPPORTED_NETWORK_FABRICS)}")
        _known_string(self.inventory_signature, "inventory_signature")
        if tuple(sorted(self.rank_capabilities, key=lambda item: item.rank)) != self.rank_capabilities:
            raise NetworkEvidenceError("rank_capabilities must be in canonical sorted order")
        ranks = tuple(capability.rank for capability in self.rank_capabilities)
        if ranks != tuple(range(self.topology.world_size)):
            raise NetworkEvidenceError(
                f"rank_capabilities contain ranks {ranks}, expected {tuple(range(self.topology.world_size))}"
            )
        for capability in self.rank_capabilities:
            capability.validate()

    def supports(self, rank: int, rail_id: str, traffic_class: str) -> bool:
        """Return whether a rank explicitly supports a rail/class binding."""

        if rank < 0 or rank >= len(self.rank_capabilities):
            return False
        rank_capability = self.rank_capabilities[rank]
        if rank_capability.rank != rank:
            return False
        return any(rail.rail_id == rail_id and traffic_class in rail.traffic_classes for rail in rank_capability.rails)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NicCapabilityFingerprint:
        """Validate and deserialize an operator-authored NIC inventory."""

        _require_schema_version(value, NIC_CAPABILITY_SCHEMA_VERSION, "NIC capability")
        _require_exact_fields(value, _NIC_CAPABILITY_FIELDS, "NIC capability")
        raw_topology = value["topology_fingerprint"]
        if not isinstance(raw_topology, Mapping):
            raise NetworkEvidenceError("topology_fingerprint must be an object")
        try:
            topology = TopologyFingerprint.from_dict(raw_topology)
        except TopologyEvidenceError as exc:
            raise NetworkEvidenceError(f"invalid topology_fingerprint: {exc}") from exc
        if value["topology_cell_id"] != topology.cell_id:
            raise NetworkEvidenceError("topology_cell_id does not match topology_fingerprint")
        network_fabric = _known_string(value["network_fabric"], "network_fabric")
        inventory_signature = _known_string(value["inventory_signature"], "inventory_signature")
        raw_rank_capabilities = value["rank_capabilities"]
        if not isinstance(raw_rank_capabilities, list):
            raise NetworkEvidenceError("rank_capabilities must be a list")
        rank_capabilities = []
        for raw_capability in raw_rank_capabilities:
            if not isinstance(raw_capability, Mapping):
                raise NetworkEvidenceError("each rank capability must be an object")
            rank_capabilities.append(RankNicCapability.from_dict(raw_capability))
        fingerprint = cls(
            topology=topology,
            network_fabric=network_fabric,
            inventory_signature=inventory_signature,
            rank_capabilities=tuple(sorted(rank_capabilities, key=lambda item: item.rank)),
        )
        fingerprint.validate()
        return fingerprint


@dataclasses.dataclass(frozen=True)
class NetworkCompatibility:
    """Exact compatibility result for two network telemetry cells."""

    compatible: bool
    reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class NetworkTelemetryFingerprint:
    """Measurement environment in which network policy evidence was observed."""

    capability: NicCapabilityFingerprint
    telemetry_source: str
    telemetry_schema_signature: str

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "schema_version": NETWORK_TELEMETRY_SCHEMA_VERSION,
            "topology_cell_id": self.capability.topology.cell_id,
            "nic_capability_id": self.capability.capability_id,
            "nic_capability_fingerprint": self.capability.to_dict(),
            "network_fabric": self.capability.network_fabric,
            "telemetry_source": self.telemetry_source,
            "telemetry_schema_signature": self.telemetry_schema_signature,
        }

    @property
    def canonical_json(self) -> str:
        """Return a stable serialization for hashing and storage."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @property
    def cell_id(self) -> str:
        """Return the content-addressed telemetry compatibility cell."""

        digest = hashlib.sha256(self.canonical_json.encode()).hexdigest()
        return f"network-telemetry-v{NETWORK_TELEMETRY_SCHEMA_VERSION}:{digest}"

    def validate(self) -> None:
        """Validate the embedded capability and telemetry semantics."""

        self.capability.validate()
        _known_string(self.telemetry_source, "network_telemetry_source")
        _known_string(self.telemetry_schema_signature, "network_telemetry_schema_signature")

    def compare(self, other: NetworkTelemetryFingerprint) -> NetworkCompatibility:
        """Compare every portability boundary without fuzzy matching."""

        reasons = []
        fields = (
            ("topology_cell_id", self.capability.topology.cell_id, other.capability.topology.cell_id),
            ("nic_capability_id", self.capability.capability_id, other.capability.capability_id),
            ("network_fabric", self.capability.network_fabric, other.capability.network_fabric),
            ("telemetry_source", self.telemetry_source, other.telemetry_source),
            (
                "telemetry_schema_signature",
                self.telemetry_schema_signature,
                other.telemetry_schema_signature,
            ),
        )
        for name, left, right in fields:
            if left != right:
                reasons.append(name)
        return NetworkCompatibility(not reasons, tuple(reasons))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NetworkTelemetryFingerprint:
        """Validate and deserialize a stored telemetry compatibility cell."""

        _require_schema_version(value, NETWORK_TELEMETRY_SCHEMA_VERSION, "network telemetry")
        _require_exact_fields(value, _TELEMETRY_FIELDS, "network telemetry")
        raw_capability = value["nic_capability_fingerprint"]
        if not isinstance(raw_capability, Mapping):
            raise NetworkEvidenceError("nic_capability_fingerprint must be an object")
        capability = NicCapabilityFingerprint.from_dict(raw_capability)
        if value["topology_cell_id"] != capability.topology.cell_id:
            raise NetworkEvidenceError("stored telemetry topology_cell_id is inconsistent")
        if value["nic_capability_id"] != capability.capability_id:
            raise NetworkEvidenceError("stored nic_capability_id does not match its fingerprint")
        if value["network_fabric"] != capability.network_fabric:
            raise NetworkEvidenceError("stored telemetry network_fabric is inconsistent")
        fingerprint = cls(
            capability=capability,
            telemetry_source=_known_string(value["telemetry_source"], "telemetry_source"),
            telemetry_schema_signature=_known_string(value["telemetry_schema_signature"], "telemetry_schema_signature"),
        )
        fingerprint.validate()
        return fingerprint


@dataclasses.dataclass(frozen=True, order=True)
class NetworkAssignment:
    """One observed operation/rank rail and traffic-class binding."""

    operation: str
    rank: int
    rail_id: str
    traffic_class: str

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        return {
            "operation": self.operation,
            "rank": self.rank,
            "rail_id": self.rail_id,
            "traffic_class": self.traffic_class,
        }

    def validate(self) -> None:
        """Validate an explicit assignment without accepting sentinels."""

        _known_string(self.operation, "operation")
        if _strict_int(self.rank, "rank") < 0:
            raise NetworkEvidenceError("rank must be non-negative")
        _known_string(self.rail_id, "rail_id")
        _known_string(self.traffic_class, "traffic_class")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NetworkAssignment:
        """Validate and deserialize one required assignment."""

        _require_exact_fields(value, _ASSIGNMENT_FIELDS, "network assignment")
        assignment = cls(
            operation=_known_string(value["operation"], "operation"),
            rank=_strict_int(value["rank"], "rank"),
            rail_id=_known_string(value["rail_id"], "rail_id"),
            traffic_class=_known_string(value["traffic_class"], "traffic_class"),
        )
        assignment.validate()
        return assignment


@dataclasses.dataclass(frozen=True)
class NetworkPolicyEligibility:
    """Measured network cell and complete bindings required by one policy."""

    telemetry: NetworkTelemetryFingerprint
    required_assignments: tuple[NetworkAssignment, ...]

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
            "network_telemetry_cell_id": self.telemetry.cell_id,
            "network_telemetry_fingerprint": self.telemetry.to_dict(),
            "required_assignments": [assignment.to_dict() for assignment in self.required_assignments],
        }

    @property
    def eligibility_id(self) -> str:
        """Return a digest binding the telemetry cell to measured assignments."""

        canonical = json.dumps(self._identity_dict(), separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"network-policy-v{NETWORK_POLICY_SCHEMA_VERSION}:{digest}"

    def to_dict(self) -> dict[str, Any]:
        """Return the stored eligibility block including its integrity digest."""

        return {"eligibility_id": self.eligibility_id, **self._identity_dict()}

    def validate(self) -> None:
        """Validate rank-complete requirements against explicit capabilities."""

        self.telemetry.validate()
        if not self.required_assignments:
            raise NetworkEvidenceError("a network policy must have explicit required_assignments")
        if tuple(sorted(self.required_assignments)) != self.required_assignments:
            raise NetworkEvidenceError("required_assignments must be in canonical sorted order")
        keys = [(assignment.operation, assignment.rank) for assignment in self.required_assignments]
        if len(set(keys)) != len(keys):
            raise NetworkEvidenceError("required_assignments contain duplicate operation/rank bindings")
        assignments_by_operation: defaultdict[str, list[NetworkAssignment]] = defaultdict(list)
        for assignment in self.required_assignments:
            assignment.validate()
            assignments_by_operation[assignment.operation].append(assignment)
            if not self.telemetry.capability.supports(assignment.rank, assignment.rail_id, assignment.traffic_class):
                raise NetworkEvidenceError(
                    f"rank {assignment.rank} does not support rail {assignment.rail_id!r} "
                    f"with traffic class {assignment.traffic_class!r}"
                )
        expected_ranks = tuple(range(self.telemetry.capability.topology.world_size))
        for operation, assignments in assignments_by_operation.items():
            observed_ranks = tuple(sorted(assignment.rank for assignment in assignments))
            if observed_ranks != expected_ranks:
                raise NetworkEvidenceError(
                    f"operation {operation!r} has assignment ranks {observed_ranks}, expected {expected_ranks}"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NetworkPolicyEligibility:
        """Validate and deserialize a policy eligibility block."""

        _require_schema_version(value, NETWORK_POLICY_SCHEMA_VERSION, "network policy")
        _require_exact_fields(value, _ELIGIBILITY_FIELDS, "network policy eligibility")
        raw_telemetry = value["network_telemetry_fingerprint"]
        if not isinstance(raw_telemetry, Mapping):
            raise NetworkEvidenceError("network_telemetry_fingerprint must be an object")
        telemetry = NetworkTelemetryFingerprint.from_dict(raw_telemetry)
        if value["network_telemetry_cell_id"] != telemetry.cell_id:
            raise NetworkEvidenceError("network_telemetry_cell_id does not match its fingerprint")
        raw_assignments = value["required_assignments"]
        if not isinstance(raw_assignments, list):
            raise NetworkEvidenceError("required_assignments must be a list")
        assignments = []
        for raw_assignment in raw_assignments:
            if not isinstance(raw_assignment, Mapping):
                raise NetworkEvidenceError("each required assignment must be an object")
            assignments.append(NetworkAssignment.from_dict(raw_assignment))
        eligibility = cls(telemetry=telemetry, required_assignments=tuple(sorted(assignments)))
        eligibility.validate()
        if value["eligibility_id"] != eligibility.eligibility_id:
            raise NetworkEvidenceError("eligibility_id does not match the eligibility evidence")
        return eligibility


def _record_value(record: Mapping[str, Any], name: str) -> Any:
    value = record.get(name)
    if value is not None:
        return value
    metadata = record.get("metadata")
    return metadata.get(name) if isinstance(metadata, Mapping) else None


def _required_consistent_value(records: Sequence[Mapping[str, Any]], name: str) -> Any:
    values = []
    for record in records:
        value = _record_value(record, name)
        if value is None:
            raise NetworkEvidenceError(f"{name} is required on every selected telemetry record")
        if isinstance(value, dict | list):
            raise NetworkEvidenceError(f"{name} must be scalar")
        values.append(value)
    unique = set(values)
    if len(unique) != 1:
        raise NetworkEvidenceError(f"conflicting {name} values: {sorted(unique, key=str)!r}")
    return values[0]


def _trace_rank(record: Mapping[str, Any]) -> int:
    rank = _record_value(record, "rank")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise NetworkEvidenceError("rank must be an integer in network telemetry")
    if rank < 0:
        raise NetworkEvidenceError("rank must be non-negative")
    return rank


def _select_records(records: Iterable[Mapping[str, Any]], operations: Sequence[str] | None) -> list[Mapping[str, Any]]:
    selected_operations = None
    if operations is not None:
        if not operations:
            raise NetworkEvidenceError("at least one operation must be selected")
        selected_operations = {_known_string(operation, "operation") for operation in operations}
    selected = []
    for record in records:
        operation = _record_value(record, "operation")
        if selected_operations is not None and operation not in selected_operations:
            continue
        _known_string(operation, "operation")
        selected.append(record)
    if not selected:
        raise NetworkEvidenceError("no trace records matched the network telemetry request")
    return selected


def _assignments_from_records(
    records: Sequence[Mapping[str, Any]], capability: NicCapabilityFingerprint
) -> tuple[NetworkAssignment, ...]:
    bindings: defaultdict[tuple[str, int], set[tuple[str, str]]] = defaultdict(set)
    for record in records:
        operation = _known_string(_record_value(record, "operation"), "operation")
        rank = _trace_rank(record)
        rail_id = _known_string(_record_value(record, "nic_rail_id"), "nic_rail_id")
        traffic_class = _known_string(_record_value(record, "nic_traffic_class"), "nic_traffic_class")
        if not capability.supports(rank, rail_id, traffic_class):
            raise NetworkEvidenceError(
                f"rank {rank} telemetry observed unsupported rail {rail_id!r} and traffic class {traffic_class!r}"
            )
        bindings[(operation, rank)].add((rail_id, traffic_class))

    assignments = []
    for (operation, rank), observed in bindings.items():
        if len(observed) != 1:
            raise NetworkEvidenceError(
                f"operation {operation!r} rank {rank} has inconsistent observed rail/class bindings"
            )
        rail_id, traffic_class = next(iter(observed))
        assignments.append(NetworkAssignment(operation, rank, rail_id, traffic_class))

    expected_ranks = tuple(range(capability.topology.world_size))
    operations = sorted({assignment.operation for assignment in assignments})
    for operation in operations:
        observed_ranks = tuple(
            sorted(assignment.rank for assignment in assignments if assignment.operation == operation)
        )
        if observed_ranks != expected_ranks:
            raise NetworkEvidenceError(
                f"operation {operation!r} has telemetry ranks {observed_ranks}, expected {expected_ranks}"
            )
    return tuple(sorted(assignments))


def _fingerprint_network_telemetry_run(
    records: Sequence[Mapping[str, Any]],
    capability: NicCapabilityFingerprint,
) -> tuple[NetworkTelemetryFingerprint, tuple[NetworkAssignment, ...]]:
    capability.validate()
    if not records:
        raise NetworkEvidenceError("cannot fingerprint an empty network telemetry run")
    try:
        topology = fingerprint_trace_run(records)
    except TopologyEvidenceError as exc:
        raise NetworkEvidenceError(str(exc)) from exc
    topology_compatibility = capability.topology.compare(topology)
    if not topology_compatibility.compatible:
        raise NetworkEvidenceError(
            "NIC capability topology does not match trace topology: " + ", ".join(topology_compatibility.reasons)
        )

    for record in records:
        if _record_value(record, "network_telemetry_observed") is not True:
            raise NetworkEvidenceError("network_telemetry_observed=true is required on every selected record")
    network_fabric = _known_string(_required_consistent_value(records, "network_fabric"), "network_fabric")
    if network_fabric != capability.network_fabric:
        raise NetworkEvidenceError("trace network_fabric does not match NIC capabilities")
    inventory_signature = _known_string(
        _required_consistent_value(records, "nic_inventory_signature"),
        "nic_inventory_signature",
    )
    if inventory_signature != capability.inventory_signature:
        raise NetworkEvidenceError("trace nic_inventory_signature does not match NIC capabilities")
    telemetry_source = _known_string(
        _required_consistent_value(records, "network_telemetry_source"),
        "network_telemetry_source",
    )
    telemetry_schema_signature = _known_string(
        _required_consistent_value(records, "network_telemetry_schema_signature"),
        "network_telemetry_schema_signature",
    )
    assignments = _assignments_from_records(records, capability)
    fingerprint = NetworkTelemetryFingerprint(
        capability=capability,
        telemetry_source=telemetry_source,
        telemetry_schema_signature=telemetry_schema_signature,
    )
    fingerprint.validate()
    return fingerprint, assignments


def _network_evidence_by_run(
    records: Iterable[Mapping[str, Any]],
    capability: NicCapabilityFingerprint,
    operations: Sequence[str] | None,
) -> dict[
    tuple[str, int | None],
    tuple[NetworkTelemetryFingerprint, tuple[NetworkAssignment, ...]],
]:
    selected = _select_records(records, operations)
    by_run: defaultdict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for record in selected:
        try:
            run_key = topology_run_key(record)
        except TopologyEvidenceError as exc:
            raise NetworkEvidenceError(str(exc)) from exc
        by_run[run_key].append(record)
    evidence = {}
    for run_key, run_records in by_run.items():
        try:
            evidence[run_key] = _fingerprint_network_telemetry_run(run_records, capability)
        except NetworkEvidenceError as exc:
            raise NetworkEvidenceError(f"run {run_key[0]!r}: {exc}") from exc
    return evidence


def fingerprint_network_telemetry_run(
    records: Sequence[Mapping[str, Any]],
    capability: NicCapabilityFingerprint,
) -> NetworkTelemetryFingerprint:
    """Fingerprint one complete run after validating every observed binding."""

    fingerprint, _ = _fingerprint_network_telemetry_run(records, capability)
    return fingerprint


def fingerprint_network_telemetry_runs(
    records: Iterable[Mapping[str, Any]],
    capability: NicCapabilityFingerprint,
    operations: Sequence[str] | None = None,
) -> dict[tuple[str, int | None], NetworkTelemetryFingerprint]:
    """Fingerprint every selected run without mixing telemetry semantics."""

    return {
        run_key: fingerprint
        for run_key, (fingerprint, _) in _network_evidence_by_run(records, capability, operations).items()
    }


def build_network_policy_eligibility(
    records: Iterable[Mapping[str, Any]],
    capability: NicCapabilityFingerprint,
    operations: Sequence[str] | None = None,
) -> NetworkPolicyEligibility:
    """Bind rank-complete observed assignments to one telemetry cell."""

    evidence = _network_evidence_by_run(records, capability, operations)
    fingerprints = {fingerprint.cell_id: fingerprint for fingerprint, _ in evidence.values()}
    if len(fingerprints) != 1:
        raise NetworkEvidenceError("policy evidence spans more than one network telemetry cell")
    assignments_by_run = {assignments for _, assignments in evidence.values()}
    if len(assignments_by_run) != 1:
        raise NetworkEvidenceError("policy evidence has inconsistent rail/class assignments across runs")
    eligibility = NetworkPolicyEligibility(
        telemetry=next(iter(fingerprints.values())),
        required_assignments=next(iter(assignments_by_run)),
    )
    eligibility.validate()
    return eligibility


def select_eligible_network_policies(
    recommendations: Sequence[Mapping[str, Any]],
    target: NetworkTelemetryFingerprint,
) -> list[Mapping[str, Any]]:
    """Return only exact, capability-supported network policy recommendations."""

    target.validate()
    matches = []
    mismatch_counts: Counter[str] = Counter()
    unannotated = 0
    for recommendation in recommendations:
        raw_eligibility = recommendation.get("network_policy_eligibility")
        if raw_eligibility is None:
            unannotated += 1
            continue
        if not isinstance(raw_eligibility, Mapping):
            raise NetworkEvidenceError("network_policy_eligibility must be an object")
        eligibility = NetworkPolicyEligibility.from_dict(raw_eligibility)
        compatibility = eligibility.telemetry.compare(target)
        if not compatibility.compatible:
            mismatch_counts.update(compatibility.reasons)
            continue
        for assignment in eligibility.required_assignments:
            if not target.capability.supports(assignment.rank, assignment.rail_id, assignment.traffic_class):
                mismatch_counts.update(("required_assignments",))
                break
        else:
            matches.append(recommendation)
    if matches:
        return matches
    details = [f"{field} ({count})" for field, count in sorted(mismatch_counts.items())]
    if unannotated:
        details.append(f"unannotated recommendations ({unannotated})")
    raise IneligibleNetworkPolicyError(
        f"no eligible network policy for {target.cell_id}; "
        f"incompatibilities: {', '.join(details) or 'no measured policies'}"
    )


def _load_capability(path: Path) -> NicCapabilityFingerprint:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise NetworkEvidenceError("NIC capability JSON must contain an object")
    return NicCapabilityFingerprint.from_dict(payload)


def _one_target_fingerprint(
    records: Iterable[Mapping[str, Any]],
    capability: NicCapabilityFingerprint,
    operations: Sequence[str],
) -> NetworkTelemetryFingerprint:
    fingerprints = fingerprint_network_telemetry_runs(records, capability, operations)
    unique = {fingerprint.cell_id: fingerprint for fingerprint in fingerprints.values()}
    if len(unique) != 1:
        raise NetworkEvidenceError("target traces contain more than one network telemetry cell")
    return next(iter(unique.values()))


def build_parser() -> argparse.ArgumentParser:
    """Build the eligibility and selection CLI."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build eligibility from measured trace shards")
    build.add_argument("--capabilities-json", type=Path, required=True)
    build.add_argument("--trace-jsonl", nargs="+", type=Path, required=True)
    build.add_argument("--operation", action="append", required=True)
    build.add_argument("--output-json", type=Path, required=True)

    select = subparsers.add_parser("select", help="select exact eligible policy recommendations")
    select.add_argument("--policy-json", type=Path, required=True)
    select.add_argument("--capabilities-json", type=Path, required=True)
    select.add_argument("--target-trace-jsonl", nargs="+", type=Path, required=True)
    select.add_argument("--operation", action="append", required=True)
    select.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build an eligibility block or select recommendations."""

    args = build_parser().parse_args(argv)
    try:
        capability = _load_capability(args.capabilities_json)
        if args.command == "build":
            eligibility = build_network_policy_eligibility(load_jsonl(args.trace_jsonl), capability, args.operation)
            output = {
                "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
                "network_policy_eligibility": eligibility.to_dict(),
            }
        else:
            policy_payload = json.loads(args.policy_json.read_text(encoding="utf-8"))
            if not isinstance(policy_payload, Mapping) or not isinstance(policy_payload.get("recommendations"), list):
                raise NetworkEvidenceError("policy JSON must contain a recommendations list")
            target = _one_target_fingerprint(load_jsonl(args.target_trace_jsonl), capability, args.operation)
            matches = select_eligible_network_policies(policy_payload["recommendations"], target)
            output = {
                "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
                "nic_capability_id": capability.capability_id,
                "network_telemetry_cell_id": target.cell_id,
                "network_telemetry_fingerprint": target.to_dict(),
                "recommendations": matches,
            }
    except (
        OSError,
        json.JSONDecodeError,
        TopologyEvidenceError,
        NetworkEvidenceError,
        IneligibleNetworkPolicyError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
