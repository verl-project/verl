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

import json
from copy import deepcopy

import pytest

from scripts.communication_network_policy import (
    IneligibleNetworkPolicyError,
    NetworkEvidenceError,
    NetworkPolicyEligibility,
    NicCapabilityFingerprint,
    build_network_policy_eligibility,
    fingerprint_network_telemetry_run,
    main,
    select_eligible_network_policies,
)
from scripts.communication_topology_policy import fingerprint_trace_run


def _hosts(world_size):
    ranks_per_node = world_size // 2
    return [f"logical-host-{rank // ranks_per_node}" for rank in range(world_size)]


def _topology_records(world_size, *, run_id="topology"):
    hosts = _hosts(world_size)
    return [
        {
            "run_id": run_id,
            "rank": rank,
            "world_size": world_size,
            "hostname": hosts[rank],
            "topology_class": "multi-node",
            "topology_signature": "logical-topology-v1",
            "accelerator_model": "logical-device",
        }
        for rank in range(world_size)
    ]


def _capability(world_size, *, inventory_signature="logical-nics-v1", telemetry_fabric="roce"):
    topology = fingerprint_trace_run(_topology_records(world_size))
    payload = {
        "schema_version": 1,
        "topology_cell_id": topology.cell_id,
        "topology_fingerprint": topology.to_dict(),
        "network_fabric": telemetry_fabric,
        "inventory_signature": inventory_signature,
        "rank_capabilities": [
            {
                "rank": rank,
                "rails": [
                    {"rail_id": "rail-a", "traffic_classes": ["tc-high", "tc-low"]},
                    {"rail_id": "rail-b", "traffic_classes": ["tc-high", "tc-low"]},
                ],
            }
            for rank in range(world_size)
        ],
    }
    return NicCapabilityFingerprint.from_dict(payload)


def _telemetry_records(
    capability,
    *,
    run_id="measured",
    telemetry_schema_signature="logical-schema-v1",
    alternate_assignments=False,
):
    world_size = capability.topology.world_size
    hosts = _hosts(world_size)
    records = []
    for operation in ("comm_a", "comm_b"):
        for rank in range(world_size):
            if alternate_assignments:
                rail_id = "rail-b" if operation == "comm_a" else "rail-a"
                traffic_class = "tc-high" if operation == "comm_a" else "tc-low"
            else:
                rail_id = "rail-a" if operation == "comm_a" else "rail-b"
                traffic_class = "tc-low" if operation == "comm_a" else "tc-high"
            records.append(
                {
                    "run_id": run_id,
                    "rank": rank,
                    "world_size": world_size,
                    "hostname": hosts[rank],
                    "topology_class": "multi-node",
                    "topology_signature": "logical-topology-v1",
                    "accelerator_model": "logical-device",
                    "operation": operation,
                    "network_fabric": capability.network_fabric,
                    "nic_inventory_signature": capability.inventory_signature,
                    "network_telemetry_source": "logical-counter-exporter",
                    "network_telemetry_schema_signature": telemetry_schema_signature,
                    "network_telemetry_observed": True,
                    "nic_rail_id": rail_id,
                    "nic_traffic_class": traffic_class,
                }
            )
    return records


def test_capability_fingerprint_is_canonical_for_four_logical_ranks():
    capability = _capability(4)
    reordered = capability.to_dict()
    reordered["rank_capabilities"].reverse()
    for rank_capability in reordered["rank_capabilities"]:
        rank_capability["rails"].reverse()
        for rail in rank_capability["rails"]:
            rail["traffic_classes"].reverse()

    loaded = NicCapabilityFingerprint.from_dict(reordered)

    assert loaded.to_dict() == capability.to_dict()
    assert loaded.capability_id == capability.capability_id
    assert loaded.topology.world_size == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rail_id", "unknown", "rail_id must be an explicit"),
        ("traffic_class", "unspecified", "traffic_class must be an explicit"),
    ],
)
def test_unknown_capability_values_fail_closed(field, value, message):
    payload = _capability(2).to_dict()
    if field == "rail_id":
        payload["rank_capabilities"][0]["rails"][0]["rail_id"] = value
    else:
        payload["rank_capabilities"][0]["rails"][0]["traffic_classes"][0] = value

    with pytest.raises(NetworkEvidenceError, match=message):
        NicCapabilityFingerprint.from_dict(payload)


def test_capability_requires_every_logical_rank():
    payload = _capability(4).to_dict()
    payload["rank_capabilities"].pop()

    with pytest.raises(NetworkEvidenceError, match="rank_capabilities contain ranks"):
        NicCapabilityFingerprint.from_dict(payload)


def test_single_node_topology_cannot_claim_a_rail_policy():
    records = [
        {
            **record,
            "hostname": "one-host",
            "topology_class": "single-node-pcie",
            "topology_signature": "single-node-layout",
        }
        for record in _topology_records(2)
    ]
    topology = fingerprint_trace_run(records)
    payload = _capability(2).to_dict()
    payload["topology_cell_id"] = topology.cell_id
    payload["topology_fingerprint"] = topology.to_dict()

    with pytest.raises(NetworkEvidenceError, match="require a multi-node topology"):
        NicCapabilityFingerprint.from_dict(payload)


def test_telemetry_cell_describes_environment_not_policy_assignment():
    capability = _capability(4)
    first = fingerprint_network_telemetry_run(_telemetry_records(capability), capability)
    second = fingerprint_network_telemetry_run(
        _telemetry_records(capability, run_id="alternate", alternate_assignments=True),
        capability,
    )

    assert first.cell_id == second.cell_id
    assert first.compare(second).compatible


def test_telemetry_schema_change_creates_an_incompatible_cell():
    capability = _capability(2)
    first = fingerprint_network_telemetry_run(_telemetry_records(capability), capability)
    second = fingerprint_network_telemetry_run(
        _telemetry_records(
            capability,
            run_id="new-schema",
            telemetry_schema_signature="logical-schema-v2",
        ),
        capability,
    )

    assert first.cell_id != second.cell_id
    assert first.compare(second).reasons == ("telemetry_schema_signature",)


def test_unobserved_network_telemetry_fails_closed():
    capability = _capability(2)
    records = _telemetry_records(capability)
    records[0]["network_telemetry_observed"] = False

    with pytest.raises(NetworkEvidenceError, match="network_telemetry_observed=true"):
        fingerprint_network_telemetry_run(records, capability)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nic_rail_id", "unknown", "nic_rail_id must be an explicit"),
        ("nic_traffic_class", "unknown", "nic_traffic_class must be an explicit"),
        ("nic_rail_id", "rail-not-in-inventory", "observed unsupported rail"),
    ],
)
def test_unknown_or_unsupported_observed_bindings_fail_closed(field, value, message):
    capability = _capability(2)
    records = _telemetry_records(capability)
    records[0][field] = value

    with pytest.raises(NetworkEvidenceError, match=message):
        fingerprint_network_telemetry_run(records, capability)


def test_each_operation_requires_telemetry_from_every_rank():
    capability = _capability(4)
    records = [
        record
        for record in _telemetry_records(capability)
        if not (record["operation"] == "comm_b" and record["rank"] == 3)
    ]

    with pytest.raises(NetworkEvidenceError, match=r"comm_b.*telemetry ranks.*expected"):
        fingerprint_network_telemetry_run(records, capability)


def test_eligibility_binds_complete_four_rank_assignments():
    capability = _capability(4)
    eligibility = build_network_policy_eligibility(
        _telemetry_records(capability),
        capability,
        ("comm_a", "comm_b"),
    )
    restored = NetworkPolicyEligibility.from_dict(eligibility.to_dict())

    assert restored == eligibility
    assert len(eligibility.required_assignments) == 8
    assert eligibility.eligibility_id.startswith("network-policy-v1:")
    assert {assignment.rank for assignment in eligibility.required_assignments} == {0, 1, 2, 3}


def test_eligibility_rejects_inconsistent_assignments_across_runs():
    capability = _capability(2)
    records = _telemetry_records(capability, run_id="first")
    records += _telemetry_records(
        capability,
        run_id="second",
        alternate_assignments=True,
    )

    with pytest.raises(NetworkEvidenceError, match="inconsistent rail/class assignments across runs"):
        build_network_policy_eligibility(records, capability, ("comm_a", "comm_b"))


def test_selector_accepts_exact_cell_and_supported_assignments():
    capability = _capability(4)
    eligibility = build_network_policy_eligibility(
        _telemetry_records(capability),
        capability,
        ("comm_a", "comm_b"),
    )
    recommendation = {
        "decision": "switch_policy",
        "network_policy_eligibility": eligibility.to_dict(),
    }
    target = fingerprint_network_telemetry_run(
        _telemetry_records(
            capability,
            run_id="target",
            alternate_assignments=True,
        ),
        capability,
    )

    assert select_eligible_network_policies([recommendation], target) == [recommendation]


def test_selector_rejects_cross_capability_extrapolation():
    source_capability = _capability(2)
    eligibility = build_network_policy_eligibility(
        _telemetry_records(source_capability),
        source_capability,
        ("comm_a", "comm_b"),
    )
    recommendation = {"network_policy_eligibility": eligibility.to_dict()}
    target_capability = _capability(2, inventory_signature="different-logical-nics")
    target = fingerprint_network_telemetry_run(
        _telemetry_records(target_capability, run_id="target"),
        target_capability,
    )

    with pytest.raises(IneligibleNetworkPolicyError, match="nic_capability_id"):
        select_eligible_network_policies([recommendation], target)


def test_selector_rejects_unannotated_policy_instead_of_falling_back():
    capability = _capability(2)
    target = fingerprint_network_telemetry_run(_telemetry_records(capability), capability)

    with pytest.raises(IneligibleNetworkPolicyError, match="unannotated recommendations"):
        select_eligible_network_policies([{"decision": "keep_baseline"}], target)


def test_stored_eligibility_digest_is_verified():
    capability = _capability(2)
    eligibility = build_network_policy_eligibility(
        _telemetry_records(capability),
        capability,
        ("comm_a", "comm_b"),
    ).to_dict()
    eligibility["required_assignments"][0]["traffic_class"] = "tc-high"

    with pytest.raises(NetworkEvidenceError, match="eligibility_id does not match"):
        NetworkPolicyEligibility.from_dict(eligibility)


def test_two_rank_build_and_select_cli_round_trip(tmp_path):
    capability = _capability(2)
    capability_path = tmp_path / "capability.json"
    source_trace_path = tmp_path / "source.jsonl"
    eligibility_path = tmp_path / "eligibility.json"
    policy_path = tmp_path / "policy.json"
    target_trace_path = tmp_path / "target.jsonl"
    selected_path = tmp_path / "selected.json"
    capability_path.write_text(json.dumps(capability.to_dict()), encoding="utf-8")
    source_trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _telemetry_records(capability)),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "build",
                "--capabilities-json",
                str(capability_path),
                "--trace-jsonl",
                str(source_trace_path),
                "--operation",
                "comm_a",
                "--operation",
                "comm_b",
                "--output-json",
                str(eligibility_path),
            ]
        )
        == 0
    )
    eligibility_payload = json.loads(eligibility_path.read_text(encoding="utf-8"))
    policy_path.write_text(
        json.dumps(
            {
                "recommendations": [
                    {
                        "decision": "switch_policy",
                        "network_policy_eligibility": eligibility_payload["network_policy_eligibility"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    target_trace_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _telemetry_records(
                capability,
                run_id="target",
                alternate_assignments=True,
            )
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "select",
                "--policy-json",
                str(policy_path),
                "--capabilities-json",
                str(capability_path),
                "--target-trace-jsonl",
                str(target_trace_path),
                "--operation",
                "comm_a",
                "--operation",
                "comm_b",
                "--output-json",
                str(selected_path),
            ]
        )
        == 0
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    assert selected["nic_capability_id"] == capability.capability_id
    assert len(selected["recommendations"]) == 1


def test_tampered_stored_telemetry_cell_is_rejected():
    capability = _capability(2)
    eligibility = build_network_policy_eligibility(
        _telemetry_records(capability),
        capability,
        ("comm_a", "comm_b"),
    ).to_dict()
    tampered = deepcopy(eligibility)
    tampered["network_telemetry_cell_id"] = "network-telemetry-v1:not-the-cell"
    recommendation = {"network_policy_eligibility": tampered}
    target = fingerprint_network_telemetry_run(_telemetry_records(capability), capability)

    with pytest.raises(NetworkEvidenceError, match="does not match its fingerprint"):
        select_eligible_network_policies([recommendation], target)
