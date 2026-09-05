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

import pytest

from scripts.autotune_communication_phase import tune_traces
from scripts.communication_topology_policy import (
    IncompatibleTopologyError,
    TopologyEvidenceError,
    TopologyFingerprint,
    fingerprint_trace_run,
    main,
    select_compatible_policy_cells,
)


def _topology_records(world_size, topology_class, hosts, *, run_id="run", signature=None):
    records = []
    for rank in range(world_size):
        record = {
            "run_id": run_id,
            "rank": rank,
            "world_size": world_size,
            "hostname": hosts[rank],
            "topology_class": topology_class,
            "accelerator_model": "logical-device",
        }
        if signature is not None:
            record["topology_signature"] = signature
        records.append(record)
    return records


def _tuning_records(
    world_size,
    topology_class,
    *,
    run_id,
    policy,
    requested_offset_us,
    duration_us,
):
    records = []
    for trial in range(3):
        for rank in range(world_size):
            a_start_ns = 1_000_000_000 + trial * 10_000_000 + rank * 1000
            b_start_ns = a_start_ns + int((requested_offset_us * 0.8 + rank) * 1000)
            common = {
                "framework": "fixture",
                "run_id": run_id,
                "rank": rank,
                "world_size": world_size,
                "hostname": "host-a",
                "accelerator_model": "logical-device",
                "topology_class": topology_class,
                "topology_signature": f"{topology_class}-layout",
                "iteration": trial,
                "microbatch": 0,
                "layer": 1,
                "requested_offset_us": requested_offset_us,
                "transport": "fixture",
                "timestamp_domain": "fixture-global-monotonic",
                "gpu_timestamp_semantics": "kernel-observed",
                "clock_sync_error_bound_us": 1.0,
                "process_group_id": "group",
                "communicator_sequence_id": trial,
                "metadata": {"policy": policy, "completion_observed": True},
            }
            records.extend(
                [
                    {
                        **common,
                        "operation": "comm_a",
                        "message_bytes": 8192,
                        "gpu_start_timestamp_ns": a_start_ns,
                        "gpu_end_timestamp_ns": a_start_ns + duration_us * 1000,
                        "consumer_timestamp_ns": a_start_ns + duration_us * 1000,
                    },
                    {
                        **common,
                        "operation": "comm_b",
                        "message_bytes": 4096,
                        "gpu_start_timestamp_ns": b_start_ns,
                        "gpu_end_timestamp_ns": b_start_ns + duration_us * 1000,
                        "consumer_timestamp_ns": b_start_ns + (duration_us + 200) * 1000,
                    },
                ]
            )
    return records


def test_single_node_pcie_and_nvlink_have_distinct_stable_cells():
    pcie_records = _topology_records(4, "single-node-pcie", ["host-a"] * 4)
    nvlink_records = _topology_records(4, "single-node-nvlink", ["host-b"] * 4)

    pcie = fingerprint_trace_run(pcie_records)
    reordered = fingerprint_trace_run(list(reversed(pcie_records)))
    nvlink = fingerprint_trace_run(nvlink_records)

    assert pcie.scope == "single_node"
    assert pcie.local_fabric == "pcie"
    assert pcie.rank_groups == ((0, 1, 2, 3),)
    assert pcie.cell_id == reordered.cell_id
    assert pcie.cell_id != nvlink.cell_id
    assert pcie.compare(nvlink).reasons == ("local_fabric",)


def test_multi_node_fingerprint_preserves_rank_placement():
    records = _topology_records(
        4,
        "multi-node",
        ["host-a", "host-a", "host-b", "host-b"],
        signature="opaque-cluster-layout",
    )

    fingerprint = fingerprint_trace_run(records)

    assert fingerprint.scope == "multi_node"
    assert fingerprint.node_count == 2
    assert fingerprint.rank_groups == ((0, 1), (2, 3))
    assert fingerprint.to_dict()["ranks_per_node"] == [2, 2]


def test_multi_node_rank_placements_are_different_cells():
    contiguous = fingerprint_trace_run(
        _topology_records(
            4,
            "multi-node",
            ["host-a", "host-a", "host-b", "host-b"],
            signature="same-cluster",
        )
    )
    interleaved = fingerprint_trace_run(
        _topology_records(
            4,
            "multi-node",
            ["host-a", "host-b", "host-a", "host-b"],
            signature="same-cluster",
        )
    )

    assert contiguous.compare(interleaved).reasons == ("rank_groups",)
    assert contiguous.cell_id != interleaved.cell_id


def test_multi_node_without_topology_signature_fails_closed():
    records = _topology_records(2, "multi-node", ["host-a", "host-b"])

    with pytest.raises(TopologyEvidenceError, match="require an opaque topology_signature"):
        fingerprint_trace_run(records)


def test_partial_topology_signature_fails_closed():
    records = _topology_records(
        2,
        "multi-node",
        ["host-a", "host-b"],
        signature="opaque-cluster-layout",
    )
    del records[1]["topology_signature"]

    with pytest.raises(TopologyEvidenceError, match="missing on only some ranks"):
        fingerprint_trace_run(records)


def test_declared_single_node_class_cannot_hide_multiple_hosts():
    records = _topology_records(2, "single-node-pcie", ["host-a", "host-b"])

    with pytest.raises(TopologyEvidenceError, match="disagrees with observed node placement"):
        fingerprint_trace_run(records)


def test_stored_fingerprint_rejects_non_integer_rank():
    fingerprint = fingerprint_trace_run(_topology_records(2, "single-node-pcie", ["host-a"] * 2)).to_dict()
    fingerprint["rank_groups"] = [[0, 1.5]]

    with pytest.raises(TopologyEvidenceError, match="malformed topology fingerprint"):
        TopologyFingerprint.from_dict(fingerprint)


def test_policy_cell_selection_rejects_cross_topology_fallback():
    pcie = fingerprint_trace_run(_topology_records(2, "single-node-pcie", ["host-a"] * 2))
    nvlink = fingerprint_trace_run(_topology_records(2, "single-node-nvlink", ["host-b"] * 2))
    pcie_cell = {
        "workload_key": {
            "topology_cell_id": pcie.cell_id,
            "topology_fingerprint": pcie.to_dict(),
        },
        "decision": "keep_baseline",
    }

    assert select_compatible_policy_cells([pcie_cell], pcie) == [pcie_cell]
    with pytest.raises(IncompatibleTopologyError, match="local_fabric"):
        select_compatible_policy_cells([pcie_cell], nvlink)


def test_policy_cell_rejects_a_mismatched_stored_digest():
    pcie = fingerprint_trace_run(_topology_records(2, "single-node-pcie", ["host-a"] * 2))
    cell = {
        "workload_key": {
            "topology_cell_id": "topology-v1:not-the-fingerprint",
            "topology_fingerprint": pcie.to_dict(),
        }
    }

    with pytest.raises(TopologyEvidenceError, match="does not match its fingerprint"):
        select_compatible_policy_cells([cell], pcie)


def test_autotuner_selects_independently_inside_four_rank_topology_cells():
    records = []
    for topology_class, shifted_duration in (("single-node-pcie", 600), ("single-node-nvlink", 1200)):
        records.extend(
            _tuning_records(
                4,
                topology_class,
                run_id=f"{topology_class}-baseline",
                policy="eager",
                requested_offset_us=0,
                duration_us=1000,
            )
        )
        records.extend(
            _tuning_records(
                4,
                topology_class,
                run_id=f"{topology_class}-shifted",
                policy="phase_shifted",
                requested_offset_us=100,
                duration_us=shifted_duration,
            )
        )

    recommendations = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
    )["recommendations"]

    assert len(recommendations) == 2
    by_topology = {item["workload_key"]["topology_class"]: item for item in recommendations}
    assert by_topology["single-node-pcie"]["decision"] == "switch_policy"
    assert by_topology["single-node-pcie"]["recommended_candidate"]["requested_offset_us"] == 100.0
    assert by_topology["single-node-nvlink"]["decision"] == "keep_baseline"
    assert (
        by_topology["single-node-pcie"]["workload_key"]["topology_cell_id"]
        != by_topology["single-node-nvlink"]["workload_key"]["topology_cell_id"]
    )


def test_selector_cli_uses_two_rank_target_trace(tmp_path):
    pcie = fingerprint_trace_run(_topology_records(2, "single-node-pcie", ["host-a"] * 2))
    policy = tmp_path / "policy.json"
    traces = tmp_path / "target.jsonl"
    output = tmp_path / "selected.json"
    policy.write_text(
        json.dumps(
            {
                "recommendations": [
                    {
                        "workload_key": {
                            "topology_cell_id": pcie.cell_id,
                            "topology_fingerprint": pcie.to_dict(),
                        },
                        "decision": "keep_baseline",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    traces.write_text(
        "".join(json.dumps(record) + "\n" for record in _topology_records(2, "single-node-pcie", ["new"] * 2)),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--policy-json",
                str(policy),
                "--target-trace-jsonl",
                str(traces),
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["topology_cell_id"] == pcie.cell_id
    assert len(selected["recommendations"]) == 1
