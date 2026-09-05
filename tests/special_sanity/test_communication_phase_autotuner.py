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

from scripts.autotune_communication_phase import TraceFormatError, main, tune_traces


def _candidate_records(
    world_size,
    *,
    run_id,
    policy,
    requested_offset_us,
    realized_offset_us,
    duration_a_us,
    duration_b_us,
    consumer_slack_us,
    critical_path_duration_us=None,
    trial_count=4,
):
    records = []
    for trial in range(trial_count):
        trial_origin_ns = 1_000_000_000 + trial * 10_000_000
        for rank in range(world_size):
            rank_skew_ns = rank * 1000
            a_start_ns = trial_origin_ns + rank_skew_ns
            b_start_ns = a_start_ns + int((realized_offset_us + rank) * 1000)
            common = {
                "framework": "fixture",
                "run_id": run_id,
                "rank": rank,
                "world_size": world_size,
                "iteration": trial,
                "microbatch": 0,
                "layer": 3,
                "requested_offset_us": requested_offset_us,
                "topology_class": "logical-fixture",
                "transport": "fixture",
                "timestamp_domain": "fixture-global-monotonic",
                "gpu_timestamp_semantics": "kernel-observed",
                "clock_sync_error_bound_us": 1.0,
                "metadata": {
                    "policy": policy,
                    "completion_observed": True,
                    "sequence_consistent": True,
                },
            }
            if critical_path_duration_us is not None:
                common["critical_path_duration_us"] = critical_path_duration_us + trial + rank
            records.extend(
                [
                    {
                        **common,
                        "operation": "comm_a",
                        "process_group_id": "group-a",
                        "message_bytes": 8192,
                        "communicator_sequence_id": trial,
                        "gpu_start_timestamp_ns": a_start_ns,
                        "gpu_end_timestamp_ns": a_start_ns + int(duration_a_us * 1000),
                        "consumer_timestamp_ns": a_start_ns + int(duration_a_us * 1000),
                    },
                    {
                        **common,
                        "operation": "comm_b",
                        "process_group_id": "group-b",
                        "message_bytes": 4096,
                        "communicator_sequence_id": trial,
                        "gpu_start_timestamp_ns": b_start_ns,
                        "gpu_end_timestamp_ns": b_start_ns + int(duration_b_us * 1000),
                        "consumer_timestamp_ns": b_start_ns + int((duration_b_us + consumer_slack_us) * 1000),
                    },
                ]
            )
    return records


def _trace_fixture(world_size):
    baseline = _candidate_records(
        world_size,
        run_id="baseline",
        policy="eager",
        requested_offset_us=0,
        realized_offset_us=0,
        duration_a_us=1200,
        duration_b_us=1100,
        consumer_slack_us=300,
    )
    safe = _candidate_records(
        world_size,
        run_id="safe-shift",
        policy="phase_shifted",
        requested_offset_us=200,
        realized_offset_us=180,
        duration_a_us=800,
        duration_b_us=650,
        consumer_slack_us=120,
    )
    deadline_regression = _candidate_records(
        world_size,
        run_id="late-shift",
        policy="phase_shifted",
        requested_offset_us=400,
        realized_offset_us=380,
        duration_a_us=700,
        duration_b_us=500,
        consumer_slack_us=-40,
    )
    return baseline + safe + deadline_regression


@pytest.mark.parametrize("world_size", [2, 4])
def test_tuner_uses_trace_evidence_for_any_logical_rank_count(world_size):
    payload = tune_traces(
        _trace_fixture(world_size),
        "comm_a",
        "comm_b",
        bootstrap_resamples=400,
    )

    assert len(payload["recommendations"]) == 1
    recommendation = payload["recommendations"][0]
    assert recommendation["workload_key"]["world_size"] == world_size
    assert recommendation["decision"] == "switch_policy"
    assert recommendation["recommended_candidate"] == {
        "policy": "phase_shifted",
        "requested_offset_us": 200.0,
        "realized_gpu_offset_us_p50": pytest.approx(180 + (world_size - 1) / 2),
    }
    by_offset = {candidate["requested_offset_us"]: candidate for candidate in recommendation["candidates"]}
    assert by_offset[200.0]["eligible"]
    assert by_offset[200.0]["confidence_method"] == "paired_mean"
    assert by_offset[200.0]["critical_path_improvement_ci_us"][0] > 0
    assert by_offset[400.0]["rejection_reasons"] == ["consumer_deadline_regression"]
    assert recommendation["refinement"]["next_requested_offsets_us"] == [100.0, 300.0]
    assert recommendation["refinement"]["next_candidates"][0] == {
        "requested_offset_us": 100.0,
        "predicted_realized_gpu_offset_us": pytest.approx(90 + (world_size - 1) / 2),
        "predicted_consumer_slack_us_p05": pytest.approx(210),
    }


def test_sequence_divergence_fails_closed_on_two_rank_trace():
    records = _trace_fixture(2)
    for record in records:
        if record["run_id"] == "safe-shift" and record["rank"] == 1:
            record["metadata"]["sequence_consistent"] = False

    recommendation = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
    )["recommendations"][0]

    assert recommendation["decision"] == "keep_baseline"
    safe_candidate = next(
        candidate for candidate in recommendation["candidates"] if candidate["requested_offset_us"] == 200.0
    )
    assert "communicator_sequence_divergence" in safe_candidate["rejection_reasons"]


def test_raw_communicator_sequence_ids_are_checked_on_four_ranks():
    records = _trace_fixture(4)
    record = next(
        record
        for record in records
        if record["run_id"] == "safe-shift"
        and record["rank"] == 2
        and record["iteration"] == 1
        and record["operation"] == "comm_b"
    )
    record["communicator_sequence_id"] += 1

    recommendation = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
    )["recommendations"][0]
    candidate = next(
        candidate for candidate in recommendation["candidates"] if candidate["requested_offset_us"] == 200.0
    )
    assert "communicator_sequence_divergence" in candidate["rejection_reasons"]


def test_rank_skew_must_fit_inside_measured_slack_on_four_ranks():
    records = _trace_fixture(4)
    for record in records:
        if record["run_id"] == "safe-shift" and record["rank"] == 3:
            record["gpu_start_timestamp_ns"] += 250_000
            record["gpu_end_timestamp_ns"] += 250_000
            record["consumer_timestamp_ns"] += 250_000

    recommendation = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
    )["recommendations"][0]
    candidate = next(
        candidate for candidate in recommendation["candidates"] if candidate["requested_offset_us"] == 200.0
    )
    assert "rank_skew_exceeds_slack_headroom" in candidate["rejection_reasons"]


def test_incomplete_four_rank_trial_is_rejected():
    records = _trace_fixture(4)
    records = [
        record
        for record in records
        if not (
            record["run_id"] == "safe-shift"
            and record["rank"] == 3
            and record["iteration"] == 1
            and record["operation"] == "comm_b"
        )
    ]

    with pytest.raises(TraceFormatError, match="has 1 'comm_a' records and 0 'comm_b' records"):
        tune_traces(records, "comm_a", "comm_b")


def test_declared_world_size_detects_an_entire_missing_rank_shard():
    records = [record for record in _trace_fixture(4) if not (record["run_id"] == "safe-shift" and record["rank"] == 3)]

    with pytest.raises(TraceFormatError, match=r"run 'safe-shift' has ranks \(0, 1, 2\), expected \(0, 1, 2, 3\)"):
        tune_traces(records, "comm_a", "comm_b")


def test_two_and_four_rank_workloads_are_tuned_independently():
    records = _trace_fixture(2) + _trace_fixture(4)

    payload = tune_traces(records, "comm_a", "comm_b", bootstrap_resamples=200)

    assert [item["workload_key"]["world_size"] for item in payload["recommendations"]] == [2, 4]


def test_negative_offset_uses_the_same_measured_policy_path():
    records = _candidate_records(
        2,
        run_id="baseline",
        policy="eager",
        requested_offset_us=0,
        realized_offset_us=0,
        duration_a_us=1200,
        duration_b_us=1100,
        consumer_slack_us=100,
    ) + _candidate_records(
        2,
        run_id="negative-shift",
        policy="phase_shifted",
        requested_offset_us=-200,
        realized_offset_us=-180,
        duration_a_us=700,
        duration_b_us=800,
        consumer_slack_us=80,
    )

    recommendation = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
    )["recommendations"][0]

    assert recommendation["decision"] == "switch_policy"
    assert recommendation["recommended_candidate"]["requested_offset_us"] == -200.0
    assert recommendation["refinement"]["next_requested_offsets_us"] == [-100.0]


def test_full_step_critical_path_overrides_pair_completion_fallback():
    records = _candidate_records(
        2,
        run_id="baseline",
        policy="eager",
        requested_offset_us=0,
        realized_offset_us=0,
        duration_a_us=1200,
        duration_b_us=1100,
        consumer_slack_us=200,
        critical_path_duration_us=2000,
    ) + _candidate_records(
        2,
        run_id="shorter-pair-slower-step",
        policy="phase_shifted",
        requested_offset_us=200,
        realized_offset_us=180,
        duration_a_us=700,
        duration_b_us=600,
        consumer_slack_us=100,
        critical_path_duration_us=2500,
    )

    recommendation = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
    )["recommendations"][0]

    assert recommendation["decision"] == "keep_baseline"
    candidate = next(
        candidate for candidate in recommendation["candidates"] if candidate["requested_offset_us"] == 200.0
    )
    assert candidate["critical_path_source"] == "trace_critical_path"
    assert candidate["pair_completion_us_p95"] < recommendation["candidates"][0]["pair_completion_us_p95"]
    assert candidate["critical_path_improvement_ci_us"][1] < 0


def test_cli_writes_machine_readable_recommendation(tmp_path):
    trace = tmp_path / "trace.jsonl"
    output = tmp_path / "recommendation.json"
    records = _trace_fixture(2)
    trace.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    assert (
        main(
            [
                "--trace-jsonl",
                str(trace),
                "--operation-a",
                "comm_a",
                "--operation-b",
                "comm_b",
                "--bootstrap-resamples",
                "200",
                "--output-json",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["max_clock_sync_error_us"] == 50.0
    assert payload["recommendations"][0]["workload_key"]["world_size"] == 2


def test_realized_offset_direction_mismatch_is_not_recommended():
    records = deepcopy(_trace_fixture(2))
    for record in records:
        if record["run_id"] == "safe-shift" and record["operation"] == "comm_b":
            a_record = next(
                candidate
                for candidate in records
                if candidate["run_id"] == record["run_id"]
                and candidate["rank"] == record["rank"]
                and candidate["iteration"] == record["iteration"]
                and candidate["operation"] == "comm_a"
            )
            duration_ns = record["gpu_end_timestamp_ns"] - record["gpu_start_timestamp_ns"]
            slack_ns = record["consumer_timestamp_ns"] - record["gpu_end_timestamp_ns"]
            record["gpu_start_timestamp_ns"] = a_record["gpu_start_timestamp_ns"] - 20_000
            record["gpu_end_timestamp_ns"] = record["gpu_start_timestamp_ns"] + duration_ns
            record["consumer_timestamp_ns"] = record["gpu_end_timestamp_ns"] + slack_ns

    recommendation = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
    )["recommendations"][0]
    candidate = next(
        candidate for candidate in recommendation["candidates"] if candidate["requested_offset_us"] == 200.0
    )
    assert "realized_offset_direction_mismatch" in candidate["rejection_reasons"]


def test_event_brackets_are_diagnostic_only_and_do_not_refine():
    records = deepcopy(_trace_fixture(2))
    for record in records:
        record["gpu_timestamp_semantics"] = "event-bracket"

    recommendation = tune_traces(records, "comm_a", "comm_b", bootstrap_resamples=200)["recommendations"][0]

    assert recommendation["decision"] == "insufficient_evidence"
    assert recommendation["refinement"]["next_candidates"] == []
    assert all(
        "kernel_observed_gpu_timestamps_required" in candidate["rejection_reasons"]
        for candidate in recommendation["candidates"]
    )


def test_clock_sync_error_bound_fails_closed():
    records = deepcopy(_trace_fixture(4))
    for record in records:
        record["clock_sync_error_bound_us"] = 75.0

    recommendation = tune_traces(
        records,
        "comm_a",
        "comm_b",
        bootstrap_resamples=200,
        max_clock_sync_error_us=50.0,
    )["recommendations"][0]

    assert recommendation["decision"] == "insufficient_evidence"
    assert recommendation["refinement"]["next_candidates"] == []
    assert all(
        "clock_sync_error_bound_exceeded" in candidate["rejection_reasons"]
        for candidate in recommendation["candidates"]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gpu_timestamp_semantics", "assumed-kernel", "gpu_timestamp_semantics"),
        ("timestamp_domain", "", "timestamp_domain"),
        ("clock_sync_error_bound_us", -1.0, "clock_sync_error_bound_us"),
    ],
)
def test_timing_provenance_contract_rejects_invalid_values(field, value, message):
    records = deepcopy(_trace_fixture(2))
    for record in records:
        record[field] = value

    with pytest.raises(TraceFormatError, match=message):
        tune_traces(records, "comm_a", "comm_b")


def test_timing_provenance_is_required_on_both_pair_members():
    records = deepcopy(_trace_fixture(2))
    record = next(item for item in records if item["operation"] == "comm_b")
    del record["gpu_timestamp_semantics"]

    with pytest.raises(TraceFormatError, match="must be present on both paired records"):
        tune_traces(records, "comm_a", "comm_b")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("world_size", "world_size"),
        ("process_group_id", "process_group_id must be a non-empty string"),
        ("communicator_sequence_id", "communicator_sequence_id must be a finite number"),
        ("completion_observed", "completion_observed=true is required"),
    ],
)
def test_required_safety_evidence_cannot_be_omitted(field, message):
    records = deepcopy(_trace_fixture(2))
    record = next(item for item in records if item["operation"] == "comm_b")
    if field == "completion_observed":
        del record["metadata"][field]
    else:
        del record[field]

    with pytest.raises(TraceFormatError, match=message):
        tune_traces(records, "comm_a", "comm_b")


def test_optional_pair_evidence_cannot_be_present_on_only_one_member():
    records = deepcopy(_trace_fixture(2))
    record = next(item for item in records if item["operation"] == "comm_b")
    del record["requested_offset_us"]

    with pytest.raises(TraceFormatError, match="present on both paired records or neither"):
        tune_traces(records, "comm_a", "comm_b")
