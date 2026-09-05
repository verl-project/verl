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

import argparse
import json
import time
from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark_collective_phase_sweep import (
    BenchmarkRunner,
    LocalObservation,
    NativeTraceWriter,
    _require_single_node,
    _topology_has_nvlink,
    build_group_specs,
    parse_size,
    percentile,
    policy_cell_id,
    resolve_group_layout,
)


def test_parse_size():
    assert parse_size("128MiB") == 128 * 1024**2
    assert parse_size("1.5KB") == 1500
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size("0")


def test_general_mesh_layout():
    groups_a, groups_b = build_group_specs("mesh", 4, (2, 2))
    assert [group.ranks for group in groups_a] == [(0, 1), (2, 3)]
    assert [group.ranks for group in groups_b] == [(0, 2), (1, 3)]
    assert all(any(rank in group.ranks for group in groups_a) for rank in range(4))
    assert all(any(rank in group.ranks for group in groups_b) for rank in range(4))


def test_auto_layout_uses_generic_factorization_for_four_ranks():
    assert resolve_group_layout("auto", 4) == ("mesh-2x2", (2, 2))


def test_prime_world_size_auto_falls_back_to_overlapping_world_groups():
    groups_a, groups_b = build_group_specs("auto", 2)
    assert groups_a == [groups_a[0]]
    assert groups_b == [groups_b[0]]
    assert groups_a[0].ranks == groups_b[0].ranks == (0, 1)


def test_ep_dp_shorthand_is_general():
    assert build_group_specs("ep2-dp2", 4) == build_group_specs("mesh", 4, (2, 2))


def test_mesh_shape_must_match_world_size():
    with pytest.raises(ValueError, match="does not match world size"):
        build_group_specs("mesh", 4, (2, 3))


def test_percentile_uses_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_topology_detection_ignores_nvlink_legend():
    pcie_topology = "GPU0 GPU1\nGPU0 X PIX\nGPU1 PIX X\nLegend:\nNV# = Connection traversing bonded NVLinks"
    nvlink_topology = "GPU0 GPU1\nGPU0 X NV4\nGPU1 NV4 X\nLegend:\nNV# = Connection traversing bonded NVLinks"
    assert not _topology_has_nvlink(pcie_topology)
    assert _topology_has_nvlink(nvlink_topology)


def test_multi_node_clock_domain_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "scripts.benchmark_collective_phase_sweep.dist.broadcast_object_list",
        lambda supported, src: None,
    )
    with pytest.raises(RuntimeError, match="perf_counter_ns anchors are not portable"):
        _require_single_node({"topology_class": "multi-node"}, rank=0)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_native_writer_preserves_raw_boundaries_and_cpu_is_not_gpu(tmp_path, world_size, backend):
    path = tmp_path / "trace.jsonl"
    writer = NativeTraceWriter(path, rank=0, world_size=world_size, run_id="logical-fixture", backend=backend)
    observed = dict(
        operation="all_reduce",
        process_group_id="explicit",
        process_group_ranks=list(range(world_size)),
        communicator_sequence_id=19,
        api_launch_timestamp_ns=100,
        api_return_timestamp_ns=110,
        completion_timestamp_ns=150,
        consumer_timestamp_ns=170,
        message_bytes=16,
        gpu_start_timestamp_ns=120,
        gpu_end_timestamp_ns=140,
        buffer_reuse_acquire_timestamp_ns=100,
        buffer_reuse_release_timestamp_ns=180,
        resource_scope="persistent-buffer-transfer-lease",
    )
    writer.emit(
        LocalObservation(collectives={"b": observed}), mode="offset", offset_us=-10, step=7, phase="measurement"
    )
    writer.close()
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 3
    assert raw["communicator_sequence_id"] == 19
    assert raw["process_group_ranks"] == list(range(world_size))
    assert raw["policy_id"] == "offset/-10us"
    assert raw["completion_timestamp_ns"] == 150
    assert raw["consumer_timestamp_ns"] == 170
    assert raw["gpu_start_timestamp_ns"] == (120 if backend == "nccl" else None)
    assert raw["clock_sync_error_bound_us"] is None
    with pytest.raises(FileExistsError):
        NativeTraceWriter(path, rank=0, world_size=world_size, run_id="second", backend=backend)


@pytest.mark.parametrize("offset", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_policy_cells_are_rejected(offset):
    with pytest.raises(ValueError, match="finite"):
        policy_cell_id("offset", offset)


@pytest.mark.parametrize("mode", ["concurrent", "offset"])
@pytest.mark.parametrize("failure_stage", ["launch", "wait"])
def test_cpu_worker_failure_cannot_publish_partial_success(mode, failure_stage):
    failure = RuntimeError("injected collective failure")

    def fail():
        raise failure

    buffer_a = SimpleNamespace(
        operation="all_reduce",
        launch=fail if failure_stage == "launch" else lambda: SimpleNamespace(wait=fail),
    )
    buffer_b = SimpleNamespace(operation="all_reduce", launch=lambda: SimpleNamespace(wait=lambda: None))
    runner = BenchmarkRunner(buffer_a, buffer_b, torch.device("cpu"), validate=False, launch_anchor_lead_us=0)
    with pytest.raises(RuntimeError, match="CPU collective worker failed") as raised:
        runner._run_cpu_trial(mode, 0.0, time.perf_counter_ns())
    assert raised.value.__cause__ is failure
