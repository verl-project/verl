# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Resource discovery must not assume a specific accelerator vendor.

These tests pin down behaviour that is easy to regress by hardcoding ``"GPU"``/``"NPU"``,
by assuming a bundle layout Ray does not guarantee, or by assuming NVML is present. They
run without a Ray cluster and without an accelerator: the Ray state APIs and the active
platform are stubbed out.
"""

import pytest
import ray

import verl.utils.distributed as distributed_utils
from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager, sort_placement_group_by_node_ip


class _FakePlatform:
    def __init__(self, resource_name, colocation=True):
        self._resource_name = resource_name
        self._colocation = colocation

    def ray_resource_name(self):
        return self._resource_name

    def supports_colocated_worker_groups(self):
        return self._colocation


class _FakePG:
    def __init__(self, pg_id):
        self.id = pg_id


@pytest.fixture()
def stub_ray_nodes(monkeypatch):
    """Give ``sort_placement_group_by_node_ip`` a two-node cluster to resolve against."""
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [
            {"NodeID": "node-a", "NodeManagerAddress": "10.0.0.1"},
            {"NodeID": "node-b", "NodeManagerAddress": "10.0.0.2"},
        ],
    )


def _stub_placement_group_table(monkeypatch, table):
    monkeypatch.setattr(
        ray._private.state.state,
        "placement_group_table",
        lambda pg_id: table[pg_id],
        raising=False,
    )


# ---------------------------------------------------------------------------
# sort_placement_group_by_node_ip
# ---------------------------------------------------------------------------


def test_sort_placement_group_orders_by_node_ip(monkeypatch, stub_ray_nodes):
    """Baseline: groups sort by the address of the node holding their bundles."""
    pg_hi, pg_lo = _FakePG("pg-hi"), _FakePG("pg-lo")
    _stub_placement_group_table(
        monkeypatch,
        {
            "pg-hi": {"bundles_to_node_id": {0: "node-b"}},
            "pg-lo": {"bundles_to_node_id": {0: "node-a"}},
        },
    )

    assert sort_placement_group_by_node_ip([pg_hi, pg_lo]) == [pg_lo, pg_hi]


def test_sort_placement_group_handles_nonzero_first_bundle_index(monkeypatch, stub_ray_nodes):
    """Bundle indices are not guaranteed to start at 0; the lowest present index is used."""
    pg = _FakePG("pg-offset")
    _stub_placement_group_table(monkeypatch, {"pg-offset": {"bundles_to_node_id": {3: "node-b", 4: "node-b"}}})

    # Indexing the map with a hardcoded 0 would raise KeyError here.
    assert sort_placement_group_by_node_ip([pg]) == [pg]


def test_sort_placement_group_tolerates_pending_group(monkeypatch, stub_ray_nodes):
    """A placement group that is still pending has no bundle mapping yet."""
    pending, placed = _FakePG("pg-pending"), _FakePG("pg-placed")
    _stub_placement_group_table(
        monkeypatch,
        {
            "pg-pending": {"bundles_to_node_id": {}},
            "pg-placed": {"bundles_to_node_id": {0: "node-a"}},
        },
    )

    # Must not raise; the unplaced group has no address and sorts first.
    assert sort_placement_group_by_node_ip([placed, pending]) == [pending, placed]


# ---------------------------------------------------------------------------
# ResourcePoolManager._check_resource_available
# ---------------------------------------------------------------------------


def _manager(process_on_nodes):
    return ResourcePoolManager(resource_pool_spec={"pool": process_on_nodes}, mapping={})


def _stub_available_resources(monkeypatch, per_node):
    monkeypatch.setattr(
        ray._private.state,
        "available_resources_per_node",
        lambda: per_node,
        raising=False,
    )


@pytest.mark.parametrize("resource_name", ["GPU", "NPU", "TPU"])
def test_check_resource_available_counts_active_platform_resource(monkeypatch, resource_name):
    """The resource key comes from the platform, not from a hardcoded list."""
    monkeypatch.setattr(
        "verl.single_controller.ray.base.get_platform",
        lambda: _FakePlatform(resource_name),
    )
    _stub_available_resources(
        monkeypatch,
        {"node-a": {"CPU": 8, resource_name: 4}, "node-b": {"CPU": 8, resource_name: 4}},
    )

    _manager([4, 4])._check_resource_available()  # 8 available, 8 required


@pytest.mark.parametrize("resource_name", ["GPU", "NPU", "TPU"])
def test_check_resource_available_raises_on_shortfall(monkeypatch, resource_name):
    """A shortfall is reported for every accelerator, naming the resource involved."""
    monkeypatch.setattr(
        "verl.single_controller.ray.base.get_platform",
        lambda: _FakePlatform(resource_name),
    )
    _stub_available_resources(monkeypatch, {"node-a": {"CPU": 8, resource_name: 2}})

    with pytest.raises(ValueError, match=resource_name):
        _manager([8])._check_resource_available()


def test_check_resource_available_ignores_other_vendors_resources(monkeypatch):
    """Devices belonging to a different vendor must not be counted toward the requirement."""
    monkeypatch.setattr(
        "verl.single_controller.ray.base.get_platform",
        lambda: _FakePlatform("TPU"),
    )
    # A cluster advertising only GPUs cannot satisfy a TPU request, even though the
    # previous hardcoded lookup would have happily counted the 8 GPUs here.
    _stub_available_resources(monkeypatch, {"node-a": {"CPU": 8, "GPU": 8}})

    with pytest.raises(ValueError, match="TPU"):
        _manager([8])._check_resource_available()


# ---------------------------------------------------------------------------
# Colocation cap
# ---------------------------------------------------------------------------


def _use_platform(monkeypatch, platform):
    monkeypatch.setattr("verl.single_controller.ray.base.get_platform", lambda: platform)


def test_resource_pool_keeps_colocate_count_where_devices_are_shareable(monkeypatch):
    """Existing accelerators time-share, so the requested colocation must be preserved."""
    _use_platform(monkeypatch, _FakePlatform("GPU", colocation=True))

    assert RayResourcePool([4], max_colocate_count=3).max_colocate_count == 3


def test_resource_pool_caps_colocate_count_where_devices_are_exclusive(monkeypatch):
    """A device claimed by a single process cannot host several WorkerGroups."""
    _use_platform(monkeypatch, _FakePlatform("TPU", colocation=False))

    # Left at 3 this builds a placement group asking for 3 CPUs per exclusive chip,
    # which never becomes schedulable.
    assert RayResourcePool([4], max_colocate_count=3).max_colocate_count == 1


@pytest.mark.parametrize("colocation,expected", [(True, 3), (False, 1)])
def test_resource_pool_manager_matches_the_pool_it_creates(monkeypatch, colocation, expected):
    """The manager sizes bundles, so its cap has to agree with RayResourcePool's."""
    _use_platform(monkeypatch, _FakePlatform("GPU" if colocation else "TPU", colocation=colocation))

    manager = ResourcePoolManager(resource_pool_spec={"pool": [4]}, mapping={}, max_colocate_count=3)

    assert manager.max_colocate_count == expected


# ---------------------------------------------------------------------------
# set_numa_affinity
# ---------------------------------------------------------------------------


def test_set_numa_affinity_skips_on_tpu(monkeypatch):
    """NUMA pinning goes through NVML, which has no TPU equivalent."""
    monkeypatch.setattr(distributed_utils, "is_npu_available", False)
    monkeypatch.setattr(distributed_utils, "is_tpu_available", True)

    def _fail(*args, **kwargs):
        raise AssertionError("libnuma/NVML must not be touched on TPU")

    monkeypatch.setattr(distributed_utils.ctypes, "CDLL", _fail)

    distributed_utils.set_numa_affinity()  # must be a no-op, not an error


def test_set_numa_affinity_skips_on_npu(monkeypatch):
    """The pre-existing NPU early return must survive the TPU addition."""
    monkeypatch.setattr(distributed_utils, "is_npu_available", True)
    monkeypatch.setattr(distributed_utils, "is_tpu_available", False)

    def _fail(*args, **kwargs):
        raise AssertionError("libnuma/NVML must not be touched on NPU")

    monkeypatch.setattr(distributed_utils.ctypes, "CDLL", _fail)

    distributed_utils.set_numa_affinity()  # must be a no-op, not an error


def test_set_numa_affinity_attempts_setup_on_gpu(monkeypatch):
    """On GPU the NVML path is still entered (and failures stay non-fatal)."""
    monkeypatch.setattr(distributed_utils, "is_npu_available", False)
    monkeypatch.setattr(distributed_utils, "is_tpu_available", False)
    calls = []

    def _record(name):
        calls.append(name)
        raise OSError("libnuma.so not present in this test environment")

    monkeypatch.setattr(distributed_utils.ctypes, "CDLL", _record)

    distributed_utils.set_numa_affinity()  # swallows the failure

    assert calls == ["libnuma.so"]
