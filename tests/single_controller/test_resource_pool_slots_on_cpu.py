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
"""Driver-local slot health: no recovery controller, cluster or accelerator required."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import ray

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
    extract_pg_from_exist,
    merge_resource_pool,
    split_resource_pool,
)


@ray.remote
class SlotProbe(Worker):
    def placement(self):
        context = ray.get_runtime_context()
        return {
            "actor_id": context.get_actor_id(),
            "node_id": context.get_node_id(),
            "placement_group_id": context.get_placement_group_id(),
        }


def make_pool(name="pool", counts=(2,)):
    pool = RayResourcePool(list(counts))
    pool.pgs = [
        SimpleNamespace(id=SimpleNamespace(hex=lambda i=i: f"{name}-{i}"), bundle_count=count, ready=Mock())
        for i, count in enumerate(counts)
    ]
    return pool


def test_marking_one_slot_keeps_other_slots_and_pg():
    pool = make_pool(counts=(2, 2))
    pg = pool.pgs[0]
    pool.mark_slot_unavailable(pg, 1, "device_failed")
    pool.mark_slot_unavailable(pg, 1, "device_failed")
    assert pool.unavailable_slots() == [{"pg_id": "pool-0", "bundle_index": 1, "reason": "device_failed"}]
    assert pool.is_slot_available(pg, 0)
    assert not pool.is_slot_available(pg, 1)
    assert pool.is_slot_available(pool.pgs[1], 1)
    assert pool.world_size == 4
    assert pool.get_placement_groups()[0] is pg


def test_unequal_bundle_counts_stay_associated_when_pgs_are_reordered():
    pool = RayResourcePool([1, 2], use_gpu=False)
    one_bundle, two_bundles = make_pool(counts=(1, 2)).pgs
    with (
        patch("verl.single_controller.ray.base.placement_group", side_effect=[one_bundle, two_bundles]),
        patch("verl.single_controller.ray.base.ray.get"),
        patch(
            "verl.single_controller.ray.base.sort_placement_group_by_node_ip",
            return_value=[two_bundles, one_bundle],
        ),
    ):
        pool.get_placement_groups()

    assert [(pg.id.hex(), index) for pg, index in pool.resource_slots()] == [
        (two_bundles.id.hex(), 0),
        (two_bundles.id.hex(), 1),
        (one_bundle.id.hex(), 0),
    ]
    pool.mark_slot_unavailable(two_bundles, 1, "device_failed")
    assert not pool.is_slot_available(two_bundles, 1)


@pytest.mark.parametrize("index", [-1, 2])
def test_out_of_range_slot_is_rejected(index):
    pool = make_pool()
    with pytest.raises(ValueError, match="outside"):
        pool.mark_slot_unavailable(pool.pgs[0], index)
    with pytest.raises(ValueError, match="outside"):
        pool.is_slot_available(pool.pgs[0], index)


def test_unknown_pg_and_empty_reason_are_rejected():
    pool, foreign = make_pool(), make_pool("foreign")
    with pytest.raises(ValueError, match="outside"):
        pool.mark_slot_unavailable(foreign.pgs[0], 0)
    with pytest.raises(ValueError, match="reason"):
        pool.mark_slot_unavailable(pool.pgs[0], 0, "")
    assert not pool.unavailable_slots()


def test_slot_queries_do_not_allocate_uninitialized_pool():
    pool = RayResourcePool([2])
    with pytest.raises(RuntimeError, match="initialized"):
        pool.resource_slots()
    assert pool.pgs is None


def test_split_and_nested_views_share_masks_in_both_directions():
    pool = make_pool(counts=(2, 2))
    first, second = split_resource_pool(pool, 2)
    healthy, bad = split_resource_pool(second, 1)
    bad.mark_slot_unavailable(pool.pgs[1], 1)
    assert not pool.is_slot_available(pool.pgs[1], 1)
    first.assert_slots_available()
    healthy.assert_slots_available()
    with pytest.raises(RuntimeError, match="unavailable"):
        second.assert_slots_available()
    with pytest.raises(ValueError, match="outside"):
        first.mark_slot_unavailable(pool.pgs[1], 1)
    pool.mark_slot_unavailable(pool.pgs[1], 0)
    with pytest.raises(RuntimeError, match="unavailable"):
        healthy.assert_slots_available()


def test_merge_shares_later_marks_with_source_pools():
    left, right = make_pool("left"), make_pool("right")
    merged = merge_resource_pool(left, right)
    right.mark_slot_unavailable(right.pgs[0], 1)
    assert len(merged.unavailable_slots()) == 1
    merged.mark_slot_unavailable(left.pgs[0], 0)
    assert not left.is_slot_available(left.pgs[0], 0)


def test_overlapping_pg_merge_is_rejected():
    pool = make_pool()
    with pytest.raises(ValueError, match="overlapping"):
        merge_resource_pool(pool, pool)


def test_extract_preserves_marks_before_and_after_reuse():
    source, target = make_pool(), RayResourcePool([2])
    source.mark_slot_unavailable(source.pgs[0], 0)
    target.pgs = extract_pg_from_exist({"actor": source}, ["actor"], target)
    assert len(target.unavailable_slots()) == 1
    target.mark_slot_unavailable(target.pgs[0], 1)
    assert len(source.unavailable_slots()) == 2


@pytest.mark.parametrize("subpool", [False, True])
def test_worker_launch_fails_before_master_or_actor_creation(subpool):
    pool = make_pool()
    pool.mark_slot_unavailable(pool.pgs[0], 0)
    worker_group = SimpleNamespace(device_name="cuda", _get_master_addr_port=Mock(), _create_worker=Mock())
    initializer = RayWorkerGroup._init_with_resource_pool
    if subpool:
        pool = split_resource_pool(pool, 1)[0]
        initializer = RayWorkerGroup._init_with_subresource_pool
    with pytest.raises(RuntimeError, match="unavailable"):
        initializer(worker_group, pool, None, False, False)
    worker_group._get_master_addr_port.assert_not_called()
    worker_group._create_worker.assert_not_called()


def test_healthy_subpool_launch_keeps_original_bundle_index():
    pool = make_pool()
    pool.mark_slot_unavailable(pool.pgs[0], 0)
    healthy = split_resource_pool(pool, 1)[1]
    worker_group = SimpleNamespace(
        device_name="cuda", _get_master_addr_port=Mock(), _create_worker=Mock(), _ray_master_port_range=None
    )
    RayWorkerGroup._init_with_subresource_pool(worker_group, healthy, None, False, False)
    assert worker_group._world_size == 1
    worker_group._create_worker.assert_called_once()
    assert worker_group._create_worker.call_args.kwargs["local_rank"] == 1
    assert worker_group._create_worker.call_args.kwargs["pg"] is pool.pgs[0]


def test_real_cpu_placement_group_rejects_bad_slot_and_launches_on_healthy_slot():
    ray.init(num_cpus=4)
    pool = RayResourcePool([2], use_gpu=False, max_colocate_count=1, name_prefix="slot_integration_")
    worker_group = None
    try:
        pg = pool.get_placement_groups(strategy="STRICT_PACK")[0]
        bad_pool, healthy_pool = split_resource_pool(pool, [1, 1])
        pool.mark_slot_unavailable(pg, 0, reason="integration_test_fault")

        with pytest.raises(RuntimeError, match="integration_test_fault"):
            RayWorkerGroup(
                bad_pool,
                RayClassWithInitArgs(cls=SlotProbe),
                name_prefix="bad_slot_",
            )

        worker_group = RayWorkerGroup(
            healthy_pool,
            RayClassWithInitArgs(cls=SlotProbe),
            name_prefix="healthy_slot_",
        )
        placement = worker_group.execute_all_sync("placement")
        assert len(placement) == 1
        assert placement[0]["actor_id"]
        assert placement[0]["node_id"]
        assert placement[0]["placement_group_id"] == pg.id.hex()
        assert healthy_pool.is_slot_available(pg, 1)
    finally:
        if worker_group is not None:
            for worker in worker_group._workers:
                ray.kill(worker)
        if pool.pgs is not None:
            for pg in pool.pgs:
                ray.util.remove_placement_group(pg)
        ray.shutdown()
