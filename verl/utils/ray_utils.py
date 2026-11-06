# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Contains commonly used utilities for ray
"""

import asyncio
import concurrent.futures
import functools
import inspect
import os
from typing import Any, Optional

import ray

# Optional Ray custom resource that constrains where lightweight loop workers
# (agent-loop / reward-loop actors) may be scheduled. In a shared or
# heterogeneous Ray cluster, ``ray.nodes()`` also returns nodes that belong to
# other jobs or worker groups. Round-robin over all of them with a soft
# NodeAffinity can place a loop worker on a foreign node that does not run this
# job's runtime image, which then crashes on import (e.g. ModuleNotFoundError).
# Set this env var to a Ray custom resource that only this job's nodes advertise
# to keep loop workers on them. Unset -> historical behavior (all alive nodes).
LOOP_WORKER_NODE_RESOURCE_ENV = "VERL_LOOP_WORKER_NODE_RESOURCE"

# Fractional amount requested from the node resource so it only acts as a
# scheduling gate, never a real capacity constraint. Mirrors the tiny reservation
# verl already uses for accelerator_type bundles in single_controller/ray/base.py.
_LOOP_WORKER_NODE_RESOURCE_AMOUNT = 1e-4


def get_loop_worker_node_resource() -> Optional[str]:
    """Return the configured loop-worker node resource, or None if unset."""
    value = os.environ.get(LOOP_WORKER_NODE_RESOURCE_ENV, "").strip()
    return value or None


def schedulable_loop_worker_node_ids(node_resource: Optional[str] = None) -> list[str]:
    """Return NodeIDs eligible to host agent-loop / reward-loop workers.

    Without ``node_resource`` this preserves the historical behavior of
    scheduling across every alive node that has CPU. When ``node_resource`` is
    given, only nodes that advertise that Ray custom resource are returned, so a
    shared/heterogeneous cluster never round-robins a loop worker onto a foreign
    node that lacks this job's runtime.
    """
    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
    if node_resource:
        nodes = [node for node in nodes if node["Resources"].get(node_resource, 0) > 0]
        if not nodes:
            raise RuntimeError(
                f"No alive Ray node advertises the resource {node_resource!r} requested via "
                f"{LOOP_WORKER_NODE_RESOURCE_ENV}; cannot place agent-loop/reward-loop workers. "
                "Ensure this job's node group exports that custom resource."
            )
    return [node["NodeID"] for node in nodes]


def loop_worker_node_affinity_resources(node_resource: Optional[str] = None) -> Optional[dict]:
    """Actor ``resources`` requirement that pins a loop worker to a node group.

    A soft NodeAffinity alone can still fall back to a foreign node when the
    targeted node is momentarily full. Requiring a tiny amount of the node
    resource makes Ray refuse to schedule the actor anywhere that does not
    advertise it, closing that leak. Returns None when no resource is configured.
    """
    if not node_resource:
        return None
    return {node_resource: _LOOP_WORKER_NODE_RESOURCE_AMOUNT}


def available_cpu_per_node() -> dict[str, float]:
    """Best-effort map of NodeID -> currently available CPU.

    Ray does not expose per-node *available* resources through the public
    ``ray.nodes()`` payload, so this reads the driver-side resource view when it
    is reachable and returns an empty mapping otherwise. Callers must treat an
    empty result as "unknown" and fall back to round-robin placement.
    """
    try:
        from ray._private.state import state as ray_state

        per_node = ray_state._available_resources_per_node()
    except Exception:
        return {}
    return {node_id: float(resources.get("CPU", 0.0)) for node_id, resources in (per_node or {}).items()}


def assign_loop_worker_nodes(
    node_ids: list[str],
    num_workers: int,
    available_cpu: Optional[dict[str, float]] = None,
    cpus_per_worker: float = 0.0,
) -> list[str]:
    """Assign each loop worker to a candidate node, spreading by available CPU.

    Prefers the node with the most currently-available CPU so workers fan out
    across a heterogeneous group instead of stacking on ``node_ids[0]``. Each
    placement provisionally debits ``cpus_per_worker`` from the chosen node so
    later workers favor the next-emptiest node. Falls back to the historical
    round-robin order when ``available_cpu`` is empty (Ray reported no per-node
    availability). ``node_ids`` order breaks ties deterministically.
    """
    if num_workers <= 0:
        return []
    if not node_ids:
        raise ValueError("assign_loop_worker_nodes requires at least one candidate node")
    if not available_cpu:
        return [node_ids[i % len(node_ids)] for i in range(num_workers)]

    remaining = {node_id: float(available_cpu.get(node_id, 0.0)) for node_id in node_ids}
    rank = {node_id: i for i, node_id in enumerate(node_ids)}
    assignments = []
    for _ in range(num_workers):
        node_id = min(node_ids, key=lambda nid: (-remaining[nid], rank[nid]))
        assignments.append(node_id)
        remaining[node_id] -= cpus_per_worker
    return assignments


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def parallel_put(data_list: list[Any], max_workers: Optional[int] = None):
    """
    Puts a list of data into the Ray object store in parallel using a thread pool.

    Args:
        data_list (List[Any]): A list of Python objects to be put into the Ray object store.
        max_workers (int, optional): The maximum number of worker threads to use.
                                     Defaults to min(len(data_list), 16).

    Returns:
        List[ray.ObjectRef]: A list of Ray object references corresponding to the input data_list,
                             maintaining the original order.
    """
    assert len(data_list) > 0, "data_list must not be empty"

    def put_data(index, data):
        return index, ray.put(data)

    if max_workers is None:
        max_workers = min(len(data_list), 16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]
        res_lst = []
        for future in concurrent.futures.as_completed(data_list_f):
            res_lst.append(future.result())

        # reorder based on index
        output = [None for _ in range(len(data_list))]
        for res in res_lst:
            index, data_ref = res
            output[index] = data_ref

    return output


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def auto_await(func):
    """Auto await a coroutine function.

    Handles three cases:
    1. When the decorated function is called with await: returns the coroutine
       so the caller can await it.
    2. When called directly and there is no running event loop: runs the
       coroutine with asyncio.run() and returns the result.
    3. When called directly and the event loop is already running: runs the
       coroutine (e.g. in a thread pool to avoid deadlock) and returns the result.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        coro = func(*args, **kwargs)

        if not inspect.iscoroutine(coro):
            return coro

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Case 1: No running loop -> run with asyncio.run()
        if loop is None:
            return asyncio.run(coro)

        # Case 2: Running loop -> return coro if caller will await
        caller_frame = inspect.currentframe()
        if caller_frame is not None:
            caller_frame = caller_frame.f_back
        caller_is_async = caller_frame is not None and (caller_frame.f_code.co_flags & inspect.CO_COROUTINE) != 0
        if caller_is_async:
            return coro

        # Case 3: Running loop -> run coro in thread pool
        # (cannot block the loop thread without deadlock)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return wrapper
