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

"""Historical ``verl.single_controller.ray.base`` import path.

Ray implementation modules live alongside this file. This module keeps the
pre-runtime public names and the worker-aliveness helpers used by existing
tests and trainers.
"""

import threading
import time

from ray.util.state import get_actor

from verl.single_controller.base.worker import Worker
from verl.single_controller.base.worker_group import check_workers_alive
from verl.single_controller.ray.actor import RayClassWithInitArgs
from verl.single_controller.ray.fused import (
    FusedWorkerCLSName,
    _bind_workers_method_to_parent,  # noqa: F401
    _unwrap_ray_remote,  # noqa: F401
    create_colocated_worker_cls,
    create_colocated_worker_cls_fused,
    create_colocated_worker_raw_cls,
)
from verl.single_controller.ray.fused import _determine_base_class as _determine_fsdp_megatron_base_class  # noqa: F401
from verl.single_controller.ray.resource_pool import (
    RayResourcePool,
    SubRayResourcePool,
    get_random_string,
    merge_resource_pool,
    sort_placement_group_by_node_ip,
    split_resource_pool,
)
from verl.single_controller.ray.resource_pool_manager import ResourcePoolManager
from verl.single_controller.ray.worker_group import RayWorkerGroup as _RayWorkerGroup
from verl.single_controller.ray.worker_group import (
    _get_master_addr_port as get_master_addr_port,
)


class RayWorkerGroup(_RayWorkerGroup):
    """RayWorkerGroup with the legacy aliveness-monitor methods."""

    def _is_worker_alive(self, worker) -> bool:
        """Check if a worker actor is still alive.

        Args:
            worker: Worker actor handle.

        Returns:
            bool: True if the worker is alive, False otherwise.
        """
        state = get_actor(worker._actor_id.hex())
        return state is not None and state.get("state", "undefined") == "ALIVE"

    def _block_until_all_workers_alive(self) -> None:
        """Blocks until all workers in the group are alive."""
        while not all(self._is_worker_alive(worker) for worker in self._workers):
            time.sleep(1)

    def start_worker_aliveness_check(self, every_n_seconds: float = 1) -> None:
        """Starts a background thread to monitor worker aliveness.

        Args:
            every_n_seconds (int): Interval between aliveness checks
        """
        # before starting checking worker aliveness, make sure all workers are already alive
        self._block_until_all_workers_alive()
        self._checker_thread = threading.Thread(
            target=check_workers_alive,
            args=(self._workers, self._is_worker_alive, every_n_seconds),
        )
        self._checker_thread.start()


__all__ = [
    "Worker",
    "FusedWorkerCLSName",
    "RayClassWithInitArgs",
    "RayResourcePool",
    "SubRayResourcePool",
    "RayWorkerGroup",
    "ResourcePoolManager",
    "create_colocated_worker_cls",
    "create_colocated_worker_cls_fused",
    "create_colocated_worker_raw_cls",
    "get_master_addr_port",
    "merge_resource_pool",
    "split_resource_pool",
    "get_random_string",
    "sort_placement_group_by_node_ip",
]
