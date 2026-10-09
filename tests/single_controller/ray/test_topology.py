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

from __future__ import annotations


def test_gpu_bundle_can_host_multiple_colocated_worker_groups():
    import ray

    from tests.runtime.workers import VisibilityProbeWorker
    from verl.runtime import Runtime

    runtime = Runtime.from_config(
        {
            "backend": "ray",
            "env_vars": {},
            "ray": {"ray_init": {"num_cpus": 10, "num_gpus": 1}},
        }
    )
    try:
        pool = runtime.create_resource_pool(nnodes=1, processes_per_node=1, device_type="gpu")
        first = runtime.create_worker_group(VisibilityProbeWorker, on=pool)
        second = runtime.create_worker_group(VisibilityProbeWorker, on=pool)

        assert first.visible_devices() == ["0"]
        assert second.visible_devices() == ["0"]
    finally:
        runtime.close()
        if ray.is_initialized():
            ray.shutdown()


def test_legacy_gpu_pool_assigns_one_reserved_bundle_per_rank():
    import ray

    from tests.runtime.workers import VisibilityProbeWorker
    from verl.runtime import Runtime

    runtime = Runtime.from_config(
        {
            "backend": "ray",
            "env_vars": {},
            "ray": {"ray_init": {"num_cpus": 20, "num_gpus": 2}},
        }
    )
    try:
        pool = runtime.create_resource_pool(nnodes=1, processes_per_node=2, device_type="gpu")
        wg = runtime.create_worker_group(VisibilityProbeWorker, on=pool)

        assert set(wg.visible_devices()) == {"0", "1"}
    finally:
        runtime.close()
        if ray.is_initialized():
            ray.shutdown()
