# Copyright 2025 Individual Contributor: Zehan Zhu
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
"""Exercise resource and teacher components without loading a model."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.single_controller.ray.base import RayResourcePool, ResourcePoolManager, merge_resource_pool
from verl.single_controller.ray.instance_naming import InstanceNameBuilder, build_weight_transfer_socket

VLLM_ROLLOUT = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout"


def _load_vllm_colocate_method(method_name: str):
    """Execute a method body from ``vllm_rollout/utils.py`` without importing vLLM.

    The module pulls in heavy dependencies at import time, so the method is
    lifted out of the source with ``ast`` and run against a small namespace.
    """
    source = (VLLM_ROLLOUT / "utils.py").read_text(encoding="utf-8")
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "vLLMColocateWorkerExtension"
    )
    method = next(node for node in cls.body if getattr(node, "name", None) == method_name)
    method.decorator_list = []
    namespace = {
        "os": os,
        "build_weight_transfer_socket": build_weight_transfer_socket,
        # The fake worker has no parallel config, so the real helper would be
        # the identity on local_rank as well.
        "_resolve_vllm_weight_sync_local_rank": lambda rank, parallel_config: int(rank),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(VLLM_ROLLOUT / "utils.py"), "exec"), namespace)
    return namespace[method_name]


def test_socket_byte_limit_and_identity():
    endpoints = {
        build_weight_transfer_socket("job", "policy_" + "x" * 200 + suffix, replica, rank)
        for suffix in ("a", "b")
        for replica in (0, 1)
        for rank in (0, 1)
    }
    assert len(endpoints) == 8
    assert all(len(endpoint.removeprefix("ipc://").encode()) <= 103 for endpoint in endpoints)
    assert build_weight_transfer_socket("job", None, 0, 0) == ("ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock")


def test_sender_and_receiver_agree_on_the_socket_path(monkeypatch):
    """The trainer reads the id from config, the vLLM worker from the environment."""
    get_zmq_handle = _load_vllm_colocate_method("_get_zmq_handle")
    worker = SimpleNamespace(local_rank=3, model_runner=None)

    monkeypatch.setenv("VERL_RAY_JOB_ID", "job-1")
    monkeypatch.setenv("VERL_REPLICA_RANK", "2")
    monkeypatch.delenv("VERL_ZMQ_BASE_TRAINER_RANK", raising=False)

    # Sender side: ServerAdapter passes the id it read from the rollout config.
    sender = build_weight_transfer_socket("job-1", "policy_a", 2, 3)
    monkeypatch.setenv("VERL_INSTANCE_ID", "policy_a")
    assert get_zmq_handle(worker) == sender

    # Without an instance id both sides fall back to the legacy path.
    monkeypatch.delenv("VERL_INSTANCE_ID", raising=False)
    assert get_zmq_handle(worker) == build_weight_transfer_socket("job-1", None, 2, 3)


def test_cross_instance_merge_rejected_before_allocating(monkeypatch):
    pools = [RayResourcePool([1], use_gpu=False, instance_name_builder=InstanceNameBuilder(i)) for i in ("a", "b")]
    for pool in pools:
        monkeypatch.setattr(pool, "get_placement_groups", lambda **kw: pytest.fail("must reject before allocation"))
    with pytest.raises(ValueError, match="instance"):
        merge_resource_pool(*pools)


def test_manager_creates_pools_with_instance_identity(monkeypatch):
    monkeypatch.setattr(ResourcePoolManager, "_check_resource_available", lambda self: None)
    manager = ResourcePoolManager({"global_pool": [1]}, {}, instance_name_builder=InstanceNameBuilder("a"))
    manager.create_resource_pool()
    assert manager.resource_pool_dict["global_pool"].instance_name_builder.instance_id == "a"


def test_teacher_inherits_pool_identity_without_mutating_shared_config(monkeypatch):
    from verl.experimental.teacher_loop import teacher_model as module
    from verl.workers.config import RolloutConfig

    inference = RolloutConfig(name="vllm", tensor_model_parallel_size=1)
    captured = []

    def replica(**kwargs):
        captured.append(kwargs["config"])
        return SimpleNamespace(init_colocated=AsyncMock(), _server_handle=None, _server_address="localhost")

    monkeypatch.setattr(module, "get_rollout_replica_class", lambda name: replica)
    monkeypatch.setattr(module, "split_resource_pool", lambda pool, split_size: [pool])
    monkeypatch.setattr(module.RLInsightLogger, "enabled", lambda: False)
    for instance_id in ("a", "b"):
        manager = object.__new__(module.TeacherModelManager)
        manager.teacher_model_config = SimpleNamespace(
            inference=inference, per_replica_world_size=1, num_replicas=1, key="teacher", model_path="unused"
        )
        manager.distillation_config = SimpleNamespace(n_gpus_per_node=1)
        manager.resource_pool = RayResourcePool(
            [1], use_gpu=False, instance_name_builder=InstanceNameBuilder(instance_id)
        )
        manager._validate_replica_node_alignment = lambda *args: None
        manager._initialize_llm_servers()
    assert [config.instance_id for config in captured] == ["a", "b"]
    assert inference.instance_id is None
