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
"""Live Monarch WorkerGroup RPC and topology-declared TorchStore."""

from __future__ import annotations

import pickle
import threading

import pytest
import torch
from omegaconf import OmegaConf

from tests.runtime.workers import EchoWorker
from tests.runtime_fixtures import monarch_runtime_config
from verl.runtime import ClassWithInitArgs, Dispatch, Runtime, Worker, register


class MixedAffinityRole(Worker):
    def __init__(self, base: int) -> None:
        super().__init__()
        self.base = base
        self.owner_thread = threading.get_ident()
        self.setup_thread: int | None = None

    def _setup_visible_devices(self) -> None:
        self.setup_thread = threading.get_ident()

    def _result(self, value: int) -> tuple[bool, bool, int]:
        current = threading.get_ident()
        return current == self.owner_thread, self.setup_thread == self.owner_thread, self.base + value

    @register(blocking=True)
    def sync_owner(self, value: int) -> tuple[bool, bool, int]:
        return self._result(value)

    @register(blocking=False)
    async def async_owner(self, value: int) -> tuple[bool, bool, int]:
        return self._result(value)

    def close(self) -> None:
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("mixed role closed outside its owner thread")


class SyncAffinityRole(Worker):
    def __init__(self) -> None:
        super().__init__()
        self.owner_thread = threading.get_ident()
        self.setup_thread: int | None = None

    def _setup_visible_devices(self) -> None:
        self.setup_thread = threading.get_ident()

    @register(blocking=True)
    def owner_is_consistent(self) -> tuple[bool, bool]:
        current = threading.get_ident()
        return current == self.owner_thread, self.setup_thread == self.owner_thread

    def close(self) -> None:
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("sync role closed outside its owner thread")


class CpuOnlyEchoWorker(EchoWorker):
    def _setup_visible_devices(self) -> None:
        # Topology DevicePools are GPU-shaped, but these integration tests run on CPU-only CI.
        pass


class NeoProtoTorchStoreWorker(CpuOnlyEchoWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def roundtrip_neoproto(self):
        from verl.experimental.neoproto.storage.torchstore import TorchStorageEngine
        from verl.experimental.neoproto.views import DataProto

        source = DataProto.from_dict(
            tensors={
                "input_ids": torch.arange(8).reshape(2, 4),
                "attention_mask": torch.ones((2, 4), dtype=torch.int64),
            },
            storage=TorchStorageEngine(),
        )
        transported = pickle.loads(pickle.dumps(source))
        values = transported.materialize(["input_ids", "attention_mask"])
        transported.release()
        return {key: value.tolist() for key, value in values.items()}


class ControllerNeoProtoWorker(Worker):
    @register(blocking=True)
    def run(self, config) -> None:
        from verl.experimental.neoproto.storage import configure_storage_engine
        from verl.experimental.neoproto.views import DataProto
        from verl.runtime.config import select_backend

        configure_storage_engine(select_backend(config.runtime))
        payload = DataProto.from_single_dict({"input_ids": torch.arange(4).reshape(1, 4)})
        values = payload.materialize(["input_ids"])
        assert values["input_ids"].tolist() == [[0, 1, 2, 3]]
        payload.release()


@pytest.fixture(scope="module")
def monarch_pkg():
    return pytest.importorskip("monarch")


@pytest.fixture(scope="module")
def torchstore_pkg():
    return pytest.importorskip("torchstore")


@pytest.fixture
def torchstore_runtime_config(monarch_local_job):
    config = monarch_runtime_config()
    config["monarch"]["object_store"] = {
        "store_name_prefix": "runtime_test_store",
        "timeout_s": 30.0,
    }
    config["topology"] = {
        "clusters": [{"name": "hybrid_pool", "nnodes": 1, "n_gpus_per_node": 1}],
        "device_pools": [{"name": "train_pool", "cluster": "hybrid_pool", "nnodes": 1, "n_gpus_per_node": 1}],
        "models": [
            {
                "name": "actor",
                "worker": "actor",
                "config_key": "actor_rollout_ref",
                "resource_pool": "train_pool",
            }
        ],
    }
    return config


def test_fused_worker_group_preserves_role_owners_across_recreate(monarch_pkg, monarch_local_job):
    cfg = monarch_runtime_config()
    for base in (5, 9):
        with Runtime.from_config(cfg) as runtime:
            pool = runtime.create_resource_pool(nnodes=1, processes_per_node=1, device_type="cpu")
            roles = runtime.create_worker_group(
                {
                    "mixed": ClassWithInitArgs(MixedAffinityRole, base),
                    "sync": SyncAffinityRole,
                },
                on=pool,
            )
            assert roles["mixed"].sync_owner(2) == [(True, True, base + 2)]
            assert roles["mixed"].async_owner(3).result() == [(True, True, base + 3)]
            assert roles["sync"].owner_is_consistent() == [(True, True)]


def test_controller_task_runner_uses_global_torchstore(
    monarch_pkg,
    torchstore_pkg,
    torchstore_runtime_config,
):
    from verl.trainer.main_ppo import _run_task

    torchstore_runtime_config["monarch"]["object_store"]["store_name_prefix"] = "controller_runtime_test"
    config = OmegaConf.create(
        {
            "runtime": {"backend": "monarch"},
        }
    )

    _run_task(torchstore_runtime_config, ControllerNeoProtoWorker, config)


def test_torchstore_is_installed_on_worker_group_by_default(
    monarch_pkg,
    torchstore_pkg,
    torchstore_runtime_config,
):
    with Runtime.from_config(torchstore_runtime_config) as runtime:
        trainer = runtime.create_worker_group(NeoProtoTorchStoreWorker, on=runtime.model_resource_pool("actor"))
        assert trainer.object_store_roundtrip("tests/trainer-roundtrip", {"value": 7}) == [{"value": 7}]
        assert trainer.roundtrip_neoproto() == [
            {
                "input_ids": [[0, 1, 2, 3], [4, 5, 6, 7]],
                "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            }
        ]
