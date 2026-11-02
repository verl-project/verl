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

from datetime import timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import torch

from verl.checkpoint_engine import nccl_collective
from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine
from verl.checkpoint_engine.nccl_collective import (
    RayNCCLCollectiveGroup,
    TorchNCCLCollectiveGroup,
    create_nccl_collective_group,
)


@pytest.fixture(autouse=True)
def clear_torch_nccl_groups():
    TorchNCCLCollectiveGroup._groups.clear()
    yield
    TorchNCCLCollectiveGroup._groups.clear()


def test_factory_keeps_ray_nccl_as_the_default_implementation():
    group = create_nccl_collective_group("ray_nccl", "weights")

    assert isinstance(group, RayNCCLCollectiveGroup)
    assert group.group_name == "weights"


def test_checkpoint_engine_keeps_ray_backend_by_default(monkeypatch):
    monkeypatch.setattr("verl.checkpoint_engine.nccl_checkpoint_engine.cp", MagicMock())

    engine = NCCLCheckpointEngine(bucket_size=1024)

    assert engine.collective_backend == "ray_nccl"
    assert isinstance(engine.collective_group, RayNCCLCollectiveGroup)


def test_checkpoint_engine_allows_torch_backend_without_cupy(monkeypatch):
    monkeypatch.setattr("verl.checkpoint_engine.nccl_checkpoint_engine.cp", None)

    engine = NCCLCheckpointEngine(
        bucket_size=1024,
        collective_backend="torch_nccl",
        collective_backend_kwargs={"timeout_s": 17},
    )

    assert isinstance(engine.collective_group, TorchNCCLCollectiveGroup)
    assert engine.collective_group.timeout == timedelta(seconds=17)


def test_collective_backend_kwargs_are_backend_specific():
    with pytest.raises(ValueError, match="ray_nccl does not accept"):
        create_nccl_collective_group("ray_nccl", "weights", {"timeout_s": 17})
    with pytest.raises(ValueError, match="unsupported backend kwargs"):
        create_nccl_collective_group("torch_nccl", "weights", {"unknown": True})


def test_ray_nccl_adapter_preserves_existing_collective_calls(monkeypatch):
    collective = MagicMock()
    collective.is_group_initialized.return_value = True
    monkeypatch.setattr(nccl_collective, "ray_collective", collective)
    group = RayNCCLCollectiveGroup("weights")
    tensor = MagicMock()

    assert group.initialized
    group.initialize(world_size=4, rank=2, master_addr="unused", master_port=None)
    group.broadcast(tensor, src_rank=1)
    group.barrier()
    group.destroy()

    collective.init_collective_group.assert_called_once_with(4, 2, "nccl", "weights")
    collective.broadcast.assert_called_once_with(tensor, src_rank=1, group_name="weights")
    collective.barrier.assert_called_once_with("weights")
    collective.destroy_collective_group.assert_called_once_with("weights")


def test_default_backend_runs_real_ray_nccl_without_process_group_nccl():
    """Exercise the unchanged default with real Ray actors and GPU collectives."""
    pytest.importorskip("cupy")
    ray = pytest.importorskip("ray")
    try:
        from cupy.cuda.nccl import get_version
    except ImportError:
        pytest.skip("CuPy was built without NCCL")
    if get_version() <= 0:
        pytest.skip("CuPy NCCL is unavailable")
    if torch.cuda.device_count() < 2:
        pytest.skip("real Ray NCCL test requires at least two GPUs")

    @ray.remote(num_gpus=1)
    def run_rank(rank: int, group_name: str):
        import cupy as actor_cp
        import torch as actor_torch

        from verl.checkpoint_engine.nccl_collective import create_nccl_collective_group as create_group

        def reject_process_group_nccl(*args, **kwargs):
            raise AssertionError("default ray_nccl path created ProcessGroupNCCL")

        actor_torch.distributed.ProcessGroupNCCL = reject_process_group_nccl
        group = create_group("ray_nccl", group_name)
        tensor = actor_cp.arange(4, dtype=actor_cp.int32) if rank == 0 else actor_cp.zeros(4, dtype=actor_cp.int32)
        try:
            group.initialize(world_size=2, rank=rank, master_addr="unused", master_port=None)
            group.broadcast(tensor, src_rank=0)
            group.barrier()
            return tensor.get().tolist()
        finally:
            group.destroy()

    ray.init(num_gpus=2)
    try:
        group_name = f"default-ray-nccl-{uuid4().hex}"
        results = ray.get([run_rank.remote(rank, group_name) for rank in range(2)])
        assert results == [[0, 1, 2, 3], [0, 1, 2, 3]]
    finally:
        ray.shutdown()


def test_torch_nccl_initializes_a_prefixed_process_group(monkeypatch):
    tcp_store = MagicMock()
    prefix_store = MagicMock()
    process_group = MagicMock()
    options = MagicMock()
    options_type = MagicMock(return_value=options)

    monkeypatch.setattr(torch.distributed, "TCPStore", tcp_store)
    monkeypatch.setattr(torch.distributed, "PrefixStore", prefix_store)
    monkeypatch.setattr(torch.distributed, "ProcessGroupNCCL", process_group, raising=False)
    monkeypatch.setattr(nccl_collective, "get_device_id", lambda: 0)
    process_group.Options = options_type

    group = TorchNCCLCollectiveGroup("weights", timeout_s=17)
    group.initialize(world_size=4, rank=2, master_addr="127.0.0.1", master_port=12345)

    timeout = timedelta(seconds=17)
    tcp_store.assert_called_once_with("127.0.0.1", 12345, 4, False, timeout)
    prefix_store.assert_called_once_with("weights", tcp_store.return_value)
    options_type.assert_called_once_with()
    assert options._timeout == timeout
    process_group.assert_called_once_with(prefix_store.return_value, 2, 4, options)
    process_group.return_value.eager_connect_single_device.assert_called_once_with(torch.device("cuda", 0))
    assert group.initialized
    assert group._state().store is prefix_store.return_value
    assert group._state().process_group is process_group.return_value


def test_torch_nccl_keeps_named_groups_independent(monkeypatch):
    process_groups = [MagicMock(), MagicMock()]
    process_group_type = MagicMock(side_effect=process_groups)
    process_group_type.Options = MagicMock(side_effect=[MagicMock(), MagicMock()])
    monkeypatch.setattr(torch.distributed, "TCPStore", MagicMock(side_effect=[MagicMock(), MagicMock()]))
    monkeypatch.setattr(torch.distributed, "PrefixStore", MagicMock(side_effect=[MagicMock(), MagicMock()]))
    monkeypatch.setattr(torch.distributed, "ProcessGroupNCCL", process_group_type, raising=False)
    monkeypatch.setattr(nccl_collective, "get_device_id", lambda: 0)

    weights = TorchNCCLCollectiveGroup("weights", timeout_s=17)
    optimizer = TorchNCCLCollectiveGroup("optimizer", timeout_s=17)
    weights.initialize(world_size=2, rank=0, master_addr="127.0.0.1", master_port=12345)
    optimizer.initialize(world_size=2, rank=0, master_addr="127.0.0.1", master_port=12346)

    weights_by_name = TorchNCCLCollectiveGroup("weights", timeout_s=17)
    assert weights_by_name.initialized
    assert weights_by_name._state().process_group is process_groups[0]
    assert optimizer._state().process_group is process_groups[1]
    weights_by_name.destroy()
    assert not weights.initialized
    assert optimizer.initialized
    process_groups[0].abort.assert_called_once_with()
    process_groups[1].abort.assert_not_called()


def test_torch_nccl_translates_collective_options_and_waits(monkeypatch):
    broadcast_options = MagicMock()
    barrier_options = MagicMock()
    monkeypatch.setattr(torch.distributed, "BroadcastOptions", MagicMock(return_value=broadcast_options))
    monkeypatch.setattr(torch.distributed, "BarrierOptions", MagicMock(return_value=barrier_options))
    monkeypatch.setattr(nccl_collective, "get_device_id", lambda: 3)

    group = TorchNCCLCollectiveGroup("weights", timeout_s=300)
    process_group = MagicMock()
    group._groups[group.group_name] = nccl_collective._TorchNCCLGroupState(MagicMock(), process_group)
    group._check_tensor = MagicMock()
    tensor = MagicMock()

    group.broadcast(tensor, src_rank=1)
    group.barrier()

    assert broadcast_options.rootRank == 1
    assert broadcast_options.rootTensor == 0
    process_group.broadcast.assert_called_once_with([tensor], broadcast_options)
    process_group.broadcast.return_value.wait.assert_called_once_with()
    assert barrier_options.device_ids == [3]
    process_group.barrier.assert_called_once_with(barrier_options)
    process_group.barrier.return_value.wait.assert_called_once_with()


def test_torch_nccl_rejects_missing_store_port_and_cleans_up():
    group = TorchNCCLCollectiveGroup("weights", timeout_s=300)

    with pytest.raises(ValueError, match="TCPStore port"):
        group.initialize(world_size=2, rank=0, master_addr="127.0.0.1", master_port=None)

    process_group = MagicMock()
    group._groups[group.group_name] = nccl_collective._TorchNCCLGroupState(MagicMock(), process_group)
    group.destroy()

    process_group.abort.assert_called_once_with()
    assert not group.initialized
