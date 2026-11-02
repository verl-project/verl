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

from unittest.mock import MagicMock

from verl.checkpoint_engine.nccl_checkpoint_engine import (
    BroadcastOperation,
    MasterMetadata,
    NCCLCheckpointEngine,
)


def test_broadcast_operation_uses_dedicated_process_group():
    operation = BroadcastOperation.__new__(BroadcastOperation)
    operation.rank = 0
    operation.process_group = MagicMock()
    operation.bucket = MagicMock()
    operation.metadata = {"length": 1}
    operation.socket = MagicMock()
    operation.topic = "metadata"

    operation._run()

    operation.process_group.broadcast.assert_called_once_with(operation.bucket, 0)
    operation.process_group.broadcast.return_value.wait.assert_called_once_with()


def test_init_process_group_builds_torch_nccl_group(monkeypatch):
    engine = NCCLCheckpointEngine.__new__(NCCLCheckpointEngine)
    engine.rebuild_group = False
    engine.process_group = None
    engine.is_master = False
    engine.rank = None
    engine.world_size = None
    engine._connect_zmq_client = MagicMock()
    store = MagicMock()
    process_group = MagicMock()

    monkeypatch.setattr("verl.checkpoint_engine.nccl_checkpoint_engine.torch.distributed.TCPStore", store)
    monkeypatch.setattr(
        "verl.checkpoint_engine.nccl_checkpoint_engine.torch.distributed.ProcessGroupNCCL",
        process_group,
        raising=False,
    )
    monkeypatch.setattr("verl.checkpoint_engine.nccl_checkpoint_engine.torch.cuda.current_device", lambda: 0)
    metadata = MasterMetadata(zmq_ip="127.0.0.1", zmq_port=1234, store_port=2345, multi_sender=False)

    engine.init_process_group(rank=1, world_size=2, master_metadata=metadata, num_senders=1)

    store.assert_called_once()
    assert store.call_args.args[:4] == ("127.0.0.1", 2345, 2, False)
    process_group.assert_called_once_with(store.return_value, 1, 2, store.call_args.args[4])
    process_group.return_value.eager_connect_single_device.assert_called_once()
    process_group.return_value.barrier.return_value.wait.assert_called_once_with()
    engine._connect_zmq_client.assert_called_once_with(metadata)


def test_finalize_aborts_rebuilt_process_group(monkeypatch):
    engine = NCCLCheckpointEngine.__new__(NCCLCheckpointEngine)
    engine.rebuild_group = True
    engine.rank = 0
    engine.world_size = 2
    engine.process_group = MagicMock()
    process_group = engine.process_group
    engine.store = MagicMock()
    engine.send_buf = MagicMock()
    engine.recv_buf = MagicMock()
    empty_cache = MagicMock()
    monkeypatch.setattr("verl.checkpoint_engine.nccl_checkpoint_engine.torch.cuda.empty_cache", empty_cache)

    engine.finalize()

    process_group.abort.assert_called_once_with()
    assert engine.process_group is None
    assert engine.store is None
    assert engine.rank is None
    assert engine.world_size is None
    assert engine.send_buf is None
    assert engine.recv_buf is None
    empty_cache.assert_called_once_with()
