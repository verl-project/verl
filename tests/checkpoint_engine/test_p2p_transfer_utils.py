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

from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import torch

from verl.checkpoint_engine.p2p.trainer_updater import P2PTrainerWeightUpdater
from verl.checkpoint_engine.p2p.transfer_utils import (
    P2PRolloutTopology,
    P2PTransferManager,
    RemoteTransferPlan,
    TransferTaskP2PMeta,
    build_rollout_topology_from_replicas,
    create_transfer_engine,
    filter_server_args_dict,
    iter_named_tensor_buckets,
    resolve_mooncake_transfer_engine_settings,
    serialize_p2p_rollout_metadata,
    unregister_cpu_memory,
)


class _FakeReplica:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.config = type("Cfg", (), {"pipeline_model_parallel_size": 1})()


def test_iter_named_tensor_buckets_splits_by_size():
    tensors = [
        ("a", torch.zeros(100, dtype=torch.float32)),
        ("b", torch.zeros(100, dtype=torch.float32)),
        ("c", torch.zeros(100, dtype=torch.float32)),
    ]
    # Each tensor is 400 bytes; bucket limit 801 fits two tensors, then one.
    buckets = list(iter_named_tensor_buckets(iter(tensors), bucket_bytes=801, clone_tensors=False))
    assert len(buckets) == 2
    assert [name for name, _ in buckets[0]] == ["a", "b"]
    assert [name for name, _ in buckets[1]] == ["c"]


def test_iter_named_tensor_buckets_oversized_tensor_gets_own_bucket():
    tensors = [
        ("big", torch.zeros(200, dtype=torch.float32)),
        ("small", torch.zeros(10, dtype=torch.float32)),
    ]
    buckets = list(iter_named_tensor_buckets(iter(tensors), bucket_bytes=100, clone_tensors=False))
    assert len(buckets) == 2
    assert buckets[0][0][0] == "big"
    assert buckets[1][0][0] == "small"


def test_iter_named_tensor_buckets_rejects_invalid_size():
    import pytest

    with pytest.raises(ValueError, match="bucket_bytes"):
        list(iter_named_tensor_buckets(iter([]), bucket_bytes=0))


def test_build_rollout_topology_from_replicas():
    replicas = [_FakeReplica(8), _FakeReplica(8)]
    topology = build_rollout_topology_from_replicas(replicas)
    assert topology.engine_count == 2
    assert topology.gpus_per_engine == 8
    assert topology.total_gpus == 16


def test_remote_transfer_plan_round_robin():
    topology = P2PRolloutTopology(engine_count=2, gpus_per_engine=2)
    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 0, 4),
    ):
        plan = RemoteTransferPlan(topology)
        tasks = plan.plan_p2p()
    assert tasks == [TransferTaskP2PMeta(engine_ind=0, engine_rank=0)]

    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 1, 4),
    ):
        plan = RemoteTransferPlan(topology)
        tasks = plan.plan_p2p()
    assert tasks == [TransferTaskP2PMeta(engine_ind=0, engine_rank=1)]


def test_serialize_p2p_rollout_metadata_roundtrip_keys():
    topology = P2PRolloutTopology(engine_count=1, gpus_per_engine=1)
    payload = serialize_p2p_rollout_metadata(
        model_path="/tmp/model",
        rollout_topology=topology,
        engine_kwargs={"p2p_transfer_num_workers": 2},
        remote_weight_infos_by_session_id={"sess": ({"w": (1, 2, 4)}, {"tp_size": 1})},
        targets_to_session_id={(0, 0): "sess"},
        session_id_to_server_args={"sess": {"model_path": "/tmp/model", "trust_remote_code": True}},
    )
    assert payload["targets_to_session_id"] == {(0, 0): "sess"}
    assert payload["session_id_to_server_args"]["sess"]["trust_remote_code"] is True


def test_filter_server_args_dict_sanitizes_non_serializable_fields():
    class _CustomConfig:
        __module__ = "transformers_modules.configuration_deepseek"

    filtered = filter_server_args_dict(
        {
            "model_path": "/tmp/model",
            "trust_remote_code": True,
            "internal_states": {"config": _CustomConfig()},
            "custom_sigquit_handler": lambda: None,
        }
    )
    assert filtered["model_path"] == "/tmp/model"
    assert filtered["trust_remote_code"] is True
    # Not a ServerArgs field: always dropped.
    assert "internal_states" not in filtered
    # A non-serializable value never survives as-is: depending on the sglang
    # version the key is either absent (no such ServerArgs field) or nullified.
    assert filtered.get("custom_sigquit_handler") is None
    # The key guarantee: the result crosses Ray/pickle boundaries safely.
    pickle.loads(pickle.dumps(filtered))


def test_serialize_p2p_rollout_metadata_is_pickle_safe():
    topology = P2PRolloutTopology(engine_count=1, gpus_per_engine=1)
    server_args = filter_server_args_dict(
        {
            "model_path": "/tmp/model",
            "trust_remote_code": True,
            "internal_states": object(),
        }
    )
    payload = serialize_p2p_rollout_metadata(
        model_path="/tmp/model",
        rollout_topology=topology,
        engine_kwargs={},
        remote_weight_infos_by_session_id={"sess": ({"w": (1, 2, 4)}, {"tp_size": 1})},
        targets_to_session_id={(0, 0): "sess"},
        session_id_to_server_args={"sess": server_args},
    )
    restored = pickle.loads(pickle.dumps(payload))
    assert restored["session_id_to_server_args"]["sess"] == {
        "model_path": "/tmp/model",
        "trust_remote_code": True,
    }


def test_resolve_mooncake_transfer_engine_settings_defaults():
    with patch.dict("os.environ", {}, clear=True):
        assert resolve_mooncake_transfer_engine_settings() == ("rdma", "")


def test_resolve_mooncake_transfer_engine_settings_reads_env():
    with patch.dict(
        "os.environ",
        {
            "MOONCAKE_PROTOCOL": "rdma",
            "MOONCAKE_DEVICE": "^=mlx5_0,mlx5_1",
        },
        clear=True,
    ):
        assert resolve_mooncake_transfer_engine_settings() == ("rdma", "^=mlx5_0,mlx5_1")


def test_resolve_mooncake_transfer_engine_settings_mc_force_tcp_clears_device():
    with patch.dict(
        "os.environ",
        {
            "MOONCAKE_PROTOCOL": "rdma",
            "MOONCAKE_DEVICE": "^=mlx5_0",
            "MC_FORCE_TCP": "1",
        },
        clear=True,
    ):
        assert resolve_mooncake_transfer_engine_settings() == ("rdma", "")


def test_create_transfer_engine_passes_mooncake_env_to_initialize():
    engine = MagicMock()
    engine.initialize.return_value = 0
    with (
        patch.dict(
            "os.environ",
            {
                "MOONCAKE_PROTOCOL": "rdma",
                "MOONCAKE_DEVICE": "^=mlx5_0",
            },
            clear=True,
        ),
        patch(
            "verl.checkpoint_engine.p2p.transfer_utils.ray._private.services.get_node_ip_address",
            return_value="10.0.0.1",
        ),
        patch("mooncake.engine.TransferEngine", return_value=engine),
    ):
        created = create_transfer_engine()
    assert created is engine
    engine.initialize.assert_called_once_with("10.0.0.1", "P2PHANDSHAKE", "rdma", "^=mlx5_0")


def test_unregister_cpu_memory_calls_unregister():
    engine = MagicMock()
    engine.unregister_memory.return_value = 0
    registry = {"weight": (123, 4, 2)}
    unregister_cpu_memory(registry, engine)
    engine.unregister_memory.assert_called_once_with(123)
    assert registry == {}


def test_p2p_updater_release_and_restore():
    from unittest.mock import PropertyMock

    topology = P2PRolloutTopology(engine_count=1, gpus_per_engine=1)
    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 0, 1),
    ):
        updater = P2PTrainerWeightUpdater(model_path="/tmp/model", rollout_topology=topology, is_master=True)
    metadata = {
        "remote_weight_infos_by_session_id": {"sid": (["ptr"], {"tp": 1})},
        "targets_to_session_id": {(0, 0): "sid"},
        "session_id_to_server_args": {"sid": {"model_path": "/tmp/model"}},
    }
    updater._rollout_metadata = metadata
    updater._connected = True
    updater._model_registered = True
    updater._transfer_engine = MagicMock()
    updater._weight_memory_registry = {"p": (1, 2, 4)}
    updater._shared_params_dict = {"p": torch.zeros(2)}
    updater._transfer_engine_meta_list = [(MagicMock(), [])]

    with patch.object(P2PTrainerWeightUpdater, "is_source", new_callable=PropertyMock, return_value=True):
        updater.release_for_checkpoint()

    assert not updater._connected
    assert updater._transfer_engine is None
    assert not updater._shared_params_dict
    assert updater._rollout_metadata["targets_to_session_id"] == {(0, 0): "sid"}

    with patch.object(updater, "connect_rollout_metadata") as connect_mock:
        updater.restore_after_checkpoint()
    connect_mock.assert_called_once_with(metadata)


def test_p2p_release_keeps_rollout_metadata_for_restore():
    from unittest.mock import PropertyMock

    topology = P2PRolloutTopology(engine_count=2, gpus_per_engine=1)
    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 0, 1),
    ):
        updater = P2PTrainerWeightUpdater(model_path="/tmp/model", rollout_topology=topology, is_master=True)

    remote_infos = {"sid0": (["ptr0"], {"tp": 1})}
    targets_to_session = {(0, 0): "sid0"}
    server_args = {"sid0": {"model_path": "/tmp/model"}}

    with (
        patch.object(P2PTrainerWeightUpdater, "is_source", new_callable=PropertyMock, return_value=True),
        patch.object(updater, "_create_cpu_replica", return_value=MagicMock(named_parameters=lambda: [])),
        patch(
            "verl.checkpoint_engine.p2p.trainer_updater.create_transfer_engine",
            return_value=MagicMock(),
        ),
        patch.object(updater.transfer_plan, "plan_p2p", return_value=[]),
    ):
        updater.connect_rollout_metadata(
            {
                "remote_weight_infos_by_session_id": remote_infos,
                "targets_to_session_id": targets_to_session,
                "session_id_to_server_args": server_args,
            }
        )

    with patch.object(P2PTrainerWeightUpdater, "is_source", new_callable=PropertyMock, return_value=True):
        updater.release_for_checkpoint()

    assert updater._rollout_metadata["targets_to_session_id"] == {(0, 0): "sid0"}


def test_p2p_updater_send_weights_uses_buckets():
    from unittest.mock import PropertyMock

    topology = P2PRolloutTopology(engine_count=1, gpus_per_engine=1)
    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 0, 1),
    ):
        updater = P2PTrainerWeightUpdater(
            model_path="/tmp/model",
            rollout_topology=topology,
            is_master=True,
            bucket_size_bytes=16,
        )
    updater._connected = True
    updater._model_registered = True
    updater._shared_params_dict = {"p": torch.zeros(2)}
    updater._transfer_engine = MagicMock()
    updater._weight_memory_registry = {"p": (1, 2, 4)}
    updater._transfer_engine_meta_list = [(MagicMock(), [MagicMock()])]

    tensors = [
        ("hf.p", torch.ones(2)),
        ("hf.q", torch.ones(2)),
        ("hf.r", torch.ones(2)),
    ]

    with (
        patch.object(P2PTrainerWeightUpdater, "is_source", new_callable=PropertyMock, return_value=True),
        patch.object(
            updater,
            "_get_transfer_ready_params",
            side_effect=[
                (["p"], [("hf.p", tensors[0][1])]),
                (["q", "r"], [("hf.q", tensors[1][1]), ("hf.r", tensors[2][1])]),
            ],
        ) as get_ready_mock,
        patch.object(updater, "_transfer_ready_bucket", return_value=1) as transfer_mock,
        patch.object(updater.transfer_manager, "wait_transfers"),
    ):
        updater.send_weights(name_tensor for name_tensor in tensors)

    assert get_ready_mock.call_count == 2
    assert transfer_mock.call_count == 2


def test_p2p_transfer_manager_wait_futures():
    from concurrent.futures import Future

    manager = P2PTransferManager()
    future = Future()
    future.set_result(None)
    manager.transfer_futures.append(future)

    manager.wait_futures([future])
    manager.wait_transfers()
    assert manager.transfer_futures == []


def test_p2p_send_weights_waits_only_at_end_like_miles():
    from unittest.mock import PropertyMock

    topology = P2PRolloutTopology(engine_count=1, gpus_per_engine=1)
    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 0, 1),
    ):
        updater = P2PTrainerWeightUpdater(
            model_path="/tmp/model",
            rollout_topology=topology,
            bucket_size_bytes=9,
        )
    updater._connected = True
    updater._model_registered = True
    updater._shared_params_dict = {"p": torch.zeros(2)}
    updater._transfer_engine = MagicMock()
    updater._weight_memory_registry = {"p": (1, 2, 4)}
    updater._transfer_engine_meta_list = [(MagicMock(), [MagicMock()])]

    tensors = [("hf.p", torch.ones(2)), ("hf.q", torch.ones(2))]

    with (
        patch.object(P2PTrainerWeightUpdater, "is_source", new_callable=PropertyMock, return_value=True),
        patch.object(
            updater,
            "_get_transfer_ready_params",
            side_effect=[
                (["p"], [tensors[0]]),
                (["q"], [tensors[1]]),
            ],
        ),
        patch.object(updater, "_transfer_ready_bucket", return_value=1) as transfer_mock,
        patch.object(updater.transfer_manager, "wait_futures") as wait_futures_mock,
        patch.object(updater.transfer_manager, "wait_transfers") as wait_transfers_mock,
    ):
        updater.send_weights(name_tensor for name_tensor in tensors)

    assert transfer_mock.call_count == 2
    wait_futures_mock.assert_not_called()
    wait_transfers_mock.assert_called_once()


def test_p2p_transfer_ready_bucket_last_engine_is_async():
    from concurrent.futures import Future
    from unittest.mock import PropertyMock

    topology = P2PRolloutTopology(engine_count=1, gpus_per_engine=2)
    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 0, 1),
    ):
        updater = P2PTrainerWeightUpdater(model_path="/tmp/model", rollout_topology=topology)

    replica0 = MagicMock()
    replica1 = MagicMock()
    session0 = MagicMock()
    session1 = MagicMock()
    updater._transfer_engine_meta_list = [
        (replica0, [session0]),
        (replica1, [session1]),
    ]
    updater._weight_memory_registry = {"w": (1, 2, 4)}
    updater._transfer_engine = MagicMock()
    updater._transfer_engine.batch_transfer_sync_write.return_value = 0

    pending = Future()
    pending.set_result(None)

    with (
        patch.object(P2PTrainerWeightUpdater, "is_source", new_callable=PropertyMock, return_value=True),
        patch.object(updater.transfer_manager, "submit_returning_future", return_value=pending) as submit_mock,
        patch.object(updater.transfer_manager, "submit") as submit_async_mock,
    ):
        writes = updater._transfer_ready_bucket(["w"], [("hf.w", torch.ones(2))])

    assert writes == 2
    submit_mock.assert_called_once()
    submit_async_mock.assert_called_once()
    replica0.load_weights.assert_called_once()
    replica1.load_weights.assert_called_once()


def test_get_transfer_ready_params_raises_on_unmapped_replica_param():
    import pytest

    topology = P2PRolloutTopology(engine_count=1, gpus_per_engine=1)
    with patch(
        "verl.checkpoint_engine.p2p.transfer_utils.resolve_trainer_parallelism",
        return_value=(0, 1, 0, 1),
    ):
        updater = P2PTrainerWeightUpdater(model_path="/tmp/model", rollout_topology=topology)

    updater._shared_params_dict = {"known.weight": torch.zeros(2)}
    mapped_result = MagicMock()
    mapped_result.sglang_name = "missing.weight"
    mapped_result.num_shards = 1
    mapped_result.num_local_experts = None
    updater._shared_param_mapper = MagicMock(map=MagicMock(return_value=mapped_result))

    with pytest.raises(RuntimeError, match="missing.weight"):
        updater._get_transfer_ready_params([("hf.missing.weight", torch.ones(2))])
