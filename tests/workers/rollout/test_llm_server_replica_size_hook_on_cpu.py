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
"""Replica-size overrides keep the manager's server lifecycle intact."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

from verl.workers.rollout import llm_server


def _config(*, name="vllm", disaggregation=None):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "name": name,
                    "tensor_model_parallel_size": 2,
                    "data_parallel_size": 2,
                    "pipeline_model_parallel_size": 2,
                    "nnodes": 4,
                    "n_gpus_per_node": 8,
                    "disaggregation": disaggregation or {"enabled": False},
                    "disable_log_stats": False,
                    "prometheus": {"enable": True},
                },
                "model": {"path": "unused"},
            }
        }
    )


def _manager(config, **kwargs):
    class Manager(llm_server.LLMServerManager):
        rollout_replica_class = Mock()

    return Manager(config, **kwargs)


@pytest.mark.parametrize(
    "disaggregation,expected",
    [
        (None, 8),
        ({"enabled": False, "prefill_replicas": 3, "decode_replicas": 2}, 8),
        ({"enabled": True, "prefill_replicas": 3, "decode_replicas": 2, "decode_tensor_model_parallel_size": None}, 40),
        ({"enabled": True, "prefill_replicas": 3, "decode_replicas": 2, "decode_tensor_model_parallel_size": 1}, 32),
    ],
)
def test_default_footprint_preserves_standard_and_pd_sizes(disaggregation, expected):
    manager = _manager(_config(disaggregation=disaggregation))
    assert manager._get_rollout_replica_world_size() == expected


def test_default_footprint_accepts_config_without_disaggregation():
    config = _config()
    del config.actor_rollout_ref.rollout.disaggregation
    assert _manager(config)._get_rollout_replica_world_size() == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["hybrid", "standalone", "trtllm"])
@pytest.mark.parametrize("start_rank", [None, 5])
@pytest.mark.parametrize("fail_init", [False, True])
async def test_size_override_preserves_initialization_and_metrics(monkeypatch, mode, start_rank, fail_init):
    factory = Mock()

    class Manager(llm_server.LLMServerManager):
        rollout_replica_class = factory

        def _get_rollout_replica_world_size(self):
            return super()._get_rollout_replica_world_size() * 2

    config = _config(name="trtllm" if mode == "trtllm" else "vllm")
    group = None if mode == "standalone" else SimpleNamespace(world_size=32)
    pool = object()
    manager = Manager(config, worker_group=group, rollout_resource_pool=pool, start_rank=3)
    error = RuntimeError("launch failed") if fail_init else None

    replicas = []

    def construct(**kwargs):
        rank = kwargs["replica_rank"]
        replica = SimpleNamespace(
            replica_rank=rank,
            init_hybrid=AsyncMock(side_effect=error),
            init_hybrid_colocated=AsyncMock(side_effect=error),
            init_standalone=AsyncMock(side_effect=error),
            _server_handle=f"handle-{rank}",
            _server_address=f"address-{rank}",
        )
        replicas.append(replica)
        return replica

    factory.side_effect = construct
    prometheus = Mock()
    insight = Mock()
    monkeypatch.setattr(llm_server, "update_prometheus_config", prometheus)
    monkeypatch.setattr(llm_server.RLInsightLogger, "enabled", lambda: True)
    monkeypatch.setattr(llm_server.RLInsightLogger, "register_rollout_metrics", insight)

    if fail_init:
        with pytest.raises(RuntimeError, match="launch failed"):
            await manager._initialize_llm_servers(start_rank=start_rank)
        prometheus.assert_not_called()
        insight.assert_not_called()
    else:
        await manager._initialize_llm_servers(start_rank=start_rank)
        assert manager.server_handles == [r._server_handle for r in replicas]
        assert manager.server_addresses == [r._server_address for r in replicas]
        prometheus.assert_called_once_with(
            config.actor_rollout_ref.rollout.prometheus, manager.server_addresses, config.actor_rollout_ref.rollout.name
        )
        insight.assert_called_once_with(
            manager.server_addresses,
            config.actor_rollout_ref.rollout.name,
            labels=[{"replica": r.replica_rank} for r in replicas],
        )

    first = 3 if start_rank is None else start_rank
    assert [r.replica_rank for r in replicas] == [first, first + 1]
    assert len(manager.rollout_replicas) == 2
    for call, replica in zip(factory.call_args_list, replicas, strict=True):
        assert call.kwargs["config"] is manager.rollout_config
        assert call.kwargs["model_config"] is manager.model_config
        assert call.kwargs["gpus_per_node"] == 8
        if mode == "hybrid":
            replica.init_hybrid.assert_awaited_once_with(group)
            replica.init_hybrid_colocated.assert_not_called()
            replica.init_standalone.assert_not_called()
        elif mode == "trtllm":
            replica.init_hybrid_colocated.assert_awaited_once_with(group, pool)
            replica.init_hybrid.assert_not_called()
            replica.init_standalone.assert_not_called()
        else:
            replica.init_standalone.assert_awaited_once_with()
            replica.init_hybrid.assert_not_called()
            replica.init_hybrid_colocated.assert_not_called()
