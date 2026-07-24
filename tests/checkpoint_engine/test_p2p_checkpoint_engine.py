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

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf

from verl.checkpoint_engine import CheckpointEngineRegistry, P2PCheckpointEngine
from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.checkpoint_engine.p2p.transfer_utils import resolve_rollout_model_path
from verl.workers.config.rollout import CheckpointEngineConfig


def test_p2p_backend_is_registered():
    assert CheckpointEngineRegistry.get("p2p") is P2PCheckpointEngine


def test_p2p_send_weights_requires_connect():
    import asyncio

    async def _run() -> None:
        engine = P2PCheckpointEngine(bucket_size=1024, is_master=True)
        with pytest.raises(RuntimeError, match="not connected"):
            await engine.send_weights(iter([]))

    asyncio.run(_run())


def test_p2p_build_topology_is_noop():
    trainer_kwargs, rollout_kwargs = P2PCheckpointEngine.build_topology(2, 4, [{}] * 6)
    assert trainer_kwargs["rank"] == [-1, -1]
    assert rollout_kwargs["rank"] == [-1, -1, -1, -1]


def test_p2p_prepare_finalize_noop():
    engine = P2PCheckpointEngine(bucket_size=1024, is_master=False)
    assert engine.prepare()["backend"] == "p2p"
    engine.init_process_group(rank=-1, world_size=0, metadata={})
    engine.finalize()


def test_resolve_rollout_model_path_from_omegaconf():
    model_config = OmegaConf.create({"path": "/tmp/hf-model", "use_shm": False})
    with patch("verl.utils.fs.copy_to_local", return_value="/tmp/hf-model-local") as copy_mock:
        assert resolve_rollout_model_path(model_config) == "/tmp/hf-model-local"
    copy_mock.assert_called_once_with("/tmp/hf-model", use_shm=False)


def test_update_weights_p2p_connects_rollout_metadata_once():
    replicas = [MagicMock(model_config=OmegaConf.create({"path": "/tmp/model", "use_shm": False}))]
    actor_wg = MagicMock()
    manager = CheckpointEngineManager(
        config=CheckpointEngineConfig(backend="p2p"),
        actor_wg=actor_wg,
        replicas=replicas,
    )

    async def _run() -> None:
        with (
            patch.object(manager, "abort_replicas", new_callable=AsyncMock),
            patch.object(manager, "release_kv_cache_replicas", new_callable=AsyncMock),
            patch.object(manager, "begin_weight_update_replicas", new_callable=AsyncMock, return_value=[]),
            patch.object(manager, "end_weight_update_replicas", new_callable=AsyncMock),
            patch.object(manager, "resume_kv_cache_replicas", new_callable=AsyncMock),
            patch.object(manager, "resume_generation_replicas", new_callable=AsyncMock),
            patch.object(manager, "update_weight_version_replicas", new_callable=AsyncMock),
            patch(
                "verl.checkpoint_engine.p2p.transfer_utils.collect_p2p_rollout_metadata",
                new_callable=AsyncMock,
                return_value={"model_path": "/tmp/model"},
            ) as collect_mock,
            patch("verl.checkpoint_engine.base.ray.get"),
        ):
            await manager._update_weights_p2p(global_steps=1)
            await manager._update_weights_p2p(global_steps=2)

        assert collect_mock.await_count == 1
        actor_wg.init_p2p_rollout_metadata.assert_called_once()
        assert manager._p2p_rollout_metadata_initialized

    asyncio.run(_run())


def test_add_replicas_invalidates_p2p_rollout_metadata():
    manager = CheckpointEngineManager(
        config=CheckpointEngineConfig(backend="p2p"),
        actor_wg=MagicMock(),
        replicas=[],
    )
    manager._p2p_rollout_metadata_initialized = True
    manager.add_replicas([MagicMock()])
    assert not manager._p2p_rollout_metadata_initialized


def test_p2p_release_restore_delegates_to_updater():
    from unittest.mock import MagicMock

    engine = P2PCheckpointEngine(bucket_size=1024, is_master=True)
    updater = MagicMock()
    engine._updater = updater

    engine.release_for_checkpoint()
    updater.release_for_checkpoint.assert_called_once()

    engine.restore_after_checkpoint()
    updater.restore_after_checkpoint.assert_called_once()
