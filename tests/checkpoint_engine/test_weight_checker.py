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
from unittest.mock import AsyncMock, patch

import pytest

from verl.checkpoint_engine.p2p.weight_checker import (
    P2PWeightChecker,
    check_weights_on_replicas,
    compare_weight_updates,
    setup_weight_checker_snapshots,
)
from verl.workers.config.rollout import CheckpointEngineConfig


class _FakeReplica:
    def __init__(self, replica_rank: int, *, success: bool = True) -> None:
        self.replica_rank = replica_rank
        self.check_weights = AsyncMock(return_value={"success": success, "message": "ok", "replica_rank": replica_rank})


def test_setup_weight_checker_snapshots_order():
    replicas = [_FakeReplica(0), _FakeReplica(1)]

    async def _run() -> None:
        await setup_weight_checker_snapshots(replicas)

    asyncio.run(_run())

    assert replicas[0].check_weights.await_count == 2
    assert replicas[1].check_weights.await_count == 2
    assert replicas[0].check_weights.await_args_list[0].kwargs == {
        "action": "snapshot",
        "allow_quant_error": False,
    }
    assert replicas[0].check_weights.await_args_list[1].kwargs == {
        "action": "reset_tensors",
        "allow_quant_error": False,
    }


def test_compare_weight_updates_passes_allow_quant_error():
    replica = _FakeReplica(0)

    async def _run() -> None:
        with patch("builtins.print") as print_mock:
            await compare_weight_updates([replica], allow_quant_error=True)
        print_mock.assert_called_once()
        assert "compare PASSED" in print_mock.call_args[0][0]

    asyncio.run(_run())

    replica.check_weights.assert_awaited_once_with(action="compare", allow_quant_error=True)


def test_check_weights_on_replicas_raises_on_failure():
    replicas = [_FakeReplica(0, success=True), _FakeReplica(1, success=False)]

    async def _run() -> None:
        await check_weights_on_replicas(replicas, action="compare")

    with pytest.raises(RuntimeError, match="replica=1"):
        asyncio.run(_run())


def test_checkpoint_engine_config_has_weight_checker_flags():
    cfg = CheckpointEngineConfig(check_weight_update_equal=True)
    assert cfg.check_weight_update_equal is True
    assert cfg.check_weight_update_allow_quant_error is False


def test_p2p_weight_checker_runs_once():
    replicas = [_FakeReplica(0)]
    checker = P2PWeightChecker(enabled=True)

    async def _run() -> None:
        with (
            patch(
                "verl.checkpoint_engine.p2p.weight_checker.setup_weight_checker_snapshots",
                new_callable=AsyncMock,
            ) as setup_mock,
            patch(
                "verl.checkpoint_engine.p2p.weight_checker.compare_weight_updates",
                new_callable=AsyncMock,
            ) as compare_mock,
        ):
            await checker.setup(replicas)
            await checker.setup(replicas)
            await checker.compare(replicas)
            await checker.compare(replicas)

        setup_mock.assert_awaited_once_with(replicas)
        compare_mock.assert_awaited_once_with(replicas, allow_quant_error=False)

    asyncio.run(_run())
    assert checker._setup_done
    assert checker._compared


def test_p2p_weight_checker_disabled_is_noop():
    replicas = [_FakeReplica(0)]
    checker = P2PWeightChecker(enabled=False)

    async def _run() -> None:
        with (
            patch(
                "verl.checkpoint_engine.p2p.weight_checker.setup_weight_checker_snapshots",
                new_callable=AsyncMock,
            ) as setup_mock,
            patch(
                "verl.checkpoint_engine.p2p.weight_checker.compare_weight_updates",
                new_callable=AsyncMock,
            ) as compare_mock,
        ):
            await checker.setup(replicas)
            await checker.compare(replicas)

        setup_mock.assert_not_awaited()
        compare_mock.assert_not_awaited()

    asyncio.run(_run())
    assert not checker._setup_done
    assert not checker._compared


def test_checkpoint_engine_manager_weight_checker_hooks():
    from unittest.mock import MagicMock, patch

    from verl.checkpoint_engine.base import CheckpointEngineManager
    from verl.workers.config.rollout import CheckpointEngineConfig

    replicas = [_FakeReplica(0)]
    manager = CheckpointEngineManager(
        config=CheckpointEngineConfig(backend="p2p", check_weight_update_equal=True),
        actor_wg=MagicMock(),
        replicas=replicas,
    )

    async def _run() -> None:
        with (
            patch(
                "verl.checkpoint_engine.p2p.weight_checker.setup_weight_checker_snapshots",
                new_callable=AsyncMock,
            ) as setup_mock,
            patch(
                "verl.checkpoint_engine.p2p.weight_checker.compare_weight_updates",
                new_callable=AsyncMock,
            ) as compare_mock,
            patch.object(manager, "_update_weights_p2p", new_callable=AsyncMock),
        ):
            await manager.update_weights(global_steps=0)
        setup_mock.assert_awaited_once_with(replicas)
        compare_mock.assert_awaited_once_with(replicas, allow_quant_error=False)

    asyncio.run(_run())
    assert manager._p2p_weight_checker._setup_done
    assert manager._p2p_weight_checker._compared


def test_checkpoint_engine_manager_skips_weight_checker_for_naive():
    from unittest.mock import MagicMock, patch

    from verl.checkpoint_engine.base import CheckpointEngineManager
    from verl.workers.config.rollout import CheckpointEngineConfig

    replicas = [_FakeReplica(0)]
    manager = CheckpointEngineManager(
        config=CheckpointEngineConfig(backend="naive", check_weight_update_equal=True),
        actor_wg=MagicMock(),
        replicas=replicas,
    )

    async def _run() -> None:
        with patch("verl.checkpoint_engine.base.ray.get"):
            await manager.update_weights(global_steps=0)

    asyncio.run(_run())
    assert manager._p2p_weight_checker is None
