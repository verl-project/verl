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
"""Weight-check digests, vLLM compatibility and lifecycle tests; no GPU required."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from verl.utils.weight_checksum import (
    WeightCheckEngineAdapter,
    compare_checksums,
    configure_weight_reference,
    merge_worker_checksums,
)


class TestChecksums:
    def test_merge_preserves_ranks_and_rejects_collisions(self):
        assert merge_worker_checksums([{"tp0:w": "a"}, {"tp1:w": "b"}]) == {"tp0:w": "a", "tp1:w": "b"}
        with pytest.raises(RuntimeError, match="Duplicate weight checksum keys"):
            merge_worker_checksums([{"tp0:w": "a"}, {"tp0:w": "a"}])

    @pytest.mark.parametrize(
        "baseline,current,mismatches",
        [
            ({}, {}, []),
            ({"w": "a"}, {"w": "a"}, []),
            ({"w": "a"}, {"w": "b"}, ["w"]),
            ({"w": "a", "v": "b"}, {"w": "a"}, ["v"]),
            ({"w": "a"}, {"w": "a", "v": "b"}, ["v"]),
            ({"tp0:w": "a", "tp1:w": "b"}, {"tp0:w": "a"}, ["tp1:w"]),
        ],
    )
    def test_compare(self, baseline, current, mismatches):
        assert compare_checksums(baseline, current) == (not mismatches, mismatches)


class TestEngineCompatibility:
    @pytest.mark.parametrize("enabled", [False, True])
    def test_reference_loader(self, enabled):
        args = {"load_format": "dummy", "dtype": "bfloat16"}
        assert configure_weight_reference(
            args, enabled=enabled, backend="naive", has_lora=False, has_quantization=False
        ) is enabled
        assert args["load_format"] == ("safetensors" if enabled else "dummy")

    @pytest.mark.parametrize(
        "overrides,args_override",
        [
            ({"backend": "delta_sharded"}, {}),
            ({"has_lora": True}, {}),
            ({"has_quantization": True}, {}),
            ({}, {"dtype": "float16"}),
            ({}, {"load_format": "pt"}),
            ({}, {"speculative_config": {"method": "mtp"}}),
        ],
    )
    def test_reference_rejects_unsupported_config(self, overrides, args_override):
        args = {"load_format": "dummy", "dtype": "bfloat16", **args_override}
        options = dict(enabled=True, backend="naive", has_lora=False, has_quantization=False)
        options.update(overrides)
        with pytest.raises(ValueError, match="independent weight check"):
            configure_weight_reference(args, **options)
        assert args["load_format"] != "safetensors"

    @pytest.mark.parametrize("api", ["all_dp", "legacy", "rpc"])
    def test_checksum_and_reset_dispatch(self, api):
        digests = {"tp0:w": "a", "tp1:w": "b"}
        engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=[digests]))
        if api == "all_dp":
            engine.compute_weight_checksums_all = AsyncMock(return_value=[digests])
            engine.reset_weights = AsyncMock()
        elif api == "legacy":
            engine.compute_weight_checksums = AsyncMock(return_value=digests)
        checker = WeightCheckEngineAdapter(engine, 2 if api == "all_dp" else 1)

        assert asyncio.run(checker.checksums()) == digests
        if api != "rpc":
            engine.collective_rpc.assert_not_called()
        else:
            engine.collective_rpc.assert_awaited_once_with("compute_weight_checksums")
        asyncio.run(checker.reset())
        if api == "all_dp":
            engine.reset_weights.assert_awaited_once()
            engine.collective_rpc.assert_not_called()
        else:
            engine.collective_rpc.assert_awaited_with("reset_weights")

    def test_validation_keeps_memory_resident(self):
        calls = Mock()
        engine = SimpleNamespace(collective_rpc=AsyncMock())
        for name in ("sleep", "wake_up", "release_kv_cache_memory"):
            method = AsyncMock()
            setattr(engine, name, method)
            calls.attach_mock(method, name)
        WeightCheckEngineAdapter(engine, 1).validate()
        assert calls.mock_calls == []

    def test_legacy_dp_rejected_before_reset(self):
        engine = SimpleNamespace(collective_rpc=AsyncMock())
        with pytest.raises(RuntimeError, match="DP>1"):
            asyncio.run(WeightCheckEngineAdapter(engine, 2).reset())
        engine.collective_rpc.assert_not_called()


@pytest.fixture
def manager_factory():
    from verl.checkpoint_engine.base import CheckpointEngineManager

    def make(*, backend="naive", failure=None):
        events = []
        manager = object.__new__(CheckpointEngineManager)
        manager.backend = backend

        async def check(action):
            events.append(action)
            if failure == "baseline" and action == "snapshot":
                return []
            return [{"success": not (failure == "compare" and action == "compare"), "num_keys": 2}]

        async def sync(step, **kwargs):
            events.append(("sync", kwargs.get("checking_weights", False)))

        async def abort():
            events.append("abort")

        async def resume():
            events.append("resume")

        async def sleep():
            events.append("sleep")

        manager.replicas = [SimpleNamespace(check_weights=check)]
        manager.update_weights = AsyncMock(side_effect=sync)
        manager.abort_replicas = AsyncMock(side_effect=abort)
        manager.resume_generation_replicas = AsyncMock(side_effect=resume)
        manager.sleep_replicas = AsyncMock(side_effect=sleep)
        return manager, events

    return make


class TestUpdateLifecycle:
    def test_repeatable_wrong_transfer_fails_independent_reference(self, manager_factory):
        manager, events = manager_factory()
        reference = {"tp0:w": "hf-reference"}
        state = dict(reference)
        baseline = {}

        async def check(action):
            if action == "snapshot":
                baseline.update(state)
            elif action == "reset":
                state["tp0:w"] = "random"
            elif action == "compare":
                matched, mismatches = compare_checksums(baseline, state)
                return [{"success": matched, "num_keys": 1, "mismatches": mismatches}]
            return [{"success": True, "num_keys": 1}]

        async def wrong_sync(*args, **kwargs):
            state["tp0:w"] = "deterministically-wrong"

        manager.replicas = [SimpleNamespace(check_weights=check)]
        manager.update_weights = AsyncMock(side_effect=wrong_sync)
        with pytest.raises(RuntimeError, match="weight update check failed"):
            manager.verify_update_path(0)
        assert baseline == reference
        manager.update_weights.assert_awaited_once()
        assert "resume" not in events

    def test_success_order(self, manager_factory):
        manager, events = manager_factory()
        assert manager.verify_update_path(0) == {"replicas": 1, "num_keys": 2, "success": True}
        assert events == [
            "abort",
            "prepare",
            "snapshot",
            "reset",
            "sleep",
            ("sync", True),
            "compare",
            "finish",
            "resume",
        ]
        manager.update_weights.assert_awaited_once_with(0, checking_weights=True)

    @pytest.mark.parametrize("failure,message", [("compare", "check failed"), ("baseline", "no valid baseline")])
    def test_failure_keeps_generation_paused(self, manager_factory, failure, message):
        manager, events = manager_factory(failure=failure)
        with pytest.raises(RuntimeError, match=message):
            manager.verify_update_path(0)
        assert "finish" not in events
        assert "resume" not in events
        if failure == "baseline":
            assert "reset" not in events

    def test_delta_rejected_before_mutation(self, manager_factory):
        manager, events = manager_factory(backend="delta_sharded")
        with pytest.raises(ValueError, match="requires full weight transfers"):
            manager.verify_update_path(0)
        assert events == []

    @pytest.mark.parametrize("checking", [False, True])
    def test_naive_check_uses_normal_worker_lifecycle(self, monkeypatch, checking):
        from verl.checkpoint_engine import base

        manager = object.__new__(base.CheckpointEngineManager)
        manager.backend = "naive"
        manager.actor_wg = SimpleNamespace(update_weights=Mock(return_value=[]))
        monkeypatch.setattr(base.ray, "get", lambda refs: refs)

        manager.update_weights(0, checking_weights=checking)
        manager.actor_wg.update_weights.assert_called_once_with(global_steps=0, mode="naive")

    def test_distributed_check_restores_kv_but_defers_generation(self, monkeypatch):
        from verl.checkpoint_engine import base

        manager = object.__new__(base.CheckpointEngineManager)
        manager.backend = "nccl"
        manager.replicas = [SimpleNamespace(workers=[object()])]
        manager.actor_wg = SimpleNamespace(
            world_size=1, update_weights=Mock(return_value=[{}]), execute_checkpoint_engine=Mock(return_value=[])
        )
        rollout = SimpleNamespace(
            world_size=1, update_weights=Mock(return_value=[{}]), execute_checkpoint_engine=Mock(return_value=[])
        )
        monkeypatch.setattr(base, "RayWorkerGroup", Mock(return_value=rollout))
        monkeypatch.setattr(base, "RayClassWithInitArgs", Mock())
        monkeypatch.setattr(base.ray, "get", lambda refs: refs)
        manager.build_process_group = Mock()
        for name in (
            "abort_replicas", "release_kv_cache_replicas", "resume_kv_cache_replicas", "resume_generation_replicas"
        ):
            setattr(manager, name, AsyncMock())

        manager.update_weights(0, checking_weights=True)

        manager.actor_wg.update_weights.assert_called_once_with(global_steps=0, mode="nccl")
        rollout.update_weights.assert_called_once_with(global_steps=0)
        manager.abort_replicas.assert_awaited_once()
        manager.release_kv_cache_replicas.assert_awaited_once()
        manager.resume_kv_cache_replicas.assert_awaited_once()
        manager.resume_generation_replicas.assert_not_called()

    @pytest.mark.parametrize("headless", [False, True])
    def test_replica_ignores_headless_servers(self, headless):
        from verl.workers.rollout.replica import RolloutReplica

        result = {"success": True, "num_keys": 2}
        remote = AsyncMock(return_value=None if headless else result)
        replica = SimpleNamespace(servers=[SimpleNamespace(check_weights=SimpleNamespace(remote=remote))])
        results = asyncio.run(RolloutReplica.check_weights(replica, "snapshot"))
        assert results == ([] if headless else [result])
        remote.assert_awaited_once_with("snapshot")
