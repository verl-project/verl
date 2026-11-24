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
"""CPU tests for the rollout-restart orchestration pieces.

Long-term-care minimum: the supervisor's cold-start recovery chain (COLD_START
fallback + sync-succeeded promotion), the terminal DEAD/fatal path with
anti-flap budget semantics, the update_weights entry probe, membership growth
re-fencing, the applicability gates, cold-start checkpoint locating, the
fatal fail-closed sync, and the bounded zero-survivor wait-and-retry loop."""

import asyncio
from types import SimpleNamespace

import pytest

import verl.checkpoint_engine.base as base_mod
from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.workers.rollout.fault_tolerance import (
    CheckpointUnavailable,
    FaultEvidence,
    MembershipSnapshot,
    ReplicaFaultSupervisor,
    ReplicaRuntimeState,
    RestartBudget,
    WeightSourceOverride,
    _locate_cold_start_checkpoint,
)


def _supervisor_with_fake_manager(config=None, rollout_config=None):
    rollout_config = rollout_config or SimpleNamespace(
        name="vllm",
        nnodes=1,
        disaggregation=SimpleNamespace(enabled=False),
        checkpoint_engine=SimpleNamespace(backend="nccl"),
        quantization=SimpleNamespace(enabled=False),
    )
    manager = SimpleNamespace(
        rollout_config=rollout_config,
        model_config=SimpleNamespace(lora_rank=0),
        rollout_replicas=[],
        server_addresses=[],
        config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=None)),
        global_load_balancer=None,
        _group_generation=0,
        _fatal_error=None,
    )
    manager.invalidate_group = lambda: manager.__setattr__("_group_generation", manager._group_generation + 1)
    manager.mark_fatal = lambda message: manager.__setattr__("_fatal_error", message)
    supervisor = ReplicaFaultSupervisor(manager=manager, config=config or SimpleNamespace())
    return supervisor, manager


class _FakeRolloutReplica:
    def __init__(self, rank, suffix=""):
        self.replica_rank = rank
        self.name_suffix = suffix
        self.workers = [SimpleNamespace()]


async def _async_value(value):
    return value


def _stub_recovery_steps(supervisor):
    """Fence the LB/register/verify edges; keep the real recovery sequencing."""
    supervisor._isolate = lambda replica, key: _async_value(None)
    supervisor._register = lambda replica, key: _async_value(None)


class TestSupervisor:
    def test_cold_start_fallback_stays_syncable_and_promotes_on_sync(self):
        """The checkpoint cold-start walk: restart(weight_source) -> stamp ->
        verify. The rebuild is up but VERIFYING fails — the replica falls back
        to the deferred-weight-sync semantics (COLD_START), never stays
        ISOLATED (a healthy probe on an ISOLATED replica is never re-convicted,
        so it would be fenced forever); the next sync's notification finishes
        recovery."""
        supervisor, manager = _supervisor_with_fake_manager()
        override = WeightSourceOverride(model_path="/ckpt/global_step_8/actor/huggingface", global_steps=8)
        supervisor._plan_weight_source = lambda: _async_value(override)
        _stub_recovery_steps(supervisor)
        replica = _FakeRolloutReplica(0)
        sources = []

        async def _restart(*, weight_source=None):
            sources.append(weight_source)
            return None

        replica.restart = _restart
        replica.servers = [SimpleNamespace(set_global_steps=SimpleNamespace(remote=lambda v: _async_value(None)))]
        manager.rollout_replicas = [replica]

        async def _failing_verify(replica, expected_version):
            raise RuntimeError("not healthy yet")

        supervisor._verify = _failing_verify
        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        snapshot = supervisor.get_membership()
        assert sources == [override]
        assert manager._group_generation == 1  # fencing generation bumped at conviction
        assert supervisor.get_state("0") == ReplicaRuntimeState.COLD_START
        # Back on the syncable set; nothing serves, so the trainer's poll hook
        # must fire the out-of-band sync.
        assert snapshot.replicas == [replica]
        assert snapshot.awaiting_weight_sync is True

        # The next sync installs weights; its notification finishes recovery.
        supervisor._verify = lambda replica, expected_version: _async_value(None)
        registered = []

        def _fake_register(replica, key):
            registered.append(key)
            supervisor._states[key] = ReplicaRuntimeState.SERVING  # the real _register owns this flip
            return _async_value(None)

        supervisor._register = _fake_register
        asyncio.run(supervisor.weight_sync_succeeded())
        assert registered == ["0"]
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING

    def test_budget_exhaustion_goes_dead_and_marks_fatal(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._budgets["0"] = RestartBudget(max_restarts=1, window_s=3600.0, cooldown_s=0.0)
        supervisor._budgets["0"].try_acquire()
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.DEAD
        # Fatal lives on the manager (the membership authority) — the
        # pulled snapshot propagates it to the checkpoint manager — and DEAD
        # replicas are fenced out of the sync group entirely.
        assert manager._fatal_error is not None
        snapshot = supervisor.get_membership()
        assert snapshot.fatal_error is not None
        assert snapshot.replicas == []

        # Anti-flap: a cooldown rejection is pacing, not a terminal verdict.
        pacing = RestartBudget(max_restarts=2, window_s=3600.0, cooldown_s=60.0)
        pacing.try_acquire()
        assert pacing.try_acquire_state() == "cooldown"

    def test_probe_membership_convicts_a_dead_replica_before_the_snapshot(self):
        """The update_weights-entry probe: one fresh detection round runs
        before the snapshot is built, so a replica that died inside the
        detection-window staleness of the periodic poll is fenced out of the
        sync group before the group is built (no commit-watchdog burn)."""
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _Replica:
            def __init__(self, rank):
                self.replica_rank = rank
                self.name_suffix = ""
                self.servers = []
                self.workers = []

        dead, healthy = _Replica(0), _Replica(1)
        manager.rollout_replicas = [dead, healthy]

        async def _probe(replica):
            if replica is dead:
                return FaultEvidence(source="detect_faults", error_class="EngineDeadError")
            return None

        supervisor._probe_replica = _probe

        snapshot = asyncio.run(supervisor.probe_membership())

        assert scheduled == ["0"]  # convicted inside the entry round
        assert snapshot.replicas == [healthy]  # fenced out before the group build

    def test_membership_growth_re_fences_the_group(self):
        """A rebuilt replica re-entering the syncable set re-fences the group:
        the survivors' cached fenced group still carries the shrunken
        world_size/rank layout and would trip the engines' idempotent-reuse
        asserts."""
        supervisor, manager = _supervisor_with_fake_manager()
        replicas = [_FakeRolloutReplica(0), _FakeRolloutReplica(1)]
        manager.rollout_replicas = replicas
        supervisor._states = {"1": ReplicaRuntimeState.ISOLATED}
        assert supervisor.get_membership().group_generation == 0  # first issue never fences
        manager._group_generation = 1  # the conviction fenced the shrink
        supervisor._states["1"] = ReplicaRuntimeState.COLD_START  # rebuilt replica re-enters
        assert supervisor.get_membership().group_generation == 2  # growth re-fence
        assert supervisor.get_membership().group_generation == 2  # unchanged set: no re-fence

    def test_applicability_gates(self):
        for bad in [
            dict(name="trtllm"),
            dict(name="sglang"),  # the server-side restart contract is not implemented this phase
            dict(nnodes=2),
            dict(disaggregation=SimpleNamespace(enabled=True)),
            dict(checkpoint_engine=SimpleNamespace(backend="delta_sharded")),
            dict(quantization=SimpleNamespace(enabled=True)),
        ]:
            base = SimpleNamespace(
                name="vllm",
                nnodes=1,
                disaggregation=SimpleNamespace(enabled=False),
                checkpoint_engine=SimpleNamespace(backend="nccl"),
                quantization=SimpleNamespace(enabled=False),
            )
            for key, value in bad.items():
                setattr(base, key, value)
            supervisor, _ = _supervisor_with_fake_manager(rollout_config=base)
            with pytest.raises(NotImplementedError):
                supervisor._validate_applicability()

        supervisor, _ = _supervisor_with_fake_manager()
        supervisor._validate_applicability()  # vLLM standalone stays in scope


def _make_megatron_step(root, step):
    model_dir = root / f"global_step_{step}" / "actor" / "model"
    hf_dir = model_dir / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / "tokenizer_config.json").write_text("{}")
    (hf_dir / "model-00001-of-00002.safetensors").write_text("x")


class TestLocateLatestValidCheckpoint:
    def test_tracker_resolution_and_defer_without_a_tracker(self, tmp_path):
        _make_megatron_step(tmp_path, 10)
        _make_megatron_step(tmp_path, 20)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("10")
        path, step = _locate_cold_start_checkpoint(str(tmp_path))
        assert step == 10
        assert path.endswith("global_step_10/actor/model/huggingface")

        # Tracker-only by design: without the tracker (first save in flight, or
        # torn) the locator defers — it never scans step directories.
        (tmp_path / "latest_checkpointed_iteration.txt").unlink()
        with pytest.raises(CheckpointUnavailable):
            _locate_cold_start_checkpoint(str(tmp_path))


def _manager_half_object():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager._fault_tolerant = False
    manager._pre_sync_hook = None
    manager._sync_succeeded_callback = None
    manager._group_generation = 0
    manager._retry_epoch = 0
    manager._fatal_error = None
    manager.replicas = [_FakeReplicaForCM(0)]
    return manager


class _FakeReplicaForCM:
    def __init__(self, rank, suffix=""):
        self.replica_rank = rank
        self.name_suffix = suffix
        self.workers = [SimpleNamespace()]


def _naive_pipeline_manager(manager, actor_update_weights):
    """Half-object plumbing for the naive pipeline (a colocated stub): the
    pipeline's gather is the only surface the wait-retry loop exercises."""
    manager.backend = "naive"
    manager.actor_wg = SimpleNamespace(update_weights=actor_update_weights)
    return manager


class TestManagerFencing:
    def test_update_weights_raises_on_fatal(self):
        """Fatal arrives via the pulled snapshot — sync_membership is the
        single membership/fatal boundary."""
        manager = _manager_half_object()
        manager.enable_fault_tolerance()
        manager.sync_membership(
            MembershipSnapshot(
                replicas=[],
                group_generation=0,
                fatal_error="all rollout replicas are DEAD",
            )
        )

        with pytest.raises(RuntimeError, match="fatal"):
            manager.update_weights(1)


class TestManagerWaitAndRetry:
    """The driver-side wait-and-retry loop: bounded, fail-closed on fatal and
    on exhausted attempts, holds only for the fault-detection grace window
    (never for a rebuild to settle — the survivors sync first); with zero
    syncable members it waits for recovery instead of burning attempts."""

    @pytest.fixture(autouse=True)
    def _fast_waits(self, monkeypatch):
        monkeypatch.setattr(base_mod, "_PROBE_GRACE_S", 0.0)

    def test_update_weights_fails_closed_after_max_attempts(self):
        calls = []

        def _always_fail(**kwargs):
            calls.append(1)
            raise RuntimeError("member still dead")

        manager = _manager_half_object()
        manager.enable_fault_tolerance()
        _naive_pipeline_manager(manager, _always_fail)

        with pytest.raises(RuntimeError, match="weight sync failed"):
            manager.update_weights(1)
        assert len(calls) == base_mod._MAX_SYNC_ATTEMPTS

    def test_zero_member_wait_recovers_then_fails_closed(self, monkeypatch):
        """Zero syncable members at entry: hold for recovery instead of burning
        the attempt budget on doomed empty-group builds (the single-replica
        regression) — and the wait is bounded, so a never-landing recovery
        fails closed."""
        monkeypatch.setattr(base_mod, "_EMPTY_MEMBERSHIP_POLL_S", 0.0)
        monkeypatch.setattr(base_mod, "_RECOVERY_WAIT_TIMEOUT_S", 0.05)
        pulls = []
        syncs = []

        async def _pull_recover():
            pulls.append(1)
            manager.sync_membership(
                MembershipSnapshot(
                    replicas=[] if len(pulls) == 1 else [_FakeReplicaForCM(0)],
                    group_generation=1,
                    fatal_error=None,
                )
            )

        def _actor_sync(**kwargs):
            syncs.append(1)
            return [_async_value({"x": 1})]

        manager = _manager_half_object()
        manager.enable_fault_tolerance(pre_sync_hook=_pull_recover)
        _naive_pipeline_manager(manager, _actor_sync)

        manager.update_weights(1)
        assert len(pulls) == 2  # the empty pull, then the wait's re-pull
        assert syncs == [1]  # the sync ran once, after a replica re-entered

        manager2 = _manager_half_object()

        async def _pull_never():
            manager2.sync_membership(MembershipSnapshot(replicas=[], group_generation=1, fatal_error=None))

        manager2.enable_fault_tolerance(pre_sync_hook=_pull_never)
        _naive_pipeline_manager(manager2, lambda **kwargs: [_async_value({})])

        with pytest.raises(RuntimeError, match="no syncable rollout replicas recovered"):
            manager2.update_weights(1)

    def test_update_weights_without_fault_tolerance_fails_fast(self):
        calls = []

        def _always_fail(**kwargs):
            calls.append(1)
            raise RuntimeError("member died")

        manager = _manager_half_object()  # fault tolerance disabled
        _naive_pipeline_manager(manager, _always_fail)

        with pytest.raises(RuntimeError, match="member died"):
            manager.update_weights(1)
        assert calls == [1]  # no retries without fault tolerance
