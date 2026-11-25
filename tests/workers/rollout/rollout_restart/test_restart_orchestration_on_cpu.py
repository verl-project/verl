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

RestartBudget semantics, the supervisor's single-channel polling detection /
conviction / recovery transitions (half-object pattern with a fake manager),
the checkpoint-cold-start vs deferred-weight-sync decision (weight source
rides the restart chain), the membership-authority snapshot (with the
membership-growth re-fence), the sync-succeeded promotion path, and the
driver-side wait-and-retry loop (zero-survivor recovery wait included).
"""

import asyncio
import time
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
    make_pre_sync_reconcile,
)


class TestRestartBudget:
    def test_allows_up_to_max_restarts(self):
        budget = RestartBudget(max_restarts=2, window_s=3600.0, cooldown_s=0.0)
        assert budget.try_acquire() is True
        assert budget.try_acquire() is True
        assert budget.try_acquire() is False

    def test_cooldown_blocks_bursts(self):
        budget = RestartBudget(max_restarts=3, window_s=3600.0, cooldown_s=60.0)
        assert budget.try_acquire() is True
        assert budget.try_acquire() is False  # inside the cooldown

    def test_window_slides(self):
        budget = RestartBudget(max_restarts=1, window_s=0.05, cooldown_s=0.0)
        assert budget.try_acquire() is True
        assert budget.try_acquire() is False
        time.sleep(0.08)
        assert budget.try_acquire() is True

    def test_try_acquire_state_distinguishes_cooldown_from_exhaustion(self):
        """A cooldown rejection is pacing, not a terminal verdict: the caller
        keeps the replica ISOLATED and retries on the next polling round
        instead of going DEAD."""
        budget = RestartBudget(max_restarts=2, window_s=3600.0, cooldown_s=60.0)
        assert budget.try_acquire_state() == "acquired"
        # Inside the anti-flap cooldown: deferred, not exhausted.
        assert budget.try_acquire_state() == "cooldown"
        assert budget.try_acquire_state() == "cooldown"

        budget = RestartBudget(max_restarts=1, window_s=3600.0, cooldown_s=0.0)
        assert budget.try_acquire_state() == "acquired"
        assert budget.try_acquire_state() == "exhausted"


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


class _RecordingLoadBalancer:
    """Plain stand-in for the embedded GlobalRequestLoadBalancer: the
    supervisor calls it in-process (no .remote indirection)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def remove_server_if_current(self, server_id, expected_handle):
        self.calls.append(("remove_server_if_current", server_id, expected_handle))
        return True

    def remove_servers(self, server_ids):
        self.calls.append(("remove_servers", tuple(server_ids)))

    def add_servers(self, servers):
        self.calls.append(("add_servers", tuple(servers)))

    def clear_sticky_cache(self):
        self.calls.append(("clear_sticky_cache",))
        return {"cleared_entries": 0, "server_loads": {}}


def _stub_recovery_steps(supervisor, restart_result=None):
    """Fence the LB/register/verify edges; keep the real recovery sequencing."""
    supervisor._isolate = lambda replica, key: _async_value(None)
    supervisor._verify = lambda replica, expected_version: _async_value(None)
    supervisor._register = lambda replica, key: _async_value(None)

    async def _restart(*, weight_source=None):
        return restart_result

    return _restart


class TestSupervisor:
    def test_checkpoint_cold_start_passes_weight_source_through_the_restart_chain(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._plan_weight_source = lambda: _async_value(
            WeightSourceOverride(model_path="/ckpt/global_step_8/actor/huggingface", global_steps=8)
        )
        restart = _stub_recovery_steps(supervisor)
        replica = _FakeRolloutReplica(0)
        replica.restart = restart
        replica.servers = [SimpleNamespace(set_global_steps=SimpleNamespace(remote=lambda v: _async_value(None)))]
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        # _register is faked (the real one owns the SERVING flip), so VERIFYING
        # proves the checkpoint cold-start walk: restart(weight_source) -> stamp -> verify.
        assert supervisor.get_state("0") == ReplicaRuntimeState.VERIFYING
        # Fencing generation bumped at conviction.
        assert manager._group_generation == 1

    def test_cold_start_verify_failure_falls_back_to_deferred_sync(self):
        """The rebuild is up when VERIFYING fails — the replica must fall back
        to the deferred-weight-sync semantics (COLD_START), not stay ISOLATED
        (a healthy probe on an ISOLATED replica is never re-convicted, so it
        would be fenced forever)."""
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._plan_weight_source = lambda: _async_value(
            WeightSourceOverride(model_path="/ckpt/global_step_8/actor/huggingface", global_steps=8)
        )
        restart = _stub_recovery_steps(supervisor)
        replica = _FakeRolloutReplica(0)
        replica.restart = restart
        replica.servers = [SimpleNamespace(set_global_steps=SimpleNamespace(remote=lambda v: _async_value(None)))]
        manager.rollout_replicas = [replica]

        async def _failing_verify(replica, expected_version):
            raise RuntimeError("not healthy yet")

        supervisor._verify = _failing_verify

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.COLD_START
        # Back on the syncable set: the next sync installs weights and the
        # sync-succeeded notification finishes recovery.
        snapshot = supervisor.get_membership()
        assert snapshot.replicas == [replica]

    def test_deferred_weight_sync_relays_empty_and_waits_for_sync(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._plan_weight_source = lambda: _async_value(None)
        restart = _stub_recovery_steps(supervisor)
        replica = _FakeRolloutReplica(0)
        replica.restart = restart
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.COLD_START
        snapshot = supervisor.get_membership()
        # COLD_START stays syncable: the next sync installs its weights.
        assert snapshot.replicas == [replica]
        assert snapshot.group_generation == 1
        assert snapshot.fatal_error is None
        # Nothing is serving while the rebuilt replica waits: the trainer's
        # poll hook must fire the out-of-band sync (the starved step cannot).
        assert snapshot.awaiting_weight_sync is True

    def test_budget_exhaustion_goes_dead_and_marks_fatal(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._budgets["0"] = RestartBudget(max_restarts=1, window_s=3600.0, cooldown_s=0.0)
        supervisor._budgets["0"].try_acquire()
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.DEAD
        # Fatal lives on the manager (the membership authority) — the
        # pulled snapshot propagates it to the checkpoint manager.
        assert manager._fatal_error is not None
        snapshot = supervisor.get_membership()
        assert snapshot.fatal_error is not None
        # DEAD replicas are fenced out of the sync group entirely — the
        # next sync proceeds without the dead member.
        assert snapshot.replicas == []

    def test_cooldown_defers_the_restart_and_keeps_the_replica_alive(self):
        """The anti-flap cooldown is pacing, not a verdict: a failed restart
        whose retry lands inside the cooldown stays ISOLATED and retries on
        the next polling round — it must not burn the budget into DEAD."""
        supervisor, manager = _supervisor_with_fake_manager()
        # One restart already happened; its event is inside the cooldown.
        supervisor._budgets["0"] = RestartBudget(max_restarts=3, window_s=3600.0, cooldown_s=60.0)
        supervisor._budgets["0"].try_acquire()
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]

        asyncio.run(supervisor._handle_confirmed_fault(replica, "0", None))

        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert manager._fatal_error is None

    def test_membership_flags_awaiting_weight_sync_only_when_nothing_serves(self):
        """The out-of-band sync trigger: true exactly when no replica is
        serving while a rebuilt one (COLD_START) waits for the next weight
        sync to promote it."""
        supervisor, manager = _supervisor_with_fake_manager()
        serving, rebuilt = _FakeRolloutReplica(0), _FakeRolloutReplica(1)
        manager.rollout_replicas = [serving, rebuilt]

        snapshot = supervisor.get_membership()
        # Nothing was ever convicted: every replica defaults to SERVING.
        assert snapshot.awaiting_weight_sync is False

        supervisor._states["1"] = ReplicaRuntimeState.COLD_START
        snapshot = supervisor.get_membership()
        # A rebuilt replica is waiting, but its peer still serves: the
        # normal step-end sync promotes it — no out-of-band trigger.
        assert snapshot.awaiting_weight_sync is False

        supervisor._states["0"] = ReplicaRuntimeState.DEAD
        snapshot = supervisor.get_membership()
        assert snapshot.awaiting_weight_sync is True

    def test_false_alarm_returns_to_serving(self):
        supervisor, manager = _supervisor_with_fake_manager()

        class _ProbeReplica:
            replica_rank = 0
            name_suffix = ""
            servers = [SimpleNamespace()]
            workers = []

            def __init__(self):
                self.probe_count = 0

            async def probe_once(self):
                self.probe_count += 1
                if self.probe_count == 1:
                    return FaultEvidence(source="detect_faults", error_class="probe_timeout")
                return None

        replica = _ProbeReplica()
        supervisor._probe_replica = lambda r: r.probe_once()
        asyncio.run(supervisor._probe_and_classify(replica, "0"))
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING
        assert replica.probe_count == 2

    def test_double_failure_confirms(self):
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _DeadReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        replica = _DeadReplica()
        supervisor._probe_replica = lambda r: _async_value(
            FaultEvidence(source="detect_faults", error_class="EngineDeadError")
        )
        asyncio.run(supervisor._probe_and_classify(replica, "0"))
        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert scheduled == ["0"]

    def test_gcs_dead_convicts_without_reprobe(self):
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _DeadReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        probe_calls = []

        async def _probe(replica):
            probe_calls.append(1)
            return FaultEvidence(source="actor_state", error_class="ActorGCSDead")

        supervisor._probe_replica = _probe
        asyncio.run(supervisor._probe_and_classify(_DeadReplica(), "0"))
        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert len(probe_calls) == 1  # authoritative: no re-verification

    def test_polling_round_convicts_a_cold_start_replica(self):
        """Polling backstop: a COLD_START replica that dies while waiting for
        the next weight sync is convicted by the poll — the all-or-nothing
        sync only fails the whole attempt; it never attributes the death, so
        the poll is the only detection path for a staged replica."""
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _DeadReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        replica = _DeadReplica()
        manager.rollout_replicas = [replica]
        supervisor._states["0"] = ReplicaRuntimeState.COLD_START
        supervisor._probe_replica = lambda r: _async_value(
            FaultEvidence(source="detect_faults", error_class="EngineDeadError")
        )

        asyncio.run(supervisor.detect_faults())
        assert supervisor.get_state("0") == ReplicaRuntimeState.ISOLATED
        assert scheduled == ["0"]

    def test_polling_round_skips_transitional_states(self):
        supervisor, manager = _supervisor_with_fake_manager()

        class _SilentReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

            def boom(self):
                raise AssertionError("probing a driven state must not happen")

        replica = _SilentReplica()
        manager.rollout_replicas = [replica]
        for state in (ReplicaRuntimeState.RESTARTING, ReplicaRuntimeState.VERIFYING, ReplicaRuntimeState.DEAD):
            supervisor._states["0"] = state
            supervisor._probe_replica = lambda r: r.boom()
            asyncio.run(supervisor.detect_faults())

    def test_probe_membership_convicts_a_dead_replica_before_the_snapshot(self):
        """The update_weights-entry probe: one fresh detection round runs
        before the snapshot is built, so a replica that died inside the
        detection-window staleness of the periodic poll is fenced out of the
        sync group before the group is built (no commit-watchdog burn)."""
        supervisor, manager = _supervisor_with_fake_manager()
        scheduled = []
        supervisor._schedule_recovery = lambda replica, key, evidence: scheduled.append(key)

        class _DeadReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        class _HealthyReplica:
            replica_rank = 1
            name_suffix = ""
            servers = []
            workers = []

        dead, healthy = _DeadReplica(), _HealthyReplica()
        manager.rollout_replicas = [dead, healthy]

        async def _probe(replica):
            if replica is dead:
                return FaultEvidence(source="detect_faults", error_class="EngineDeadError")
            return None

        supervisor._probe_replica = _probe

        snapshot = asyncio.run(supervisor.probe_membership())

        assert scheduled == ["0"]  # convicted inside the entry round
        assert snapshot.replicas == [healthy]  # fenced out before the group build

    def test_probe_membership_falls_back_to_the_last_snapshot_on_a_hung_round(self, monkeypatch):
        supervisor, manager = _supervisor_with_fake_manager()
        monkeypatch.setattr("verl.workers.rollout.fault_tolerance._ENTRY_ROUND_TIMEOUT_S", 0.05)

        class _SilentReplica:
            replica_rank = 0
            name_suffix = ""
            servers = []
            workers = []

        replica = _SilentReplica()
        manager.rollout_replicas = [replica]

        async def _hung_round():
            await asyncio.sleep(10)

        supervisor.detect_faults = _hung_round

        snapshot = asyncio.run(supervisor.probe_membership())
        # The round was abandoned; the last view (still SERVING) answers.
        assert snapshot.replicas == [replica]

    def test_weight_sync_succeeded_promotes_every_cold_start_replica(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._verify = lambda replica, expected_version: _async_value(None)
        registered = []

        def _fake_register(replica, key):
            registered.append(key)
            supervisor._states[key] = ReplicaRuntimeState.SERVING  # the real _register owns this flip
            return _async_value(None)

        supervisor._register = _fake_register
        replicas = [_FakeRolloutReplica(0), _FakeRolloutReplica(1)]
        manager.rollout_replicas = replicas
        supervisor._states["0"] = ReplicaRuntimeState.COLD_START
        supervisor._states["1"] = ReplicaRuntimeState.COLD_START

        asyncio.run(supervisor.weight_sync_succeeded())

        assert registered == ["0", "1"]
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING
        assert supervisor.get_state("1") == ReplicaRuntimeState.SERVING

    def test_weight_sync_succeeded_ignores_non_cold_start_replicas(self):
        supervisor, manager = _supervisor_with_fake_manager()
        supervisor._verify = lambda replica, expected_version: _async_value(None)
        registered = []
        supervisor._register = lambda replica, key: registered.append(key) or _async_value(None)
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]
        supervisor._states["0"] = ReplicaRuntimeState.SERVING

        asyncio.run(supervisor.weight_sync_succeeded())

        assert registered == []
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING

    def test_weight_sync_succeeded_verification_failure_retries_next_sync(self):
        supervisor, manager = _supervisor_with_fake_manager()

        async def _failing_verify(replica, expected_version):
            raise RuntimeError("not healthy yet")

        supervisor._verify = _failing_verify
        registered = []
        supervisor._register = lambda replica, key: registered.append(key) or _async_value(None)
        replica = _FakeRolloutReplica(0)
        manager.rollout_replicas = [replica]
        supervisor._states["0"] = ReplicaRuntimeState.COLD_START

        asyncio.run(supervisor.weight_sync_succeeded())

        assert registered == []
        # Back to COLD_START: the next sync's notification retries — the
        # derived state self-heals the promotion loop.
        assert supervisor.get_state("0") == ReplicaRuntimeState.COLD_START

    def test_isolate_and_register_call_the_embedded_balancer_in_process(self):
        supervisor, manager = _supervisor_with_fake_manager()
        lb = _RecordingLoadBalancer()
        manager.global_load_balancer = lb
        replica = _FakeRolloutReplica(0)
        replica._server_address = "addr-0"
        replica._server_handle = "handle-0"
        replica.server_address = "addr-0"
        replica.server_handle = "handle-0"
        replica.abort_all_requests = lambda **kwargs: _async_value(None)

        asyncio.run(supervisor._isolate(replica, "0"))
        asyncio.run(supervisor._register(replica, "0"))

        assert lb.calls == [
            ("remove_server_if_current", "addr-0", "handle-0"),
            ("add_servers", ("addr-0",)),
            ("clear_sticky_cache",),
        ]
        assert supervisor.get_state("0") == ReplicaRuntimeState.SERVING

    def test_membership_snapshot_filters_non_syncable_replicas(self):
        supervisor, manager = _supervisor_with_fake_manager()
        replicas = [_FakeRolloutReplica(0), _FakeRolloutReplica(1), _FakeRolloutReplica(2), _FakeRolloutReplica(3)]
        manager.rollout_replicas = replicas
        manager._group_generation = 2
        supervisor._states = {
            "0": ReplicaRuntimeState.SERVING,
            "1": ReplicaRuntimeState.COLD_START,
            "2": ReplicaRuntimeState.ISOLATED,
            "3": ReplicaRuntimeState.DEAD,
        }

        snapshot = supervisor.get_membership()
        assert snapshot.replicas == [replicas[0], replicas[1]]
        # ISOLATED / DEAD are fenced out of the snapshot — the wait-retry
        # loop never holds a retry for them; the survivors sync and they
        # rejoin at a later sync.
        assert snapshot.group_generation == 2
        assert isinstance(snapshot, MembershipSnapshot)

    def test_membership_growth_re_fences_the_group(self):
        """A rebuilt replica re-entering the syncable set re-fences the
        group: the next sync builds under a fresh name, because the
        survivors' cached fenced group still carries the shrunken
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

    def test_membership_pure_shrink_does_not_re_fence(self):
        """Shrink fencing belongs to the conviction path — the snapshot
        itself never bumps the generation."""
        supervisor, manager = _supervisor_with_fake_manager()
        replicas = [_FakeRolloutReplica(0), _FakeRolloutReplica(1)]
        manager.rollout_replicas = replicas
        assert supervisor.get_membership().group_generation == 0
        supervisor._states["1"] = ReplicaRuntimeState.DEAD  # fenced out without a fresh conviction
        snapshot = supervisor.get_membership()
        assert snapshot.group_generation == 0
        assert snapshot.replicas == [replicas[0]]

    def test_membership_oscillation_re_fences_on_re_entry(self):
        """COLD_START -> VERIFYING (not syncable) -> COLD_START: re-entry is
        growth against the snapshot issued while the replica was fenced out,
        so it re-fences; a member that never left the issued set does not."""
        supervisor, manager = _supervisor_with_fake_manager()
        manager.rollout_replicas = [_FakeRolloutReplica(0)]
        supervisor._states = {"0": ReplicaRuntimeState.COLD_START}
        assert supervisor.get_membership().group_generation == 0  # first issue never fences
        assert supervisor.get_membership().group_generation == 0  # unchanged set: no re-fence
        supervisor._states["0"] = ReplicaRuntimeState.VERIFYING
        supervisor.get_membership()  # issued empty: pure shrink, no re-fence
        supervisor._states["0"] = ReplicaRuntimeState.COLD_START
        assert supervisor.get_membership().group_generation == 1  # re-entry re-fences

    def test_applicability_gates(self):
        for bad in [
            dict(name="trtllm"),
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

    def test_applicability_accepts_vllm(self):
        rollout_config = SimpleNamespace(
            name="vllm",
            nnodes=1,
            disaggregation=SimpleNamespace(enabled=False),
            checkpoint_engine=SimpleNamespace(backend="nccl"),
            quantization=SimpleNamespace(enabled=False),
        )
        supervisor, _ = _supervisor_with_fake_manager(rollout_config=rollout_config)
        supervisor._validate_applicability()

    def test_applicability_rejects_sglang(self):
        # The SGLang server-side restart contract is not implemented yet.
        rollout_config = SimpleNamespace(
            name="sglang",
            nnodes=1,
            disaggregation=SimpleNamespace(enabled=False),
            checkpoint_engine=SimpleNamespace(backend="nccl"),
            quantization=SimpleNamespace(enabled=False),
        )
        supervisor, _ = _supervisor_with_fake_manager(rollout_config=rollout_config)
        with pytest.raises(NotImplementedError, match="sglang"):
            supervisor._validate_applicability()

    def test_unmerged_lora_rejected(self):
        supervisor, manager = _supervisor_with_fake_manager()
        manager.model_config = SimpleNamespace(lora_rank=16, lora={"merge": False})
        with pytest.raises(NotImplementedError):
            supervisor._validate_applicability()


# ── Checkpoint locator (cold-start discovery) ───────────────────────────────────


def _make_megatron_step(root, step, with_hf=True):
    step_dir = root / f"global_step_{step}"
    model_dir = step_dir / "actor" / "model"
    (model_dir / "huggingface").mkdir(parents=True, exist_ok=True)
    if with_hf:
        (model_dir / "huggingface" / "config.json").write_text("{}")
        (model_dir / "huggingface" / "tokenizer_config.json").write_text("{}")
        (model_dir / "huggingface" / "model-00001-of-00002.safetensors").write_text("x")
    return step_dir


def _make_fsdp_step(root, step):
    step_dir = root / f"global_step_{step}"
    hf_dir = step_dir / "actor" / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / "tokenizer_config.json").write_text("{}")
    (hf_dir / "model.safetensors").write_text("x")
    return step_dir


class TestLocateLatestValidCheckpoint:
    def test_prefers_tracker_step(self, tmp_path):
        _make_megatron_step(tmp_path, 10)
        _make_megatron_step(tmp_path, 20)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("10")
        path, step = _locate_cold_start_checkpoint(str(tmp_path))
        assert step == 10
        assert path.endswith("global_step_10/actor/model/huggingface")

    def test_resolves_fsdp_layout(self, tmp_path):
        # FSDP writes the HF export directly under actor/ (no model/ nesting).
        _make_fsdp_step(tmp_path, 10)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("10")
        path, step = _locate_cold_start_checkpoint(str(tmp_path))
        assert step == 10
        assert path.endswith("global_step_10/actor/huggingface")

    def test_no_readable_tracker_raises(self, tmp_path):
        # Tracker-only by design: the step directory exists but the tracker was
        # never written (first save in flight, or torn) — defer, never scan.
        _make_megatron_step(tmp_path, 20)
        with pytest.raises(CheckpointUnavailable):
            _locate_cold_start_checkpoint(str(tmp_path))

    def test_tracker_step_without_hf_export_raises(self, tmp_path):
        # The tracker points at a save that carried no engine-loadable HF export:
        # defer — the older valid step is deliberately not scanned into.
        _make_megatron_step(tmp_path, 30, with_hf=False)
        _make_megatron_step(tmp_path, 24)
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("30")
        with pytest.raises(CheckpointUnavailable):
            _locate_cold_start_checkpoint(str(tmp_path))

    def test_no_checkpoint_raises_unavailable(self, tmp_path):
        with pytest.raises(CheckpointUnavailable):
            _locate_cold_start_checkpoint(str(tmp_path))
        with pytest.raises(CheckpointUnavailable):
            _locate_cold_start_checkpoint(str(tmp_path / "does-not-exist"))


# ── CheckpointEngineManager fault-tolerance surface (half-object pattern) ──────


def _manager_half_object():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager._fault_tolerant = False
    manager._pre_sync_hook = None
    manager._sync_succeeded_callback = None
    manager._group_generation = 0
    manager._retry_epoch = 0
    manager._fatal_error = None
    # A real manager always carries its replica list; one serving replica is
    # the baseline for the retry tests (the zero-survivor wait tests replace
    # it wholesale via sync_membership).
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
    def test_replica_key_includes_suffix(self):
        replica = _FakeReplicaForCM(3, "_reward")
        assert CheckpointEngineManager._replica_key(replica) == "3_reward"

    def test_sync_membership_adopts_the_authoritative_snapshot(self):
        manager = _manager_half_object()
        manager.enable_fault_tolerance()
        replicas = [_FakeReplicaForCM(0), _FakeReplicaForCM(1)]
        snapshot = MembershipSnapshot(
            replicas=replicas,
            group_generation=4,
            fatal_error=None,
        )

        manager.sync_membership(snapshot)

        assert manager.replicas == replicas
        assert manager._group_generation == 4
        assert manager._fatal_error is None

        # A later snapshot replaces the earlier one wholesale (derived state).
        fresh = MembershipSnapshot(replicas=[replicas[0]], group_generation=5, fatal_error="dead")
        manager.sync_membership(fresh)
        assert manager.replicas == [replicas[0]]
        assert manager._group_generation == 5
        assert manager._fatal_error == "dead"

    def test_guarded_resume_swallows_per_replica_failures(self):
        manager = _manager_half_object()
        resumed = []

        class _Resumes:
            def __init__(self, key, fail=False):
                self.key, self.fail = key, fail

            async def resume_kv_cache(self):
                resumed.append((self.key, "kv"))
                if self.fail:
                    raise RuntimeError("dead replica")

            async def resume_generation(self):
                resumed.append((self.key, "gen"))

        replica_ok, replica_dead = _FakeReplicaForCM(0), _FakeReplicaForCM(1)
        ok, dead = _Resumes("0"), _Resumes("1", fail=True)
        replica_ok.resume_kv_cache, replica_ok.resume_generation = ok.resume_kv_cache, ok.resume_generation
        replica_dead.resume_kv_cache, replica_dead.resume_generation = dead.resume_kv_cache, dead.resume_generation

        asyncio.run(manager._guarded_resume([replica_dead, replica_ok]))
        # The dead replica did not block the healthy one.
        assert resumed == [("1", "kv"), ("0", "kv"), ("0", "gen")]

    def test_update_weights_raises_on_fatal(self):
        """Fatal arrives via the pulled snapshot (the production channel) —
        sync_membership is the single membership/fatal boundary, so a direct
        _fatal_error write is not part of the contract."""
        manager = _manager_half_object()
        manager.enable_fault_tolerance()
        manager.sync_membership(
            MembershipSnapshot(
                replicas=[],
                group_generation=0,
                fatal_error="all rollout replicas are DEAD",
            )
        )

        async def scenario():
            try:
                await manager.update_weights(1)
                raised = None
            except RuntimeError as exc:
                raised = exc
            assert raised is not None and "fatal" in str(raised)

        asyncio.run(scenario())

    def test_update_weights_runs_pre_sync_reconcile_first(self):
        """The pre-sync hook (the membership pull) runs before anything else and
        its snapshot is what the sync reads; a hook failure is non-fatal."""
        manager = _manager_half_object()
        manager.enable_fault_tolerance()
        manager.backend = "naive"
        manager.actor_wg = SimpleNamespace(
            update_weights=lambda **kwargs: [_async_value({"x": 1})],
        )
        pulls = []

        async def _reconcile():
            pulls.append(1)
            replica = _FakeReplicaForCM(0)
            manager.sync_membership(
                MembershipSnapshot(replicas=[replica], group_generation=1, fatal_error=None)
            )

        manager._pre_sync_hook = _reconcile

        manager.update_weights(1)  # @auto_await: sync call runs the coroutine
        assert pulls == [1]
        assert manager._group_generation == 1


class TestManagerWaitAndRetry:
    """The driver-side wait-and-retry loop: all-or-nothing attempts, bounded,
    fail-closed on fatal, hold only for the fault-detection grace window
    (never for a rebuild to settle — the survivors sync first); with zero
    syncable members it waits for recovery instead of burning attempts."""

    @pytest.fixture(autouse=True)
    def _fast_waits(self, monkeypatch):
        monkeypatch.setattr(base_mod, "_PROBE_GRACE_S", 0.0)

    def test_update_weights_retries_after_failed_attempt(self):
        calls = []

        def _fail_first(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("member died mid-sync")
            return [_async_value({"x": 1})]

        manager = _manager_half_object()
        manager.enable_fault_tolerance()
        _naive_pipeline_manager(manager, _fail_first)

        manager.update_weights(1)
        assert len(calls) == 2

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

    def test_update_weights_waits_when_nothing_is_syncable(self, monkeypatch):
        """Zero syncable members at entry: hold for recovery instead of
        burning the attempt budget on doomed empty-group builds (the
        single-replica regression); the wait itself is not an attempt."""
        monkeypatch.setattr(base_mod, "_EMPTY_MEMBERSHIP_POLL_S", 0.0)
        pulls = []
        syncs = []

        async def _pull():
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
        manager.enable_fault_tolerance(pre_sync_hook=_pull)
        _naive_pipeline_manager(manager, _actor_sync)

        manager.update_weights(1)
        assert len(pulls) == 2  # the empty pull, then the wait's re-pull
        assert syncs == [1]  # the sync ran once, after a replica re-entered

    def test_recovery_wait_fails_closed_when_nothing_recovers(self, monkeypatch):
        monkeypatch.setattr(base_mod, "_EMPTY_MEMBERSHIP_POLL_S", 0.0)
        monkeypatch.setattr(base_mod, "_RECOVERY_WAIT_TIMEOUT_S", 0.05)
        manager = _manager_half_object()

        async def _pull():
            manager.sync_membership(MembershipSnapshot(replicas=[], group_generation=1, fatal_error=None))

        manager.enable_fault_tolerance(pre_sync_hook=_pull)
        _naive_pipeline_manager(manager, lambda **kwargs: [_async_value({})])

        with pytest.raises(RuntimeError, match="no syncable rollout replicas recovered"):
            manager.update_weights(1)

    def test_recovery_wait_terminates_on_fatal(self, monkeypatch):
        """All replicas DEAD while the loop waits: the fatal flag raised by
        the next pull fails the step — the wait never outlives a terminal
        membership."""
        monkeypatch.setattr(base_mod, "_EMPTY_MEMBERSHIP_POLL_S", 0.0)
        manager = _manager_half_object()
        pulls = []

        async def _pull():
            pulls.append(1)
            manager.sync_membership(
                MembershipSnapshot(
                    replicas=[],
                    group_generation=1,
                    fatal_error=None if len(pulls) == 1 else "all rollout replicas are DEAD",
                )
            )

        manager.enable_fault_tolerance(pre_sync_hook=_pull)
        _naive_pipeline_manager(manager, lambda **kwargs: [_async_value({})])

        with pytest.raises(RuntimeError, match="fatal"):
            manager.update_weights(1)
        assert len(pulls) >= 2  # the fatal arrived while waiting, not at entry

    def test_submit_update_rejects_an_empty_member_set(self):
        """Backstop: a path that reaches submit_update with nothing syncable
        gets a clear error, not an opaque single-controller TypeError on an
        empty worker list."""
        manager = _manager_half_object()
        manager.replicas = []

        with pytest.raises(RuntimeError, match="no rollout replicas available"):
            asyncio.run(manager.submit_update(1))

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

    def test_wait_for_detection_pulls_once_and_gates_on_fatal(self):
        manager = _manager_half_object()
        pulls = []

        async def _pull():
            pulls.append(1)
            manager.sync_membership(
                MembershipSnapshot(
                    replicas=[],
                    group_generation=len(pulls),
                    fatal_error="all rollout replicas are DEAD",
                )
            )

        manager._pre_sync_hook = _pull

        with pytest.raises(RuntimeError, match="fatal"):
            asyncio.run(manager._wait_for_detection())
        # One pull after the grace window: the wait never polls for a
        # rebuild to settle — a fenced-out replica rejoins at the next sync.
        assert len(pulls) == 1

    def test_wait_for_detection_returns_after_a_single_pull(self):
        manager = _manager_half_object()
        pulls = []

        async def _pull():
            pulls.append(1)
            manager.sync_membership(MembershipSnapshot(replicas=[], group_generation=1, fatal_error=None))

        manager._pre_sync_hook = _pull

        asyncio.run(manager._wait_for_detection())
        assert len(pulls) == 1

    def test_sync_succeeded_notification_fires_callback(self):
        manager = _manager_half_object()
        fired = []
        manager.enable_fault_tolerance(sync_succeeded_callback=lambda: fired.append(1))

        manager._notify_sync_succeeded()
        assert fired == [1]

    def test_sync_succeeded_notification_swallows_callback_errors(self):
        manager = _manager_half_object()
        manager.enable_fault_tolerance(sync_succeeded_callback=lambda: 1 / 0)

        manager._notify_sync_succeeded()  # must not raise


class TestPreSyncReconcile:
    def test_reconcile_pulls_a_fresh_probe_snapshot(self):
        """The hook drives the fresh-probe RPC (not the cheap stale pull)
        and writes the result into the checkpoint manager."""
        adopted = []

        snapshot = MembershipSnapshot(replicas=[], group_generation=3, fatal_error=None)
        handle = SimpleNamespace(
            probe_and_get_membership=SimpleNamespace(remote=lambda: _async_value(snapshot))
        )
        checkpoint_manager = SimpleNamespace(sync_membership=adopted.append)

        reconcile = make_pre_sync_reconcile(checkpoint_manager, handle)
        asyncio.run(reconcile())

        assert adopted == [snapshot]
