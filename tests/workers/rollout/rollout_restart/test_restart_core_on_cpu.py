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
"""CPU tests for the rollout-restart lifecycle core (verl/workers/rollout/fault_tolerance.py).

Owned-process accounting baseline and cleanup ladder against real
subprocesses, the (pid, create_time) identity probe (zombie and PID-recycle
cases), and the frozen evidence/weight-source dataclasses.
"""

import dataclasses
import os
import time

import psutil
import pytest

from verl.workers.rollout.fault_tolerance import (
    EngineProcessAccounting,
    FaultEvidence,
    ReplicaRestartError,
    WeightSourceOverride,
    verify_process_dead,
)
from verl.workers.rollout.replica import engine_failure_receipt


def _spawn_child(lifetime: int = 300) -> psutil.Popen:
    return psutil.Popen(["sleep", str(lifetime)])


class TestErrorTaxonomy:
    def test_hierarchy(self):
        assert issubclass(ReplicaRestartError, RuntimeError)


class TestDataclasses:
    def test_frozen_evidence(self):
        ev = FaultEvidence(source="detect_faults", error_class="check_health_failed")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.source = "other"

    def test_weight_source_override_defaults(self):
        override = WeightSourceOverride(model_path="/ckpt/global_step_12/actor/huggingface")
        assert override.global_steps is None
        stamped = WeightSourceOverride(model_path="/p", global_steps=9)
        assert stamped.global_steps == 9


class TestVerifyProcessDead:
    def test_live_process_is_not_dead(self):
        child = _spawn_child()
        try:
            assert not verify_process_dead(child.pid, child.create_time())
        finally:
            child.kill()
            child.wait()

    def test_dead_process_is_dead(self):
        child = _spawn_child()
        pid, create_time = child.pid, child.create_time()
        child.kill()
        child.wait()
        assert verify_process_dead(pid, create_time)

    def test_pid_reuse_counts_as_dead(self):
        child = _spawn_child()
        try:
            # Same pid, wrong create_time: the identity is gone (recycled PID).
            assert verify_process_dead(child.pid, child.create_time() + 999.0)
        finally:
            child.kill()
            child.wait()

    def test_zombie_counts_as_dead(self):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                try:
                    proc = psutil.Process(pid)
                    if proc.status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.05)
            assert verify_process_dead(pid, psutil.Process(pid).create_time())
        finally:
            os.waitpid(pid, 0)


class TestEngineProcessAccounting:
    def test_baseline_excludes_preexisting_children(self):
        preexisting = _spawn_child()
        try:
            accounting = EngineProcessAccounting()
            assert accounting.owned_pids == []
            accounting.record()
            assert preexisting.pid not in accounting.owned_pids
        finally:
            preexisting.kill()
            preexisting.wait()

    def test_record_captures_new_children(self):
        accounting = EngineProcessAccounting()
        child_a, child_b = _spawn_child(), _spawn_child()
        try:
            owned = accounting.record()
            assert {proc.pid for proc in owned} == {child_a.pid, child_b.pid}
        finally:
            for child in (child_a, child_b):
                if child.is_running():
                    child.kill()
                    child.wait()

    def test_shutdown_owned_cleans_real_subprocesses(self):
        accounting = EngineProcessAccounting()
        children = [_spawn_child() for _ in range(3)]
        accounting.record()
        identities = [(child.pid, child.create_time()) for child in children]
        accounting.shutdown_owned(wait_s=1.0, term_s=1.0, kill_s=1.0)
        for pid, create_time in identities:
            assert verify_process_dead(pid, create_time)

    def test_shutdown_owned_no_children_is_noop(self):
        accounting = EngineProcessAccounting()
        accounting.shutdown_owned()

    def test_shutdown_owned_raises_on_survivor(self, monkeypatch):
        accounting = EngineProcessAccounting()

        class _Immortal:
            pid = 424242

            def is_running(self):
                return True

            def status(self):
                return psutil.STATUS_RUNNING

            def terminate(self):
                pass

            def kill(self):
                pass

        accounting._owned = [_Immortal()]
        monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: (list(procs), list(procs)))
        with pytest.raises(ReplicaRestartError):
            accounting.shutdown_owned(wait_s=0.1, term_s=0.1, kill_s=0.1)


class _FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class _FakeOutput:
    def __init__(self, token_ids, logprobs):
        self.token_ids = token_ids
        self.logprobs = logprobs


class _FakeRequestOutput:
    def __init__(self, output):
        self.outputs = [output]


class TestEngineFailureReceipt:
    def test_keeps_complete_prefix(self):
        partial = _FakeRequestOutput(
            _FakeOutput(
                [5, 6, 7],
                [{5: _FakeLogprob(-1.0)}, {6: _FakeLogprob(-2.0)}, {7: _FakeLogprob(-3.0)}],
            )
        )
        receipt = engine_failure_receipt(42, partial)
        assert receipt.token_ids == [5, 6, 7]
        assert receipt.log_probs == [-1.0, -2.0, -3.0]
        assert receipt.stop_reason == "engine_failed"
        assert receipt.extra_fields["engine_failed"] is True
        assert receipt.extra_fields["global_steps"] == 42

    def test_drops_prefix_when_logprobs_lag_tokens(self):
        partial = _FakeRequestOutput(_FakeOutput([5, 6, 7], [{5: _FakeLogprob(-1.0)}]))
        receipt = engine_failure_receipt(None, partial)
        assert receipt.token_ids == []
        assert receipt.log_probs is None
        assert receipt.extra_fields["engine_failed"] is True

    def test_drops_prefix_on_unparsable_logprobs(self):
        partial = _FakeRequestOutput(_FakeOutput([5, 6], "not-a-logprob-list"))
        receipt = engine_failure_receipt(None, partial)
        assert receipt.token_ids == []

    def test_none_partial_gives_empty_receipt(self):
        receipt = engine_failure_receipt(7, None)
        assert receipt.token_ids == []
        assert receipt.log_probs is None
        assert receipt.extra_fields["global_steps"] == 7
