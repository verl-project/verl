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

Long-term-care minimum: the (pid, create_time) identity probe with the
owned-process cleanup ladder against real subprocesses, and the engine
failure receipt contract (verified-prefix preservation).
"""

import psutil

from verl.workers.rollout.fault_tolerance import EngineProcessAccounting, verify_process_dead
from verl.workers.rollout.replica import engine_failure_receipt


def _spawn_child(lifetime: int = 300) -> psutil.Popen:
    return psutil.Popen(["sleep", str(lifetime)])


def test_process_identity_probe_and_owned_cleanup_ladder():
    child = _spawn_child()
    pid, create_time = child.pid, child.create_time()
    assert not verify_process_dead(pid, create_time)  # a live process is never dead
    assert verify_process_dead(pid, create_time + 999.0)  # recycled pid: the identity is gone
    child.kill()
    child.wait()
    assert verify_process_dead(pid, create_time)

    accounting = EngineProcessAccounting()
    children = [_spawn_child() for _ in range(2)]
    accounting.record()
    identities = [(c.pid, c.create_time()) for c in children]
    accounting.shutdown_owned(wait_s=1.0, term_s=1.0, kill_s=1.0)
    for owned_pid, owned_ct in identities:
        assert verify_process_dead(owned_pid, owned_ct)


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


def test_engine_failure_receipt_keeps_a_verified_prefix():
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


def test_engine_failure_receipt_drops_an_unverified_prefix():
    partial = _FakeRequestOutput(_FakeOutput([5, 6, 7], [{5: _FakeLogprob(-1.0)}]))
    receipt = engine_failure_receipt(None, partial)
    assert receipt.token_ids == []
    assert receipt.log_probs is None
    assert receipt.extra_fields["engine_failed"] is True
