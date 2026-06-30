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
"""CPU integration tests for the streaming replay buffers against a *real* TransferQueue.

Wires the *real* replay buffers (default prompt-level :class:`ReplayBuffer` and opt-in rollout-level
:class:`SessionReplayBuffer`) against a *real* TransferQueue (SimpleStorage / ZMQ in-memory, no GPU)
backed by a local Ray cluster. The only simulated pieces are the parts that inherently need a GPU
rollout server:

- the trainer-side ``_feed_one_batch`` glue (here: register a prompt metadata key in TQ, mirroring
  ``PPOTrainerFullyAsync._feed_one_batch``: ``{is_prompt, status, global_steps, n}``), and
- the agent-loop worker (here: write the trajectory data keys + the prompt status / per-session
  completion markers, mirroring the prompt-level and rollout-level workers).

The autonomous feeder thread itself lives inside ``PPOTrainerFullyAsync`` (which pulls the full GPU
serving stack and is not importable on a CPU-only box); its throttle/pause behavior is covered by
the GPU e2e scripts (tests/special_e2e/run_v1_fully_async*.sh). What is validated here is the buffer
logic the feeder depends on — session-counting readiness, abnormal-rollout discard, and that a
concurrent produce -> complete -> sample -> clear cycle reaches a bounded steady state without
deadlock.

Run:  /cbs/cua/.venv/bin/python -m pytest \
          tests/trainer/ppo/v1/test_streaming_feeder_tq_integration_on_cpu.py -v
"""

import importlib.util
import sys
import threading
import time
import types
import uuid
from pathlib import Path

import pytest
import ray
import transfer_queue as tq
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve()
while not (_ROOT / "verl" / "trainer" / "ppo" / "v1").exists():
    _ROOT = _ROOT.parent


def _load_pkg_module(modname: str, rel: str):
    """File-load a ``verl.trainer.ppo.v1.*`` module under its real dotted name without triggering
    the heavy ``verl.trainer.ppo.v1`` package __init__ (which imports the GPU rollout stack).

    A lightweight stub package for ``verl.trainer.ppo.v1`` is registered first so absolute imports
    between the loaded modules (e.g. ``SessionReplayBuffer`` importing ``ReplayBuffer``) resolve to
    these file-loaded copies instead of importing the real package.
    """
    if "verl.trainer.ppo.v1" not in sys.modules:
        stub = types.ModuleType("verl.trainer.ppo.v1")
        stub.__path__ = [str(_ROOT / "verl" / "trainer" / "ppo" / "v1")]
        sys.modules["verl.trainer.ppo.v1"] = stub
    spec = importlib.util.spec_from_file_location(modname, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


_rb = _load_pkg_module("verl.trainer.ppo.v1.replay_buffer", "verl/trainer/ppo/v1/replay_buffer.py")
_rbs = _load_pkg_module("verl.trainer.ppo.v1.replay_buffer_session", "verl/trainer/ppo/v1/replay_buffer_session.py")
ReplayBuffer = _rb.ReplayBuffer
SessionReplayBuffer = _rbs.SessionReplayBuffer
compute_complete_uids = _rbs.compute_complete_uids


def setup_module(module):
    ray.init(num_cpus=6, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    conf = OmegaConf.create(
        {
            "enable": True,
            "metrics": {"enabled": False, "port": 0},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"total_storage_size": 10000, "num_data_storage_units": 2},
            },
        }
    )
    tq.init(conf)


def teardown_module(module):
    try:
        tq.close()
    finally:
        ray.shutdown()


def _clear_all():
    data = tq.kv_list() or {}
    for partition_id, items in data.items():
        if items:
            tq.kv_clear(partition_id=partition_id, keys=list(items.keys()))


@pytest.fixture(autouse=True)
def _clean_tq():
    _clear_all()
    yield
    _clear_all()


def _make_session_rb(strategy="drop", threshold=8, parameter_sync_step=1):
    return SessionReplayBuffer(
        trainer_mode="fully_async",
        trainer_config=OmegaConf.create({"parameter_sync_step": parameter_sync_step}),
        max_off_policy_threshold=threshold,
        max_off_policy_strategy=strategy,
        sampler_kwargs=OmegaConf.create({}),
        poll_interval=0.05,
    )


def _make_status_rb(strategy="drop", threshold=8, parameter_sync_step=1):
    return ReplayBuffer(
        trainer_mode="fully_async",
        trainer_config=OmegaConf.create({"parameter_sync_step": parameter_sync_step}),
        max_off_policy_threshold=threshold,
        max_off_policy_strategy=strategy,
        sampler_kwargs=OmegaConf.create({}),
        poll_interval=0.05,
    )


def _register_prompt(uid, global_steps, n=1, partition_id="train"):
    """Mirror ``_feed_one_batch``'s prompt metadata write (carries both status and n)."""
    tq.kv_batch_put(
        keys=[uid],
        partition_id=partition_id,
        tags=[{"is_prompt": True, "status": "pending", "global_steps": global_steps, "n": n}],
    )


def _register_session(uid, session_id=0, status="success", global_steps=1, partition_id="train"):
    """Mirror the rollout-level worker completing one rollout: write the trajectory data key
    (success only) then the per-session completion marker."""
    if status == "success":
        tq.kv_batch_put(
            keys=[f"{uid}_{session_id}_0"],
            partition_id=partition_id,
            tags=[{"global_steps": global_steps, "seq_len": 8, "status": "success"}],
        )
    tq.kv_batch_put(
        keys=[f"{uid}_sess{session_id}"],
        partition_id=partition_id,
        tags=[{"is_session": True, "session_id": session_id, "status": status}],
    )


def _set_prompt_status(uid, status, global_steps=1, n=1, partition_id="train", with_data=True):
    """Mirror the prompt-level worker driving the {uid} status marker, plus its data keys."""
    if status in ("finished", "failure") and with_data:
        tq.kv_batch_put(
            keys=[f"{uid}_0_0"],
            partition_id=partition_id,
            tags=[{"global_steps": global_steps, "seq_len": 8, "status": "success"}],
        )
    tq.kv_batch_put(
        keys=[uid],
        partition_id=partition_id,
        tags=[{"is_prompt": True, "status": status, "global_steps": global_steps, "n": n}],
    )


# ======================================================================================
# Default prompt-level ReplayBuffer: status-based readiness
# ======================================================================================
def test_status_buffer_count_inflight_reflects_tq():
    rb = _make_status_rb()
    _set_prompt_status("p0", "pending")
    _set_prompt_status("p1", "running")
    _set_prompt_status("p2", "finished")
    _set_prompt_status("p3", "failure")
    counts = rb.count_inflight("train")
    assert counts == {"pending": 1, "running": 1, "finished": 1, "failure": 1}, counts


def test_status_buffer_samples_finished_and_failure():
    rb = _make_status_rb(strategy="none")
    _set_prompt_status("good", "finished", global_steps=1)
    _set_prompt_status("bad", "failure", global_steps=1)
    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=2)
    assert {k.split("_")[0] for k in batch.keys} == {"good", "bad"}, batch.keys


def test_status_buffer_dead_prompt_keys_is_noop():
    """The prompt-level buffer never discards (a failure prompt is still sampleable)."""
    rb = _make_status_rb()
    _set_prompt_status("bad", "failure")
    rb.count_inflight("train")
    assert rb.dead_prompt_keys("train") == []


# ======================================================================================
# Opt-in rollout-level SessionReplayBuffer: session-counting readiness
# ======================================================================================
def test_session_count_inflight_reflects_real_tq():
    rb = _make_session_rb()
    for uid in ["p0", "p1", "p2", "p3", "p4"]:
        _register_prompt(uid, 1, n=1)
    _register_session("p2", 0)
    _register_session("p3", 0)
    counts = rb.count_inflight("train")
    assert counts == {"incomplete": 3, "complete": 2}, counts


def test_session_prompt_not_ready_until_all_sessions_complete():
    rb = _make_session_rb(strategy="none")
    _register_prompt("g", 1, n=3)
    _register_session("g", 0)
    _register_session("g", 1)
    assert rb.count_inflight("train") == {"incomplete": 1, "complete": 0}
    _register_session("g", 2)
    assert rb.count_inflight("train") == {"incomplete": 0, "complete": 1}


def test_session_failed_session_counts_toward_completion():
    rb = _make_session_rb(strategy="none")
    _register_prompt("g", 1, n=2)
    _register_session("g", 0, status="success")
    _register_session("g", 1, status="failure")  # no data key, but marker completes the group
    assert rb.count_inflight("train")["complete"] == 1

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    assert {k.split("_")[0] for k in batch.keys} == {"g"}, batch.keys
    assert len(batch.keys) == 1, batch.keys  # only the successful session contributes data


def test_session_all_failed_prompt_not_sampleable():
    rb = _make_session_rb(strategy="none")
    _register_prompt("dead", 1, n=2)
    _register_session("dead", 0, status="failure")
    _register_session("dead", 1, status="failure")
    _register_prompt("good", 1, n=1)
    _register_session("good", 0, status="success")

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    assert {k.split("_")[0] for k in batch.keys} == {"good"}, batch.keys


def test_session_dead_prompt_keys_lists_only_all_failed():
    rb = _make_session_rb(strategy="none")
    _register_prompt("dead", 1, n=2)
    _register_session("dead", 0, status="failure")
    _register_session("dead", 1, status="failure")
    _register_prompt("incomplete", 1, n=2)
    _register_session("incomplete", 0, status="success")  # still generating (1/2)
    _register_prompt("good", 1, n=1)
    _register_session("good", 0, status="success")

    rb.count_inflight("train")  # syncs the buffer view (as the feeder loop does before discarding)
    keys = rb.dead_prompt_keys("train")
    assert set(keys) == {"dead", "dead_sess0", "dead_sess1"}, keys

    tq.kv_clear(partition_id="train", keys=keys)  # the feeder clears exactly these to discard it
    leftover = set(((tq.kv_list() or {}).get("train") or {}).keys())
    assert not any(k.startswith("dead") for k in leftover), leftover
    assert any(k.startswith("incomplete") for k in leftover)
    assert any(k.startswith("good") for k in leftover)


def test_session_sample_clears_prompt_and_session_markers():
    rb = _make_session_rb(strategy="none")
    _register_prompt("g", 1, n=1)
    _register_session("g", 0)

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    tq.kv_clear(keys=batch.keys, partition_id="train")  # trainer clears the data keys post-consume

    leftover = list(((tq.kv_list() or {}).get("train") or {}).keys())
    assert leftover == [], leftover  # prompt key + marker cleared by sample, data by trainer


def test_compute_complete_uids_pure():
    f = compute_complete_uids
    assert f({"a": 2, "b": 1}, {"a": {0, 1}, "b": set()}) == {"a"}
    assert f({"a": 3}, {"a": {0, 1}}) == set()
    assert f({"a": 1, "b": 1}, {"a": {0}, "b": {0}}) == {"a", "b"}
    assert f({}, {}) == set()


def test_session_strategy_none_no_staleness_gate():
    """`none`: sample the oldest batch_size usable prompts regardless of staleness — nothing is
    dropped. Note: uids must not contain '_' (sample() matches trajectories via key.split('_')[0])."""
    rb = _make_session_rb(strategy="none", threshold=2, parameter_sync_step=1)
    global_steps = 10
    for i in range(3):
        uid = f"old{i}"
        _register_prompt(uid, 1, n=1)
        _register_session(uid, 0, "success", 1)
    for i in range(3):
        uid = f"new{i}"
        _register_prompt(uid, 10, n=1)
        _register_session(uid, 0, "success", 10)

    batch, metrics = rb.sample(global_steps=global_steps, partition_id="train", batch_size=4)
    selected = {k.split("_")[0] for k in batch.keys}
    assert len(selected) == 4, selected
    assert {"old0", "old1", "old2"}.issubset(selected), selected
    assert metrics == {}, metrics


# ======================================================================================
# Buffer under concurrent produce/consume reaches a bounded steady state without deadlock.
# This drives the REAL SessionReplayBuffer the way the inlined feeder + worker + trainer do, but
# the "feed while in-flight < budget" decision is made inline here (the feeder thread itself is
# GPU-e2e covered). It guards the deadlock / unbounded-growth risk at the buffer level.
# ======================================================================================
def test_buffer_steady_state_bounded_and_progresses():
    rb = _make_session_rb(strategy="drop", threshold=8, parameter_sync_step=1)
    budget = 6
    batch_size = 2

    state_lock = threading.Lock()
    pending_uids = []
    fed_count = 0
    max_inflight_seen = 0
    param_version = [1]
    stop = threading.Event()

    def feeder():
        nonlocal fed_count
        while not stop.is_set():
            inflight = sum(rb.count_inflight("train").values())  # mirrors _feeder_loop's throttle
            if inflight < budget:
                uid = uuid.uuid4().hex[:8]
                _register_prompt(uid, param_version[0], n=1)
                with state_lock:
                    pending_uids.append((uid, param_version[0]))
                    fed_count += 1
            else:
                time.sleep(0.01)

    def worker():
        while not stop.is_set():
            item = None
            with state_lock:
                if pending_uids:
                    item = pending_uids.pop(0)
            if item is None:
                time.sleep(0.01)
                continue
            uid, gs = item
            time.sleep(0.01)  # simulate generation latency
            _register_session(uid, 0, "success", gs)

    consumed = [0]

    def consumer():
        while not stop.is_set():
            try:
                batch, _metrics = rb.sample(global_steps=param_version[0], partition_id="train", batch_size=batch_size)
            except Exception:
                time.sleep(0.02)
                continue
            if not batch.keys:
                continue
            consumed[0] += len(batch.keys)
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

    def monitor():
        nonlocal max_inflight_seen
        while not stop.is_set():
            total = sum(rb.count_inflight("train").values())
            with state_lock:
                max_inflight_seen = max(max_inflight_seen, total)
            time.sleep(0.01)

    threads = [threading.Thread(target=fn, daemon=True) for fn in (feeder, worker, consumer, monitor)]
    for t in threads:
        t.start()
    time.sleep(0.4)
    param_version[0] = 2  # simulate periodic weight syncs
    time.sleep(0.4)
    param_version[0] = 3
    time.sleep(0.4)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert fed_count > budget, f"feeder should have produced many batches, got {fed_count}"
    assert consumed[0] > 0, "consumer should have sampled complete prompts"
    assert max_inflight_seen <= budget + batch_size + 2, f"in-flight overshoot: {max_inflight_seen}"


def _run_all():
    setup_module(None)
    try:
        tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
        failures = 0
        for fn in tests:
            _clear_all()
            try:
                fn()
                print(f"PASS {fn.__name__}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                import traceback

                traceback.print_exc()
                print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            finally:
                _clear_all()
        print(f"\n{len(tests) - failures}/{len(tests)} passed")
        return failures
    finally:
        teardown_module(None)


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
