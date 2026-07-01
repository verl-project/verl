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
"""CPU tests for the V1 ReplayBuffer readiness models (no GPU / rollout server).

The replay buffer supports two readiness models, selected by ``rollout_level_dispatch``:

- ``False`` (default, legacy): a prompt is sampleable once its prompt-status tag flips to
  ``finished``/``failure`` (the single owning worker fans out all n sessions and sets it).
- ``True`` (rollout-level): a prompt is sampleable once all ``n`` of its per-session completion
  markers (``{uid}_sess{session_id}``) are present — readiness is *derived* by counting sessions,
  so rollouts can be dispatched one session at a time, decoupled from any single worker.

These wire the *real* ``ReplayBuffer`` against a *real* TransferQueue (SimpleStorage / ZMQ
in-memory, no GPU) backed by a local Ray cluster, exercising both models end to end.

Run:  /cbs/cua/.venv/bin/python -m pytest \
          tests/trainer/ppo/v1/test_replay_buffer_session_counting_on_cpu.py -v
"""

import importlib.util
from pathlib import Path

import pytest
import ray
import transfer_queue as tq
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve()
while not (_ROOT / "verl" / "trainer" / "ppo" / "v1").exists():
    _ROOT = _ROOT.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_rb = _load("replay_buffer_sct", "verl/trainer/ppo/v1/replay_buffer.py")
ReplayBuffer = _rb.ReplayBuffer


def setup_module(module):
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
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


def _make_replay_buffer(strategy="none", threshold=8, parameter_sync_step=1, rollout_level_dispatch=True):
    return ReplayBuffer(
        trainer_mode="separate_async",
        trainer_config=OmegaConf.create({"parameter_sync_step": parameter_sync_step}),
        max_off_policy_threshold=threshold,
        max_off_policy_strategy=strategy,
        sampler_kwargs=OmegaConf.create({}),
        poll_interval=0.05,
        rollout_level_dispatch=rollout_level_dispatch,
    )


# ----- rollout-level (session-counting) helpers -----
def _register_prompt(uid, global_steps, n=1, partition_id="train"):
    """Mirror trainer_base's rollout-level feed: prompt metadata tag carrying n."""
    tq.kv_batch_put(
        keys=[uid],
        partition_id=partition_id,
        tags=[{"is_prompt": True, "global_steps": global_steps, "n": n}],
    )


def _register_session(uid, session_id=0, status="success", global_steps=1, partition_id="train"):
    """Mirror AgentLoopWorkerTQ completing one rollout: trajectory data key (success only) +
    the per-session completion marker."""
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


# ----- legacy (prompt-status) helpers -----
def _register_prompt_legacy(uid, status, global_steps, partition_id="train"):
    tq.kv_batch_put(
        keys=[uid],
        partition_id=partition_id,
        tags=[{"is_prompt": True, "status": status, "global_steps": global_steps}],
    )


def _register_trajectory_legacy(uid, global_steps, partition_id="train"):
    tq.kv_batch_put(
        keys=[f"{uid}_0_0"],
        partition_id=partition_id,
        tags=[{"global_steps": global_steps, "seq_len": 8, "status": "success"}],
    )


# ======================================================================================
# Pure helper
# ======================================================================================
def test_compute_complete_uids_pure():
    f = _rb.compute_complete_uids
    assert f({"a": 2, "b": 1}, {"a": {0, 1}, "b": set()}) == {"a"}
    assert f({"a": 3}, {"a": {0, 1}}) == set()
    assert f({"a": 1, "b": 1}, {"a": {0}, "b": {0}}) == {"a", "b"}
    assert f({}, {}) == set()


# ======================================================================================
# Rollout-level (session-counting) readiness
# ======================================================================================
def test_session_counting_prompt_ready_when_all_sessions_complete():
    rb = _make_replay_buffer(rollout_level_dispatch=True)
    _register_prompt("g", 1, n=3)
    rb._sync_metadata_from_transfer_queue()
    assert not rb._has_enough_samples(1, "train", batch_size=1)

    _register_session("g", 0)
    _register_session("g", 1)
    rb._sync_metadata_from_transfer_queue()
    assert not rb._has_enough_samples(1, "train", batch_size=1)  # 2/3 sessions -> not ready

    _register_session("g", 2)
    rb._sync_metadata_from_transfer_queue()
    assert rb._has_enough_samples(1, "train", batch_size=1)  # 3/3 -> ready

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    assert {k.split("_")[0] for k in batch.keys} == {"g"}, batch.keys
    assert len(batch.keys) == 3, batch.keys  # all three sessions' trajectory data collected


def test_session_counting_failed_session_completes_group():
    rb = _make_replay_buffer(rollout_level_dispatch=True)
    _register_prompt("g", 1, n=2)
    _register_session("g", 0, status="success")
    _register_session("g", 1, status="failure")  # no data key, but marker completes the group
    rb._sync_metadata_from_transfer_queue()
    assert rb._has_enough_samples(1, "train", batch_size=1)

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    # only the successful session contributes a trajectory data key
    assert len(batch.keys) == 1, batch.keys
    assert {k.split("_")[0] for k in batch.keys} == {"g"}, batch.keys


def test_session_counting_sample_clears_prompt_and_markers():
    rb = _make_replay_buffer(rollout_level_dispatch=True)
    _register_prompt("g", 1, n=1)
    _register_session("g", 0)

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    tq.kv_clear(keys=batch.keys, partition_id="train")  # trainer clears the data keys post-consume

    leftover = list(((tq.kv_list() or {}).get("train") or {}).keys())
    assert leftover == [], leftover  # prompt key + session marker cleared by sample(), data by trainer


def test_session_counting_oldest_first():
    rb = _make_replay_buffer(strategy="none", rollout_level_dispatch=True)
    for i in range(3):
        _register_prompt(f"old{i}", 1, n=1)
        _register_session(f"old{i}", 0, "success", 1)
    for i in range(3):
        _register_prompt(f"new{i}", 10, n=1)
        _register_session(f"new{i}", 0, "success", 10)

    batch, _ = rb.sample(global_steps=10, partition_id="train", batch_size=4)
    selected = {k.split("_")[0] for k in batch.keys}
    assert len(selected) == 4, selected
    assert {"old0", "old1", "old2"}.issubset(selected), selected  # stale-first, kept (strategy=none)


# ======================================================================================
# Legacy (prompt-status) readiness — backward compatibility (rollout_level_dispatch=False)
# ======================================================================================
def test_legacy_status_bucket_readiness():
    rb = _make_replay_buffer(rollout_level_dispatch=False)
    # not ready: prompts only pending/running
    _register_prompt_legacy("p0", "pending", 1)
    _register_prompt_legacy("p1", "running", 1)
    rb._sync_metadata_from_transfer_queue()
    assert not rb._has_enough_samples(1, "train", batch_size=1)

    # flip to finished + write its trajectory -> sampleable
    _register_trajectory_legacy("p1", 1)
    _register_prompt_legacy("p1", "finished", 1)
    rb._sync_metadata_from_transfer_queue()
    assert rb._has_enough_samples(1, "train", batch_size=1)

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    assert {k.split("_")[0] for k in batch.keys} == {"p1"}, batch.keys


def test_legacy_ignores_session_markers():
    # Under the legacy model, session markers must not make a prompt sampleable; only the
    # prompt-status tag does. (Guards against the two models leaking into each other.)
    rb = _make_replay_buffer(rollout_level_dispatch=False)
    _register_prompt_legacy("p", "pending", 1)
    _register_session("p", 0, "success", 1)  # writes a marker + data, but status stays pending
    rb._sync_metadata_from_transfer_queue()
    assert not rb._has_enough_samples(1, "train", batch_size=1)


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
    import sys

    sys.exit(1 if _run_all() else 0)
