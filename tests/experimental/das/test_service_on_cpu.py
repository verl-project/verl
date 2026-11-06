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
"""SuffixDataService driven as a plain object (no Ray)."""

from verl.experimental.das import service as service_module
from verl.experimental.das.contracts import SHADOW_OFFSET, TrajectoryPush, deserialize_delta_batch
from verl.experimental.das.service import SuffixDataService
from verl.workers.config.das import CLS_LONG, CLS_SHORT, DASConfig


def push(problem_id, tokens):
    return TrajectoryPush(problem_id=problem_id, token_ids=tuple(tokens))


def drain(svc, replica="r1", since=-1):
    """Pull batches until caught up; returns (batches, final_version)."""
    batches = []
    while True:
        payload = svc.get_deltas(replica, since)
        if payload is None:
            return batches, since
        batch = deserialize_delta_batch(payload)
        batches.append(batch)
        since = batch.to_version


def make_service(**overrides):
    overrides.setdefault("window_iterations", 4)
    overrides.setdefault("fresh_iterations", 1)
    return SuffixDataService(DASConfig(**overrides))


def test_add_emits_base_and_shadow():
    svc = make_service(fresh_iterations=2)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.begin_iteration(2)  # per-iteration serving: data ships at the boundary
    batches, _ = drain(svc)
    adds = [a for b in batches for a in b.adds]
    assert len(adds) == 2
    base, shadow = adds
    assert base.token_ids == (1, 2, 3)
    assert shadow.seq_id == base.seq_id + SHADOW_OFFSET
    assert shadow.token_ids == base.token_ids


def test_current_iteration_withheld_until_boundary():
    svc = make_service()
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    # Mid-iteration: everything is withheld, replicas see nothing.
    assert svc.get_deltas("r1", -1) is None
    svc.begin_iteration(2)
    batches, _ = drain(svc)
    assert [a for b in batches for a in b.adds]


def test_no_shadow_when_fresh_disabled():
    svc = make_service(fresh_iterations=0)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.begin_iteration(2)
    batches, _ = drain(svc)
    assert len([a for b in batches for a in b.adds]) == 1


def test_empty_trajectories_ignored():
    svc = make_service()
    svc.add_trajectories("w", [push("p1", [])])
    assert svc.get_deltas("r1", -1) is None


def test_shadow_expires_after_fresh_window():
    svc = make_service(fresh_iterations=2, window_iterations=10)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.begin_iteration(2)
    _, version = drain(svc)

    svc.begin_iteration(3)  # cutoff = 3 - 2 = 1: iteration-1 shadows expire
    batches, _ = drain(svc, since=version)
    removals = [r for b in batches for r in b.removals]
    assert len(removals) == 1
    assert removals[0][1] >= SHADOW_OFFSET  # only the shadow died


def test_window_eviction_removes_base_seq():
    svc = make_service(fresh_iterations=1, window_iterations=2)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.begin_iteration(2)
    _, version = drain(svc)

    svc.begin_iteration(4)  # cutoff = 4 - 2 = 2 >= 1: iteration 1 evicted
    batches, _ = drain(svc, since=version)
    removals = {r[1] for b in batches for r in b.removals}
    base_ids = {r for r in removals if r < SHADOW_OFFSET}
    assert len(base_ids) == 1
    assert svc.get_metrics()["das_total_tokens"] == 0.0


def test_begin_iteration_idempotent():
    svc = make_service()
    svc.begin_iteration(3)
    svc.add_trajectories("w", [push("p1", [1, 2])])
    v1 = svc.get_metrics()["das_log_version"]
    info = svc.begin_iteration(3)
    assert svc.get_metrics()["das_log_version"] == v1
    assert info["iteration"] == 3


def test_incremental_delta_versioning():
    svc = make_service()
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.begin_iteration(2)
    batches, version = drain(svc)
    assert batches
    assert not batches[0].snapshot
    assert svc.get_deltas("r1", version) is None

    svc.add_trajectories("w", [push("p1", [4, 5, 6])])
    svc.begin_iteration(3)
    batches, _ = drain(svc, since=version)
    adds = [a for b in batches for a in b.adds]
    assert {tuple(a.token_ids) for a in adds} == {(4, 5, 6)}


def test_bounded_batches_drain_fully():
    svc = make_service(max_delta_batch_tokens=5, fresh_iterations=2)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1] * 4), push("p2", [2] * 4)])
    svc.begin_iteration(2)
    batches, _ = drain(svc)
    assert len(batches) > 1
    adds = [a for b in batches for a in b.adds]
    assert len(adds) == 4  # 2 base + 2 shadows across batches


def test_snapshot_when_log_truncated(monkeypatch):
    monkeypatch.setattr(service_module, "_LOG_CAPACITY", 4)
    svc = make_service(fresh_iterations=2)
    svc.begin_iteration(1)
    for i in range(6):
        svc.add_trajectories("w", [push("p1", [i, i, i])])
    svc.begin_iteration(2)
    payload = svc.get_deltas("r1", -1)
    batch = deserialize_delta_batch(payload)
    assert batch.snapshot
    assert batch.to_version == svc.get_metrics()["das_log_version"]
    # Snapshot carries the full live state: 6 base + 6 shadow sequences.
    assert len(batch.adds) == 12
    assert "p1" in batch.problem_cls


def test_snapshot_excludes_current_iteration(monkeypatch):
    monkeypatch.setattr(service_module, "_LOG_CAPACITY", 1)
    svc = make_service(fresh_iterations=0)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.begin_iteration(2)
    svc.add_trajectories("w", [push("p1", [4, 5, 6])])  # in-progress iteration

    payload = svc.get_deltas("r1", -1)
    batch = deserialize_delta_batch(payload)
    assert batch.snapshot
    # Only the servable (prior-iteration) sequence ships in the snapshot...
    assert {tuple(a.token_ids) for a in batch.adds} == {(1, 2, 3)}
    # ...and the withheld one arrives incrementally after the next boundary.
    svc.begin_iteration(3)
    batches, _ = drain(svc, since=batch.to_version)
    assert {tuple(a.token_ids) for b in batches for a in b.adds} == {(4, 5, 6)}


def test_max_seqs_per_problem_evicts_oldest():
    svc = make_service(fresh_iterations=0, max_total_tokens=10**9, max_problems=10, max_seqs_per_problem=2)
    svc.begin_iteration(1)
    for i in range(2):
        svc.add_trajectories("w", [push("p1", [i, i])])
    # Sync a replica first: eviction removals only ship to replicas that
    # already hold the sequence (fresh drains compact them away instead).
    svc.begin_iteration(2)
    _, version = drain(svc)
    svc.add_trajectories("w", [push("p1", [9, 9])])
    svc.begin_iteration(3)
    batches, _ = drain(svc, since=version)
    removals = [r for b in batches for r in b.removals]
    # Oldest base seq evicted (no shadows: fresh_iterations=0).
    assert len(removals) == 1
    # A fresh replica sees only the two live sequences, no removals.
    fresh_batches, _ = drain(svc, replica="fresh", since=-1)
    assert [r for b in fresh_batches for r in b.removals] == []
    assert len([a for b in fresh_batches for a in b.adds]) == 2  # 2 live base seqs


def test_max_problems_drops_lru():
    svc = make_service(max_total_tokens=10**9, max_problems=2, max_seqs_per_problem=8)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1]), push("p2", [2]), push("p3", [3])])
    svc.begin_iteration(2)
    batches, _ = drain(svc)
    dropped = [d for b in batches for d in b.dropped_problems]
    assert dropped == ["p1"]
    assert svc.get_metrics()["das_problem_count"] == 2.0


def test_token_cap_evicts_old_iterations():
    svc = make_service(max_total_tokens=10, max_problems=10, max_seqs_per_problem=64)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1] * 8)])
    svc.begin_iteration(2)
    svc.add_trajectories("w", [push("p1", [2] * 8)])
    assert svc.get_metrics()["das_total_tokens"] <= 10


def test_length_class_transitions_ship_to_replicas():
    svc = make_service(short_max_tokens=4, long_min_tokens=8)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [0] * 2)])  # Short so far
    for _ in range(20):
        svc.add_trajectories("w", [push("p1", [0] * 20)])  # EMA climbs past the Long threshold
    svc.begin_iteration(2)
    batches, _ = drain(svc)
    cls_updates = {}
    for b in batches:
        cls_updates.update(b.problem_cls)
    assert cls_updates["p1"] == CLS_LONG

    # A problem that stays short is reported Short in the snapshot/class map.
    svc.add_trajectories("w", [push("p2", [0] * 2)])
    svc.begin_iteration(3)
    batches, _ = drain(svc, replica="fresh", since=-1)
    cls_updates = {}
    for b in batches:
        cls_updates.update(b.problem_cls)
    assert cls_updates["p2"] == CLS_SHORT


def test_fresh_replica_replay_compacts_cancelled_history():
    """Regression: a fresh replica draining add+evict history in one batch
    must not resurrect evicted sequences (wire format loses log order, so
    the service must cancel add/removal pairs within the batch range)."""
    svc = make_service(fresh_iterations=2, window_iterations=2)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.begin_iteration(5)  # evicts iteration 1 entirely
    svc.add_trajectories("w", [push("p1", [7, 8, 9])])
    svc.begin_iteration(6)

    batches, _ = drain(svc, replica="fresh", since=-1)
    adds = [a for b in batches for a in b.adds]
    removals = [r for b in batches for r in b.removals]
    # Only the live sequence (base + shadow) ships; the evicted one and its
    # removals cancel out entirely.
    assert {tuple(a.token_ids) for a in adds} == {(7, 8, 9)}
    assert len(adds) == 2
    assert removals == []


def test_dropped_problem_purges_in_range_adds():
    svc = make_service(fresh_iterations=0, max_total_tokens=10**9, max_problems=1, max_seqs_per_problem=8)
    svc.begin_iteration(1)
    svc.add_trajectories("w", [push("p1", [1, 2, 3])])
    svc.add_trajectories("w", [push("p2", [4, 5, 6])])  # drops p1 (LRU cap 1)
    svc.begin_iteration(2)
    batches, _ = drain(svc, replica="fresh", since=-1)
    adds = [a for b in batches for a in b.adds]
    dropped = [d for b in batches for d in b.dropped_problems]
    assert {tuple(a.token_ids) for a in adds} == {(4, 5, 6)}
    assert dropped == ["p1"]
