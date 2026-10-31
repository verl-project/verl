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
import pickle

import pytest

from verl.experimental.das.contracts import (
    SEQ_ID_LIMIT,
    SHADOW_OFFSET,
    TRANSIENT_BASE,
    DeltaBatch,
    TrajectoryDelta,
    deserialize_delta_batch,
    serialize_delta_batch,
)


def test_round_trip():
    batch = DeltaBatch(
        to_version=9,
        iteration=4,
        snapshot=False,
        adds=[TrajectoryDelta("p1", 7, (1, 2, 3))],
        removals=[("p2", 5)],
        dropped_problems=["p3"],
        problem_cls={"p1": 2},
    )
    out = deserialize_delta_batch(serialize_delta_batch(batch))
    assert out.to_version == 9
    assert out.iteration == 4
    assert out.adds == [TrajectoryDelta("p1", 7, (1, 2, 3))]
    assert out.removals == [("p2", 5)]
    assert out.dropped_problems == ["p3"]
    assert out.problem_cls == {"p1": 2}


def test_schema_version_mismatch_raises():
    payload = serialize_delta_batch(DeltaBatch(to_version=1, iteration=0))
    bad = bytes([payload[0] + 1]) + payload[1:]
    with pytest.raises(ValueError, match="schema mismatch"):
        deserialize_delta_batch(bad)


def test_empty_payload_raises():
    with pytest.raises(ValueError, match="empty"):
        deserialize_delta_batch(b"")


def test_non_batch_payload_raises():
    payload = bytes([1]) + pickle.dumps({"not": "a batch"})
    with pytest.raises(TypeError):
        deserialize_delta_batch(payload)


def test_seq_id_namespaces_are_disjoint():
    # Canonical [0, SHADOW_OFFSET), shadows [SHADOW_OFFSET, 2*SHADOW_OFFSET),
    # transients [TRANSIENT_BASE, SEQ_ID_LIMIT).
    assert 2 * SHADOW_OFFSET <= TRANSIENT_BASE
    assert TRANSIENT_BASE < SEQ_ID_LIMIT
