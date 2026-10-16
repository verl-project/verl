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
"""``NCCLCheckpointEngine.init_process_group`` under a reused group name.

``ray.util.collective`` groups are process-global and keyed only by name, so a
second engine built in one process with the default ``group_name`` finds a group
it never created. The reuse guard compares ``self.rank``, which used to be set
only on the branch that creates the group -- so that second engine raised
``AttributeError: 'DeltaShardedCheckpointEngine' object has no attribute 'rank'``
instead of anything actionable. These tests pin the three paths apart: create,
re-init of the same instance, and a foreign group.

CPU-only: ``collective`` is stubbed, so no NCCL, GPU or Ray is needed.
"""

import sys
import types

import pytest


@pytest.fixture
def engine_cls(monkeypatch):
    """Import the engine with ray.util.collective and zmq stubbed out."""

    class FakeCollective:
        def __init__(self):
            self.groups = {}

        def is_group_initialized(self, name):
            return name in self.groups

        def init_collective_group(self, world_size, rank, backend, name):
            self.groups[name] = (world_size, rank, backend)

        def barrier(self, name):
            pass

    fake = FakeCollective()
    monkeypatch.setitem(sys.modules, "ray.util.collective", fake)
    # cupy is imported at module scope but only used to allocate the send/recv
    # buffers, which init_process_group never touches. Stubbing it keeps this
    # test runnable without CUDA. `ndarray` has to exist because it appears in an
    # annotation that is evaluated when the module's classes are defined.
    cupy_stub = types.ModuleType("cupy")
    cupy_stub.ndarray = type("ndarray", (), {})
    monkeypatch.setitem(sys.modules, "cupy", cupy_stub)

    from verl.checkpoint_engine.nccl_checkpoint_engine import NCCLCheckpointEngine

    monkeypatch.setattr("verl.checkpoint_engine.nccl_checkpoint_engine.collective", fake, raising=False)
    return NCCLCheckpointEngine


def _engine(engine_cls, **kwargs):
    # is_master=False keeps __init__ from starting a ZeroMQ publisher.
    return engine_cls(bucket_size=1 << 20, is_master=False, **kwargs)


def test_rank_is_defined_before_init_process_group(engine_cls):
    engine = _engine(engine_cls)
    assert engine.rank is None
    assert engine.world_size is None


def test_creating_engine_takes_the_group(engine_cls):
    engine = _engine(engine_cls)
    engine.init_process_group(rank=0, world_size=4, master_metadata=None, num_senders=1)
    assert (engine.rank, engine.world_size) == (0, 4)


def test_same_instance_may_reinit(engine_cls):
    """The production path: one engine, many update_weights cycles."""
    engine = _engine(engine_cls)
    engine.init_process_group(rank=0, world_size=4, master_metadata=None, num_senders=1)
    engine.init_process_group(rank=0, world_size=4, master_metadata=None, num_senders=1)
    assert (engine.rank, engine.world_size) == (0, 4)


def test_foreign_group_raises_actionable_error(engine_cls):
    """A second engine on the same group name must not raise AttributeError."""
    first = _engine(engine_cls)
    first.init_process_group(rank=0, world_size=4, master_metadata=None, num_senders=1)

    second = _engine(engine_cls)
    with pytest.raises(RuntimeError, match="already exists but was not created"):
        second.init_process_group(rank=0, world_size=4, master_metadata=None, num_senders=1)


def test_distinct_group_name_lets_a_second_engine_init(engine_cls):
    first = _engine(engine_cls)
    first.init_process_group(rank=0, world_size=4, master_metadata=None, num_senders=1)

    second = _engine(engine_cls, group_name="second")
    second.init_process_group(rank=0, world_size=4, master_metadata=None, num_senders=1)
    assert (second.rank, second.world_size) == (0, 4)


def test_excluded_rank_still_records_state(engine_cls):
    """Actor workers left out of the group get rank -1, not an exception."""
    engine = _engine(engine_cls)
    engine.init_process_group(rank=-1, world_size=4, master_metadata=None, num_senders=1)
    assert (engine.rank, engine.world_size) == (-1, 4)
