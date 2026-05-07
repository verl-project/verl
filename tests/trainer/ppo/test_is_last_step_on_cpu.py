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

import pytest

from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

# FullyAsyncTrainer is decorated with @ray.remote; reach through Ray's wrapper
# to test the underlying class without spinning up Ray.
_FullyAsyncCls = FullyAsyncTrainer.__ray_metadata__.modified_class


class _Stub:
    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


@pytest.mark.parametrize(
    ("global_steps", "total_training_steps", "expected"),
    [
        (0, 5, False),
        (3, 5, False),
        (4, 5, False),
        (5, 5, True),
        (6, 5, True),
        (0, 0, True),
    ],
)
@pytest.mark.parametrize("trainer_cls", [RayPPOTrainer, PPOTrainer])
def test_is_last_step_predicate(trainer_cls, global_steps, total_training_steps, expected):
    """is_last_step is True iff global_steps >= total_training_steps."""
    stub = _Stub(global_steps=global_steps, total_training_steps=total_training_steps)
    assert trainer_cls.is_last_step.fget(stub) is expected


@pytest.mark.parametrize(
    ("global_steps", "total_train_steps", "expected"),
    [
        (0, None, False),
        (5, None, False),
        (0, 5, False),
        (4, 5, False),
        (5, 5, True),
        (6, 5, True),
    ],
)
def test_fully_async_is_last_step(global_steps, total_train_steps, expected):
    """FullyAsyncTrainer overrides the property to use total_train_steps and tolerate None."""
    stub = _Stub(global_steps=global_steps, total_train_steps=total_train_steps)
    assert _FullyAsyncCls.is_last_step.fget(stub) is expected


@pytest.mark.parametrize("trainer_cls", [RayPPOTrainer, PPOTrainer, _FullyAsyncCls])
def test_is_last_step_is_read_only(trainer_cls):
    """Setter would let stale snapshots shadow the live predicate; keep it read-only."""
    assert trainer_cls.is_last_step.fset is None


def test_subclasses_inherit_property():
    assert SeparateRayPPOTrainer.is_last_step is RayPPOTrainer.is_last_step
    assert OneStepOffRayTrainer.is_last_step is RayPPOTrainer.is_last_step


def test_fully_async_overrides_property():
    assert _FullyAsyncCls.is_last_step is not RayPPOTrainer.is_last_step
