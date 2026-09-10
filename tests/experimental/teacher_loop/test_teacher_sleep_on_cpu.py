# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""CPU tests for multi-teacher shared GPU group (share_gpu_group) support.

Covers:
- ``DistillationConfig`` shared-mode validation (equal per-teacher world_size,
  world_size == pool size, free_cache_engine/enable_sleep_mode requirements,
  max_awake_teachers >= 1), and that the non-shared sum check is unchanged.
- ``TeacherSleepState``: acquire/release, LRU eviction, pinned teachers are never
  evicted, acquire failure when all awake teachers are pinned, and sleep_all.
"""

import pytest

from verl.experimental.teacher_loop.teacher_controller import TeacherSleepState
from verl.workers.config.distillation import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)
from verl.workers.config.rollout import RolloutConfig


def _teacher(key, num_replicas=1, tp=2, free_cache_engine=True, enable_sleep_mode=True):
    return DistillationTeacherModelConfig(
        key=key,
        model_path="Qwen/Qwen3-0.6B",
        num_replicas=num_replicas,
        inference=RolloutConfig(
            name="vllm",
            tensor_model_parallel_size=tp,
            free_cache_engine=free_cache_engine,
            enable_sleep_mode=enable_sleep_mode,
        ),
    )


def _two_teachers(**kwargs):
    """Two named teachers plus the default ``teacher_model`` entry.

    Mirrors the YAML layout: the default entry exists (resolution asserts on it) and is
    popped once other teacher entries are present, leaving the two named teachers.
    """
    return {
        "teacher_model": _teacher("default_placeholder"),
        "teacher_a": _teacher("teacher_a", **kwargs),
        "teacher_b": _teacher("teacher_b", **kwargs),
    }


def _distill_config(teacher_models, n_gpus_per_node, nnodes, share_gpu_group=True, max_awake_teachers=None):
    return DistillationConfig(
        enabled=True,
        n_gpus_per_node=n_gpus_per_node,
        nnodes=nnodes,
        teacher_models=teacher_models,
        distillation_loss=DistillationLossConfig(loss_mode="k1", use_policy_gradient=True),
        share_gpu_group=share_gpu_group,
        max_awake_teachers=max_awake_teachers,
    )


class TestSharedGpuGroupValidation:
    def test_shared_ok(self):
        # Two teachers, world_size 2 each; pool size 2. Valid in shared mode.
        config = _distill_config(_two_teachers(), n_gpus_per_node=2, nnodes=1)
        assert set(config.teacher_models) == {"teacher_a", "teacher_b"}

    def test_shared_single_teacher_ok(self):
        # Single teacher auto-fills num_replicas = pool / per_replica = 4 / 2 = 2,
        # so its world_size equals the pool size.
        config = _distill_config({"teacher_model": _teacher(None, num_replicas=0)}, n_gpus_per_node=4, nnodes=1)
        assert config.teacher_models["default"].world_size == 4

    def test_shared_unequal_world_size_raises(self):
        teacher_models = {
            "teacher_model": _teacher("default_placeholder"),
            "teacher_a": _teacher("teacher_a", num_replicas=1),
            "teacher_b": _teacher("teacher_b", num_replicas=2),
        }
        with pytest.raises(ValueError, match="world_size.*to be equal"):
            _distill_config(teacher_models, n_gpus_per_node=4, nnodes=1)

    def test_shared_pool_size_mismatch_raises(self):
        # Each teacher world_size is 2 but the pool is 4.
        with pytest.raises(ValueError, match="resource pool size"):
            _distill_config(_two_teachers(), n_gpus_per_node=4, nnodes=1)

    def test_shared_requires_free_cache_engine(self):
        with pytest.raises(ValueError, match="free_cache_engine"):
            _distill_config(_two_teachers(free_cache_engine=False), n_gpus_per_node=2, nnodes=1)

    def test_shared_requires_enable_sleep_mode(self):
        with pytest.raises(ValueError, match="enable_sleep_mode"):
            _distill_config(_two_teachers(enable_sleep_mode=False), n_gpus_per_node=2, nnodes=1)

    def test_shared_max_awake_teachers_must_be_positive(self):
        with pytest.raises(ValueError, match="max_awake_teachers"):
            _distill_config(_two_teachers(), n_gpus_per_node=2, nnodes=1, max_awake_teachers=0)

    def test_shared_max_awake_teachers_ok(self):
        config = _distill_config(_two_teachers(), n_gpus_per_node=2, nnodes=1, max_awake_teachers=1)
        assert config.max_awake_teachers == 1

    def test_non_shared_sum_check_unchanged(self):
        # Non-shared: sum of world sizes (2 + 2 = 4) must equal the pool size.
        config = _distill_config(_two_teachers(), n_gpus_per_node=4, nnodes=1, share_gpu_group=False)
        assert set(config.teacher_models) == {"teacher_a", "teacher_b"}
        with pytest.raises(ValueError, match="resource pool size"):
            _distill_config(_two_teachers(), n_gpus_per_node=2, nnodes=1, share_gpu_group=False)


class TestTeacherSleepState:
    def test_max_awake_must_be_positive(self):
        with pytest.raises(ValueError, match="max_awake"):
            TeacherSleepState(["a"], max_awake=0)

    def test_unknown_key_raises(self):
        state = TeacherSleepState(["a"], max_awake=1)
        with pytest.raises(KeyError, match="Unknown teacher key"):
            state.try_acquire("zzz")

    def test_first_acquire_needs_wake(self):
        state = TeacherSleepState(["a", "b"], max_awake=2)
        assert state.try_acquire("a") == (True, [], True)
        assert state.awake == ["a"]

    def test_acquire_awake_refreshes_lru_without_wake(self):
        state = TeacherSleepState(["a", "b"], max_awake=2)
        state.try_acquire("a")
        state.try_acquire("b")
        state.release("a")
        state.release("b")
        # Re-acquiring "a" (already awake) needs no wake and moves it to most-recent.
        assert state.try_acquire("a") == (True, [], False)
        assert state.awake == ["b", "a"]

    def test_lru_eviction(self):
        state = TeacherSleepState(["a", "b", "c"], max_awake=2)
        state.try_acquire("a")
        state.release("a")
        state.try_acquire("b")
        state.release("b")
        # "a" is least-recently-used, so it is evicted to make room for "c".
        assert state.try_acquire("c") == (True, ["a"], True)
        assert state.awake == ["b", "c"]

    def test_pinned_teacher_not_evicted(self):
        state = TeacherSleepState(["a", "b", "c"], max_awake=2)
        state.try_acquire("a")  # stays pinned
        state.try_acquire("b")
        state.release("b")
        # "a" is LRU but pinned, so unpinned "b" is evicted instead.
        assert state.try_acquire("c") == (True, ["b"], True)
        assert state.awake == ["a", "c"]

    def test_acquire_fails_when_all_awake_pinned(self):
        state = TeacherSleepState(["a", "b"], max_awake=1)
        state.try_acquire("a")  # pinned, occupies the only awake slot
        assert state.try_acquire("b") == (False, [], False)
        # Nothing changed: "a" is still awake and pinned.
        assert state.awake == ["a"]
        state.release("a")
        assert state.try_acquire("b") == (True, ["a"], True)

    def test_release_without_acquire_raises(self):
        state = TeacherSleepState(["a"], max_awake=1)
        with pytest.raises(ValueError, match="without a matching acquire"):
            state.release("a")
        state.try_acquire("a")
        state.release("a")
        with pytest.raises(ValueError, match="without a matching acquire"):
            state.release("a")

    def test_keys_to_sleep_all(self):
        state = TeacherSleepState(["a", "b", "c"], max_awake=3)
        state.try_acquire("a")
        state.try_acquire("b")
        assert state.keys_to_sleep_all() == ["a", "b"]
        assert state.awake == []
        assert state.keys_to_sleep_all() == []
        # Teachers can be re-acquired (and woken) after sleep_all.
        assert state.try_acquire("c") == (True, [], True)
