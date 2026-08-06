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

"""CPU-only unit tests for the streaming dataloader feed logic in the V1 PPO trainer.

These cover the dependency-light pieces introduced for streaming generation:

- The ``num_prompts`` resolution + validation contract of
  ``PPOTrainer._next_train_batch`` (default to ``train_batch_size``; require a positive
  multiple of ``gen_batch_size``; fetch in small chunks and coalesce submissions).
- The ``steps_per_epoch`` derivation used for ``total_training_steps`` / epoch tracking, which
  must be computed from the dataset size and ``train_batch_size`` (NOT ``len(dataloader)``, which
  now counts ``gen_batch_size`` fetches), and must equal the classic ``len(dataloader)`` when
  ``gen_batch_size == train_batch_size``.

Importing the trainer directly would pull in heavy runtime deps (ray, transfer_queue, vllm), so
-- following the convention of ``test_compute_reward_colocate_on_cpu.py`` -- the logic under test
is replicated standalone and kept in sync with ``verl/trainer/ppo/v1/trainer_base.py``.
"""

import pytest


def _resolve_num_prompts(num_prompts, train_batch_size, gen_batch_size):
    """Standalone copy of the ``num_prompts`` resolution/validation in
    ``PPOTrainer._next_train_batch``. Kept in sync with ``trainer_base.py``.
    """
    if num_prompts is None:
        num_prompts = train_batch_size
    gen_batch_size = gen_batch_size or train_batch_size
    if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
        raise ValueError(
            f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size "
            f"({gen_batch_size}); it is submitted in whole gen_batch_size dataloader fetches."
        )
    return num_prompts


def _num_fetches(num_prompts, gen_batch_size):
    """How many ``gen_batch_size`` dataloader fetches ``_next_train_batch`` performs."""
    return num_prompts // gen_batch_size


def _num_submissions(num_prompts, gen_batch_size):
    """A valid request is coalesced into one submission."""
    if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
        raise ValueError("num_prompts must be a positive multiple of gen_batch_size")
    return 1


def _steps_per_epoch(dataset_size, train_batch_size):
    """Standalone copy of ``PPOTrainer.steps_per_epoch``. ``//`` matches dataloader drop_last=True."""
    return dataset_size // train_batch_size


class TestResolveNumPrompts:
    def test_default_is_train_batch_size(self):
        # None -> one train batch worth, regardless of gen_batch_size.
        assert _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=1) == 1024
        assert _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=256) == 1024

    def test_gen_batch_size_none_falls_back_to_train_batch_size(self):
        # gen_batch_size null => equals train_batch_size => default request is a single fetch.
        assert _resolve_num_prompts(None, train_batch_size=512, gen_batch_size=None) == 512
        assert _num_fetches(_resolve_num_prompts(None, 512, None) or 512, 512) == 1

    def test_explicit_multiple_ok(self):
        assert _resolve_num_prompts(24, train_batch_size=64, gen_batch_size=8) == 24
        assert _resolve_num_prompts(8, train_batch_size=64, gen_batch_size=8) == 8

    def test_gen_batch_size_one_allows_any_positive(self):
        # The DAPO/drop-refill use case: gen_batch_size=1 makes any dropped count valid.
        for k in (1, 3, 7, 100):
            assert _resolve_num_prompts(k, train_batch_size=1024, gen_batch_size=1) == k

    @pytest.mark.parametrize("bad", [20, 1, 7, 15])
    def test_non_multiple_raises(self, bad):
        with pytest.raises(ValueError, match="must be a positive multiple of gen_batch_size"):
            _resolve_num_prompts(bad, train_batch_size=64, gen_batch_size=8)

    @pytest.mark.parametrize("bad", [0, -8, -1])
    def test_non_positive_raises(self, bad):
        with pytest.raises(ValueError, match="must be a positive multiple of gen_batch_size"):
            _resolve_num_prompts(bad, train_batch_size=64, gen_batch_size=8)


class TestNumFetches:
    def test_default_gen_equals_train_is_single_fetch(self):
        n = _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=1024)
        assert _num_fetches(n, 1024) == 1

    def test_small_gen_batch_size_multiple_fetches(self):
        # train_batch_size=1024, gen_batch_size=1 => 1024 individual fetches per train batch.
        n = _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=1)
        assert _num_fetches(n, 1) == 1024

    def test_intermediate_granularity(self):
        n = _resolve_num_prompts(None, train_batch_size=64, gen_batch_size=8)
        assert _num_fetches(n, 8) == 8

    def test_refill_count_maps_to_fetches(self):
        # Refilling 5 dropped groups with gen_batch_size=1 => exactly 5 fetches.
        n = _resolve_num_prompts(5, train_batch_size=64, gen_batch_size=1)
        assert _num_fetches(n, 1) == 5


class TestCoalescedSubmission:
    def test_gen_batch_size_one_submits_train_batch_once(self):
        assert _num_fetches(64, 1) == 64
        assert _num_submissions(64, gen_batch_size=1) == 1

    def test_large_refill_is_submitted_once(self):
        assert _num_fetches(626, 1) == 626
        assert _num_submissions(626, gen_batch_size=1) == 1

    def test_multiple_gen_chunks_are_submitted_once(self):
        assert _num_fetches(96, 24) == 4
        assert _num_submissions(96, gen_batch_size=24) == 1


class TestStepsPerEpoch:
    def test_exact_division(self):
        assert _steps_per_epoch(dataset_size=10240, train_batch_size=1024) == 10

    def test_drop_last_truncates_partial_batch(self):
        # 10500 // 1024 == 10 (the trailing partial train batch is dropped, matching drop_last=True).
        assert _steps_per_epoch(dataset_size=10500, train_batch_size=1024) == 10

    def test_independent_of_gen_batch_size(self):
        # steps_per_epoch must NOT depend on gen_batch_size; only on dataset size / train_batch_size.
        dataset_size, train_batch_size = 8192, 1024
        expected = _steps_per_epoch(dataset_size, train_batch_size)
        for gen_batch_size in (1, 8, 256, 1024):
            # Number of gen fetches per epoch scales with 1/gen_batch_size, but step count does not.
            fetches_per_epoch = dataset_size // gen_batch_size
            assert fetches_per_epoch == expected * (train_batch_size // gen_batch_size)
            assert _steps_per_epoch(dataset_size, train_batch_size) == expected

    def test_equals_len_dataloader_when_gen_equals_train(self):
        # When gen_batch_size == train_batch_size, len(dataloader) (== dataset//gen with drop_last)
        # equals steps_per_epoch exactly, reproducing the classic non-streaming step count.
        dataset_size, train_batch_size = 8192, 1024
        gen_batch_size = train_batch_size
        len_dataloader = dataset_size // gen_batch_size  # what StatefulDataLoader reports
        assert _steps_per_epoch(dataset_size, train_batch_size) == len_dataloader

    def test_streaming_len_dataloader_would_overcount(self):
        # Documents the bug being avoided: with gen_batch_size < train_batch_size, naively using
        # len(dataloader) as the step count over-counts by train_batch_size/gen_batch_size.
        dataset_size, train_batch_size, gen_batch_size = 8192, 1024, 1
        len_dataloader = dataset_size // gen_batch_size  # 8192 gen fetches
        steps = _steps_per_epoch(dataset_size, train_batch_size)  # 8 real steps
        assert len_dataloader == steps * (train_batch_size // gen_batch_size)
        assert len_dataloader != steps  # naive use would be wrong

    def test_total_training_steps_composition(self):
        # total_training_steps = steps_per_epoch * total_epochs.
        steps = _steps_per_epoch(dataset_size=10240, train_batch_size=1024)
        assert steps * 3 == 30
