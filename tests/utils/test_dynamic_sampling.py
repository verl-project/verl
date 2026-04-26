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

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.utils.filtering import DynamicSamplingAccumulator


def _batch(uids, reward_values, extra_infos=None, meta_info=None, token_level_rewards=None):
    reward_values = torch.tensor(reward_values, dtype=torch.float32)
    tensors = {
        "responses": torch.zeros((len(uids), reward_values.shape[-1]), dtype=torch.long),
        "attention_mask": torch.ones((len(uids), reward_values.shape[-1] + 1), dtype=torch.long),
        "rm_scores": reward_values,
    }
    if token_level_rewards is not None:
        tensors["token_level_rewards"] = torch.tensor(token_level_rewards, dtype=torch.float32)
    non_tensors = {"uid": np.array(uids, dtype=object)}
    if extra_infos:
        non_tensors.update({key: np.array(value, dtype=object) for key, value in extra_infos.items()})
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info or {})


def test_filters_uniform_prompt_groups_and_keeps_aligned_rewards():
    batch = _batch(
        ["a", "a", "b", "b", "c", "c"],
        [[0, 1], [0, 2], [0, 5], [0, 5], [0, 0], [0, 0]],
        extra_infos={"acc": [0, 1, 1, 1, 0, 0]},
        meta_info={"timing": {"rollout": 1.0}, "reward_extra_keys": ["acc"]},
    )
    sampler = DynamicSamplingAccumulator({"metric": "acc", "max_num_gen_batches": 1}, 1, 2)

    result = sampler.add_candidate(batch, batch.batch["rm_scores"], {"acc": batch.non_tensor_batch["acc"]})
    final_batch, reward_tensor, reward_extra_infos, metrics = sampler.finalize()

    assert result.ready
    assert final_batch.non_tensor_batch["uid"].tolist() == ["a", "a"]
    assert reward_tensor.tolist() == [[0, 1], [0, 2]]
    assert reward_extra_infos["acc"].tolist() == [0, 1]
    assert final_batch.meta_info == {"reward_extra_keys": ["acc"]}
    assert metrics["dynamic_sampling/kept_prompts"] == 1.0
    assert metrics["dynamic_sampling/dropped_prompts"] == 2.0


def test_accumulates_multiple_candidate_batches_and_truncates_to_train_size():
    first = _batch(
        ["a", "a", "b", "b"],
        [[0, 1], [0, 2], [0, 5], [0, 5]],
        extra_infos={"acc": [0, 1, 1, 1]},
        meta_info={"timing": {"rollout": 1.0}, "reward_extra_keys": ["acc"]},
    )
    second = _batch(
        ["c", "c", "d", "d"],
        [[0, 0], [0, 0], [0, 3], [0, 4]],
        extra_infos={"acc": [0, 0, 0, 1]},
        meta_info={"timing": {"rollout": 2.0}, "reward_extra_keys": ["acc"]},
    )
    sampler = DynamicSamplingAccumulator({"metric": "acc", "max_num_gen_batches": 3}, 2, 2)

    assert not sampler.add_candidate(first, first.batch["rm_scores"], {"acc": first.non_tensor_batch["acc"]}).ready
    assert sampler.should_continue()
    assert sampler.add_candidate(second, second.batch["rm_scores"], {"acc": second.non_tensor_batch["acc"]}).ready
    final_batch, reward_tensor, reward_extra_infos, metrics = sampler.finalize()

    assert final_batch.non_tensor_batch["uid"].tolist() == ["a", "a", "d", "d"]
    assert reward_tensor.sum(dim=-1).tolist() == [1, 2, 3, 4]
    assert reward_extra_infos["acc"].tolist() == [0, 1, 0, 1]
    assert metrics["dynamic_sampling/num_gen_batches"] == 2.0
    assert metrics["dynamic_sampling/num_prompt_in_batch"] == 2.0


def test_seq_reward_uses_reward_tensor_when_token_level_scores_are_absent():
    batch = _batch(
        ["a", "a", "b", "b"],
        [[0, 1], [0, 1], [0, 2], [0, 3]],
        meta_info={"timing": {"rollout": 1.0}},
    )
    sampler = DynamicSamplingAccumulator({"metric": "seq_reward", "max_num_gen_batches": 1}, 1, 2)

    sampler.add_candidate(batch, batch.batch["rm_scores"], {})
    final_batch, reward_tensor, reward_extra_infos, _ = sampler.finalize()

    assert final_batch.non_tensor_batch["uid"].tolist() == ["b", "b"]
    assert reward_tensor.sum(dim=-1).tolist() == [2, 3]
    assert reward_extra_infos == {}


def test_seq_final_reward_requires_token_level_rewards():
    batch = _batch(["a", "a"], [[0, 1], [0, 2]])
    sampler = DynamicSamplingAccumulator({"metric": "seq_final_reward", "max_num_gen_batches": 1}, 1, 2)

    with pytest.raises(ValueError, match="seq_final_reward"):
        sampler.add_candidate(batch, batch.batch["rm_scores"], {})


def test_max_num_gen_batches_stops_underfilled_sampling():
    batch = _batch(["a", "a", "b", "b"], [[0, 1], [0, 1], [0, 2], [0, 2]], extra_infos={"acc": [1, 1, 0, 0]})
    sampler = DynamicSamplingAccumulator({"metric": "acc", "max_num_gen_batches": 1}, 1, 2)

    result = sampler.add_candidate(batch, batch.batch["rm_scores"], {"acc": batch.non_tensor_batch["acc"]})

    assert not result.ready
    assert not sampler.should_continue()
    with pytest.raises(ValueError, match="num_prompt_in_batch=0"):
        sampler.finalize()


def test_prompt_groups_must_match_rollout_n():
    batch = _batch(["a", "a", "b"], [[0, 1], [0, 2], [0, 3]], extra_infos={"acc": [0, 1, 1]})
    sampler = DynamicSamplingAccumulator({"metric": "acc", "max_num_gen_batches": 1}, 1, 2)

    with pytest.raises(ValueError, match="rollout_n=2"):
        sampler.add_candidate(batch, batch.batch["rm_scores"], {"acc": batch.non_tensor_batch["acc"]})


def test_missing_metric_raises_clear_error():
    batch = _batch(["a", "a"], [[0, 1], [0, 2]])
    sampler = DynamicSamplingAccumulator({"metric": "acc", "max_num_gen_batches": 1}, 1, 2)

    with pytest.raises(ValueError, match="Could not find filter metric 'acc'"):
        sampler.add_candidate(batch, batch.batch["rm_scores"], {})


def test_reward_extra_keys_must_be_stable_across_candidates():
    first = _batch(["a", "a"], [[0, 1], [0, 2]], extra_infos={"acc": [0, 1]})
    second = _batch(["b", "b"], [[0, 3], [0, 4]], extra_infos={"acc": [0, 1], "score": [0, 1]})
    sampler = DynamicSamplingAccumulator({"metric": "acc", "max_num_gen_batches": 2}, 2, 2)

    sampler.add_candidate(first, first.batch["rm_scores"], {"acc": first.non_tensor_batch["acc"]})
    with pytest.raises(ValueError, match="Reward extra info keys must be stable"):
        sampler.add_candidate(
            second,
            second.batch["rm_scores"],
            {"acc": second.non_tensor_batch["acc"], "score": second.non_tensor_batch["score"]},
        )


def test_reward_extra_infos_must_match_batch_length():
    batch = _batch(["a", "a"], [[0, 1], [0, 2]], extra_infos={"acc": [0, 1]})
    sampler = DynamicSamplingAccumulator({"metric": "acc", "max_num_gen_batches": 1}, 1, 2)

    with pytest.raises(ValueError, match="length 1 must match batch length 2"):
        sampler.add_candidate(batch, batch.batch["rm_scores"], {"score": np.array([0])})
