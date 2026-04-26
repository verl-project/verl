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

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    from verl.trainer.ppo.dynamic_filter import (
        FILTER_TYPE_BAND_PASS,
        FILTER_TYPE_SAME_VALUE,
        DynamicFilterState,
        apply_dynamic_filter,
        get_reward_extra_infos_from_batch,
        profile_state_after_dynamic_filter_skip,
        resolve_filter_metric,
        validate_filter_groups_config,
    )
except ModuleNotFoundError:
    _DYNAMIC_FILTER_PATH = Path(__file__).resolve().parents[3] / "verl/trainer/ppo/dynamic_filter.py"
    _SPEC = importlib.util.spec_from_file_location("dynamic_filter_for_test", _DYNAMIC_FILTER_PATH)
    assert _SPEC is not None and _SPEC.loader is not None
    dynamic_filter = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = dynamic_filter
    _SPEC.loader.exec_module(dynamic_filter)

    FILTER_TYPE_BAND_PASS = dynamic_filter.FILTER_TYPE_BAND_PASS
    FILTER_TYPE_SAME_VALUE = dynamic_filter.FILTER_TYPE_SAME_VALUE
    DynamicFilterState = dynamic_filter.DynamicFilterState
    apply_dynamic_filter = dynamic_filter.apply_dynamic_filter
    get_reward_extra_infos_from_batch = dynamic_filter.get_reward_extra_infos_from_batch
    profile_state_after_dynamic_filter_skip = dynamic_filter.profile_state_after_dynamic_filter_skip
    resolve_filter_metric = dynamic_filter.resolve_filter_metric
    validate_filter_groups_config = dynamic_filter.validate_filter_groups_config


class _FakeDataProto:
    def __init__(self, *, tensors: dict[str, torch.Tensor], non_tensors: dict[str, np.ndarray], meta_info=None):
        self.batch = tensors
        self.non_tensor_batch = non_tensors
        self.meta_info = meta_info or {}

    def __len__(self):
        if self.batch:
            return len(next(iter(self.batch.values())))
        if self.non_tensor_batch:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0

    def __getitem__(self, item):
        tensor_item = torch.from_numpy(item) if isinstance(item, np.ndarray) else item
        selected_tensors = {key: value[tensor_item] for key, value in self.batch.items()}
        selected_non_tensors = {key: value[item] for key, value in self.non_tensor_batch.items()}
        return type(self)(tensors=selected_tensors, non_tensors=selected_non_tensors, meta_info=self.meta_info)

    @staticmethod
    def concat(data: list[_FakeDataProto]) -> _FakeDataProto:
        first = data[0]
        tensors = {key: torch.cat([item.batch[key] for item in data], dim=0) for key in first.batch}
        non_tensors = {
            key: np.concatenate([item.non_tensor_batch[key] for item in data], axis=0) for key in first.non_tensor_batch
        }
        return _FakeDataProto(tensors=tensors, non_tensors=non_tensors, meta_info=first.meta_info)


def _config(
    *,
    train_batch_size: int = 1,
    rollout_n: int = 2,
    metric: str = "score",
    filter_type: str = FILTER_TYPE_SAME_VALUE,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    max_num_gen_batches: int = 1,
):
    return SimpleNamespace(
        data=SimpleNamespace(train_batch_size=train_batch_size),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=rollout_n)),
        algorithm=SimpleNamespace(
            adv_estimator="grpo",
            filter_groups=SimpleNamespace(
                enable=True,
                metric=metric,
                max_num_gen_batches=max_num_gen_batches,
                filter_type=filter_type,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
        ),
    )


def _batch(
    *,
    uids: list[str],
    score: list[float] | None = None,
    token_level_scores: torch.Tensor | None = None,
    extra: dict[str, list] | None = None,
    reward_extra_keys: list[str] | None = None,
):
    batch_size = len(uids)
    if token_level_scores is None:
        token_level_scores = torch.zeros(batch_size, 2)
    non_tensors = {"uid": np.asarray(uids, dtype=object)}
    if score is not None:
        non_tensors["score"] = np.asarray(score, dtype=np.float32)
    if extra is not None:
        for key, value in extra.items():
            non_tensors[key] = np.asarray(value, dtype=object)

    return _FakeDataProto(
        tensors={
            "responses": torch.ones(batch_size, 2, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
            "token_level_scores": token_level_scores,
        },
        non_tensors=non_tensors,
        meta_info={"reward_extra_keys": reward_extra_keys or []},
    )


def test_same_value_filter_keeps_only_non_constant_prompt_groups():
    batch = _batch(uids=["easy", "easy", "mixed", "mixed"], score=[1.0, 1.0, 0.0, 1.0])
    result = apply_dynamic_filter(batch, _config(metric="score"), DynamicFilterState())

    assert not result.should_generate_more
    assert result.batch is not None
    assert result.batch.non_tensor_batch["uid"].tolist() == ["mixed", "mixed"]
    assert result.metrics["dynamic_filter/accepted_prompt_groups"] == 1.0
    assert result.metrics["dynamic_filter/rejected_prompt_groups"] == 1.0
    assert result.metrics["dynamic_filter/same_value_reject_ratio"] == 0.5


def test_band_pass_filter_keeps_groups_with_mean_metric_inside_bounds():
    batch = _batch(
        uids=[
            "low",
            "low",
            "lower_edge",
            "lower_edge",
            "medium",
            "medium",
            "upper_edge",
            "upper_edge",
            "high",
            "high",
        ],
        score=[0.1, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 0.8, 0.9, 1.0],
    )
    config = _config(
        train_batch_size=3,
        metric="score",
        filter_type=FILTER_TYPE_BAND_PASS,
        lower_bound=0.2,
        upper_bound=0.8,
    )
    result = apply_dynamic_filter(batch, config, DynamicFilterState())

    assert not result.should_generate_more
    assert result.batch is not None
    assert result.batch.non_tensor_batch["uid"].tolist() == [
        "lower_edge",
        "lower_edge",
        "medium",
        "medium",
        "upper_edge",
        "upper_edge",
    ]
    assert result.metrics["dynamic_filter/accepted_prompt_groups"] == 3.0
    assert result.metrics["dynamic_filter/rejected_prompt_groups"] == 2.0
    assert result.metrics["dynamic_filter/too_low_ratio"] == pytest.approx(1 / 5)
    assert result.metrics["dynamic_filter/too_high_ratio"] == pytest.approx(1 / 5)


def test_seq_reward_metric_is_derived_from_token_level_scores():
    token_scores = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.0, 0.5]], dtype=torch.float32)
    batch = _batch(uids=["a", "a", "b"], token_level_scores=token_scores)

    np.testing.assert_allclose(resolve_filter_metric(batch, "seq_reward"), np.asarray([0.3, 0.7, 0.5]))


def test_missing_metric_error_lists_available_keys():
    batch = _batch(uids=["a", "a"], score=[0.0, 1.0])

    expected_message = "Available batch keys.*token_level_scores.*Available non_tensor_batch keys.*score"
    with pytest.raises(ValueError, match=expected_message):
        resolve_filter_metric(batch, "unknown_metric")


def test_filter_accumulates_generation_batches_until_train_batch_size_is_met():
    config = _config(train_batch_size=2, max_num_gen_batches=2)
    state = DynamicFilterState()
    first = _batch(uids=["easy", "easy", "mixed1", "mixed1"], score=[1.0, 1.0, 0.0, 1.0])
    second = _batch(uids=["hard", "hard", "mixed2", "mixed2"], score=[0.0, 0.0, 0.0, 1.0])

    first_result = apply_dynamic_filter(first, config, state)
    assert first_result.should_generate_more
    assert first_result.batch is None

    second_result = apply_dynamic_filter(second, config, state)
    assert not second_result.should_generate_more
    assert second_result.batch is not None
    assert second_result.batch.non_tensor_batch["uid"].tolist() == ["mixed1", "mixed1", "mixed2", "mixed2"]
    assert second_result.metrics["dynamic_filter/gen_batches"] == 2.0


def test_filter_raises_when_max_generation_batches_cannot_fill_train_batch():
    config = _config(train_batch_size=1, max_num_gen_batches=1)
    batch = _batch(uids=["easy", "easy"], score=[1.0, 1.0])

    with pytest.raises(ValueError, match="accepted 0 prompt groups.*requires 1"):
        apply_dynamic_filter(batch, config, DynamicFilterState())


def test_reward_extra_info_stays_aligned_after_filtering():
    batch = _batch(
        uids=["low", "low", "medium", "medium"],
        score=[0.0, 0.1, 0.4, 0.6],
        extra={"acc": [0.0, 0.1, 0.4, 0.6], "label": ["a0", "a1", "b0", "b1"]},
        reward_extra_keys=["acc", "label"],
    )
    config = _config(
        metric="score",
        filter_type=FILTER_TYPE_BAND_PASS,
        lower_bound=0.2,
        upper_bound=0.8,
    )

    result = apply_dynamic_filter(batch, config, DynamicFilterState())
    assert result.batch is not None
    reward_extra_infos = get_reward_extra_infos_from_batch(result.batch)

    assert reward_extra_infos["acc"].tolist() == [0.4, 0.6]
    assert reward_extra_infos["label"].tolist() == ["b0", "b1"]


def test_filter_groups_config_validation_rejects_invalid_band_pass_settings():
    config = _config(filter_type=FILTER_TYPE_BAND_PASS, lower_bound=None, upper_bound=None)

    with pytest.raises(ValueError, match="requires at least one"):
        validate_filter_groups_config(config.algorithm.filter_groups, "grpo")

    config = _config(filter_type=FILTER_TYPE_BAND_PASS, lower_bound=0.9, upper_bound=0.1)
    with pytest.raises(ValueError, match="less than or equal"):
        validate_filter_groups_config(config.algorithm.filter_groups, "grpo")


def test_filter_groups_config_validation_rejects_remax():
    config = _config()

    with pytest.raises(ValueError, match="not supported with REMAX"):
        validate_filter_groups_config(config.algorithm.filter_groups, "remax")


def test_profile_state_after_dynamic_filter_skip_preserves_active_continuous_profile():
    prev_step_profile, curr_step_profile = profile_state_after_dynamic_filter_skip(
        completed_step_profile=True,
        next_step_profile=True,
        retry_step_profile=True,
        profile_continuous_steps=True,
    )

    assert prev_step_profile is True
    assert curr_step_profile is True


def test_profile_state_after_dynamic_filter_skip_restarts_when_profile_was_stopped():
    prev_step_profile, curr_step_profile = profile_state_after_dynamic_filter_skip(
        completed_step_profile=True,
        next_step_profile=False,
        retry_step_profile=True,
        profile_continuous_steps=True,
    )

    assert prev_step_profile is False
    assert curr_step_profile is True
