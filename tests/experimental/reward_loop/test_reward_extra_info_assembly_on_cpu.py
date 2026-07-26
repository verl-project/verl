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
"""Regression tests for batch assembly of per-sample ``reward_extra_info`` dicts.

``RewardLoopManager.compute_rm_score`` used to infer the extra-info schema from
sample 0 only (``list(reward_extra_infos[0].keys())``), which either crashed with
``KeyError`` or silently dropped columns whenever samples emitted different keys —
e.g. a reward function that adds a diagnostic key only on parse success, or a
mixed-dataset batch where ``default_compute_score`` returns a dict for one data
source (``math_dapo``) and a bare float for another (``openai/gsm8k``).
"""

import numpy as np

from verl.experimental.reward_loop.reward_loop import assemble_reward_extra_info


def test_uniform_keys_preserve_alignment_and_dtype():
    infos = [
        {"acc": 1.0, "pred": "42"},
        {"acc": 0.0, "pred": "7"},
        {"acc": 1.0, "pred": "42"},
    ]
    out = assemble_reward_extra_info(infos)

    assert set(out.keys()) == {"acc", "pred"}
    assert out["acc"].tolist() == [1.0, 0.0, 1.0]
    assert out["acc"].dtype == np.float64  # dense keys keep the natural numpy dtype
    assert out["pred"].tolist() == ["42", "7", "42"]
    assert all(len(v) == len(infos) for v in out.values())


def test_key_missing_from_sample_zero_is_not_dropped():
    """The old schema-from-sample-0 logic silently discarded this column."""
    infos = [
        {"acc": 1.0},
        {"acc": 0.0, "cp": -30.0},
        {"acc": 1.0, "cp": 120.0},
    ]
    out = assemble_reward_extra_info(infos)

    assert "cp" in out
    assert out["cp"].tolist() == [None, -30.0, 120.0]
    assert out["cp"].dtype == object  # None fill requires object dtype


def test_key_missing_from_later_sample_does_not_raise():
    """The old logic raised KeyError: info[key] on the sample that omitted the key."""
    infos = [
        {"score": 1.0, "acc": True, "pred": "42"},  # e.g. math_dapo sample
        {"acc": 0.0},  # e.g. gsm8k sample (float score -> only "acc")
    ]
    out = assemble_reward_extra_info(infos)

    assert set(out.keys()) == {"score", "acc", "pred"}
    assert out["score"].tolist() == [1.0, None]
    assert out["pred"].tolist() == ["42", None]
    assert out["acc"].tolist() == [True, 0.0]
    assert all(len(v) == len(infos) for v in out.values())


def test_key_order_is_first_seen():
    infos = [{"b": 1}, {"a": 2, "b": 3}]
    assert list(assemble_reward_extra_info(infos).keys()) == ["b", "a"]


def test_all_empty_infos():
    assert assemble_reward_extra_info([{}, {}, {}]) == {}


def test_explicit_none_value_is_kept_dense():
    """A key uniformly present with a None value is not the same as a missing key."""
    infos = [{"cp": None}, {"cp": 1.0}]
    out = assemble_reward_extra_info(infos)
    assert out["cp"].tolist() == [None, 1.0]
