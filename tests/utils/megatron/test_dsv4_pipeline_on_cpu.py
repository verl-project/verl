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

from verl.utils.megatron.dsv4_pipeline import dsv4_logical_layers_per_stage


def test_even_split_kept_when_hash_layers_fit_on_first_stage():
    # DeepSeek-V4-Flash: 43 layers over PP8 -> 6,6,6,5,5,5,5,5; the 3 hash layers fit on stage 0.
    assert dsv4_logical_layers_per_stage(43, 8, 3) is None
    assert dsv4_logical_layers_per_stage(6, 1, 3) is None


def test_first_stage_grows_to_hold_all_hash_layers():
    counts = dsv4_logical_layers_per_stage(16, 8, 3)
    assert counts[0] == 3 and sum(counts) == 16 and min(counts[1:]) >= 1
    assert counts == [3, 2, 2, 2, 2, 2, 2, 1]
    counts = dsv4_logical_layers_per_stage(6, 2, 4)
    assert counts == [4, 2]


def test_impossible_split_raises():
    with pytest.raises(ValueError, match="hash MoE layers"):
        dsv4_logical_layers_per_stage(8, 8, 3)
