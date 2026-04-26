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

from verl.utils.reward_score import default_compute_score


def test_gsm8k_default_reward_scores_correct_answer():
    assert default_compute_score("openai/gsm8k", "reasoning #### 42", "42") == 1.0


def test_gsm8k_default_reward_scores_incorrect_answer():
    assert default_compute_score("openai/gsm8k", "reasoning #### 41", "42") == 0.0


@pytest.mark.parametrize("data_source", ["", "   ", None, "/some/path/gsm8k"])
def test_unsupported_reward_source_error_is_actionable(data_source):
    with pytest.raises(NotImplementedError) as exc_info:
        default_compute_score(data_source, "reasoning #### 42", "42")

    message = str(exc_info.value)
    assert f"data_source={data_source!r}" in message
    assert "openai/gsm8k" in message
    assert "examples/data_preprocess/gsm8k.py" in message
    assert "reward.custom_reward_function.path" in message
