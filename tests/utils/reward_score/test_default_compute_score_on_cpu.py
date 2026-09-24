# Copyright 2025 Individual Contributor: LiFangBo2003
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

"""Tests that config kwargs (e.g. the GSM8K extraction method) reach the default compute_score function."""

from omegaconf import OmegaConf

import verl.experimental.reward_loop.reward_manager  # noqa: F401  # registers the built-in reward managers
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.reward_score import default_compute_score

GSM8K = "openai/gsm8k"


class TestGsm8kMethodForwarding:
    """`default_compute_score` should forward `method` to the GSM8K scorer."""

    def test_default_is_strict(self):
        """Without the `#### <answer>` marker, strict extraction yields zero reward."""
        assert default_compute_score(GSM8K, "The answer is 42", "42") == 0.0

    def test_strict_extraction(self):
        """A well-formatted strict response gets the full reward."""
        assert default_compute_score(GSM8K, "reasoning... #### 42", "42") == 1.0

    def test_flexible_extraction(self):
        """With method='flexible', the last number is extracted and no `####` marker is needed."""
        assert default_compute_score(GSM8K, "The answer is 42", "42", method="flexible") == 1.0

    def test_flexible_wrong_answer(self):
        """A wrong flexible answer only gets format_score (0 by default)."""
        assert default_compute_score(GSM8K, "The answer is 43", "42", method="flexible") == 0.0


class TestComputeScoreKwargsConfig:
    """`reward.compute_score_kwargs` in the config should reach the default compute_score function."""

    @staticmethod
    def _build_reward_manager(compute_score_kwargs=None):
        reward_config = {"reward_manager": {"source": "register", "name": "naive"}}
        if compute_score_kwargs is not None:
            reward_config["compute_score_kwargs"] = compute_score_kwargs
        config = OmegaConf.create({"reward": reward_config})
        return load_reward_manager(config, tokenizer=None)

    def test_kwargs_forwarded_to_default_compute_score(self):
        """method=flexible set through the config makes the scorer accept a bare trailing number."""
        reward_manager = self._build_reward_manager(compute_score_kwargs={"method": "flexible"})
        score = reward_manager.compute_score(data_source=GSM8K, solution_str="The answer is 42", ground_truth="42")
        assert float(score) == 1.0

    def test_default_behaviour_unchanged(self):
        """Without compute_score_kwargs, the strict behaviour is preserved."""
        reward_manager = self._build_reward_manager()
        score = reward_manager.compute_score(data_source=GSM8K, solution_str="The answer is 42", ground_truth="42")
        assert float(score) == 0.0
