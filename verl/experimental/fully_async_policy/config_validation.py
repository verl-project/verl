# Copyright 2025 Meituan Ltd. and/or its affiliates
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


def validate_rollout_log_prob_config(config) -> None:
    """Require rollout log-probs only when they are used as the PPO anchor."""
    bypass_mode = config.algorithm.rollout_correction.bypass_mode
    if bypass_mode and not config.actor_rollout_ref.rollout.calculate_log_probs:
        raise ValueError(
            "[FullyAsyncRollouter] algorithm.rollout_correction.bypass_mode=True requires "
            "actor_rollout_ref.rollout.calculate_log_probs=True"
        )
