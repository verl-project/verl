# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


from omegaconf import DictConfig

from verl.trainer.ppo.core_algos import AdvantageEstimator


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic"""
    if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    elif config.algorithm.adv_estimator in [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_PASSK,
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        # AdvantageEstimator.REMAX, # TODO:REMAX advantage estimator is not yet supported in one_step_off_policy
        AdvantageEstimator.RLOO,
        AdvantageEstimator.OPO,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        AdvantageEstimator.GPG,
    ]:
        return False
    else:
        raise NotImplementedError
