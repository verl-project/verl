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
"""Entry point for IFGym multi-turn PPO training.

Thin wrapper over ``verl.trainer.main_ppo`` that ensures the IFGym custom
advantage estimators (``ifgym_per_turn_grpo`` / ``ifgym_per_turn_rloo``) and the
``ifgym_agent`` agent loop are registered in every process that needs them.

The estimators register via ``@register_adv_est`` import side effects and the
advantage computation runs inside the ``TaskRunner`` Ray actor. Because Ray
imports this module in the actor to deserialize ``TaskRunner``, the module-level
imports below register the estimators there as well as in the launcher.
"""

import hydra
import ray

# Import side effects: register custom advantage estimators + agent loop.
import apertus.ifgym.ifgym_advantage  # noqa: F401
import apertus.ifgym.ifgym_agent_loop  # noqa: F401
import apertus.ifgym.ifgym_per_turn_rloo  # noqa: F401
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner as _BaseTaskRunner
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device


class TaskRunner(_BaseTaskRunner):
    """TaskRunner that re-imports the IFGym registrations inside the actor."""

    def run(self, config):
        import apertus.ifgym.ifgym_advantage  # noqa: F401
        import apertus.ifgym.ifgym_agent_loop  # noqa: F401
        import apertus.ifgym.ifgym_per_turn_rloo  # noqa: F401

        return super().run(config)


@hydra.main(config_path=".", config_name="ifgym_multiturn", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(TaskRunner))


if __name__ == "__main__":
    main()
