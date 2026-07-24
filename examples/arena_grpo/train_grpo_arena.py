# Copyright 2025 Individual Contributor: albert-lv
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
"""Launcher for GRPO training with the OpenAgora (Arena) sandbox agent loop.

verl discovers agent loops by importing modules that decorate subclasses of
``AgentLoopBase`` with ``@register(...)``. Setting
``actor_rollout_ref.rollout.agent.default_agent_loop=arena_agent`` in the Hydra
config is not enough if the module that performs the registration has never
been imported in the trainer process.

This wrapper imports ``verl.experimental.agent_loop.arena_agent_loop`` (which
registers ``ArenaAgentLoop`` as ``arena_agent``) and then delegates to verl's
``main_ppo`` entry point. All command line arguments and environment variables
are forwarded unchanged. Ray workers that do not inherit this process's imports
register the loop via ``actor_rollout_ref.rollout.agent.agent_loop_config_path``
(see ``arena_agent_loop.yaml`` next to this file).
"""

import verl.experimental.agent_loop.arena_agent_loop  # noqa: F401
from verl.trainer.main_ppo import main

if __name__ == "__main__":
    main()
