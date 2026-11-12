# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import copy
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from verl.workers.rollout.llm_server import LLMServerManager


def test_manager_resolves_profiler_in_a_private_rollout_config():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"nnodes": 1, "profiler": {"tool": "nsys"}},
                "model": {},
            }
        }
    )
    original = copy.deepcopy(config)
    resolved = OmegaConf.create({"tool": "nsys", "ray_actor_nsight_options": {"o": "resolved"}})
    # The replica class is unrelated to configuration resolution; avoid backend imports.
    with (
        patch.object(LLMServerManager, "rollout_replica_class", Mock(), create=True),
        patch("verl.workers.rollout.llm_server.resolve_ray_actor_profiler_config", return_value=resolved) as resolve,
    ):
        manager = LLMServerManager(config)

    resolve.assert_called_once_with(original.actor_rollout_ref.rollout.profiler, config)
    assert manager.rollout_config.profiler == resolved
    # The resolver must receive a private copy of the nested profiler config.
    resolve.call_args.args[0].tool = "modified-by-test"
    assert config == original
