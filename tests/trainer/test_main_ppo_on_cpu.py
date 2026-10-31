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

from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl.trainer import main_ppo


def test_run_ppo_uses_resolved_controller_nsight_options(monkeypatch):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"full_determinism": False}},
            "reward": {"reward_model": {"enable": False, "rollout": {}}},
            "trainer": {"logger": []},
            "global_profiler": {"tool": "nsys", "steps": [1]},
            "ray_kwargs": {},
        }
    )
    resolved_options = {"o": "resolved"}
    resolve_options = MagicMock(return_value=resolved_options)
    task_runner_class = MagicMock()
    monkeypatch.setattr(main_ppo, "build_controller_nsight_launch_options", resolve_options)
    monkeypatch.setattr(main_ppo, "is_cuda_available", True)
    monkeypatch.setattr(main_ppo.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(main_ppo.ray, "get", MagicMock())
    monkeypatch.setattr("verl.utils.import_utils.is_nvtx_available", lambda: True)

    main_ppo.run_ppo(config, task_runner_class)

    resolve_options.assert_called_once_with(config)
    task_runner_class.options.assert_called_once_with(runtime_env={"nsight": resolved_options})
