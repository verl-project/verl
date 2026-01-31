# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import unittest

import numpy as np
import pytest
from omegaconf import OmegaConf


# @pytest.mark.parametrize("simulator_type", ["libero", "isaac"])
@pytest.mark.parametrize("simulator_type", ["isaac"])
def test_sim_env_creation_and_step(simulator_type):
    num_envs = 8
    actions = np.array(
        [
            [5.59112417e-01, 8.06460073e-02, 1.36817226e-02, -4.64279854e-04, -1.72158767e-02, -6.57548380e-04, -1],
            [2.12711899e-03, -3.13366604e-01, 3.41386353e-04, -4.64279854e-04, -8.76528812e-03, -6.57548380e-04, -1],
            [7.38182960e-02, -4.64548351e-02, -6.63602950e-02, -4.64279854e-04, -2.32520114e-02, -6.57548380e-04, -1],
            [7.38182960e-02, -1.60845593e-01, 3.41386353e-04, -4.64279854e-04, 1.05503430e-02, -6.57548380e-04, -1],
            [7.38182960e-02, -3.95982152e-01, -7.97006313e-02, -5.10713711e-03, 3.22804279e-02, -6.57548380e-04, -1],
            [2.41859427e-02, -3.64206941e-01, -6.63602950e-02, -4.64279854e-04, 1.05503430e-02, -6.57548380e-04, -1],
            [4.62447664e-02, -5.16727952e-01, -7.97006313e-02, -4.64279854e-04, 1.05503430e-02, 8.73740975e-03, -1],
            [4.62447664e-02, -5.73923331e-01, 3.41386353e-04, -4.64279854e-04, 6.92866212e-03, -6.57548380e-04, -1],
        ]
    )
    cfg = OmegaConf.create(
        {
            "max_episode_steps": 512,
            "only_eval": False,
            "reward_coef": 1.0,
            "init_params": {
                "camera_names": ["agentview"],
            },
            "video_cfg": {
                "save_video": True,
                "video_base_dir": "/tmp/test_sim_env_creation_and_step",
            },
            "task_suite_name": "libero_10",
            "num_envs": num_envs,
            "num_group": 1,
            "group_size": num_envs,
            "seed": 0,
        },
    )

    sim_env = None
    if simulator_type == "isaac":
        from verl.experimental.vla.envs.isaac_env.isaac_env import IsaacEnv

        sim_env = IsaacEnv(cfg, rank=0, world_size=1)
    elif simulator_type == "libero":
        from verl.experimental.vla.envs.libero_env.libero_env import LiberoEnv

        sim_env = LiberoEnv(cfg, rank=0, world_size=1)
    else:
        raise ValueError(f"simulator_type {simulator_type} is not supported")

    video_count = 0
    for i in [0]:
        # The first call to step with actions=None will reset the environment
        step = 0
        sim_env.reset_envs_to_state_ids([0] * num_envs, [i] * num_envs)
        for action in actions:
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = sim_env.step(
                np.array([action] * num_envs)
            )

            assert isinstance(obs_venv, dict)
            assert reward_venv.shape == (num_envs,)
            assert terminated_venv.shape == (num_envs,)
            assert truncated_venv.shape == (num_envs,)
            assert isinstance(info_venv, dict)

            if terminated_venv.any() or truncated_venv.any():
                break
            step += 1

        sim_env.flush_video(video_sub_dir=f"task_{i}")
        assert os.path.exists(os.path.join(cfg.video_cfg.video_base_dir, f"rank_0/task_{i}/{video_count}.mp4"))
        os.remove(os.path.join(cfg.video_cfg.video_base_dir, f"rank_0/task_{i}/{video_count}.mp4"))
        video_count += 1

    print("test passed")
    sim_env.close()


if __name__ == "__main__":
    unittest.main()
