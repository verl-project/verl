# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import omni
import torch

from verl.experimental.vla.envs.action_utils import (
    put_info_on_image,
    save_rollout_video,
    tile_images,
    to_tensor,
)

logger = logging.getLogger(__name__)


class IsaacEnv(gym.Env):
    def __init__(self, cfg, rank, world_size):
        self.rank = rank
        self.cfg = cfg
        self.world_size = world_size
        self.seed = self.cfg.seed + rank
        self.num_envs = self.cfg.num_envs
        self.action_dim = self.cfg.get("action_dim", 7)
        self.device = self.cfg.get("device", "cuda:0")

        self._generator = np.random.default_rng(seed=self.seed)

        self.task_suite_name = self.cfg.task_suite_name

        self.env = None
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = False

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.max_episode_steps = cfg.max_episode_steps
        self.video_cfg = cfg.video_cfg

        self.render_images = []
        self.video_cnt = 0
        self.camera_name = cfg.init_params.camera_names

        # sys env must be set before import isaaclab
        from isaaclab.app import AppLauncher

        launch_args = {"headless": True, "enable_cameras": True}
        app_launcher = AppLauncher(**launch_args)
        self.app = app_launcher.app
        # force franka registration
        import isaaclab_playground.tasks.manipulation.libero.config.franka  # noqa

    def _init_env(self, task_id=0):
        """Initializes the Isaac Sim environment."""

        self.task_name = self.cfg.get("task_name")
        self.task_id = task_id
        # FIXME since isaac use env to set task id, all env have to use the same task id
        if self.task_suite_name.startswith("libero"):
            os.environ["LIBERO_TASK_SUITE"] = self.task_suite_name
            os.environ["LIBERO_TASK_ID"] = str(task_id)
            os.environ["LIBERO_OSC_TYPE"] = "pose_rel"

            if not self.task_name:
                self.task_name = "Isaac-Libero-Franka-OscPose-v0"

        from isaaclab_tasks.utils import parse_env_cfg

        self.env_cfg = parse_env_cfg(self.task_name, num_envs=self.num_envs)
        self.env_cfg.env_name = self.cfg.get("env_name", str(self.task_id))
        self.env_cfg.sim.device = self.device
        self.env_cfg.sim.physx.enable_ccd = True
        self.env_cfg.terminations.time_out = None
        self.env_cfg.observations.policy.concatenate_terms = False

        # create environment from loaded config
        if self.env:
            self.env.close()
            omni.usd.get_context().new_stage()
        self.env = gym.make(self.task_name, cfg=self.env_cfg).unwrapped

        if self.cfg.video_cfg.save_video:
            video_dir = os.path.join(self.cfg.video_cfg.video_base_dir, f"rank_{self.rank}")
            os.makedirs(video_dir, exist_ok=True)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # TODO support other task suite
        if self.task_suite_name.startswith("libero"):
            self.task_descriptions = self.env.cfg.libero_config.task_info["language_instruction"]
            assert self.env_cfg.osc_type == "pose_rel", (
                f"Only pose_rel osc type is supported for libero. Received: {self.env_cfg.osc_type}"
            )
        else:
            raise ValueError(f"Task suite {self.task_suite_name} is not supported.")
        logger.info("Isaac Sim environment initialized")

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        # Ensure terminations is a numpy array before the bitwise OR
        if isinstance(terminations, torch.Tensor):
            terminations = terminations.cpu().numpy()
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        if any(self.elapsed_steps > 0):
            episode_info["reward"] = episode_info["return"] / self.elapsed_steps
        else:
            episode_info["reward"] = 0
        infos["episode"] = to_tensor(episode_info)
        return infos

    def reset(self, env_idx: Optional[int | list[int] | np.ndarray] = None, options: Optional[dict] = None):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        raw_obs, infos = self.env.reset()

        obs = self._wrap_obs(raw_obs)

        self._reset_metrics(env_idx)

        return obs, infos

    def step(self, actions=None):
        if actions is None:
            # isaac should start with reset_envs_to_initial_state
            # do nothing for None
            return (None, None, None, None, None)

        truncations = self.elapsed_steps >= self.max_episode_steps
        # _actions = torch.zeros(self.action_space.shape)

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, _, infos = self.env.step(actions)
        self.last_obs = raw_obs
        self.last_infos = infos

        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(_reward.cpu().numpy())

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "task": self.task_descriptions,
            }
            self.add_new_frames(obs, plot_infos)

        infos = self._record_metrics(step_reward, terminations, infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(actions)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)  # [num_envs, chunk_steps]

        chunk_terminations = raw_chunk_terminations.clone()
        chunk_truncations = raw_chunk_truncations.clone()
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _calc_step_reward(self, reward):
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward

    def _wrap_obs(self, raw_obs):
        images_and_states = self._extract_image_and_state(raw_obs)

        obs = {
            "images_and_states": to_tensor(images_and_states),
            "task_descriptions": [self.task_descriptions] * self.num_envs,
        }
        return obs

    def _extract_image_and_state(self, obs):
        # TODO support multiple camera
        camera_name = self.camera_name[0]
        for key in self.env.unwrapped.scene.keys():
            if key.startswith(camera_name):
                cam = self.env.unwrapped.scene[key]
                break
        assert cam is not None, f"camera {camera_name} not found in scene"

        rgb = cam.data.output["rgb"]

        full_image = rgb.cpu().numpy()
        return {
            "full_image": full_image,
            "state": np.concatenate(
                [
                    obs["policy"]["eef_pose"].cpu(),
                    # quat2axisangle(obs["robot0_eef_quat"]), # isaac do not return robot0_eef_quat
                    # obs["policy"]["gripper_pos"].cpu(),
                ],
                axis=-1,
            ),
        }

    def add_new_frames(self, obs, plot_infos):
        images = []
        for env_id, img in enumerate(obs["images_and_states"]["full_image"]):
            info_item = {k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()}
            img = put_info_on_image(img.cpu().numpy(), info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"rank_{self.rank}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []

    def close(self):
        if self.env is not None:
            self.env.close()
            self.app.close()

    def load_state(self, state_buffer: bytes):
        self.env.load_state(state_buffer)

    def get_state(self):
        return None

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        logger.info(f"IsaacEnv reset_envs_to_state_ids task_ids_list: {task_ids_list}")
        assert len(set(task_ids_list)) == 1, "Isaac env only support single task"

        self._init_env(task_ids_list[0])

        # In Isaac, reset to random status in groups to have more test coverage
        # TODO support reset in group with options = {"group": len(set(state_ids_list))}
        raw_obs, infos = self.env.reset()
        env_idx = np.arange(self.num_envs)
        self._reset_metrics(env_idx)

        self.elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        # stablize the environment
        for _ in range(10):
            zero_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
            raw_obs, _, _, _, infos = self.env.step(zero_actions)

        obs = self._wrap_obs(raw_obs)
        return obs, infos
