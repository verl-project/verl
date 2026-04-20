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
import math
import os
import re
from pathlib import Path
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

HDF5_SEARCH_PATHS = [
    Path("/root/data/IsaacLabPlayGround_Dataset/libero/assembled_hdf5"),
    Path("/root/RobotLearningLab/benchmarks/datasets/libero/assembled_hdf5"),
]


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
        self.reset_mode = self.cfg.get("reset_mode", "random")  # "hdf5" | "random"

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

        # Set correct environment variables for RobotLearningLab paths
        os.environ["LIBERO_CONFIG_DIR"] = "/root/RobotLearningLab/benchmarks/datasets/libero/config"
        os.environ["LIBERO_ASSETS_DATA_DIR"] = "/root/RobotLearningLab/benchmarks/datasets/libero/USD"

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

            if not self.task_name:
                self.task_name = "Isaac-Libero-Franka-IK-Abs-RL-v0"

            # For OSC-based tasks, set the controller type env var that
            # OscPoseLiberoCameraEnvCfg reads during __post_init__.
            if "OscPose" in self.task_name:
                os.environ.setdefault("LIBERO_OSC_TYPE", "pose_rel")

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
        # use step reward to determine success
        self.success_once = self.success_once | (step_reward > 0)
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

    @staticmethod
    def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to axis-angle (ax, ay, az). Batched: (..., 4) -> (..., 3)."""
        w = np.clip(quat[..., 0:1], -1.0, 1.0)
        xyz = quat[..., 1:4]
        angle = 2.0 * np.arccos(np.abs(w))
        den = np.sqrt(1.0 - w * w)
        small = (den < 1e-8)
        axis_angle = np.where(small, np.zeros_like(xyz), xyz / den * angle * np.sign(w))
        return axis_angle

    @staticmethod
    def _axisangle2quat(axisangle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle (..., 3) to quaternion (..., 4) in (w, x, y, z) convention (Isaac Lab)."""
        angle = torch.norm(axisangle, dim=-1, keepdim=True).clamp(min=1e-8)
        axis = axisangle / angle
        half = angle * 0.5
        sin_half = torch.sin(half)
        cos_half = torch.cos(half)
        return torch.cat([cos_half, axis * sin_half], dim=-1)

    def _convert_actions_for_ik_abs(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert model output (pos3 + axisangle3 + gripper1) to env input (pos3 + quat4 + gripper1)."""
        pos = actions[..., :3]
        axisangle = actions[..., 3:6]
        gripper = actions[..., 6:7]
        quat = self._axisangle2quat(axisangle)
        return torch.cat([pos, quat, gripper], dim=-1)

    def step(self, actions=None, critic_values=None):
        if actions is None:
            # isaac should start with reset_envs_to_initial_state
            # do nothing for None
            return (None, None, None, None, None)

        truncations = self.elapsed_steps >= self.max_episode_steps

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        is_ik_abs = "IK-Abs" in (self.task_name or "")

        if self._elapsed_steps[0] < 3:
            logger.info(
                f"[step {self._elapsed_steps[0]}] VLA action (pos|aa|grip): "
                f"{actions[0, :3].tolist()} | {actions[0, 3:6].tolist()} | {actions[0, 6:].tolist()}"
            )

        if is_ik_abs:
            actions = self._convert_actions_for_ik_abs(actions.to(self.device))
        else:
            actions = actions.to(self.device)

        if self._elapsed_steps[0] < 3:
            eef = self.env.unwrapped.scene["robot"].data.body_state_w[:, -1, :7]
            logger.info(
                f"[step {self._elapsed_steps[0]}] Current EEF pos: {eef[0, :3].tolist()}, "
                f"Action to env: {actions[0].tolist()}"
            )

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, _, infos = self.env.step(actions)
        self.last_obs = raw_obs
        self.last_infos = infos

        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(_reward.cpu().numpy())

        infos = self._record_metrics(step_reward, terminations, infos)

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": self.success_once,
                "task": self.task_descriptions,
            }
            if critic_values is not None:
                plot_infos["critic_value"] = np.asarray(critic_values, dtype=np.float32)
            self.add_new_frames(obs, plot_infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions, chunk_values=None):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        ever_done = torch.zeros(self.num_envs, dtype=torch.bool)
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            step_values = None
            if chunk_values is not None:
                if len(chunk_values.shape) == 1:
                    step_values = chunk_values
                elif len(chunk_values.shape) == 2:
                    step_values = chunk_values[:, i]

            extracted_obs, step_reward, terminations, truncations, infos = self.step(actions, critic_values=step_values)

            # Derive success from reward signal (since DoneTerm is disabled to
            # prevent IsaacLab auto-reset).  Once an env succeeds, mark it as
            # terminated for all remaining sub-steps so verl sees monotonic
            # terminations — matching MuJoCo/LIBERO semantics.
            reward_val = step_reward.cpu() if isinstance(step_reward, torch.Tensor) else torch.as_tensor(step_reward)
            ever_done = ever_done | (reward_val > 0).view(-1)
            raw_chunk_terminations.append(ever_done.clone())

            chunk_rewards.append(step_reward)
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
        images = {}

        available_cameras = list(self.env.unwrapped.scene.keys())

        for camera_name in self.camera_name:
            found = False
            for key in available_cameras:
                if key.startswith(camera_name):
                    cam = self.env.unwrapped.scene[key]
                    rgb = cam.data.output["rgb"]
                    images[camera_name] = rgb.cpu().numpy()
                    found = True
                    break

            if camera_name == self.camera_name[0]:
                assert camera_name in images, f"camera {camera_name} not found in scene"

        eef_pose = obs["policy"]["eef_pose"].cpu().numpy()        # (num_envs, 7): pos3 + quat_wxyz4
        eef_pos = eef_pose[..., :3]                                  # (num_envs, 3)
        eef_axisangle = self._quat2axisangle(eef_pose[..., 3:7])    # (num_envs, 3)
        gripper_pos = obs["policy"]["gripper_pos"].cpu().numpy()     # (num_envs, 2)
        gripper_state = gripper_pos[..., 0:1]                        # (num_envs, 1): first finger

        output = {
            "full_image": images[self.camera_name[0]],
            "state": np.concatenate(
                [eef_pos, eef_axisangle, gripper_state],
                axis=-1,
            ),
        }

        if "eye_in_hand_cam" in images:
            output["wrist_image"] = images["eye_in_hand_cam"]
        elif "robot0_eye_in_hand" in images:
            output["wrist_image"] = images["robot0_eye_in_hand"]
        else:
            wrist_cam_found = False
            for cam_key in available_cameras:
                if "eye_in_hand" in cam_key or "wrist" in cam_key:
                    cam = self.env.unwrapped.scene[cam_key]
                    rgb = cam.data.output["rgb"]
                    output["wrist_image"] = rgb.cpu().numpy()
                    wrist_cam_found = True
                    break
            assert wrist_cam_found, "wrist camera not found in scene"

        return output

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

    def _find_hdf5_file(self, task_suite_name: str, task_id: int) -> Optional[Path]:
        """Locate the HDF5 demo file for the given task suite and task id."""
        pattern = f"{task_suite_name}_task{task_id}_*_demo.hdf5"
        for search_dir in HDF5_SEARCH_PATHS:
            if not search_dir.exists():
                continue
            matches = list(search_dir.glob(pattern))
            if matches:
                logger.info(f"Found HDF5 file: {matches[0]}")
                return matches[0]
        logger.warning(f"No HDF5 file found for {task_suite_name} task {task_id}")
        return None

    def _load_hdf5_initial_state(self, task_id: int, episode_hint: int = 0):
        """Load initial state from the HDF5 demo dataset.

        Returns the initial_state dict (on self.device) or None if not available.
        """
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        hdf5_path = self._find_hdf5_file(self.task_suite_name, task_id)
        if hdf5_path is None:
            return None

        handler = HDF5DatasetFileHandler()
        handler.open(str(hdf5_path))
        try:
            num_episodes = handler.get_num_episodes()
            if num_episodes == 0:
                return None

            episode_names = list(handler.get_episode_names())
            episode_map = {}
            for name in episode_names:
                m = re.search(r"(\d+)", name)
                if m:
                    episode_map[int(m.group(1))] = name

            ep_idx = episode_hint % num_episodes
            sorted_indices = sorted(episode_map.keys())
            if ep_idx < len(sorted_indices):
                ep_name = episode_map[sorted_indices[ep_idx]]
            else:
                ep_name = episode_names[0]

            episode_data = handler.load_episode(ep_name, self.device)
            if episode_data is None:
                return None

            initial_state = episode_data.get_initial_state()
            if initial_state is None:
                logger.warning(f"No initial_state in episode {ep_name}")
            else:
                logger.info(f"Loaded initial state from {hdf5_path.name} episode {ep_name}")
            return initial_state
        finally:
            handler.close()

    @staticmethod
    def _expand_state(state, num_envs):
        """Recursively expand all tensors with batch dim 1 to num_envs."""
        if isinstance(state, torch.Tensor):
            if state.shape[0] == 1:
                return state.expand(num_envs, *state.shape[1:]).contiguous()
            return state
        if isinstance(state, dict):
            return {k: IsaacEnv._expand_state(v, num_envs) for k, v in state.items()}
        return state

    def reset_envs_to_state_ids(self, state_ids_list, task_ids_list):
        logger.info(
            f"IsaacEnv reset_envs_to_state_ids task_ids={task_ids_list}, "
            f"reset_mode={self.reset_mode}"
        )
        assert len(set(task_ids_list)) == 1, "Isaac env only support single task"

        task_id = task_ids_list[0]
        self._init_env(task_id)

        raw_obs, infos = self.env.reset()

        if self.reset_mode == "hdf5":
            raw_obs, infos = self._reset_from_hdf5(task_id, state_ids_list, raw_obs, infos)

        raw_obs = self._stabilize_arm(raw_obs)
        self._reset_metrics()
        self.elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        obs = self._wrap_obs(raw_obs)
        return obs, infos

    def _reset_from_hdf5(self, task_id, state_ids_list, raw_obs, infos):
        """Override env state with a fixed initial state loaded from HDF5 demo data."""
        episode_hint = int(state_ids_list[0]) if state_ids_list else 0
        initial_state = self._load_hdf5_initial_state(task_id, episode_hint=episode_hint)
        if initial_state is None:
            logger.warning("No HDF5 initial state available, falling back to random reset")
            return raw_obs, infos

        env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        if self.num_envs > 1:
            initial_state = self._expand_state(initial_state, self.num_envs)
        raw_obs, infos = self.env.reset_to(initial_state, env_ids_tensor, is_relative=True)
        logger.info("Reset environment to HDF5 initial state")
        return raw_obs, infos

    def _stabilize_arm(self, raw_obs, steps=10):
        """Run a few hold-position steps so the arm settles after reset."""
        is_abs_action = "IK-Abs" in (self.task_name or "")
        for _ in range(steps):
            if is_abs_action:
                eef_pose = raw_obs["policy"]["eef_pose"].to(self.device)
                gripper_pos = raw_obs["policy"]["gripper_pos"].to(self.device)
                hold_action = torch.cat([eef_pose, gripper_pos[..., 0:1]], dim=-1)
            else:
                env_action_dim = self.env.action_space.shape[-1]
                hold_action = torch.zeros((self.num_envs, env_action_dim), device=self.device)
            raw_obs, _, _, _, _ = self.env.step(hold_action)
        return raw_obs