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

import asyncio
import logging
import os

import numpy as np
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EnvLoop:
    """An env loop manages interactions between models and vectorized environments. It's designed for computationally
    intensive environments, such as robotics simulators."""

    def __init__(self, env_wg: RayWorkerGroup, rollout_wg: RayWorkerGroup, config: DictConfig):
        """
        Initialize the EnvLoop.

        Args:
            env_wg (RayWorkerGroup): Environment worker group.
            rollout_wg (RayWorkerGroup): Rollout worker group for model inference.
            config (DictConfig): YAML config.
        """
        self.env_wg = env_wg
        self.rollout_wg = rollout_wg
        self.config = config
        # Extract relevant configuration
        self.max_interactions = config.env.train.max_episode_steps // config.env.actor.model.num_action_chunks
        self.stage_num = config.env.rollout.pipeline_stage_num
        self.num_envs_per_worker = config.env.train.num_envs
        self.action_dim = config.env.actor.model.action_dim
        self.num_action_chunks = config.env.actor.model.num_action_chunks
        self.isaac_server_mode = config.env.train.get("isaac_server_mode", False)
        # Derived properties
        # In server/ray mode, total_trajs is configured explicitly
        # In local mode, it's computed from world_size * num_envs_per_worker
        # Backward compatible: support both total_trajs (new) and total_envs (legacy)
        default_total = self.env_wg.world_size * self.num_envs_per_worker
        self.total_trajs = config.env.train.get("total_trajs", config.env.train.get("total_envs", default_total))
        if self.total_trajs % self.stage_num != 0:
            raise ValueError(f"total_trajs ({self.total_trajs}) must be divisible by stage_num ({self.stage_num})")
        self.trajs_per_stage = self.total_trajs // self.stage_num

        self.env_wg.init_worker()
        self.env_wg.init_simulator()

    def generate_sequences(self, prompts: DataProto, reset_future: asyncio.Future) -> DataProto:
        """Split input batch and dispatch to env loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        reset_results = reset_future.get()

        loop = asyncio.get_event_loop()
        self.rollout_wg.switch_to_rollout()
        output = loop.run_until_complete(self.run(prompts, reset_results))
        self.rollout_wg.switch_to_train()
        # TODO(caiyunke.astra): add timing metrics
        return output

    async def run(self, prompts: DataProto, reset_results: DataProto) -> DataProto:
        """
        Run the environment interaction loop.
        This method orchestrates a pipelined process:
        1. Resets environments to specified initial states.
        2. In a loop, it gets actions from the rollout workers and applies them to the environments.
        3. Collects all trajectory data (observations, actions, rewards, dones).
        4. Formats and returns the collected trajectories as a single batch.
        Args:
            prompts (DataProto): Contains initial state IDs and other settings.
                                 - 'non_tensor_batch.state_ids': A numpy array of state IDs to reset envs.
        Returns:
            DataProto: A batch containing the complete trajectories.
        """
        initial_state_ids = prompts.non_tensor_batch["state_ids"]

        staged_obs = self._restructure_obs_data(reset_results)
        # --- Pipeline state ---
        trajectories = {i: [] for i in range(self.stage_num)}  # To store (obs, action, rew, done) tuples
        rollout_futures = {}
        # is_complete = torch.zeros((self.total_trajs,), dtype=torch.bool)

        # Extract traj_keys for each stage (important for action-env mapping)
        # Each traj_key maps a rollout traj to exactly one sim env
        staged_traj_keys = {}
        for stage_id in range(self.stage_num):
            stage_data = staged_obs[stage_id]
            if "traj_keys" in stage_data.non_tensor_batch:
                staged_traj_keys[stage_id] = stage_data.non_tensor_batch["traj_keys"]
            else:
                staged_traj_keys[stage_id] = None

        for stage_id in range(self.stage_num):
            # trajectories[stage_id].append({'obs': staged_obs[stage_id]})
            trajectories[stage_id].append({})
            vla_input = staged_obs[stage_id]
            vla_input.meta_info = prompts.meta_info  # Pass along rollout config
            rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        async def _stage_loop(stage_id: int):
            for step_idx in range(self.max_interactions):
                action_result: DataProto = await asyncio.to_thread(rollout_futures[stage_id].get)

                trajectories[stage_id][-1]["action"] = action_result

                # Include traj_keys to ensure action-env mapping (1:1 traj-env correspondence)
                non_tensors = {"actions": action_result.batch["action"].cpu().numpy()}
                if staged_traj_keys[stage_id] is not None:
                    non_tensors["traj_keys"] = staged_traj_keys[stage_id]

                action_data = DataProto.from_dict(
                    non_tensors=non_tensors,
                    meta_info={
                        "stage_id": stage_id,
                        "step_idx": step_idx,
                        "max_steps": self.max_interactions,
                    },
                )

                env_ref = self.env_wg.env_interact_step(action_data)
                env_result: DataProto = await asyncio.to_thread(env_ref.get)

                trajectories[stage_id][-1]["rew"] = env_result.batch["rews"]
                trajectories[stage_id][-1]["done"] = env_result.batch["terminations"]

                next_obs = DataProto(
                    batch=env_result.batch.select("full_image", "state"),
                    non_tensor_batch={"task_descriptions": env_result.non_tensor_batch["task_descriptions"]},
                )

                if step_idx < self.max_interactions - 1:
                    trajectories[stage_id].append({})
                    vla_input = next_obs
                    vla_input.meta_info = prompts.meta_info
                    rollout_futures[stage_id] = self.rollout_wg.generate_sequences(vla_input)

        await asyncio.gather(*[asyncio.create_task(_stage_loop(sid)) for sid in range(self.stage_num)])
        self.env_wg.finish_rollout()

        return self._collate_trajectories(trajectories, initial_state_ids, meta_info=prompts.meta_info)

    def _restructure_obs_data(self, data_proto: DataProto) -> list[DataProto]:
        """Reshapes flat observation data from env_wg into a list of per-stage DataProto objects.

        Reset returns data in interleaved format:
            traj_0 (stage_0), traj_1 (stage_1), traj_2 (stage_0), traj_3 (stage_1), ...

        In isaac_server_mode: each traj maps to exactly one env (1:1), so we chunk by total_trajs.
        In local mode: each trajectory has num_envs_per_worker environments.
        """
        if self.isaac_server_mode:
            # Isaac server mode: traj and env are 1:1, chunk by total_trajs
            trajectory_chunks = data_proto.chunk(self.total_trajs)
        else:
            # Local mode: each trajectory has num_envs_per_worker envs
            assert self.total_trajs % self.num_envs_per_worker == 0, (
                f"total_trajs ({self.total_trajs}) must be divisible by "
                f"num_envs_per_worker ({self.num_envs_per_worker})"
            )
            num_trajectories = self.total_trajs // self.num_envs_per_worker
            trajectory_chunks = data_proto.chunk(num_trajectories)

        # Group by stage (trajectories are assigned round-robin to stages)
        staged_data = [[] for _ in range(self.stage_num)]
        for traj_idx, traj_data in enumerate(trajectory_chunks):
            stage_id = traj_idx % self.stage_num
            staged_data[stage_id].append(traj_data)

        # Concatenate trajectories for each stage
        return [DataProto.concat(data_list) for data_list in staged_data]

    def _collate_trajectories(self, trajectories: dict, initial_state_ids: np.ndarray, meta_info) -> DataProto:
        """
        Collates the collected trajectory data into the final batch format.
        """
        flat_trajs = [{} for _ in range(len(trajectories[0]))]
        for stage_id in range(self.stage_num):
            for step_idx, step_data in enumerate(trajectories[stage_id]):
                if not flat_trajs[step_idx]:  # if dict is empty
                    flat_trajs[step_idx] = step_data
                else:
                    # Concatenate DataProto objects
                    for key, value in step_data.items():
                        if isinstance(value, DataProto):
                            flat_trajs[step_idx][key] = DataProto.concat([flat_trajs[step_idx][key], value])
                        elif isinstance(value, torch.Tensor):
                            flat_trajs[step_idx][key] = torch.cat([flat_trajs[step_idx][key], value], dim=0)

        all_pixel_values = [step["action"].batch["pixel_values"] for step in flat_trajs]
        all_responses = [step["action"].batch["responses"] for step in flat_trajs]
        all_input_ids = [step["action"].batch["input_ids"] for step in flat_trajs]
        all_attn_masks = [step["action"].batch["attention_mask"] for step in flat_trajs]
        all_actions = [step["action"].batch["action"] for step in flat_trajs]
        all_dones = [step["done"] for step in flat_trajs]

        pixel_values = torch.stack(all_pixel_values, dim=1)
        responses = torch.stack(all_responses, dim=1)
        input_ids = torch.stack(all_input_ids, dim=1)
        attention_mask = torch.stack(all_attn_masks, dim=1)
        actions = torch.stack(all_actions, dim=1)
        complete = torch.stack(all_dones, dim=1).squeeze(-1)  # Shape [bs, steps]
        batch_dict = {
            "pixel_values": pixel_values,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "complete": complete,
            "action": actions,
            "env_state_id": torch.from_numpy(initial_state_ids.astype(int)),
        }

        return DataProto.from_single_dict(batch_dict, meta_info=meta_info)
