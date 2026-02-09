# Copyright 2025 The RLinf Authors.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import itertools

import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.experimental.vla.workers.env.env_manager import EnvManager
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_name,
)
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig


def put_tensor_cpu(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_cpu(value)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().contiguous()
    return data_dict


def create_env_batch(obs, rews, dones, infos, meta=None):
    ret_dict = {"obs": obs, "rews": rews, "dones": dones, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    ret_dict = put_tensor_cpu(ret_dict)
    return ret_dict


def create_env_batch_dataproto(obs, rews, terminations, truncations, infos, meta=None):
    ret_dict = {"obs": obs, "rews": rews, "terminations": terminations, "truncations": truncations, "infos": infos}
    if meta is not None:
        ret_dict.update(meta=meta)

    ret_dict = put_tensor_cpu(ret_dict)
    tensor_batch = {
        "full_image": ret_dict["obs"]["images_and_states"]["full_image"],
        "wrist_image": ret_dict["obs"]["images_and_states"]["wrist_image"],
        "state": ret_dict["obs"]["images_and_states"]["state"],
        "rews": ret_dict["rews"],
        "terminations": ret_dict["terminations"],
        "truncations": ret_dict["truncations"],
    }
    non_tensor_batch = {"task_descriptions": obs["task_descriptions"]}
    output = DataProto.from_dict(tensors=tensor_batch, non_tensors=non_tensor_batch)

    return output


class EnvWorker(Worker, DistProfilerExtension):
    def __init__(self, config: DictConfig):
        Worker.__init__(self)
        self.cfg = config
        self.train_video_cnt = 0
        self.eval_video_cnt = 0

        self.simulator_list = []
        self.last_obs_list = []
        self.last_dones_list = []
        self.eval_simulator_list = []

        self.stage_num = self.cfg.rollout.pipeline_stage_num
        initialize_global_process_group_ray(timeout_second=None)
        device_name = get_device_name()
        env_device_mesh = init_device_mesh(device_name, mesh_shape=(self.world_size, 1), mesh_dim_names=["dp", "tp"])
        self._register_dispatch_collect_info("env", dp_rank=env_device_mesh["dp"].get_local_rank(), is_collect=True)

        # Initialize profiler
        omega_profiler_config = config.train.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="green", role="env_init")
    def init_worker(self):
        if self.cfg.train.simulator_type == "libero":
            from verl.experimental.vla.envs.libero_env.libero_env import LiberoEnv

            for _ in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.cfg.train,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=LiberoEnv,
                    )
                )

        elif self.cfg.train.simulator_type == "isaac":
            from verl.experimental.vla.envs.isaac_env.isaac_env import IsaacEnv

            for _ in range(self.stage_num):
                self.simulator_list.append(
                    EnvManager(
                        self.cfg.train,
                        rank=self._rank,
                        world_size=self._world_size,
                        env_cls=IsaacEnv,
                    )
                )
        else:
            raise NotImplementedError(f"Simulator type {self.cfg.train.simulator_type} not implemented")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="green", role="env_init_simulator")
    def init_simulator(self):
        for i in range(self.stage_num):
            self.simulator_list[i].start_simulator()
        return

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="env"), blocking=False)
    @DistProfiler.annotate(color="red", role="env_interact_step")
    def env_interact_step(self, data: DataProto) -> dict:
        """
        This function is used to interact with the environment.
        """
        chunk_actions: torch.Tensor = data.non_tensor_batch["actions"]
        stage_id: int = data.meta_info["stage_id"]

        # Pi0.5 Libero is not required
        # TODO: prepare actions according to simulator type
        # chunk_actions = prepare_actions(
        #     simulator_type=self.cfg.train.simulator_type,
        #     raw_chunk_actions=chunk_actions,
        #     num_action_chunks=self.cfg.actor.model.num_action_chunks,
        #     action_dim=self.cfg.actor.model.action_dim,
        # )

        env_info_list = {}

        extracted_obs, chunk_rewards, chunk_terminations, chunk_truncations, infos = self.simulator_list[
            stage_id
        ].chunk_step(chunk_actions)
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        if chunk_dones.any():
            if "final_info" in infos:
                final_info = infos["final_info"]
                for key in final_info["episode"]:
                    env_info_list[key] = final_info["episode"][key][chunk_dones[:, -1]].cpu()

        env_batch = create_env_batch_dataproto(
            obs=extracted_obs,
            rews=chunk_rewards,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            infos=infos,
            meta=env_info_list,
        )
        return env_batch

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_all_state_ids(self):
        """Get all available state IDs from the environment."""
        state_ids = self.simulator_list[0].get_all_state_ids()
        return state_ids

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="env"), blocking=False)
    @DistProfiler.annotate(color="blue", role="env_reset_envs_to_state_ids")
    def reset_envs_to_state_ids(self, data: DataProto):
        """Reset environments to specified state IDs.

        Args:
            state_ids: State IDs to reset environments to
        """
        state_ids_list = list(data.non_tensor_batch["state_ids"])
        task_ids_list = list(data.non_tensor_batch["task_ids"])

        assert len(state_ids_list) == self.cfg.train.num_envs * self.stage_num, (
            f"state_ids_list length is {len(state_ids_list)}, but should be {self.cfg.train.num_envs * self.stage_num}"
        )
        result_list = []
        for stage_id in range(self.stage_num):
            if self.cfg.train.simulator_type == "isaac":
                assert (
                    len(
                        set(
                            state_ids_list[
                                stage_id * self.cfg.train.num_envs : (stage_id + 1) * self.cfg.train.num_envs
                            ]
                        )
                    )
                    == 1
                ), "rollout.n should equal to num_envs for isaac"

            result = self.simulator_list[stage_id].reset_envs_to_state_ids(
                state_ids_list[stage_id * self.cfg.train.num_envs : (stage_id + 1) * self.cfg.train.num_envs],
                task_ids_list[stage_id * self.cfg.train.num_envs : (stage_id + 1) * self.cfg.train.num_envs],
            )
            result_list.append(result)
        output_tensor_dict = {}
        output_non_tensor_dict = {}

        # Handle nested 'images_and_states'
        images_and_states_list = [d[0]["images_and_states"] for d in result_list]
        if images_and_states_list:
            # Assuming all dicts in the list have the same keys
            for k in images_and_states_list[0].keys():
                if isinstance(images_and_states_list[0][k], torch.Tensor):
                    output_tensor_dict[k] = torch.cat([d[k] for d in images_and_states_list])

        # Handle 'task_descriptions'
        task_descriptions_list = [d[0]["task_descriptions"] for d in result_list]
        output_non_tensor_dict["task_descriptions"] = list(itertools.chain.from_iterable(task_descriptions_list))

        output = DataProto.from_dict(tensors=output_tensor_dict, non_tensors=output_non_tensor_dict)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(color="gray", role="env_finish_rollout")
    def finish_rollout(self, mode="train"):
        # reset
        if mode == "train":
            if self.cfg.train.video_cfg.save_video:
                for i in range(self.stage_num):
                    self.simulator_list[i].flush_video(video_sub_dir=f"stage_{i}")
