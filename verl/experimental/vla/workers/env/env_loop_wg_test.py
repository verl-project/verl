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

import asyncio

import ray
from omegaconf import OmegaConf

from verl import DataProto
from verl.experimental.vla.naive_rollout_rob import NaiveRolloutRob

# from verl.workers.env.env_worker import EnvWorker
from verl.experimental.vla.workers.env.env_worker import EnvWorker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

if not ray.is_initialized():
    ray.init()

    # for debugging
    # ray.init(
    #     runtime_env={
    #         "env_vars": {"RAY_DEBUG_POST_MORTEM": "1"},
    #     }
    # )

ENV_WORKERS_NUM = 1
STAGE_NUM = 1
# NUM_ENVS_PER_ITER = 32

# NUM_ENVS_PER_STAGE = 8
# NUM_ENVS_PER_ITER = STAGE_NUM * NUM_ENVS_PER_STAGE
# NUM_ENVS_PER_ITER = 8
# NUM_ENVS_PER_ITER = 32
NUM_ENVS_PER_ITER = 2
NUM_ENVS_PER_WORKER = NUM_ENVS_PER_ITER // ENV_WORKERS_NUM
# NUM_ENVS_PER_WORKER_PER_STAGE = NUM_ENVS_PER_STAGE // ENV_WORKERS_NUM
GROUP_SIZE = 2  # real group size = GROUP_SIZE * STAGE_NUM
GROUP_NUM_PER_ITER = NUM_ENVS_PER_ITER * STAGE_NUM // GROUP_SIZE
BATCH_SIZE_PER_GPU = 2
NUM_ACTS_CHUNKS = 8
MAX_EPISODE_STEPS = 32
MAX_INFER_STEPS = MAX_EPISODE_STEPS // NUM_ACTS_CHUNKS
cfg_dict = {
    "rollout": {"pipeline_stage_num": STAGE_NUM},
    "train": {
        "use_fixed_reset_state_ids": False,
        "ignore_terminations": False,
        # "auto_reset": True,
        "auto_reset": False,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "use_rel_reward": False,
        "reward_coef": 1.0,
        "only_eval": False,
        "use_ordered_reset_state_ids": False,
        # "num_images_in_input": 1,
        "init_params": {
            "camera_depths": False,
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
        },
        "video_cfg": {
            "save_video": True,
            "video_base_dir": "/tmp/videos",
        },
        "task_suite_name": "libero_10",
        "num_envs": NUM_ENVS_PER_WORKER,
        "simulator_type": "isaac",
        "seed": 0,
    },
    "enable_offload": False,
    "actor": {"model": {"num_action_chunks": NUM_ACTS_CHUNKS, "action_dim": 7}},
    "runner": {"only_eval": False},
}
env_cfg = OmegaConf.create(cfg_dict)

gpu_pool = RayResourcePool([ENV_WORKERS_NUM], use_gpu=True)
# RayEnvWorker = ray.remote(num_gpus=1)(EnvWorker)
ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(EnvWorker), config=env_cfg)

env_wg = RayWorkerGroup(gpu_pool, ray_cls_with_init)


def restructure_data_proto(data_proto: DataProto) -> list[DataProto]:
    total_batch_size = len(data_proto)
    tensors = data_proto.batch
    non_tensors = data_proto.non_tensor_batch

    full_image_tensor = tensors["full_image"]
    state_tensor = tensors["state"]
    task_descriptions_np = non_tensors["task_descriptions"]
    if total_batch_size != ENV_WORKERS_NUM * STAGE_NUM * NUM_ENVS_PER_WORKER:
        raise ValueError(
            f"Total batch size {total_batch_size} does not match the expected size "
            f"ENV_WORKERS_NUM * STAGE_NUM * NUM_ENVS_PER_WORKER = "
            f"{ENV_WORKERS_NUM * STAGE_NUM * NUM_ENVS_PER_WORKER}"
        )

    image_rest_shape = (ENV_WORKERS_NUM, STAGE_NUM, NUM_ENVS_PER_WORKER) + full_image_tensor.shape[1:]
    state_rest_shape = (ENV_WORKERS_NUM, STAGE_NUM, NUM_ENVS_PER_WORKER) + state_tensor.shape[1:]
    reshaped_full_image = full_image_tensor.view(image_rest_shape)
    reshaped_state = state_tensor.view(state_rest_shape)

    reshaped_task_descriptions = task_descriptions_np.reshape(ENV_WORKERS_NUM, STAGE_NUM, NUM_ENVS_PER_WORKER)
    stages_data_list = []
    for stage_idx in range(STAGE_NUM):
        stage_images = reshaped_full_image[:, stage_idx, :]
        stage_states = reshaped_state[:, stage_idx, :]
        stage_tasks = reshaped_task_descriptions[:, stage_idx, :]
        final_images = stage_images.reshape(ENV_WORKERS_NUM * NUM_ENVS_PER_WORKER, *full_image_tensor.shape[1:])
        final_states = stage_states.reshape(ENV_WORKERS_NUM * NUM_ENVS_PER_WORKER, *state_tensor.shape[1:])
        final_tasks = stage_tasks.flatten().tolist()

        stage_dp = DataProto.from_dict(
            tensors={"full_image": final_images, "state": final_states},
            non_tensors={"task_descriptions": final_tasks},
            meta_info={"do_sample": True, "temperature": 1.6, "prompt_length": 512},
        )
        stages_data_list.append(stage_dp)
    return stages_data_list


async def run():
    # breakpoint()
    env_wg.init_worker()
    env_wg.init_simulator()

    reset_state_ids_tensordict = DataProto.from_dict(
        non_tensors={"state_ids": [0] * NUM_ENVS_PER_ITER * STAGE_NUM, "task_ids": [0] * NUM_ENVS_PER_ITER * STAGE_NUM}
    )

    reset_result = env_wg.reset_envs_to_state_ids(reset_state_ids_tensordict)
    print(f"reset_envs_to_state_ids result: {reset_result}")
    stages_data_list = restructure_data_proto(reset_result)

    RayNaiveRolloutRob = ray.remote(num_gpus=1)(NaiveRolloutRob)

    model_config = {"path": "Haozhan72/Openvla-oft-SFT-libero10-trajall"}
    rollout_workers = RayNaiveRolloutRob.remote(model_config)

    env_obs_refs = {}
    rollout_refs = {}
    traj = [[], []]

    for _ in range(MAX_INFER_STEPS):
        for stage_id in range(STAGE_NUM):
            if _ == 0:
                rollout_refs[stage_id] = rollout_workers.generate_sequences.remote(stages_data_list[stage_id])
            else:
                # env_batch = env_obs_refs[stage_id]
                env_batch: DataProto = env_obs_refs[stage_id].get()
                env_batch_traj = env_batch.select(batch_keys=["rews", "terminations", "truncations"])
                traj[stage_id][-1].update({"env": env_batch_traj})
                obs = env_batch
                obs.meta_info.update({"do_sample": True, "temperature": 1.6, "prompt_length": 512})
                rollout_refs[stage_id] = rollout_workers.generate_sequences.remote(obs)
        for stage_id in range(STAGE_NUM):
            batch: DataProto = ray.get(rollout_refs[stage_id])
            traj[stage_id].append({"model": batch})
            action = batch.batch["action"]
            action = action.cpu().numpy()
            # already in env
            data = DataProto.from_dict(non_tensors={"actions": action}, meta_info={"stage_id": stage_id})
            env_obs_refs[stage_id] = env_wg.env_interact_step(data)

    env_wg.finish_rollout()


asyncio.run(run())
ray.timeline(filename="2stage_pipeline_timeline_wg.json")
