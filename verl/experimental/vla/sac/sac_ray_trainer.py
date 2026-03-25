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
from pprint import pprint
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


def compute_avg_positive_trajectory_length(batch: DataProto) -> float:
    dones = batch.batch["dones"].bool()  # (B, T)
    positive_mask = batch.batch["positive_sample_mask"]  # (B, T)
    positive_traj = positive_mask.any(dim=1)  # (B,)

    if positive_traj.sum() == 0:
        return 0.0

    B, T = dones.shape
    done_idx = torch.argmax(dones.int(), dim=1)  # (B,)
    traj_lens = done_idx + 1

    return traj_lens[positive_traj].float().mean().item()


def flatten_trajectories(data: DataProto) -> DataProto:
    batch_size, num_steps = data.batch["action"].shape[:2]
    new_batch_fields = {}
    for key, tensor in data.batch.items():
        if len(tensor.shape) >= 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_steps:
            # (B, S, H, W) -> (B*S, H, W)
            new_shape = (batch_size * num_steps, *tensor.shape[2:])
            new_batch_fields[key] = tensor.reshape(new_shape)
        elif len(tensor.shape) == 1 and tensor.shape[0] == batch_size:
            # [e1, e2] -> [e1, e1, ..., e2, e2, ...] (S times each)
            new_batch_fields[key] = tensor.repeat_interleave(num_steps)
        else:
            new_batch_fields[key] = tensor
    new_data = DataProto.from_dict(tensors=new_batch_fields, meta_info=data.meta_info)
    return new_data


def add_transition_prefixes(data: DataProto) -> DataProto:
    batch = data.batch
    step_key = "action" if "action" in batch else "full_action"
    if step_key not in batch:
        return data

    num_steps = batch[step_key].shape[1]
    if num_steps <= 1:
        return data

    def drop_last(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, :-1, ...]

    def shift_next(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, 1:, ...]

    state_keys = ["states", "images", "image_masks", "lang_tokens", "lang_masks"]
    action_keys = ["full_action", "action"]

    for key in state_keys:
        if key in batch:
            batch[f"s0.{key}"] = drop_last(batch[key])
            batch[f"s1.{key}"] = shift_next(batch[key])

    for key in action_keys:
        if key in batch:
            batch[f"a0.{key}"] = drop_last(batch[key])
            batch[f"a1.{key}"] = shift_next(batch[key])

    batch_size = batch[step_key].shape[0]
    for key, tensor in list(batch.items()):
        if tensor.ndim >= 2 and tensor.shape[0] == batch_size and tensor.shape[1] == num_steps:
            batch[key] = drop_last(tensor)

    return data


class RobRaySACTrainer(RayPPOTrainer):
    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups including env workers."""
        super()._start_profiling(do_profile)
        if do_profile and hasattr(self, "env_wg"):
            self.env_wg.start_profile(role="env", profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups including env workers."""
        super()._stop_profiling(do_profile)
        if do_profile and hasattr(self, "env_wg"):
            self.env_wg.stop_profile()

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()

        if self.config.env.disagg_sim.enable:
            # pin EnvWorker to Simulator GPU nodes
            self.resource_pool_manager.get_resource_pool(Role.Env).accelerator_type = "sim"
            self.resource_pool_manager.get_resource_pool(Role.ActorRollout).accelerator_type = "train_rollout"

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        assert Role.Env in self.role_worker_mapping
        if Role.Env in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Env)
            env_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.Env], config=self.config.env)
            self.resource_pool_to_cls[resource_pool]["env"] = env_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        self.env_wg = all_wg["env"]

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async_envloop":
            from verl.experimental.vla.env_loop import EnvLoop

            self.async_rollout_mode = True
            self.async_rollout_manager = EnvLoop(
                config=self.config, rollout_wg=self.actor_rollout_wg, env_wg=self.env_wg
            )

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys())
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        return gen_batch

    def _reset_envs(self, gen_batch: DataProto) -> asyncio.Future:
        initial_state_ids = gen_batch.non_tensor_batch["state_ids"]
        task_ids = gen_batch.non_tensor_batch["task_ids"]
        reset_prompts = DataProto.from_dict(non_tensors={"state_ids": initial_state_ids, "task_ids": task_ids})
        reset_future = self.env_wg.reset_envs_to_state_ids(reset_prompts)
        return reset_future

    def _next_rollout_batch(self, train_iter) -> Optional[DataProto]:
        try:
            batch_dict = next(train_iter)
        except StopIteration:
            return None

        rollout_batch = DataProto.from_single_dict(batch_dict)
        rollout_batch = self._get_gen_batch(rollout_batch)
        rollout_batch = rollout_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        rollout_batch.meta_info["task_ids"] = np.asarray(rollout_batch.non_tensor_batch["task_ids"], dtype=np.int64)
        rollout_batch.meta_info["global_steps"] = self.global_steps

        return rollout_batch

    def _prepare_actor_input(self, rollout_output: Optional[DataProto]) -> DataProto:
        # dones
        complete_any = rollout_output.batch["complete"].any(dim=-1)  # (B, T)
        dones_step = complete_any.clone()
        dones_step[:, -2] = True
        rollout_output.batch["dones"] = dones_step.float()

        # reward (sparse reward with step penalty)
        sparse_rewards = complete_any.float()
        rollout_output.batch["valids"] = (~rollout_output.batch["complete"]).any(dim=-1).float()
        step_penalty = float(self.config.env.train.get("step_penalty", 0.0))
        rollout_output.batch["rewards"] = sparse_rewards - step_penalty * rollout_output.batch["valids"]
        rollout_output.batch["rewards"][:, -2] = -1.0

        # mark samples in successful trajectories as positive samples
        rollout_output.batch["positive_sample_mask"] = (
            sparse_rewards.any(dim=-1).unsqueeze(-1).repeat_interleave(rollout_output.batch["action"].shape[1], dim=-1)
        )

        # task id
        rollout_output.batch["task_ids"] = torch.as_tensor(
            rollout_output.meta_info["task_ids"],
            dtype=torch.long,
            device=rollout_output.batch["action"].device,
        )

        rollout_output.meta_info["global_token_num"] = [0]
        rollout_output.meta_info["data/trajectory_avg_reward"] = (
            sparse_rewards.any(dim=-1).mean(dtype=torch.float32).item()
        )
        rollout_output.meta_info["data/avg_positive_trajectory_length"] = compute_avg_positive_trajectory_length(
            rollout_output
        )

        rollout_output = add_transition_prefixes(rollout_output)
        rollout_output = flatten_trajectories(rollout_output)

        return rollout_output

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        self.total_training_steps = (
            self.config.trainer.total_epochs * len(self.train_dataloader) * self.config.trainer.rollout_interval
        )
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            train_iter = iter(self.train_dataloader)
            reset_future = None
            next_rollout_batch = self._next_rollout_batch(train_iter)
            if next_rollout_batch is None:
                continue

            print(f"Starting epoch {epoch}, dataloader length: {len(self.train_dataloader)}")
            while next_rollout_batch is not None:
                for training_step in range(self.config.trainer.rollout_interval):
                    metrics = {}
                    timing_raw = {}

                    # === start profiling ===
                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )

                    with marked_timer("step", timing_raw):
                        # === rollout ===
                        # Determine whether to perform rollout:
                        # enable at start and early warmup, disable during critic warmup phase
                        warm_rollout_steps = int(getattr(self.config.actor_rollout_ref.actor, "warm_rollout_steps", 0))
                        need_rollout = (training_step == 0) or self.global_steps < warm_rollout_steps
                        if (
                            warm_rollout_steps
                            <= self.global_steps
                            < self.config.actor_rollout_ref.actor.critic_warmup_steps
                        ):
                            need_rollout = False
                        if need_rollout and next_rollout_batch is None:
                            break

                        actor_input = None
                        if need_rollout:
                            with marked_timer("rollout", timing_raw):
                                # execute rollout
                                rollout_batch = next_rollout_batch
                                assert rollout_batch is not None
                                if reset_future is None:
                                    reset_future = self._reset_envs(rollout_batch)
                                with marked_timer("generate", timing_raw, color="red"):
                                    rollout_output = self.async_rollout_manager.generate_sequences(
                                        rollout_batch, reset_future
                                    )

                                # prepare for next batch's env reset
                                next_rollout_batch = self._next_rollout_batch(train_iter)
                                if next_rollout_batch is not None:
                                    reset_future = self._reset_envs(next_rollout_batch)

                                # compute rewards and other metrics, and prepare for actor update
                                actor_input = self._prepare_actor_input(rollout_output)

                        # === update policy ===
                        with marked_timer("update_actor", timing_raw, color="red"):
                            if actor_input is not None:
                                actor_output = self.actor_rollout_wg.update_actor(actor_input)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(
                                    DataProto(
                                        meta_info={
                                            "empty_batch": True,
                                            "global_steps": self.global_steps,
                                            "global_token_num": [0],
                                        }
                                    )
                                )
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # === validate ===
                    is_last_step = self.global_steps >= self.total_training_steps
                    if (
                        self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        and self.global_steps >= self.config.actor_rollout_ref.actor.critic_warmup_steps
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        reset_future = None

                    # === save checkpoint ===
                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # === stop profiling ===
                    with marked_timer("stop_profile", timing_raw):
                        next_step_profile = (
                            self.global_steps + 1 in self.config.global_profiler.steps
                            if self.config.global_profiler.steps is not None
                            else False
                        )
                        self._stop_profiling(
                            curr_step_profile and not next_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )
                        prev_step_profile = curr_step_profile
                        curr_step_profile = next_step_profile

                    steps_duration = timing_raw["step"]
                    self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                    # === training metrics ===
                    metrics.update(
                        {
                            "training/global_step": self.global_steps,
                            "training/epoch": epoch,
                        }
                    )
                    metrics.update({f"timing_s/{name}": value for name, value in timing_raw.items()})
                    if actor_input is not None:
                        metrics["data/trajectory_avg_reward"] = actor_input.meta_info["data/trajectory_avg_reward"]
                        metrics["data/avg_positive_trajectory_length"] = actor_input.meta_info[
                            "data/avg_positive_trajectory_length"
                        ]
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1

                    if (
                        hasattr(self.config.actor_rollout_ref.actor, "profiler")
                        and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                    ):
                        self.actor_rollout_wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                        )

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return

    def _validate(self) -> dict:
        metric_list = []
        val_iter = iter(self.val_dataloader)
        test_batch = self._next_rollout_batch(val_iter)
        while test_batch is not None:
            if len(test_batch) < self.config.data.val_batch_size:
                print(f"drop last batch in val_dataloader, len {len(test_batch)}")
                break

            test_batch.meta_info["validate"] = True
            reset_future = self._reset_envs(test_batch)
            rollout_output = self.async_rollout_manager.generate_sequences(test_batch, reset_future)
            self._prepare_actor_input(rollout_output)
            test_batch = self._next_rollout_batch(val_iter)
            actor_input = self._prepare_actor_input(rollout_output)

            metric_list.append(
                {
                    "val/avg_reward": actor_input.meta_info["data/trajectory_avg_reward"],
                    "val/avg_positive_trajectory_length": actor_input.meta_info["data/avg_positive_trajectory_length"],
                }
            )

        metrics = {}
        if metric_list:
            metrics["val/avg_reward"] = np.mean([m["val/avg_reward"] for m in metric_list])
            metrics["val/avg_positive_trajectory_length"] = np.mean(
                [m["val/avg_positive_trajectory_length"] for m in metric_list]
            )

        return metrics
