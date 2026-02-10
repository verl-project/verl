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
import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


def compute_response_mask(config, data: DataProto) -> torch.Tensor:
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    complete = data.batch["complete"]  # shape: [batch_size, num_steps, chunk_size]

    complete_traj = complete.view(complete.shape[0], -1)  # # shape: [batch_size, num_steps * chunk_size]
    batch_size, action_steps = complete_traj.shape

    step_indices = torch.arange(action_steps, device=complete.device).unsqueeze(0).expand(batch_size, -1)

    first_true_idx_approx = torch.argmax(complete_traj.long(), dim=1)

    has_any_true = complete_traj.any(dim=1)

    final_first_true_idx = torch.where(
        has_any_true, first_true_idx_approx, torch.tensor(action_steps - 1, device=complete.device)
    )

    mask_traj = step_indices <= final_first_true_idx.unsqueeze(1)

    mask = mask_traj.view(complete.shape)  # shape: [batch_size, num_steps, chunk_size]
    mask = mask.repeat_interleave(config.env.actor.model.action_dim, dim=-1)  # eapand to action dim
    return mask


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

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
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
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
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
            next_batch_dict = next(train_iter)
            need_validate = False
            dataloader_len = len(self.train_dataloader)
            print(f"Starting epoch {epoch}, dataloader length: {dataloader_len}")
            for step_idx in range(dataloader_len):
                batch_dict = next_batch_dict
                try:
                    next_batch_dict = next(train_iter)
                except StopIteration:
                    next_batch_dict = None

                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["do_sample"] = True
                gen_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                gen_batch.meta_info["prompt_length"] = self.config.actor_rollout_ref.rollout.prompt_length
                gen_batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
                gen_batch.meta_info["n_samples"] = self.config.actor_rollout_ref.rollout.n
                gen_batch.meta_info["pad_token_id"] = self.tokenizer.pad_token_id

                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                if step_idx == 0 or need_validate:
                    # reset env workers in first step
                    # if validation on last step, the reset was not executed and need to be done here
                    reset_future = self._reset_envs(gen_batch)

                need_validate = (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                )

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch, reset_future)

                    # prepare for next batch's env reset
                    if step_idx != dataloader_len - 1 and not need_validate:
                        next_batch: DataProto = DataProto.from_single_dict(next_batch_dict)
                        next_gen_batch = self._get_gen_batch(next_batch)
                        next_gen_batch = next_gen_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )
                        reset_future = self._reset_envs(next_gen_batch)

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = gen_batch_output

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(self.config, batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    batch.batch["rewards"] = reward_tensor
                    average_reward = reward_tensor.any(-1).mean(dtype=torch.float32).item()
                    metrics["data/trajectory_avg_reward"] = average_reward

                    batch = add_transition_prefixes(batch)
                    batch = flatten_trajectories(batch)

                    # batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["global_token_num"] = [0]

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if need_validate:
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

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
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

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

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
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

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if len(test_batch) < self.config.data.val_batch_size:
                print(f"drop last batch in val_dataloader, len {len(test_batch)}")
                break

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch))], dtype=object
                )

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "prompt_length": self.config.actor_rollout_ref.rollout.prompt_length,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "n_samples": self.config.actor_rollout_ref.rollout.n,
                "validate": True,
                "global_steps": self.global_steps,
            }

            test_gen_batch = test_gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )

            sample_uids.extend(test_gen_batch.non_tensor_batch["uid"])

            # pad to be divisible by dp_size
            size_divisor = self.config.env.train.num_envs * self.config.env.rollout.pipeline_stage_num
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                reset_future = self._reset_envs(test_gen_batch_padded)
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                    test_gen_batch_padded, reset_future
                )

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            test_batch = test_output_gen_batch
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict
