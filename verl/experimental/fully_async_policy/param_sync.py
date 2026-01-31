# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import logging
import time

import ray
from ray.util.collective import collective

from verl.utils.device import get_nccl_backend

logger = logging.getLogger(__name__)


@ray.remote
class ParameterSynchronizer:
    """
    Unified parameter synchronizer, responsible for synchronizing model parameters between actor and rollout
    Based on the mature synchronization mode implementation of one_step_off_policy
    Merges the functions of the original multiple synchronizer classes
    """

    def __init__(self, config, trainer, rollouter, mq):
        self.config = config
        self.trainer = trainer
        self.rollouter = rollouter
        self.mq_client = mq
        self.actor_wg = ray.get(trainer.get_actor_wg.remote())
        self.rollout_wg = ray.get(rollouter.get_rollout_wg.remote())

        # Basic attributes
        self.weights_info = None
        self.sync_group_initialized = False
        self.sync_group_name = "actor_rollout"
        self.wait_last_update = None
        self.wait_last_resume = None
        self.validate_task = None

        # Statistics
        self.current_version = 0

        self._init_weights_info()
        self._init_sync_group()

        if self.config.async_training.checkpoint_engine.enable:
            self._init_actor_rollout_checkpoint_engine()

    def get_current_param_version(self) -> int:
        """Get current parameter version number"""
        return self.current_version

    def get_weights_info(self):
        """Get weights info"""
        return self.weights_info

    def _init_weights_info(self):
        self.weights_info = self.actor_wg.get_actor_weights_info()[0]
        self.rollout_wg.set_actor_weights_info(self.weights_info)

    def _init_sync_group(self):
        print("[ParameterSynchronizer] Initializing parameter synchronization group...")
        actor_rollout_workers = self.actor_wg.workers + self.rollout_wg.workers
        n_workers = len(self.actor_wg.workers + self.rollout_wg.workers)
        if self.config.trainer.device == "npu":
            master_address = ray.get(self.actor_wg.workers[0]._get_node_ip.remote()).strip("[]")
            master_port = ray.get(self.actor_wg.workers[0]._get_free_port.remote())
            self.actor_wg.create_weight_sync_group(
                master_address,
                master_port,
                0,
                n_workers,
            )
            ray.get(
                self.rollout_wg.create_weight_sync_group(
                    master_address,
                    master_port,
                    len(self.actor_wg.workers),
                    n_workers,
                )
            )
        else:
            collective.create_collective_group(
                actor_rollout_workers,
                n_workers,
                list(range(0, n_workers)),
                backend=get_nccl_backend(),
                group_name=self.sync_group_name,
            )

    def _init_actor_rollout_checkpoint_engine(self):
        ray.get(
            self.actor_wg.init_checkpoint_engine(
                rank_offset=0,
                actor_num=len(self.actor_wg.workers),
                rollout_num=len(self.rollout_wg.workers),
            )
        )
        ray.get(
            self.rollout_wg.init_checkpoint_engine(
                rank_offset=len(self.actor_wg.workers),
                actor_num=len(self.actor_wg.workers),
                rollout_num=len(self.rollout_wg.workers),
            )
        )

    def sync_weights(self, version, validate=False, global_steps=0, use_trainer_do_validate=False):
        """Sync weights between trainer and rollouter, and update parameter version"""
        start_time = time.time()

        self.current_version = version
        ray.get(self.rollouter.pause.remote())

        print(f"[ParameterSynchronizer] rollout paused. cost {time.time() - start_time:.2f} seconds")
        # Update MQ version
        self.mq_client.update_param_version_sync(version)

        pause_time = time.time()

        # sync weights
        # For sglang, always use sync_rollout_weights instead of sync_rollout_weights_by_checkpoint
        rollout_name = getattr(self.config.actor_rollout_ref.rollout, "name", None)
        use_checkpoint_engine = self.config.async_training.checkpoint_engine.enable and rollout_name != "sglang"

        if use_checkpoint_engine:
            self.actor_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name)
            ray.get(self.rollout_wg.sync_rollout_weights_by_checkpoint(self.sync_group_name))
        else:
            self.actor_wg.sync_rollout_weights(self.sync_group_name)
            ray.get(self.rollout_wg.sync_rollout_weights(self.sync_group_name))
        end_time = time.time()
        print(
            f"[ParameterSynchronizer] sync_weights success. cost {end_time - start_time:.2f} seconds, "
            f"pause:{pause_time - start_time:.2f}s, sync:{end_time - pause_time:.2f}s"
        )
        # async train do validate
        print(f"[ParameterSynchronizer] validate: {validate}, use_trainer_do_validate: {use_trainer_do_validate}")
        if validate and use_trainer_do_validate:
            print("[ParameterSynchronizer] use trainer to do validate")
            self.validate_task = self.trainer._validate_process.remote()
        else:
            self.validate_task = None
        # Async Update rollout version & validation
        self.wait_last_update = self.rollouter.update_param_version.remote(
            version, validate, global_steps, use_trainer_do_validate
        )
        self.wait_last_resume = self.rollouter.resume.remote(self.wait_last_update)

    def wait_last_valid(self):
        print("[ParameterSynchronizer] Waiting last sync and validate...")
        start_time = time.time()
        if self.wait_last_update:
            ray.get(self.wait_last_update)
        if self.wait_last_resume:
            ray.get(self.wait_last_resume)
        if self.validate_task:
            ray.get(self.validate_task)
        print(f"[ParameterSynchronizer] Wait last validate cost: {time.time() - start_time:.2f} seconds")

    def rollouter_save_checkpoint(self, local_global_step_folder: str):
        """Trigger rollout to save checkpoint(dataloader)"""
        print(f"[ParameterSynchronizer] Triggering checkpoint save at {local_global_step_folder} ...")
        return ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
