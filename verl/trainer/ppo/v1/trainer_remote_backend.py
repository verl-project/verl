# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""V1 PPO trainer that drives a plugin-provided :class:`RemoteBackend`.

Selected via ``trainer.v1.trainer_mode=remote_backend`` or the Hydra
``remote_backend=<name>`` group choice (mirrored onto the trainer_mode
by :func:`verl.trainer.main_ppo.main`).

The verl side is CPU-only on this path: all GPU compute (training,
sampling, log-prob, weight sync) happens in the plugin's remote client.
The trainer marshals batches through TransferQueue and forwards
requests to the backend. One driver-side :class:`RemoteBackend` is
built in ``__init__``; its
:meth:`~RemoteBackend.reconnect_handle` is passed to the forwarder
worker(s) and to every ``RolloutReplica`` so both re-attach instead of
building a second copy. Weight sync short-circuits in
:meth:`CheckpointEngineManager.update_weights` for backend
``"remote_backend"``.
"""

from __future__ import annotations

import logging
import os

import ray
from omegaconf import DictConfig, open_dict

from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.import_utils import import_external_libs

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("remote_backend")
class PPOTrainerRemoteBackend(PPOTrainerSync):
    """Sync PPO trainer for out-of-process RL backends.

    Payload shape, loss, and weight sync all live inside the plugin's
    ``RemoteBackend`` and its per-backend forwarder worker.
    """

    def __init__(self, config: DictConfig):
        self._backend_name: str = self._resolve_backend_name(config)

        # Give the plugin a chance to register its RemoteBackend / worker /
        # rollout replica before PPOTrainer.__init__, so role_worker_mapping
        # resolution below can call RemoteBackendRegistry.get_worker().
        import_external_libs(config.actor_rollout_ref.rollout.checkpoint_engine.get("custom_backend_module", None))
        rb_cfg = config.get("remote_backend")
        if rb_cfg is not None:
            import_external_libs(rb_cfg.get("custom_backend_module", None))

        # RemoteBackend is sync semantics only (no partial rollout, no
        # async drop); pin trainer_mode before base __init__ reads it.
        with open_dict(config.trainer.v1):
            config.trainer.v1.trainer_mode = "sync"

        self._backend: RemoteBackend = RemoteBackendRegistry.get(self._backend_name).from_config(config)
        self._backend_handle: dict = self._backend.reconnect_handle()

        super().__init__(config)

    # -- Trainer_base extension hooks -------------------------------------

    def _actor_rollout_wg_extra_kwargs(self) -> dict:
        return {"main_config": self.config, "backend_handle": self._backend_handle}

    def _llm_server_replica_init_kwargs(self) -> dict:
        return {"main_config": self.config, "backend_handle": self._backend_handle}

    def _checkpoint_engine_backend(self) -> str:
        return "remote_backend"

    def _init_resource_pool_mgr(self):
        """Install the plugin's forwarder as the ActorRollout worker and
        mark the pool CPU-only.

        The forwarder does no GPU work: it marshals batches and delegates
        compute to the plugin's remote client. Reserving a GPU here
        would double-book with the plugin's own actors and trip Ray's
        "Total available GPUs 0" placement check.
        """
        super()._init_resource_pool_mgr()
        self.resource_pool_manager.use_gpu = False

        worker_cls = RemoteBackendRegistry.get_worker(self._backend_name)
        if worker_cls is None:
            raise RuntimeError(
                f"Remote backend '{self._backend_name}' did not register an "
                "ActorRollout forwarder worker via RemoteBackendRegistry.register_worker."
            )

        actor_role = Role.ActorRolloutRef if need_reference_policy(self.config) else Role.ActorRollout
        self.role_worker_mapping[actor_role] = ray.remote(worker_cls)

        # Backends whose payload uses mesh dispatch break with n_workers > 1:
        # ONE_TO_ALL calls duplicate against the single backend, and
        # mesh-dispatched compute fragments the global batch across
        # forwarders that each forward the whole batch.
        if self._backend.requires_single_forwarder():
            n_gpus = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
            assert n_gpus == 1, (
                f"Backend '{self._backend_name}' requires a single forwarder worker "
                f"(trainer.n_gpus_per_node * trainer.nnodes == 1); got {n_gpus}. "
                "Scale the backend's own training_gpus / sampling_gpus for parallelism."
            )

    # -- Config resolution ------------------------------------------------

    @staticmethod
    def _resolve_backend_name(config: DictConfig) -> str:
        """Return the backend name from ``trainer.remote_backend`` or
        ``remote_backend.name``, mirroring the Hydra group choice onto
        the canonical ``trainer.remote_backend`` field.
        """
        name = config.trainer.get("remote_backend")
        if not name:
            rb_cfg = config.get("remote_backend")
            if rb_cfg is not None and rb_cfg.get("name"):
                name = rb_cfg.name
                with open_dict(config.trainer):
                    config.trainer.remote_backend = name
        if not name:
            raise ValueError(
                "PPOTrainerRemoteBackend requires either trainer.remote_backend=<name> "
                "or a remote_backend=<name> Hydra group choice."
            )
        return name

    # -- Lifecycle --------------------------------------------------------

    def fit(self, agent_loop_manager):
        try:
            super().fit(agent_loop_manager)
        finally:
            self._destroy_backend()

    def _destroy_backend(self):
        backend, self._backend = getattr(self, "_backend", None), None
        if backend is None:
            return
        try:
            import asyncio

            destroy = getattr(backend, "destroy", None)
            if destroy is None:
                return
            result = destroy()
            if asyncio.iscoroutine(result):
                asyncio.run(result)
        except Exception as exc:  # noqa: BLE001 -- shutdown path
            logger.warning("RemoteBackend.destroy raised %s; ignoring on shutdown.", exc)
