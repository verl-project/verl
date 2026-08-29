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
"""``RemoteBackend`` ABC + registry for out-of-process RL backends.

The ABC only defines the lifecycle contract verl drives:
``from_config`` / ``reconnect_handle`` / ``destroy``, plus
``update_weights`` / ``save_checkpoint`` and a ``requires_single_forwarder``
parallelism assertion. Compute/update op signatures
(``compute_log_prob``, ``update_actor``, ``generate``) live on the concrete
adapter and its matching forwarder worker so backends can shape payloads
however they need without growing this class.

Backends live in their own packages and are wired in via
``VERL_USE_EXTERNAL_MODULES=<pkg>.integrations.verl.register``. That
module registers:

1. The ``RemoteBackend`` subclass, via
   ``@RemoteBackendRegistry.register("<name>")`` at class definition time.
2. The ActorRollout forwarder worker class, via
   :meth:`RemoteBackendRegistry.register_worker` (lazy loader; read back
   from ``main_ppo`` at bootstrap).
3. The rollout replica class, via
   :class:`verl.workers.rollout.replica.RolloutReplicaRegistry`.
"""

from __future__ import annotations

import abc
from typing import Any, Callable

from omegaconf import DictConfig


class RemoteBackend(abc.ABC):
    """Out-of-process RL backend that owns its own GPUs.

    Created once on the driver by ``PPOTrainerRemoteBackend`` via
    ``from_config(main_config)``; re-attached inside every forwarder
    worker via ``from_config(main_config, handle=...)``.
    """

    @classmethod
    @abc.abstractmethod
    def from_config(
        cls,
        main_config: DictConfig,
        *,
        handle: dict[str, Any] | None = None,
    ) -> RemoteBackend:
        """Sole public constructor.

        Args:
            main_config: full verl config tree. Backend-specific knobs
                live under ``main_config.remote_backend.<name>``.
            handle: when supplied, re-attach to an existing backend
                described by a previous :meth:`reconnect_handle` (used by
                forwarder workers / rollout replicas that share the
                driver-side instance). When ``None``, create fresh.
        """

    @abc.abstractmethod
    def reconnect_handle(self) -> dict[str, Any]:
        """Serializable handle that, passed back to :meth:`from_config`
        as ``handle=...``, yields a reference to this backend.

        ``PPOTrainerRemoteBackend`` puts this into ``wg_kwargs`` so each
        forwarder worker can re-attach.
        """

    @abc.abstractmethod
    def destroy(self) -> None:
        """Tear down cleanly. Must be idempotent."""

    @abc.abstractmethod
    async def update_weights(self) -> dict[str, Any]:
        """Sync trained weights from training to rollout engine.

        No-op for colocated backends.
        """

    @abc.abstractmethod
    async def save_checkpoint(self) -> dict[str, Any]:
        """Persist current model + optimizer state."""

    @abc.abstractmethod
    def requires_single_forwarder(self) -> bool:
        """Whether ``PPOTrainerRemoteBackend`` should assert
        ``n_gpus_per_node * nnodes == 1`` and a single rollout replica.

        With more than one forwarder, ``ONE_TO_ALL`` hooks
        (``save_checkpoint``, ``update_weights``) duplicate against the
        single backend and mesh-dispatched compute calls fragment the
        global batch. Return ``True`` to enable the assert; ``False``
        opts out (backend takes responsibility for validating its own
        worker-group config).
        """


class RemoteBackendRegistry:
    """Process-wide registry: name -> (backend class, forwarder worker class).

    Backend classes register themselves via the
    ``@RemoteBackendRegistry.register(name)`` decorator at class
    definition time. Forwarder workers register imperatively via
    :meth:`register_worker` with a lazy loader, so an adapter plugin's
    ``register.py`` doesn't force an import of vLLM / DeepSpeed /
    tensordict just to wire a name.
    """

    _backends: dict[str, type[RemoteBackend]] = {}
    _worker_loaders: dict[str, Callable[[], type]] = {}
    _resolved_workers: dict[str, type] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[RemoteBackend]], type[RemoteBackend]]:
        """Decorator: register the decorated class as backend ``name``.

        Re-registering the same class object is a no-op (tolerates
        module re-import); a different class under an already-taken name
        raises.
        """

        def _decorator(backend_cls: type[RemoteBackend]) -> type[RemoteBackend]:
            existing = cls._backends.get(name)
            if existing is not None and existing is not backend_cls:
                raise ValueError(
                    f"Remote backend name '{name}' is already registered to "
                    f"{existing!r}; cannot re-register to {backend_cls!r}."
                )
            cls._backends[name] = backend_cls
            return backend_cls

        return _decorator

    @classmethod
    def get(cls, name: str) -> type[RemoteBackend]:
        if name not in cls._backends:
            raise KeyError(
                f"Unknown remote backend '{name}'. Registered: "
                f"{sorted(cls._backends)}. Wire the adapter package in via "
                "VERL_USE_EXTERNAL_MODULES=<pkg>.integrations.verl.register "
                "before starting verl."
            )
        return cls._backends[name]

    @classmethod
    def create(cls, name: str, main_config: DictConfig) -> RemoteBackend:
        return cls.get(name).from_config(main_config)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._backends)

    @classmethod
    def register_worker(cls, name: str, loader: Callable[[], type]) -> None:
        """Register a lazy loader for the ActorRollout forwarder class for
        backend ``name``.

        ``loader`` is a zero-arg callable returning the concrete worker
        class; invoked once at first :meth:`get_worker` and cached.
        """
        existing = cls._worker_loaders.get(name)
        if existing is not None and existing is not loader:
            raise ValueError(
                f"Remote backend '{name}' worker loader already registered to "
                f"{existing!r}; cannot re-register to {loader!r}."
            )
        cls._worker_loaders[name] = loader

    @classmethod
    def get_worker(cls, name: str) -> type | None:
        """Return the ActorRollout forwarder class for ``name``, or ``None``
        if the backend didn't register one (in which case ``main_ppo``
        falls back to verl's stock ``ActorRolloutRefWorker`` -- only
        correct for backends whose payload/loss shape matches).
        """
        if name in cls._resolved_workers:
            return cls._resolved_workers[name]
        loader = cls._worker_loaders.get(name)
        if loader is None:
            return None
        cls._resolved_workers[name] = loader()
        return cls._resolved_workers[name]
