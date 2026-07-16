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
"""`RemoteBackend` ABC + `RemoteBackendRegistry`.

The ABC is intentionally minimal -- it only enforces the *lifecycle*
contract that verl's trainer side needs to know about. Compute/update op
signatures (``compute_log_prob``, ``update_actor``, ``generate``)
intentionally live on the concrete per-backend adapter and its matching
per-backend worker, not on the ABC, so different backends can shape
those calls however suits them (one backend for training and another
for sampling, different payload schemas, etc.) without growing this
base class.

What the ABC owns (what verl drives):

* Lifecycle: ``from_config`` (sole constructor, takes an optional
  ``handle=`` for re-attach) / ``reconnect_handle`` / ``destroy``.
* Weight sync + checkpoint: ``update_weights`` / ``save_checkpoint``
  (called from ``ONE_TO_ALL`` worker hooks).
* Parallelism contract: ``requires_single_forwarder`` (read by
  :class:`verl.remote_backend.trainer.RemoteBackendTrainer` to decide
  whether to assert ``n_gpus_per_node * nnodes == 1``).

What the ABC does NOT own: payload schemas, wire formats, loss-function
plumbing, compute/update method signatures -- those live entirely on
the concrete backend + its per-backend worker.

Registration model
------------------

Backends live in their own packages -- verl core carries none -- and
are wired in via the ``VERL_USE_EXTERNAL_MODULES`` hook. Users set::

    VERL_USE_EXTERNAL_MODULES=my_pkg.integrations.verl.register

That module's top level:

1. Imports the adapter module, which at class-definition time is
   decorated with ``@RemoteBackendRegistry.register("<name>")`` and thus
   inserts itself into the class registry as a side effect.
2. Calls :meth:`RemoteBackendRegistry.register_worker` with the
   backend's ActorRollout(Ref) forwarder worker class. The trainer
   ``main_ppo`` reads this back via
   :meth:`RemoteBackendRegistry.get_worker` to select
   ``actor_rollout_cls`` at bootstrap time, without hard-coding a
   per-backend if-branch.
3. Registers the rollout replica class with
   :class:`verl.workers.rollout.replica.RolloutReplicaRegistry` (which
   uses its own lazy-loader signature for vLLM/SGLang/... parity).
"""

from __future__ import annotations

import abc
from typing import Any, Callable

from omegaconf import DictConfig


class RemoteBackend(abc.ABC):
    """Out-of-process RL backend that owns its own GPUs.

    Created once on the driver by ``RemoteBackendTrainer`` (via
    ``from_config(main_config)``); re-attached inside every forwarder
    worker via ``from_config(main_config, handle=...)``.
    """

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

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
            main_config: the full verl config tree. Backend-specific knobs
                live under ``main_config.remote_backend.<name>``; backends
                MUST NOT read outside their own namespace plus the small set
                of standard fields under ``main_config.{trainer, data,
                actor_rollout_ref}``.
            handle: when supplied, re-attach to an existing backend
                instance described by a previous
                :meth:`reconnect_handle` (used by forwarder workers /
                rollout replicas that share the driver-side backend
                instead of creating a second one). When ``None``,
                create a fresh backend on the driver.
        """

    @abc.abstractmethod
    def reconnect_handle(self) -> dict[str, Any]:
        """A serializable handle that, when passed back to
        :meth:`from_config` as ``handle=...``, yields a reference to
        *this* backend.

        Typically contains a Ray actor handle and a small config blob.
        ``RemoteBackendTrainer`` puts this dict into ``wg_kwargs`` so each
        forwarder worker can re-attach.
        """

    @abc.abstractmethod
    def destroy(self) -> None:
        """Tear down the backend cleanly. Must be idempotent.

        Called from ``RemoteBackendTrainer.destroy()`` after ``fit()``.
        """

    # ------------------------------------------------------------------ #
    # Weight sync + checkpoint (called from ONE_TO_ALL worker hooks).
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    async def update_weights(self) -> dict[str, Any]:
        """Sync trained weights from the training engine to the rollout
        engine. May be a no-op for colocated backends.
        """

    @abc.abstractmethod
    async def save_checkpoint(self) -> dict[str, Any]:
        """Persist current model + optimizer state.

        ``async`` so the underlying RPC (typically a Ray actor call) can be
        awaited without blocking the forwarder's event loop.
        """

    # ------------------------------------------------------------------ #
    # Parallelism contract
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def requires_single_forwarder(self) -> bool:
        """Whether ``RemoteBackendTrainer`` should assert
        ``n_gpus_per_node * nnodes == 1`` and a single rollout replica.

        With more than one forwarder worker, ``ONE_TO_ALL`` calls
        (``save_checkpoint``, ``update_weights``, ``to``, ``set_loss_fn``)
        get duplicated against the single backend, and mesh-dispatched
        compute/update calls fragment the global batch across forwarders
        that each forward the whole batch downstream.

        Returning ``True`` enables the assert; returning ``False`` opts
        out (the backend takes responsibility for validating its own
        worker-group config).
        """


class RemoteBackendRegistry:
    """Process-wide registry of name -> (:class:`RemoteBackend` class,
    ActorRollout forwarder worker class).

    Backend classes register themselves via the
    ``@RemoteBackendRegistry.register(name)`` decorator at class
    definition time. Forwarder worker classes are registered
    imperatively by the same plugin's entry-point module, via
    :meth:`register_worker`; that keeps the decorator on the backend
    class simple, and lets the worker module retain its own eager
    imports without having to know about registry mechanics.

    There is intentionally no eager MODULES table that pre-imports every
    known adapter -- that would force the process to take on the
    transitive deps (vLLM, arctic-training, tinker, ...) of every
    backend even when only one is in use.
    """

    _backends: dict[str, type[RemoteBackend]] = {}
    _worker_loaders: dict[str, Callable[[], type]] = {}
    _resolved_workers: dict[str, type] = {}

    # -- Backend class registry -------------------------------------------

    @classmethod
    def register(cls, name: str) -> Callable[[type[RemoteBackend]], type[RemoteBackend]]:
        """Decorator: register the decorated class as backend ``name``.

        Duplicate registrations of the same name with the identical class
        object are a no-op (so a re-import of the plugin module during
        test teardown / hot-reload doesn't blow up); different classes
        under the same name raise, so the collision surfaces at import
        time.
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

    # -- ActorRollout forwarder worker registry ---------------------------

    @classmethod
    def register_worker(cls, name: str, loader: Callable[[], type]) -> None:
        """Register a lazy loader for the ActorRollout forwarder worker
        class matching backend ``name``.

        ``loader`` is a zero-arg callable returning the concrete worker
        class; it is invoked once on the driver at first
        :meth:`get_worker` and its result cached. Keeps this symmetric
        with :class:`verl.workers.rollout.replica.RolloutReplicaRegistry`
        (also lazy-loader) so an adapter plugin's ``register.py`` never
        forces an import of vLLM / DeepSpeed / tensordict just to wire a
        name into the registry.

        Duplicate registrations of the same name with the same loader
        object are a no-op; different loaders raise, so the collision
        surfaces at import time.
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
        """Return the ActorRollout forwarder worker class for ``name``,
        or ``None`` if the backend didn't register one (in which case
        ``main_ppo`` falls back to verl's stock ``ActorRolloutRefWorker``
        -- only correct for backends whose payload/loss shape matches
        the stock worker).
        """
        if name in cls._resolved_workers:
            return cls._resolved_workers[name]
        loader = cls._worker_loaders.get(name)
        if loader is None:
            return None
        cls._resolved_workers[name] = loader()
        return cls._resolved_workers[name]
