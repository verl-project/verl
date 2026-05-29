"""`RemoteBackend` ABC + `RemoteBackendRegistry`.

The ABC is intentionally minimal — it only enforces the *lifecycle*
contract that verl's trainer side needs to know about. Compute/update
op signatures (``compute_log_prob``, ``update_actor``, ``generate``)
intentionally live on the concrete per-backend adapter and its
matching per-backend worker, not on the ABC, so different backends can
shape those calls however suits them (one backend for training and
another for sampling, different payload schemas, etc.) without growing
this base class.

What the ABC owns (what verl drives):

* Lifecycle: ``from_config`` (sole constructor, takes an optional
  ``handle=`` for re-attach) / ``reconnect_handle`` / ``destroy``.
* Weight sync + checkpoint: ``update_weights`` / ``save_checkpoint``
  (called from ``ONE_TO_ALL`` worker hooks).
* Parallelism contract: ``requires_single_forwarder`` (read by
  :class:`verl.remote_backend.trainer.RemoteBackendTrainer` to decide
  whether to assert ``n_gpus_per_node × nnodes == 1``).

What the ABC does NOT own: payload schemas, wire formats, loss-function
plumbing, compute/update method signatures — those live entirely on the
concrete backend + its per-backend worker.
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
    ) -> "RemoteBackend":
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
        ``n_gpus_per_node × nnodes == 1`` and a single rollout replica.

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
    """Process-wide registry of name → :class:`RemoteBackend` class.

    Registration is explicit and happens when the user (or their entry
    script) imports the adapter module they want to use::

        # in main_ppo.py, conditioned on `trainer.remote_backend == "arctic"`:
        from verl.workers.remote_client import arctic_rl  # noqa: F401

        # arctic_rl decorates its class with
        # @RemoteBackendRegistry.register("arctic"), so by import time the
        # name is available to `get()` / `create()`.

    There is intentionally no eager `MODULES` table that pre-imports
    every known adapter — that would force the process to take on the
    transitive deps (vLLM, arctic-training, tinker, ...) of every
    backend even when only one is in use.
    """

    _backends: dict[str, type[RemoteBackend]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[RemoteBackend]], type[RemoteBackend]]:
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
                f"{sorted(cls._backends)}. Import the adapter module "
                "(e.g. `from verl.workers.remote_client import arctic_rl`) "
                "before calling `get()`."
            )
        return cls._backends[name]

    @classmethod
    def create(cls, name: str, main_config: DictConfig) -> RemoteBackend:
        return cls.get(name).from_config(main_config)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._backends)
