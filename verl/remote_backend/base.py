"""`RemoteBackend` ABC + `RemoteBackendRegistry`.

The ABC is intentionally minimal: every method is abstract, so each
backend declares its own implementation explicitly. There are no
default behaviours to silently inherit.

What the abstraction is responsible for (what verl drives):

* Lifecycle: ``from_config`` (sole constructor, takes an optional
  ``handle=`` for re-attach) / ``reconnect_handle`` / ``destroy``.
* Core RL ops: ``compute_log_prob`` / ``update_actor`` /
  ``generate`` / ``update_weights`` / ``save_checkpoint``.
* Parallelism contract: ``requires_single_forwarder`` (read by
  :class:`verl.remote_backend.trainer.RemoteBackendTrainer` to decide
  whether to assert ``n_gpus_per_node × nnodes == 1``).

What the abstraction is NOT responsible for: payload schemas, wire
formats, loss-function plumbing — those live entirely inside each
backend.
"""

from __future__ import annotations

import abc
from typing import Any, Callable

import torch
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
    # Core RL operations called by `RemoteBackendActorRolloutRefWorker`
    # ------------------------------------------------------------------ #
    # Inputs are verl `TensorDict`s; outputs are plain Python dicts that
    # the worker wraps / augments with FlopsCounter MFU before returning a
    # `TensorDict` to the trainer. Backends pick their own wire format
    # below this layer.

    @abc.abstractmethod
    async def compute_log_prob(
        self,
        data,
        *,
        ref: bool,
        calculate_entropy: bool,
        rollout_n: int,
        temperature,
        pad_token_id: int,
    ) -> dict[str, Any]:
        """Forward-only pass; either the actor (``ref=False``) or the
        reference model (``ref=True``).

        ``async`` so the underlying RPC (typically a Ray actor call) can be
        awaited without blocking the forwarder's event loop.

        Returns:
            ``{"model_output": {"log_probs": Tensor,
            "entropy": Tensor?}, "metrics": dict}``. The generic worker
            reconstructs nested-jagged tensors from
            ``model_output["log_probs"]`` / ``["entropy"]`` and attaches
            MFU.
        """

    @abc.abstractmethod
    async def update_actor(
        self,
        data,
        *,
        actor_config,
        pad_token_id: int,
        rollout_n: int,
        temperature,
    ) -> dict[str, Any]:
        """Forward-backward + optimizer step on a global batch.

        ``async`` so the underlying RPC (typically a Ray actor call) can be
        awaited without blocking the forwarder's event loop.

        Returns:
            ``{"loss": float|list[float], "metrics": dict,
            "global_token_num": list[int]}``. ``metrics["loss"]`` may
            also be present; the generic worker computes MFU from
            ``global_token_num`` and aggregates metrics.
        """

    @abc.abstractmethod
    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
    ) -> list:
        """Sample a rollout. Called from the rollout server that owns
        this backend (e.g. ``ArcticLLMEngine``)."""

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
        calls (``compute_log_prob``, ``update_actor``) fragment the
        global batch across forwarders that each forward the whole
        batch downstream.

        Returning ``True`` enables the assert; returning ``False`` opts
        out (the backend takes responsibility for validating its own
        worker-group config).
        """


class RemoteBackendRegistry:
    """Process-wide registry of name → :class:`RemoteBackend` class.

    Backends register themselves at import time::

        @RemoteBackendRegistry.register("arctic")
        class ArcticBackend(RemoteBackend): ...

    Callers don't need to know which module to import for a given
    backend name — :meth:`create` and :meth:`get` lazy-import the
    adapter module listed in :attr:`MODULES`. To plug in a new
    backend (e.g. ``"tinker"``), add an entry to ``MODULES`` and
    decorate the class with ``@RemoteBackendRegistry.register("tinker")``.
    """

    _backends: dict[str, type[RemoteBackend]] = {}

    # Backend name → dotted module path to import on first use. Ray
    # child procs (and the driver) only need to know the name; the
    # registry resolves to the right adapter module without forcing
    # `main_ppo.py` to grow a per-backend `import` line.
    MODULES: dict[str, str] = {
        "arctic": "verl.trainer.ppo.arctic_rl_client",
    }

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
        """Resolve ``name`` to a registered backend class, lazy-importing
        the adapter module if needed."""
        if name not in cls._backends:
            module_path = cls.MODULES.get(name)
            if module_path is None:
                raise KeyError(
                    f"Unknown remote backend '{name}'. Registered: "
                    f"{sorted(cls._backends)}. Known modules: "
                    f"{sorted(cls.MODULES)}. Either add an entry to "
                    f"RemoteBackendRegistry.MODULES or import the module "
                    "that registers the backend before calling get()."
                )
            import importlib
            importlib.import_module(module_path)
        return cls._backends[name]

    @classmethod
    def create(cls, name: str, main_config: DictConfig) -> RemoteBackend:
        return cls.get(name).from_config(main_config)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._backends)
