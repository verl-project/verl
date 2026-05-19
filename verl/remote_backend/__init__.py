"""Generic remote-backend abstraction for verl.

Lets verl drive an out-of-process RL backend (training + rollout +
log-prob + checkpoint) that owns its own GPUs. Verl talks to a CPU-only
forwarder worker group; the forwarder forwards every dispatched call to
a :class:`RemoteBackend` implementation behind a Ray actor (or any other
RPC the backend prefers).

Pieces:

* :class:`RemoteBackend` (``base.py``) — the all-abstract contract
  every backend implements.
* :class:`RemoteBackendRegistry` (``base.py``) — name → class lookup so
  ``trainer.remote_backend="<name>"`` resolves to a concrete adapter.
* :class:`RemoteBackendTrainer` (``trainer.py``) — `RayPPOTrainer` subclass
  that creates the backend on the driver and threads its reconnect handle
  to every worker.
* :class:`RemoteBackendActorRolloutRefWorker` (``worker.py``) — the
  backend-agnostic CPU forwarder.
* ``worker_utils.py`` — small generic tensor / metric helpers shared by
  the forwarder and backend adapters.

See :mod:`verl.trainer.ppo.arctic_rl_client` for a reference adapter.
"""

from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry

__all__ = ["RemoteBackend", "RemoteBackendRegistry"]
