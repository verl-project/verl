"""Generic remote-backend abstraction for verl.

Lets verl drive an out-of-process RL backend (training + rollout +
log-prob + checkpoint) that owns its own GPUs. Verl talks to a CPU-only
forwarder worker group; the forwarder forwards every dispatched call to
a :class:`RemoteBackend` implementation behind a Ray actor (or any other
RPC the backend prefers).

Pieces:

* :class:`RemoteBackend` (``base.py``) — minimal ABC: lifecycle
  (``from_config`` / ``reconnect_handle`` / ``destroy``) + weight sync
  + checkpoint + a single-forwarder parallelism flag. Compute/update
  op signatures intentionally live on the per-backend adapter, not
  here.
* :class:`RemoteBackendRegistry` (``base.py``) — name → class lookup
  populated by explicit adapter imports (no lazy MODULES table).
* :class:`RemoteBackendTrainer` (``trainer.py``) — `RayPPOTrainer`
  subclass that creates the backend on the driver and threads its
  reconnect handle to every worker.
* ``workers/<backend_name>/`` — per-backend forwarder worker (Arctic
  ships as ``workers/arctic_rl/`` -> :class:`ArcticRLActorRolloutRefWorker`).
* ``worker_utils.py`` — small generic tensor / metric helpers shared
  across per-backend workers.

Adapter modules (the concrete :class:`RemoteBackend` implementations)
live under :mod:`verl.workers.remote_client`. See
:mod:`verl.workers.remote_client.arctic_rl_client` for a reference adapter.
"""

from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry

__all__ = ["RemoteBackend", "RemoteBackendRegistry"]
