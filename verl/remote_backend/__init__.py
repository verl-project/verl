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
"""Generic remote-backend abstraction for verl.

Lets verl drive an out-of-process RL backend (training + rollout +
log-prob + checkpoint) that owns its own GPUs. Verl talks to a CPU-only
forwarder worker group; the forwarder forwards every dispatched call to
a :class:`RemoteBackend` implementation behind a Ray actor (or any other
RPC the backend prefers).

Pieces:

* :class:`RemoteBackend` (``base.py``) -- minimal ABC: lifecycle
  (``from_config`` / ``reconnect_handle`` / ``destroy``) + weight sync
  + checkpoint + a single-forwarder parallelism flag. Compute/update
  op signatures intentionally live on the per-backend adapter, not
  here.
* :class:`RemoteBackendRegistry` (``base.py``) -- name -> backend class
  registry (populated by the adapter's ``@register`` decorator) plus a
  parallel ``register_worker`` / ``get_worker`` slot for the matching
  ActorRollout forwarder worker class. Populated by adapter packages
  via the ``VERL_USE_EXTERNAL_MODULES`` hook.
* :class:`RemoteBackendTrainer` (``trainer.py``) -- ``RayPPOTrainer``
  subclass that creates the backend on the driver and threads its
  reconnect handle to every worker.
* ``worker_utils.py`` -- small generic tensor / metric helpers shared
  across per-backend workers, which live in the adapter packages.

Verl-core carries no concrete backends. The reference implementation for
the ABC lives in ``arctic_platform.integrations.verl`` (Arctic RL);
plug it in with
``VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register``.
"""

from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry

__all__ = ["RemoteBackend", "RemoteBackendRegistry"]
