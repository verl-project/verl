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
"""RemoteBackend abstraction: pluggable out-of-process RL backends.

See :mod:`verl.remote_backend.base` for the ABC and registry, and
:class:`verl.trainer.ppo.v1.trainer_remote_backend.PPOTrainerRemoteBackend`
for the V1 trainer that drives such backends. Concrete backends
(training + sampling implementations) live outside verl core and are
loaded via ``VERL_USE_EXTERNAL_MODULES``.
"""

from .base import RemoteBackend, RemoteBackendRegistry

__all__ = ["RemoteBackend", "RemoteBackendRegistry"]
