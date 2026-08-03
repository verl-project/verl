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

import logging
import os
import tempfile
import uuid
from contextlib import suppress
from unittest.mock import patch

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

logger = logging.getLogger(__name__)


class RaceFreeMultiprocExecutor(MultiprocExecutor):
    """Use a unique FileStore rendezvous for vLLM's single-node workers.

    vLLM's ``MultiprocExecutor`` normally reserves an unused TCP port, releases
    it, and lets ``TCPStore`` bind it after the workers have spawned. Concurrent
    rollout engines can select the same port during that gap, causing one of
    them to fail with ``EADDRINUSE``.

    Replace only that rendezvous method with a unique file URI. vLLM passes the
    executor class to its spawned EngineCore process, so the override runs in
    the process that constructs ``MultiprocExecutor``.
    """

    _verl_rendezvous_path: str | None = None

    def _init_executor(self) -> None:
        def get_file_init_method(_ip: str, _port: int) -> str:
            path = os.path.join(tempfile.gettempdir(), f"verl-vllm-dist-{os.getpid()}-{uuid.uuid4().hex}")
            self._verl_rendezvous_path = path
            init_method = f"file://{path}"
            logger.info("Using race-free vLLM worker rendezvous: %s", init_method)
            return init_method

        # MultiprocExecutor resolves this name from its defining module. Keep
        # the override active only while the base initializer creates workers.
        with patch.object(multiproc_executor, "get_distributed_init_method", get_file_init_method):
            super()._init_executor()

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            if path := self._verl_rendezvous_path:
                with suppress(OSError):
                    os.unlink(path)
                self._verl_rendezvous_path = None
