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
"""
Test that the _control_method decorator used by vLLMHttpServer schedules
control methods on a dedicated concurrency group to prevent deadlock.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_control_concurrency_on_cpu.py -v
"""

import asyncio
import sys

import pytest

ray = pytest.importorskip("ray")
ray_cloudpickle = pytest.importorskip("ray.cloudpickle")
pytest.importorskip("vllm")

from verl.workers.rollout.replica import CONTROL_METHOD_CONCURRENCY  # noqa: E402
from verl.workers.rollout.vllm_rollout.vllm_async_server import _control_method  # noqa: E402

GENERATE_CONCURRENCY = 8
RAY_TIMEOUT = 30


class _FakeHttpServer:
    def __init__(self):
        # Required by _control_method.
        self._default_asyncio_loop = asyncio.get_running_loop()

        self._num_enqueued_generations = 0
        self._all_generations_enqueued = asyncio.Event()
        self._release_generations = asyncio.Event()

    @_control_method
    async def release_generations(self):
        await self._all_generations_enqueued.wait()

        assert asyncio.get_running_loop() is self._default_asyncio_loop
        self._release_generations.set()

    async def generate(self):
        assert asyncio.get_running_loop() is self._default_asyncio_loop

        self._num_enqueued_generations += 1
        if self._num_enqueued_generations == GENERATE_CONCURRENCY:
            self._all_generations_enqueued.set()

        await self._release_generations.wait()


@pytest.fixture(scope="module", autouse=True)
def serialize_test_module_by_value():
    """Tells Ray to serialize helpers in this file (e.g. _FakeHttpServer) by
    value to avoid import errors in the actor process."""

    test_module = sys.modules[__name__]
    ray_cloudpickle.register_pickle_by_value(test_module)
    yield
    ray_cloudpickle.unregister_pickle_by_value(test_module)


def test_control_rpc_is_not_starved_by_generate():
    """Test the _control_method decorator using _FakeHttpServer."""

    server = None

    try:
        server_actor_class = ray.remote(
            concurrency_groups={"control": CONTROL_METHOD_CONCURRENCY},
        )(_FakeHttpServer)
        server = server_actor_class.options(max_concurrency=GENERATE_CONCURRENCY).remote()

        # Saturate server with GENERATE_CONCURRENCY requests.
        generation_refs = [server.generate.remote() for _ in range(GENERATE_CONCURRENCY)]

        # Verify that release_generations() is not blocked and that it runs on the
        # default asyncio loop.
        ray.get(server.release_generations.remote(), timeout=RAY_TIMEOUT)

        # Verify all generation_refs are fulfilled.
        ray.get(generation_refs, timeout=RAY_TIMEOUT)

    finally:
        if server is not None:
            ray.kill(server)
