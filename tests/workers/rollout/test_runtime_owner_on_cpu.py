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
"""Exercise production rollout actor ownership without a GPU or vLLM installation."""

import subprocess
import sys


def test_runtime_server_follows_controller_lifetime_and_preserves_name_owner():
    code = r"""
import asyncio
import os
import sys
import time
from types import ModuleType, SimpleNamespace
import ray
from ray.job_config import JobConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.workers.config import RolloutConfig
from verl.workers.rollout import vllm_replica
from verl.workers.rollout.runtime import VLLMServerLauncher, runtime_identity

class StubServer:
    async def ping(self):
        return os.getpid()

class StubLauncher(VLLMServerLauncher):
    def __init__(self, expected):
        # Only replace the heavy backend import. Production launcher, replica,
        # Ray scheduling, actor handles and ownership all run unchanged.
        backend = ModuleType('verl.workers.rollout.vllm_rollout.vllm_async_server')
        backend.vLLMHttpServer = StubServer
        sys.modules[backend.__name__] = backend
        super().__init__(expected)

@ray.remote(num_cpus=0)
class Controller:
    async def start(self):
        vllm_replica.VLLMServerLauncher = StubLauncher
        config = RolloutConfig(name='vllm', runtime={'py_executable': sys.executable})
        self.replica = vllm_replica.vLLMReplica(0, config, SimpleNamespace(), gpus_per_node=1)
        self.replica.runtime_env = {'py_executable': sys.executable}
        self.replica.runtime_expected = runtime_identity()
        self.options = {
            'name': 'isolated-runtime-owner-test', 'num_cpus': 0,
            'runtime_env': {'env_vars': {}},
            'scheduling_strategy': NodeAffinitySchedulingStrategy(
                ray.get_runtime_context().get_node_id(), soft=False),
        }
        server = await self.replica._create_server(dict(self.options), {})
        return server, self.replica._server_launchers[0]

    async def collide(self):
        try:
            await self.replica._create_server(dict(self.options), {})
        except ray.exceptions.RayTaskError:
            return len(self.replica._server_launchers)
        raise AssertionError('duplicate server name was accepted')

ray.init(num_cpus=2, include_dashboard=False, namespace='verl-runtime-owner-test',
         job_config=JobConfig(default_actor_lifetime='detached'))
try:
    controller = Controller.options(lifetime='non_detached').remote()
    server, launcher = ray.get(controller.start.remote(), timeout=90)
    pid = ray.get(server.ping.remote(), timeout=15)
    assert ray.get(controller.collide.remote(), timeout=30) == 1
    assert ray.get(server.ping.remote(), timeout=15) == pid
    # Retain borrowed handles while the training controller exits: both the
    # launcher and native server must die, even with a detached job default.
    ray.kill(controller, no_restart=True)
    pending = [server, launcher]
    deadline = time.monotonic() + 20
    while pending and time.monotonic() < deadline:
        for actor in list(pending):
            try:
                ref = actor.ping.remote() if actor is server else actor.create.remote({}, {})
                ray.get(ref, timeout=2)
            except ray.exceptions.RayActorError:
                pending.remove(actor)
            except (ray.exceptions.GetTimeoutError, ray.exceptions.RayTaskError):
                pass
        time.sleep(0.1)
    assert not pending, 'rollout actors outlived the training controller'
    print('OWNER_TEARDOWN_PASSED')
finally:
    ray.shutdown()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OWNER_TEARDOWN_PASSED" in result.stdout
