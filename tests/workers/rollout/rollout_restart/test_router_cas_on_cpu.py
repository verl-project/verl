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
"""CPU tests for the load-balancer CAS pair (rollout-restart ABA defense).

Long-term-care minimum: late operations issued against a replaced actor
must never act on its replacement. Real Ray actors so ``ActorHandle``
identity survives serialization round-trips — the exact hazard the CAS
defends against.
"""

import pytest
import ray

from verl.workers.rollout.router import GlobalRequestLoadBalancer


@pytest.fixture(scope="module")
def ray_ctx():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@ray.remote
class _DummyServer:
    def ping(self):
        return "pong"


def _make_lb(servers: dict) -> "ray.actor.ActorHandle":
    return ray.remote(GlobalRequestLoadBalancer).remote(servers=servers)


def test_remove_server_if_current_rejects_foreign_handle(ray_ctx):
    old_server, replacement = _DummyServer.remote(), _DummyServer.remote()
    lb = _make_lb({"s1": replacement})
    # A late removal issued against the OLD actor at the same address must not
    # remove the replacement.
    assert ray.get(lb.remove_server_if_current.remote("s1", old_server)) is False
    assert ray.get(lb.get_all_servers.remote()) == ["s1"]


def test_release_server_with_expected_handle_ignores_late_release(ray_ctx):
    old_server, replacement = _DummyServer.remote(), _DummyServer.remote()
    lb = _make_lb({"s1": replacement})
    ray.get(lb.acquire_server.remote("req-1"))  # replacement now has inflight 1
    # Late release from the replaced actor: must not decrement the replacement.
    ray.get(lb.release_server.remote("s1", expected_handle=old_server))
    status = ray.get(lb.get_status.remote())
    assert status["servers"]["s1"] == 1

    # Release with the correct handle still works.
    ray.get(lb.release_server.remote("s1", expected_handle=replacement))
    status = ray.get(lb.get_status.remote())
    assert status["servers"]["s1"] == 0
