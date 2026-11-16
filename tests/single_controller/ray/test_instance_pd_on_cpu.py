# Copyright 2025 Individual Contributor: Zehan Zhu
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
"""Execute production PD methods with transport doubles, without importing vLLM.

AST extraction avoids importing the GPU engine, but executes the complete method
bodies from source. It is not a substitute for a live vLLM acceptance test.
"""

import ast
import asyncio
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl.single_controller.ray.instance_naming import build_vllm_server_prefix

ROOT = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout"


def method(filename, class_name, method_name, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if getattr(n, "name", None) == method_name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(ROOT / filename), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("instance_id", [None, "policy_a", "policy_b"])
def test_pd_created_servers_match_adapter_lookup(instance_id):
    registry = {}

    def spawn(**kwargs):
        server = SimpleNamespace(
            launch_server=SimpleNamespace(remote=AsyncMock()),
            set_pd_peer=SimpleNamespace(remote=AsyncMock()),
            get_server_address=SimpleNamespace(remote=AsyncMock(return_value=("127.0.0.1", 8000))),
        )
        registry[kwargs["actor_name"]] = server
        return server

    worker = SimpleNamespace(__ray_call__=SimpleNamespace(remote=AsyncMock(return_value=("node", "0", "127.0.0.1"))))
    prefix = lambda: build_vllm_server_prefix(instance_id)
    replica = SimpleNamespace(
        workers=[worker, worker],
        world_size=2,
        _prefill_tp=1,
        _decode_tp=1,
        _n_decode=1,
        replica_rank=0,
        name_suffix="",
        _decode_servers=[],
        config=SimpleNamespace(disaggregation=SimpleNamespace(transfer_backend="nixl", mooncake_protocol="tcp")),
        _collect_cuda_devices=lambda infos: "0",
        _build_kv_transfer_config=lambda **kw: {},
        _spawn_pd_server=spawn,
        _get_server_name_prefix=prefix,
    )
    launch = method(
        "vllm_pd_replica.py",
        "vLLMPDReplica",
        "launch_servers",
        {
            "asyncio": asyncio,
            "uuid": uuid,
            "is_torch_npu_available": lambda **kw: False,
            "get_free_port": lambda *args, **kw: (9000, SimpleNamespace(close=lambda: None)),
            "is_valid_ipv6_address": lambda addr: False,
            "logger": logging.getLogger(__name__),
        },
    )
    asyncio.run(launch(replica))
    lookup = method(
        "vllm_rollout.py",
        "ServerAdapter",
        "_ensure_server_handle",
        {
            "ray": SimpleNamespace(get_actor=registry.__getitem__),
        },
    )
    for role in ("prefill", "decode"):
        adapter = SimpleNamespace(
            _has_server=True,
            server_handle=None,
            _get_server_name_prefix=prefix,
            _pd_role=role,
            replica_rank=0,
            node_rank=0,
            _pd_server_index=0,
        )
        assert lookup(adapter)
        assert adapter.server_handle in registry.values()
