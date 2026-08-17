# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl.checkpoint_engine.sglang_hccl_checkpoint_engine import (
    BACKEND_NAME,
    SGLangHCCLCheckpointEngineManager,
    _post_json,
    deranged_rollout_indices,
    iter_weight_buckets,
)


def test_weight_buckets_preserve_complete_tensors_and_order():
    weights = [
        ("a", torch.zeros(4, dtype=torch.uint8)),
        ("b", torch.zeros(6, dtype=torch.uint8)),
        ("c", torch.zeros(3, dtype=torch.uint8)),
    ]

    buckets = list(iter_weight_buckets(weights, bucket_size=10))

    assert [[name for name, _ in bucket] for bucket in buckets] == [["a", "b"], ["c"]]


def test_weight_larger_than_bucket_is_rejected():
    weights = [("embedding", torch.zeros(11, dtype=torch.uint8))]

    with pytest.raises(ValueError, match="embedding"):
        list(iter_weight_buckets(weights, bucket_size=10))


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_deranged_rollout_indices_avoid_colocated_devices(world_size):
    rollout_indices = deranged_rollout_indices(world_size)

    assert sorted(rollout_indices) == list(range(world_size))
    assert all(trainer_rank != rollout_rank for trainer_rank, rollout_rank in enumerate(rollout_indices))
    if world_size == 4:
        assert rollout_indices == [1, 2, 3, 0]


def test_deranged_rollout_indices_require_multiple_ranks():
    with pytest.raises(ValueError, match="at least two ranks"):
        deranged_rollout_indices(1)


def test_post_json_preserves_error_response(monkeypatch):
    class Response:
        ok = False
        status_code = 502
        text = "upstream scheduler unavailable"

        @staticmethod
        def json():
            return {"message": "group initialization failed"}

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            assert self.trust_env is False
            return Response()

    monkeypatch.setattr("verl.checkpoint_engine.sglang_hccl_checkpoint_engine.requests.Session", Session)

    with pytest.raises(RuntimeError, match="HTTP 502: group initialization failed"):
        _post_json("http://127.0.0.1:30000", "init_weights_update_group", {}, 1)


def test_post_json_accepts_null_success_response(monkeypatch):
    class Response:
        ok = True
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return None

    class Session:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            assert self.trust_env is False
            return Response()

    monkeypatch.setattr("verl.checkpoint_engine.sglang_hccl_checkpoint_engine.requests.Session", Session)

    assert _post_json("http://127.0.0.1:30000", "resume_memory_occupation", {}, 1) == {}


def test_manager_builds_paired_hccl_groups(monkeypatch):
    class ActorWorkerGroup:
        world_size = 4

        def __init__(self):
            self.calls = []

        def execute_checkpoint_engine(self, method=None, *args, **kwargs):
            self.calls.append((method, args, kwargs))
            if method == ["prepare"] * self.world_size:
                return [
                    {"master_address": f"10.0.0.{rank + 1}", "master_port": 30000 + rank}
                    for rank in range(self.world_size)
                ]
            return [None] * self.world_size

    actor_wg = ActorWorkerGroup()
    replicas = [SimpleNamespace(server_address=f"127.0.0.1:{31000 + rank}", world_size=1) for rank in range(4)]
    config = SimpleNamespace(
        backend=BACKEND_NAME,
        custom_backend_module=None,
        engine_kwargs={BACKEND_NAME: {"request_timeout": 5}},
    )
    posts = []
    monkeypatch.setattr("verl.checkpoint_engine.sglang_hccl_checkpoint_engine.ray.get", lambda value: value)
    monkeypatch.setattr(
        "verl.checkpoint_engine.sglang_hccl_checkpoint_engine._post_json",
        lambda url, endpoint, payload, timeout: posts.append((url, endpoint, payload, timeout)) or {"success": True},
    )

    manager = SGLangHCCLCheckpointEngineManager(config=config, actor_wg=actor_wg, replicas=replicas)
    asyncio.run(manager._initialize_process_group())

    assert actor_wg.calls[1][1][0] == [
        ["http://127.0.0.1:31001"],
        ["http://127.0.0.1:31002"],
        ["http://127.0.0.1:31003"],
        ["http://127.0.0.1:31000"],
    ]
    assert actor_wg.calls[2][2] == {
        "rank": [0, 0, 0, 0],
        "world_size": [2, 2, 2, 2],
        "master_address": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        "master_port": [30000, 30001, 30002, 30003],
        "group_name": ["verl_sglang_hccl_0", "verl_sglang_hccl_1", "verl_sglang_hccl_2", "verl_sglang_hccl_3"],
    }
    assert sorted((url, payload["master_port"], payload["group_name"]) for url, _, payload, _ in posts) == [
        ("http://127.0.0.1:31000", 30003, "verl_sglang_hccl_3"),
        ("http://127.0.0.1:31001", 30000, "verl_sglang_hccl_0"),
        ("http://127.0.0.1:31002", 30001, "verl_sglang_hccl_1"),
        ("http://127.0.0.1:31003", 30002, "verl_sglang_hccl_2"),
    ]
