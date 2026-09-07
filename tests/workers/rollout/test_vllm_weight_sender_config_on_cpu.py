# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")


@pytest.mark.asyncio
async def test_update_weights_forwards_gc_setting_to_sender(monkeypatch):
    from verl.workers.rollout.vllm_rollout import vllm_rollout

    sender_args = {}

    class FakeSender:
        def __init__(self, **kwargs):
            sender_args.update(kwargs)

        async def async_send_weights(self, weights):
            assert list(weights) == []

    async def execute_method(*args, **kwargs):
        return None

    adapter = SimpleNamespace(
        config=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(
                update_weights_bucket_megabytes=128,
                gc_on_weight_transfer_cleanup=1,
            )
        ),
        zmq_handle="tcp://unused",
        use_shm=True,
        _has_server=False,
        replica_rank=1,
        rollout_rank=1,
        _execute_method=execute_method,
    )
    monkeypatch.setattr(vllm_rollout, "BucketedWeightSender", FakeSender)

    await vllm_rollout.ServerAdapter.update_weights(adapter, iter(()), gc_diagnostics=True)

    assert sender_args == {
        "zmq_handle": "tcp://unused",
        "bucket_size_mb": 128,
        "use_shm": True,
        "gc_on_cleanup": 1,
        "gc_diagnostics": True,
    }
