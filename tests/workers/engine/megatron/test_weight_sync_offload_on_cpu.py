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

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from verl.workers.engine.megatron import transformer_impl
from verl.workers.engine.megatron.transformer_impl import MegatronEngine
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets


def test_megatron_offloads_after_sglang_flushes_final_bucket(monkeypatch):
    backing = torch.tensor([1.0, 2.0])
    param = torch.nn.Parameter(torch.empty(1))
    param.data = backing[:1]
    tensor = param.detach()
    assert tensor._base is None
    assert tensor.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    engine = object.__new__(MegatronEngine)
    engine.module = object()
    engine.peft_cls = None
    engine.model_config = SimpleNamespace(lora={})
    engine.vanilla_bridge = True
    engine.bridge = SimpleNamespace(export_weights=lambda _module: iter([("weight", tensor)]))
    engine._qat_enabled = False
    engine._is_offload_param = True

    load = Mock()
    offload = Mock()
    monkeypatch.setattr(transformer_impl, "load_megatron_model_to_gpu", load)
    monkeypatch.setattr(transformer_impl, "offload_megatron_model_to_cpu", offload)

    async def _consume():
        weights, _ = engine.get_per_tensor_param()
        async for bucket in get_named_tensor_buckets(weights, bucket_bytes=1024):
            # SGLang yields its final bucket only after the source iterator has
            # ended. The tensor must remain valid until this consumer returns.
            offload.assert_not_called()
            assert bucket[0][0] == "weight"
            torch.testing.assert_close(bucket[0][1], tensor)

    asyncio.run(_consume())

    load.assert_called_once_with(engine.module, load_grad=False, load_frozen_params=True)
    offload.assert_not_called()

    engine.finalize_weight_sync()
    offload.assert_called_once_with(engine.module)
