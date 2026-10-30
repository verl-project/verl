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
"""FSDP dtype defaults must reach both wrapper policies, autocast and loss scaling."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl.workers.config import FSDPEngineConfig
from verl.workers.engine.fsdp import transformer_impl


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
@pytest.mark.parametrize(
    "dtype,mixed_precision,expected_param,expected_reduce,expected_buffer",
    [
        ("bfloat16", None, torch.bfloat16, torch.float32, torch.float32),
        ("float16", None, torch.float16, torch.float32, torch.float32),
        ("fp16", None, torch.float16, torch.float32, torch.float32),
        ("float32", None, torch.float32, torch.float32, torch.float32),
        ("float16", {}, torch.float16, torch.float32, torch.float32),
        ("float16", {"reduce_dtype": "bf16"}, torch.float16, torch.bfloat16, torch.float32),
        ("float16", {"buffer_dtype": "bf16"}, torch.float16, torch.float32, torch.bfloat16),
        ("float16", {"param_dtype": "bf16"}, torch.bfloat16, torch.float32, torch.float32),
        ("bfloat16", {"param_dtype": "fp16"}, torch.float16, torch.float32, torch.float32),
        ("bfloat16", {"param_dtype": "fp32"}, torch.float32, torch.float32, torch.float32),
    ],
)
def test_fsdp_compute_dtype_precedence(
    monkeypatch, strategy, dtype, mixed_precision, expected_param, expected_reduce, expected_buffer
):
    engine = object.__new__(transformer_impl.FSDPEngineWithLMHead)
    engine.engine_config = FSDPEngineConfig(strategy=strategy, dtype=dtype, mixed_precision=mixed_precision)
    engine.model_config = SimpleNamespace(lora_rank=0, enable_activation_offload=False)
    engine.device_mesh = object()
    module = torch.nn.Linear(4, 4)

    fsdp = Mock(return_value=module)
    apply_fsdp2 = Mock()
    scaler = Mock()
    monkeypatch.setattr(transformer_impl, "FSDP", fsdp)
    monkeypatch.setattr(transformer_impl, "apply_fsdp2", apply_fsdp2)
    monkeypatch.setattr(transformer_impl, "fsdp2_load_full_state_dict", Mock())
    monkeypatch.setattr(transformer_impl, "get_fsdp_wrap_policy", Mock(return_value=None))
    monkeypatch.setattr(transformer_impl, "get_sharding_strategy", Mock(return_value=None))
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda module: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.fsdp.sharded_grad_scaler.ShardedGradScaler", scaler)

    assert engine._build_fsdp_module(module) is module
    if strategy == "fsdp":
        policy = fsdp.call_args.kwargs["mixed_precision"]
        assert policy.buffer_dtype == expected_buffer
    else:
        policy = apply_fsdp2.call_args.args[1]["mp_policy"]
        assert policy.cast_forward_inputs
    assert policy.param_dtype == expected_param
    assert policy.reduce_dtype == expected_reduce
    assert engine._autocast_dtype == expected_param
    if expected_param == torch.float16:
        scaler.assert_called_once_with(growth_interval=400)
        assert engine.scaler is scaler.return_value
    else:
        scaler.assert_not_called()
        assert engine.scaler is None
