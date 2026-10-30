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
"""CPU unit tests for FSDPEngine dtype and mixed-precision resolution (verl#8076)."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from verl.workers.config.engine import FSDPEngineConfig
from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _DummyFSDPWrap:
    """Capture arguments passed to FSDP wrapper during _build_fsdp_module."""

    def __init__(self, module, **kwargs):
        """Store wrapped module and keyword arguments."""
        self.module = module
        self.kwargs = kwargs


def _make_fsdp_engine_stub(engine_config: FSDPEngineConfig) -> FSDPEngine:
    """Create a minimal FSDPEngine stub sufficient to run _build_fsdp_module on CPU."""
    eng = object.__new__(FSDPEngine)
    eng.engine_config = engine_config
    eng.model_config = SimpleNamespace(
        lora_rank=0,
        enable_activation_offload=False,
        hf_config=SimpleNamespace(model_type="qwen2"),
    )
    eng.device_mesh = SimpleNamespace(ndim=1, size=lambda: 1)
    eng.ulysses_sequence_parallel_size = 1
    eng._is_offload_param = False
    eng._is_offload_optimizer = False
    eng._uses_fsdp2_cpu_offload_policy = False
    eng._autocast_dtype = torch.bfloat16
    eng.scaler = None
    return eng


@pytest.mark.parametrize(
    ("dtype", "mixed_precision", "expected_param_dtype", "expect_scaler"),
    [
        ("bfloat16", None, torch.bfloat16, False),
        ("float16", None, torch.float16, True),
        ("float32", None, torch.float32, False),
        ("float16", {"reduce_dtype": "fp32"}, torch.float16, True),
        ("float16", {"param_dtype": "bf16", "reduce_dtype": "fp32"}, torch.bfloat16, False),
        ("bfloat16", {"param_dtype": "fp16", "reduce_dtype": "fp32"}, torch.float16, True),
    ],
)
def test_build_fsdp_module_honors_engine_config_dtype(
    monkeypatch, dtype, mixed_precision, expected_param_dtype, expect_scaler
):
    """Verify _build_fsdp_module derives param_dtype from engine_config.dtype when mixed_precision omits it."""
    import torch.distributed.fsdp as torch_fsdp

    monkeypatch.setattr(transformer_impl, "FSDP", _DummyFSDPWrap)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "get_fsdp_wrap_policy", lambda **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_sharding_strategy", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    class_scaler = type("DummyShardedGradScaler", (), {"__init__": lambda self, **kw: setattr(self, "kw", kw)})
    monkeypatch.setattr(
        torch_fsdp.sharded_grad_scaler,
        "ShardedGradScaler",
        class_scaler,
    )

    cfg = FSDPEngineConfig(strategy="fsdp", dtype=dtype, mixed_precision=mixed_precision)
    eng = _make_fsdp_engine_stub(cfg)
    wrapped = eng._build_fsdp_module(torch.nn.Linear(4, 4))

    mp = wrapped.kwargs["mixed_precision"]
    assert mp.param_dtype == expected_param_dtype
    assert mp.reduce_dtype == torch.float32
    assert mp.buffer_dtype == torch.float32
    assert eng._autocast_dtype == expected_param_dtype
    if expect_scaler:
        assert isinstance(eng.scaler, (ShardedGradScaler, class_scaler))
    else:
        assert eng.scaler is None


def test_build_fsdp2_module_honors_engine_config_dtype(monkeypatch):
    """Verify FSDP2 MixedPrecisionPolicy uses engine_config.dtype when mixed_precision is None."""
    import torch.distributed.fsdp as torch_fsdp

    captured_fsdp_kwargs = {}

    def _capture_apply_fsdp2(module, fsdp_kwargs, config):
        """Record fsdp_kwargs passed to apply_fsdp2."""
        captured_fsdp_kwargs.update(fsdp_kwargs)

    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(transformer_impl, "get_fsdp_wrap_policy", lambda **kwargs: None)
    monkeypatch.setattr(transformer_impl, "get_sharding_strategy", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "apply_fsdp2", _capture_apply_fsdp2)
    monkeypatch.setattr(transformer_impl, "fsdp2_load_full_state_dict", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    class_scaler = type("DummyShardedGradScaler", (), {"__init__": lambda self, **kw: setattr(self, "kw", kw)})
    monkeypatch.setattr(
        torch_fsdp.sharded_grad_scaler,
        "ShardedGradScaler",
        class_scaler,
    )

    cfg = FSDPEngineConfig(strategy="fsdp2", dtype="float16", mixed_precision=None)
    eng = _make_fsdp_engine_stub(cfg)
    eng._build_fsdp_module(torch.nn.Linear(4, 4))

    mp_policy = captured_fsdp_kwargs["mp_policy"]
    assert mp_policy.param_dtype == torch.float16
    assert mp_policy.reduce_dtype == torch.float32
    assert eng._autocast_dtype == torch.float16
    assert isinstance(eng.scaler, (ShardedGradScaler, class_scaler))
