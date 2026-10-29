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
"""Exercise VeOmni engine initialization without a real distributed mesh.

VeOmni is optional in CPU CI. Stub its import surfaces as in the router-replay
engine tests, then exercise the config-based parallel-state API explicitly in the tests.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

for _mod in (
    "veomni",
    "veomni.arguments",
    "veomni.distributed",
    "veomni.distributed.torch_parallelize",
    "veomni.models",
    "veomni.models.auto",
    "veomni.models.checkpoint_tensor_loading",
    "veomni.optim",
    "veomni.utils",
    "veomni.utils.seqlen_pos_transform_utils",
):
    sys.modules.setdefault(_mod, MagicMock())

from verl.workers.config import VeOmniEngineConfig  # noqa: E402
from verl.workers.engine.veomni import transformer_impl  # noqa: E402


def _init_engine(monkeypatch, parallel_state, world_size=8, **config_kwargs):
    monkeypatch.setattr(transformer_impl, "parallel_state", parallel_state)
    monkeypatch.setattr(transformer_impl.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(transformer_impl.dist, "get_world_size", lambda: world_size)
    return transformer_impl.VeOmniEngine(
        model_config=SimpleNamespace(use_remove_padding=True, lora_rank=0),
        engine_config=VeOmniEngineConfig(use_torch_compile=False, **config_kwargs),
        optimizer_config=None,
        checkpoint_config=None,
    )


@pytest.fixture
def parallel_state():
    """A current VeOmni API, intentionally missing the removed initializer."""
    state = SimpleNamespace(sp_enabled=False)
    return SimpleNamespace(
        get_parallel_state=MagicMock(return_value=state),
        init_parallel_state_from_config=MagicMock(return_value=state),
    )


@pytest.mark.parametrize(
    ("world_size", "fsdp_size", "ulysses_size", "ep_size", "replicate_size", "shard_size"),
    [(1, -1, 1, 1, 1, 1), (8, -1, 2, 2, 1, 4), (16, 4, 2, 4, 2, 4), (8, 8, 1, 2, 1, 8)],
)
def test_current_api_preserves_parallel_topology(
    monkeypatch, parallel_state, world_size, fsdp_size, ulysses_size, ep_size, replicate_size, shard_size
):
    accelerator = SimpleNamespace(dp_size=world_size // ulysses_size)
    make_accelerator = MagicMock(return_value=accelerator)
    fsdp_config = object()
    make_fsdp_config = MagicMock(return_value=fsdp_config)
    monkeypatch.setattr(transformer_impl, "AcceleratorConfig", make_accelerator)
    monkeypatch.setattr(transformer_impl, "FSDPConfig", make_fsdp_config)

    _init_engine(
        monkeypatch,
        parallel_state,
        world_size=world_size,
        fsdp_size=fsdp_size,
        ulysses_parallel_size=ulysses_size,
        expert_parallel_size=ep_size,
    )

    make_fsdp_config.assert_called_once_with(fsdp_mode="fsdp2")
    make_accelerator.assert_called_once_with(
        dp_replicate_size=replicate_size,
        dp_shard_size=shard_size,
        ep_size=ep_size,
        ulysses_size=ulysses_size,
        init_device="meta",
        fsdp_config=fsdp_config,
    )
    parallel_state.init_parallel_state_from_config.assert_called_once_with(accelerator, name=None)


def test_current_api_rejects_world_size_mismatch(monkeypatch, parallel_state):
    monkeypatch.setattr(transformer_impl, "AcceleratorConfig", MagicMock(return_value=SimpleNamespace(dp_size=1)))
    monkeypatch.setattr(transformer_impl, "FSDPConfig", MagicMock())

    with pytest.raises(ValueError, match="distributed process group requires dp_size=8"):
        _init_engine(monkeypatch, parallel_state)
    parallel_state.init_parallel_state_from_config.assert_not_called()


def test_invalid_hsdp_fails_before_mesh_creation(monkeypatch, parallel_state):
    with pytest.raises(ValueError, match="must be divisible by fsdp_size"):
        _init_engine(monkeypatch, parallel_state, fsdp_size=3)
    parallel_state.init_parallel_state_from_config.assert_not_called()


@pytest.mark.parametrize("enable_full_shard", [False, True])
def test_full_shard_uses_veomni_reshard_option(monkeypatch, enable_full_shard):
    module = SimpleNamespace(_no_split_modules=["DecoderLayer"])
    engine = object.__new__(transformer_impl.VeOmniEngine)
    engine.engine_config = VeOmniEngineConfig(forward_only=True, enable_full_shard=enable_full_shard)
    engine.model_config = SimpleNamespace(
        local_hf_config_path="/model", local_path="/model", enable_gradient_checkpointing=True
    )
    monkeypatch.setattr(transformer_impl, "_build_ops_implementation_config", MagicMock())
    monkeypatch.setattr(transformer_impl, "MixedPrecisionConfig", MagicMock(return_value=SimpleNamespace(enable=False)))
    monkeypatch.setattr(transformer_impl, "build_foundation_model", MagicMock(return_value=module))
    monkeypatch.setattr(transformer_impl, "load_safetensors_index", MagicMock(return_value={}))
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", MagicMock())
    parallelize = MagicMock(return_value=module)
    monkeypatch.setattr(transformer_impl, "build_parallelize_model", parallelize)

    engine._build_model_optimizer()

    assert parallelize.call_args.kwargs["enable_reshard_after_forward"] is enable_full_shard
    assert "enable_full_shard" not in parallelize.call_args.kwargs
    assert engine.module is module
    assert engine.optimizer is None
