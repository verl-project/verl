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

"""The existing engine API exports full HF by default, including delta seeds."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from verl.workers.engine.fsdp import transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead

_checkpoint = pytest.importorskip("fsdp_turbo.models.kimi.kimi_k3.checkpoint")
iter_checkpoint_weights = _checkpoint.iter_checkpoint_weights
map_checkpoint_indices = _checkpoint.map_checkpoint_indices


@pytest.fixture(autouse=True)
def cpu_fsdp2(monkeypatch):
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "fsdp_version", lambda *args: 2)
    monkeypatch.setattr(transformer_impl, "get_device_id", lambda: "cpu")


def _engine():
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = torch.nn.Module()
    engine.module.iter_checkpoint_weights = iter_checkpoint_weights
    engine.module.map_checkpoint_indices = map_checkpoint_indices
    engine.module.experts = torch.nn.Module()
    engine.module.experts.gate_up_proj = torch.nn.Parameter(torch.arange(48).reshape(2, 3, 8).bfloat16())
    engine.module.experts.down_proj = torch.nn.Parameter(torch.arange(24).reshape(2, 4, 3).bfloat16())
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3"), lora={})
    engine._uses_fsdp2_cpu_offload_policy = engine._is_offload_param = engine._qat_enabled = engine._is_lora = False
    return engine


def test_default_hf_and_delta_steady_reconstruct_same_slots():
    engine = _engine()
    seed = dict(engine.get_per_tensor_param()[0])
    checkpoint = dict(engine._get_hf_export_per_tensor_param()[0])
    assert seed.keys() == checkpoint.keys()
    assert len(seed) == 6
    assert all(".__verl_packed_local__." not in key for key in seed)
    engine.prime_delta_snapshots()
    with torch.no_grad():
        engine.module.experts.gate_up_proj[1, 2, 5] += 2
        engine.module.experts.down_proj[0, 3, 1] += 3
    entries, _ = engine.get_per_tensor_param_delta_shard()
    for slots, _, counts, indices, values, _ in entries:
        offset = 0
        for (key, shape), count in zip(slots, counts.tolist(), strict=True):
            assert tuple(seed[key].shape) == shape
            seed[key].view(-1)[indices[offset : offset + count]] = values[offset : offset + count]
            offset += count
    fresh = dict(engine.get_per_tensor_param()[0])
    for key in fresh:
        torch.testing.assert_close(seed[key], fresh[key], rtol=0, atol=0)


def test_colocated_layout_is_explicit_and_bypasses_generic_unfuse():
    engine = _engine()
    layout = {"expert_parallel_size": 2}
    marker = "experts.gate_up_proj.__verl_packed_local__.1"

    def export(params, *, device, rollout_layout):
        assert rollout_layout == layout
        yield marker, torch.ones(1)

    engine.module.iter_checkpoint_weights = export
    values = dict(engine.get_per_tensor_param(rollout_layout=layout)[0])
    assert values.keys() == {marker}
    torch.testing.assert_close(values[marker], torch.ones(1))


def test_native_export_capability_keeps_generic_state_fallback():
    ordinary = object.__new__(FSDPEngine)
    ordinary.module = torch.nn.Linear(2, 2)
    ordinary.module.register_buffer("persisted_buffer", torch.ones(3))
    ordinary._uses_fsdp2_cpu_offload_policy = False
    assert "persisted_buffer" in ordinary._get_parameter_export()
    assert ordinary._get_parameter_export()["weight"] is not ordinary.module.weight
    ordinary.module.iter_checkpoint_weights = iter_checkpoint_weights
    assert ordinary._get_parameter_export()["weight"] is ordinary.module.weight


def test_kimi_physical_export_preserves_buffers_and_merged_values():
    from contextlib import contextmanager

    engine = _engine()
    engine.module.register_buffer("persistent", torch.tensor([11.0]))
    engine.module.register_buffer("alias", engine.module.persistent)
    engine.module.register_buffer("cache", torch.tensor([99.0]), persistent=False)
    before = engine.module.experts.gate_up_proj.detach().clone()

    @contextmanager
    def merge(module, **kwargs):
        with torch.no_grad():
            module.experts.gate_up_proj.add_(2)
        try:
            yield
        finally:
            with torch.no_grad():
                module.experts.gate_up_proj.sub_(2)

    with (
        patch("verl.workers.engine.fsdp.transformer_impl.merged_lora_context", side_effect=merge),
    ):
        params = engine._get_parameter_export()
        assert params["persistent"] is params["alias"]
        assert "cache" not in params
        merged = dict(engine._merged_lora_per_tensor_param())
    assert "persistent" in merged and "alias" in merged and "cache" not in merged
    torch.testing.assert_close(engine.module.experts.gate_up_proj, before)
    torch.testing.assert_close(merged["experts.0.w1.weight"], (before[0, :, :4] + 2).T)


@pytest.mark.parametrize("load_after_shard", [False, True])
def test_model_build_loads_native_checkpoint_only_after_wrap(load_after_shard):
    calls = []

    class Model(torch.nn.Linear):
        def load_hf_checkpoint_shards(self, path, *, device):
            calls.append(("load", path, device))

    model = Model(2, 2)
    engine = object.__new__(FSDPEngine)
    engine.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(load_checkpoint_after_shard=load_after_shard), local_path="checkpoint"
    )
    engine.engine_config = SimpleNamespace(forward_only=False)
    engine._is_lora = engine._uses_fsdp2_cpu_offload_policy = False
    engine.rank = 1
    engine._build_module = lambda: calls.append("build") or model
    engine._build_fsdp_module = lambda module: calls.append("wrap") or module
    engine._build_optimizer = lambda module: calls.append("optimizer") or object()
    engine._build_lr_scheduler = lambda optimizer: object()
    with patch("verl.workers.engine.fsdp.transformer_impl.torch.distributed.barrier"):
        engine._build_model_optimizer()
    assert calls == (
        ["build", "wrap", ("load", "checkpoint", "cpu"), "optimizer"]
        if load_after_shard
        else ["build", "wrap", "optimizer"]
    )


@pytest.mark.parametrize("load_after_shard", [None, False, True])
@pytest.mark.parametrize("loader", ["missing", "noncallable", "supported"])
def test_after_shard_declaration_validates_loader_before_model_patches(load_after_shard, loader):
    from unittest.mock import Mock

    def build_model(*args, **kwargs):
        module = torch.nn.Linear(2, 2)
        if loader == "supported":
            module.load_hf_checkpoint_shards = Mock()
        elif loader == "noncallable":
            module.load_hf_checkpoint_shards = None
        return module

    auto_class = SimpleNamespace(
        from_config=Mock(side_effect=build_model),
        from_pretrained=Mock(side_effect=build_model),
    )
    hf_config = SimpleNamespace(tie_word_embeddings=False)
    if load_after_shard is not None:
        hf_config.load_checkpoint_after_shard = load_after_shard
    engine = object.__new__(FSDPEngine)
    engine.device_mesh = None
    engine.engine_config = SimpleNamespace(model_dtype="fp32", forward_only=False)
    engine.model_config = SimpleNamespace(
        model_type="language_model",
        hf_config=hf_config,
        trust_remote_code=False,
        local_path="checkpoint",
        use_liger=False,
        fused_kernel_options={},
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
    )
    engine.use_remove_padding = False
    engine.ulysses_sequence_parallel_size = 1
    with (
        patch("verl.utils.model.get_hf_auto_model_class", return_value=auto_class),
        patch("verl.workers.engine.fsdp.transformer_impl.get_init_weight_context_manager", return_value=nullcontext),
        patch("verl.workers.engine.fsdp.transformer_impl.apply_monkey_patch") as patch_model,
    ):
        if load_after_shard and loader != "supported":
            with pytest.raises(ValueError, match="load_checkpoint_after_shard=True.*load_hf_checkpoint_shards"):
                engine._build_module()
            patch_model.assert_not_called()
        else:
            module = engine._build_module()
            assert isinstance(module, torch.nn.Linear)
            assert module.weight.is_meta == bool(load_after_shard)
            patch_model.assert_called_once()
    if load_after_shard:
        auto_class.from_config.assert_called_once()
        auto_class.from_pretrained.assert_not_called()
    else:
        auto_class.from_pretrained.assert_called_once()
        auto_class.from_config.assert_not_called()
