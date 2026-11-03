# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Megatron LoRA compatibility tests with CPU tensors."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")
peft_utils = pytest.importorskip("megatron.bridge.peft.utils")


@pytest.mark.parametrize("min_lr", [None, 0.001])
def test_lora_plus_without_bridge_helper_preserves_native_groups(monkeypatch, min_lr):
    from megatron.bridge.peft import lora
    from megatron.core.optimizer import OptimizerConfig, ParamKey

    from verl.utils.megatron import optimizer as opt

    monkeypatch.delattr(lora, "get_lora_plus_config_overrides", raising=False)
    monkeypatch.setattr(opt, "get_megatron_optimizer_native", lambda **kwargs: kwargs)
    monkeypatch.setattr(opt, "print_rank_0", lambda msg: None)
    config = OptimizerConfig(lr=0.01, min_lr=min_lr)
    assert "config_overrides" not in opt.get_megatron_optimizer([], config)

    overrides = opt.get_megatron_optimizer([], config, lora_plus_ratio=4.0)["config_overrides"]
    for linear, factor in (("linear_in", 1), ("linear_out", 4)):
        assert overrides[ParamKey(name=f"*.{linear}.weight")] == {
            "max_lr": 0.01 * factor,
            "min_lr": (min_lr or 0) * factor,
        }
    assert any(override.get("wd_mult") == 0 for override in overrides.values())


@pytest.mark.parametrize(
    "adapter_cls,grouped,ep,fused,train_shared,needs_finalizer",
    [
        ("ParallelLinearAdapter", True, 2, True, True, True),
        ("SharedOuterGroupedExpertAdapter", True, 2, True, True, True),
        ("SharedOuterGroupedExpertAdapter", True, 2, True, False, False),
        ("GroupedExpertLinearAdapter", True, 2, True, True, False),
        ("ParallelLinearAdapter", False, 2, True, True, False),
        ("ParallelLinearAdapter", True, 1, True, True, False),
        ("ParallelLinearAdapter", True, 2, False, True, False),
    ],
)
def test_legacy_bridge_expert_grad_hooks(monkeypatch, adapter_cls, grouped, ep, fused, train_shared, needs_finalizer):
    from megatron.core.distributed import finalize_model_grads

    from verl.utils.megatron_utils import _get_megatron_grad_finalizer

    monkeypatch.delattr(peft_utils, "enable_expert_parallel_grad_sync_in_finalize", raising=False)
    monkeypatch.delattr(peft_utils, "finalize_model_grads_with_expert_adapter_sync", raising=False)
    # Construct a CPU-only adapter without initializing TP/EP process groups.
    cls = getattr(peft_utils, adapter_cls)
    adapter = cls.__new__(cls)
    torch.nn.Module.__init__(adapter)
    adapter._is_fc1 = True
    adapter.is_expert = grouped
    adapter.base_linear_name = "layers.0.mlp.experts.linear_fc1" if grouped else "layers.0.mlp.linear_fc1"
    adapter.linear_in = torch.nn.Linear(1, 1, bias=False)
    adapter.linear_in.weight.requires_grad_(train_shared)
    adapter.linear_out = torch.nn.Linear(1, 1, bias=False)
    model = torch.nn.Sequential(adapter)
    model.config = SimpleNamespace(expert_model_parallel_size=ep, gradient_accumulation_fusion=fused)

    if needs_finalizer:
        with pytest.raises(RuntimeError, match="expert-adapter gradient finalizer"):
            _get_megatron_grad_finalizer([model])
    else:
        assert _get_megatron_grad_finalizer([model]) is finalize_model_grads
