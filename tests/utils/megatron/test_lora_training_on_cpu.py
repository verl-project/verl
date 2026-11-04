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
    from megatron.bridge.peft.lora_layers import LinearAdapter
    from megatron.core.optimizer import OptimizerConfig, _get_param_groups

    from verl.utils.megatron import optimizer as opt

    monkeypatch.delattr(lora, "get_lora_plus_config_overrides", raising=False)
    monkeypatch.setattr(opt, "get_megatron_optimizer_native", lambda **kwargs: kwargs)
    monkeypatch.setattr(opt, "print_rank_0", lambda msg: None)
    config = OptimizerConfig(lr=0.01, min_lr=min_lr)
    model = torch.nn.Module()
    model.proj = lora.LoRA(dim=2, target_modules=["proj"]).transform(torch.nn.Linear(4, 4), name="proj")
    model.offset = torch.nn.Parameter(torch.zeros(1))
    adapter = next(module for module in model.proj.modules() if isinstance(module, LinearAdapter))
    assert "config_overrides" not in opt.get_megatron_optimizer([model], config)

    kwargs = opt.get_megatron_optimizer([model], config, lora_plus_ratio=4.0)
    # MCore aligns optimizer groups across ranks even for a CPU-only model.
    created_group = not torch.distributed.is_initialized()
    if created_group:
        torch.distributed.init_process_group("gloo", store=torch.distributed.HashStore(), rank=0, world_size=1)
    try:
        groups = _get_param_groups(**kwargs)
    finally:
        if created_group:
            torch.distributed.destroy_process_group()

    by_param = {id(param): group for group in groups for param in group["params"]}
    for linear, factor in ((adapter.linear_in, 1), (adapter.linear_out, 4)):
        assert by_param[id(linear.weight)]["max_lr"] == config.lr * factor
        assert by_param[id(linear.weight)]["min_lr"] == (min_lr or 0) * factor
    assert by_param[id(model.offset)]["wd_mult"] == 0


@pytest.fixture
def legacy_bridge(monkeypatch):
    monkeypatch.delattr(peft_utils, "enable_expert_parallel_grad_sync_in_finalize", raising=False)
    monkeypatch.delattr(peft_utils, "finalize_model_grads_with_expert_adapter_sync", raising=False)


def _adapter_model(adapter_cls, *, grouped=True, ep=2):
    # Exercise hook selection without building a distributed transformer.
    cls = getattr(peft_utils, adapter_cls)
    adapter = cls.__new__(cls)
    torch.nn.Module.__init__(adapter)
    adapter._is_fc1 = True
    adapter.is_expert = grouped
    adapter.base_linear_name = "layers.0.mlp.experts.linear_fc1" if grouped else "layers.0.mlp.linear_fc1"
    adapter.linear_in = torch.nn.Linear(1, 1, bias=False)
    adapter.linear_out = torch.nn.Linear(1, 1, bias=False)
    for linear in (adapter.linear_in, adapter.linear_out):
        linear.gradient_accumulation_fusion = True
    model = torch.nn.Sequential(adapter)
    model.config = SimpleNamespace(expert_model_parallel_size=ep, gradient_accumulation_fusion=True)
    return model


@pytest.mark.parametrize(
    "adapter_cls,grouped,ep,train_in,expected",
    [
        ("ParallelLinearAdapter", True, 2, True, (False, False)),
        ("ParallelLinearAdapter", True, 2, False, (True, False)),
        ("GroupedExpertLinearAdapter", True, 2, True, (True, True)),
        ("ParallelLinearAdapter", False, 2, True, (True, True)),
        ("ParallelLinearAdapter", True, 1, True, (True, True)),
    ],
)
def test_legacy_bridge_expert_grad_hooks(legacy_bridge, adapter_cls, grouped, ep, train_in, expected):
    from megatron.core.distributed import finalize_model_grads

    from verl.utils.megatron_utils import _get_megatron_grad_finalizer

    model = _adapter_model(adapter_cls, grouped=grouped, ep=ep)
    a, b = model[0].linear_in, model[0].linear_out
    a.weight.requires_grad_(train_in)
    assert _get_megatron_grad_finalizer([model]) is finalize_model_grads
    assert (a.gradient_accumulation_fusion, b.gradient_accumulation_fusion) == expected
    assert model.config.gradient_accumulation_fusion


@pytest.mark.parametrize(
    "ep,stage,fc1,trainable,fused,expected_fusion",
    [
        (2, [0, 1], True, True, True, False),
        (2, [0, 1], False, True, False, False),
        (2, [0, 1], False, True, True, False),
        (2, [0, 1], True, False, True, True),
        (1, [0], True, True, True, True),
        (2, [0, 2], True, True, True, None),
        (2, [0, 1, 2, 3], True, True, False, None),
        (1, [0, 1], False, False, False, None),
    ],
)
def test_legacy_bridge_shared_outer_group(
    legacy_bridge, monkeypatch, ep, stage, fc1, trainable, fused, expected_fusion
):
    from megatron.core import parallel_state
    from megatron.core.distributed import finalize_model_grads

    from verl.utils.megatron_utils import _get_megatron_grad_finalizer

    model = _adapter_model("SharedOuterGroupedExpertAdapter", ep=ep)
    adapter = model[0]
    adapter._is_fc1 = fc1
    shared, per_expert = (adapter.linear_in, adapter.linear_out) if fc1 else (adapter.linear_out, adapter.linear_in)
    shared.weight.requires_grad_(trainable)
    shared.gradient_accumulation_fusion = fused
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_group", lambda: list(range(ep)))
    monkeypatch.setattr(parallel_state, "get_tensor_and_data_parallel_group", lambda with_context_parallel: stage)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group)
    if expected_fusion is None:
        with pytest.raises(RuntimeError, match="different ranks"):
            _get_megatron_grad_finalizer([model])
    else:
        assert _get_megatron_grad_finalizer([model]) is finalize_model_grads
        assert shared.gradient_accumulation_fusion == expected_fusion
        assert per_expert.gradient_accumulation_fusion and model.config.gradient_accumulation_fusion


@pytest.mark.parametrize("replicated", [0, 1])
def test_native_bridge_expert_grad_finalizer(monkeypatch, replicated):
    from megatron.core.distributed import finalize_model_grads

    from verl.utils.megatron_utils import _get_megatron_grad_finalizer

    model = _adapter_model("SharedOuterGroupedExpertAdapter")
    monkeypatch.setattr(
        peft_utils, "enable_expert_parallel_grad_sync_in_finalize", lambda chunks: replicated, raising=False
    )
    native = object()
    monkeypatch.setattr(peft_utils, "finalize_model_grads_with_expert_adapter_sync", native, raising=False)
    assert _get_megatron_grad_finalizer([model]) is (native if replicated else finalize_model_grads)
    assert model[0].linear_in.gradient_accumulation_fusion
