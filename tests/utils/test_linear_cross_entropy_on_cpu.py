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

import builtins
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "kernel" / "linear_cross_entropy.py"
_SPEC = importlib.util.spec_from_file_location("_linear_cross_entropy", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lce = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lce)


def test_triton_backend_preserves_existing_autograd_dispatch(monkeypatch):
    expected = (torch.tensor([1.0]), torch.tensor([2.0]))
    calls = []

    class FakeLinearCrossEntropy:
        @staticmethod
        def apply(*args):
            calls.append(args)
            return expected

    monkeypatch.setattr(lce, "LinearCrossEntropy", FakeLinearCrossEntropy)
    hidden = torch.randn(3, 5)
    weight = torch.randn(7, 5)
    labels = torch.randint(7, (3,))
    group = object()

    output = lce.linear_cross_entropy(
        hidden,
        weight,
        labels,
        0.8,
        "none",
        group,
        impl_backend="triton",
    )

    assert output is expected
    assert calls == [(hidden, weight, labels, 0.8, "none", group)]


def test_liger_tp_delegates_full_tensor_to_public_frontend(monkeypatch):
    calls = []

    class FakeTPFunction:
        @staticmethod
        def apply(
            hidden,
            weight,
            labels,
            process_group,
            temperature=1.0,
            ignore_index=-100,
            return_entropy=False,
        ):
            calls.append(
                {
                    "hidden": hidden,
                    "weight": weight,
                    "labels": labels,
                    "process_group": process_group,
                    "temperature": temperature,
                    "ignore_index": ignore_index,
                    "return_entropy": return_entropy,
                }
            )
            nll = torch.arange(hidden.shape[0], dtype=torch.float32)
            entropy = torch.arange(hidden.shape[0], dtype=torch.float32) + 100
            return nll, entropy

    monkeypatch.setattr(lce, "_require_liger_tp_runtime", lambda: FakeTPFunction)
    process_group = object()

    hidden = torch.arange(35, dtype=torch.float32).reshape(1, 7, 5)
    weight = torch.randn(11, 5)
    labels = torch.arange(7, dtype=torch.int32).reshape(1, 7)

    log_probs, entropy = lce.linear_cross_entropy(
        hidden,
        weight,
        labels,
        0.7,
        "none",
        process_group,
        impl_backend="liger_tp",
    )

    assert len(calls) == 1
    assert tuple(calls[0]["hidden"].shape) == (7, 5)
    assert calls[0]["labels"].tolist() == list(range(7))
    assert calls[0]["labels"].dtype == torch.int64
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["ignore_index"] == -100
    assert calls[0]["return_entropy"] is True
    assert calls[0]["process_group"] is process_group
    torch.testing.assert_close(log_probs, -torch.arange(7, dtype=torch.float32))
    torch.testing.assert_close(entropy, torch.arange(7, dtype=torch.float32) + 100)


def test_liger_tp_propagates_public_frontend_gradients(monkeypatch):
    class FakeTPAutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden,
            weight,
            labels,
            process_group,
            temperature=1.0,
            ignore_index=-100,
            return_entropy=False,
        ):
            del labels, process_group, temperature, ignore_index, return_entropy
            ctx.save_for_backward(hidden, weight)
            nll = (hidden * weight[0]).sum(dim=-1)
            entropy = (hidden * weight[1]).sum(dim=-1)
            return nll, entropy

        @staticmethod
        def backward(ctx, grad_nll, grad_entropy):
            hidden, weight = ctx.saved_tensors
            grad_hidden = grad_nll[:, None] * weight[0] + grad_entropy[:, None] * weight[1]
            grad_weight = torch.stack(
                (
                    (grad_nll[:, None] * hidden).sum(dim=0),
                    (grad_entropy[:, None] * hidden).sum(dim=0),
                )
            )
            return grad_hidden, grad_weight, None, None, None, None, None

    class FakeTPFunction:
        @staticmethod
        def apply(
            hidden,
            weight,
            labels,
            process_group,
            temperature=1.0,
            ignore_index=-100,
            return_entropy=False,
        ):
            return FakeTPAutogradFunction.apply(
                hidden,
                weight,
                labels,
                process_group,
                temperature,
                ignore_index,
                return_entropy,
            )

    monkeypatch.setattr(lce, "_require_liger_tp_runtime", lambda: FakeTPFunction)

    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4).requires_grad_(True)
    weight = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [-1.0, 0.5, 0.25, -0.75],
        ],
        requires_grad=True,
    )
    labels = torch.arange(5)
    grad_log_probs = torch.linspace(-0.5, 0.5, 5)
    grad_entropy = torch.linspace(0.3, -0.2, 5)
    process_group = object()

    log_probs, entropy = lce.linear_cross_entropy(
        hidden,
        weight,
        labels,
        dist_process_group=process_group,
        impl_backend="liger_tp",
    )
    torch.autograd.backward((log_probs, entropy), (grad_log_probs, grad_entropy))

    expected_hidden_grad = -grad_log_probs[:, None] * weight.detach()[0]
    expected_hidden_grad += grad_entropy[:, None] * weight.detach()[1]
    expected_weight_grad = torch.stack(
        (
            (-grad_log_probs[:, None] * hidden.detach()).sum(dim=0),
            (grad_entropy[:, None] * hidden.detach()).sum(dim=0),
        )
    )
    torch.testing.assert_close(hidden.grad, expected_hidden_grad)
    torch.testing.assert_close(weight.grad, expected_weight_grad)


def test_liger_tp_runtime_uses_public_ops_frontend(monkeypatch):
    public_function = object()
    liger_module = types.ModuleType("liger_kernel")
    liger_module.__path__ = []
    ops_module = types.ModuleType("liger_kernel.ops")
    ops_module.LigerFusedLinearScaledCrossEntropyTPFunction = public_function
    liger_module.ops = ops_module

    monkeypatch.setattr(lce, "_LIGER_TP_FUNCTION", None)
    monkeypatch.setitem(sys.modules, "liger_kernel", liger_module)
    monkeypatch.setitem(sys.modules, "liger_kernel.ops", ops_module)

    assert lce._require_liger_tp_runtime() is public_function


def test_liger_tp_configuration_skips_when_lck_is_not_installed(monkeypatch):
    monkeypatch.setattr(lce, "_LIGER_TP_CONFIGURATION", None)
    monkeypatch.setattr(lce, "_load_lck_runtime", lambda: None)
    monkeypatch.setattr(lce.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lce.torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(lce.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(lce.dist, "get_global_rank", lambda group, rank: 0)

    assert (
        lce.configure_liger_tp_flsce(
            max_tokens=4096,
            hidden_size=2048,
            local_vocab_size=151936,
            process_group=object(),
            device=torch.device("cuda:0"),
        )
        is False
    )


@pytest.mark.parametrize("missing_module", ["liger_cute_kernels", "tvm_ffi"])
def test_load_lck_runtime_only_skips_missing_optional_package(monkeypatch, missing_module):
    original_import = builtins.__import__

    def import_without_dependency(name, *args, **kwargs):
        if name == "liger_cute_kernels":
            raise ModuleNotFoundError(f"No module named '{missing_module}'", name=missing_module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)
    if missing_module == "liger_cute_kernels":
        assert lce._load_lck_runtime() is None
    else:
        with pytest.raises(ModuleNotFoundError, match=missing_module):
            lce._load_lck_runtime()


def test_liger_tp_configuration_uses_existing_maximum(monkeypatch):
    process_group = object()
    device = torch.device("cuda:2")
    calls = []

    class FakeNvshmem:
        @staticmethod
        def init_from_pg(group):
            calls.append(("init", group))

        @staticmethod
        def resolve_team(group):
            calls.append(("team", group))
            return 17

    class FakeTvmFfi:
        @staticmethod
        def fused_linear_scaled_cross_entropy_configure_backward(*args):
            calls.append(("backward", args))

        @staticmethod
        def fused_linear_scaled_cross_entropy_configure_forward(*args):
            calls.append(("forward", args))

    monkeypatch.setattr(lce, "_LIGER_TP_CONFIGURATION", None)
    monkeypatch.setattr(lce, "_load_lck_runtime", lambda: (FakeNvshmem, FakeTvmFfi))
    monkeypatch.setattr(lce.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lce.torch.cuda, "get_device_capability", lambda actual: (10, 3))
    monkeypatch.setattr(lce.torch.cuda, "device", lambda actual: nullcontext())
    monkeypatch.setattr(lce.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(lce.dist, "get_global_rank", lambda group, rank: 4 + rank)

    kwargs = {
        "max_tokens": 4096,
        "hidden_size": 5120,
        "local_vocab_size": 62080,
        "process_group": process_group,
        "device": device,
    }
    assert lce.configure_liger_tp_flsce(**kwargs) is True
    assert lce.configure_liger_tp_flsce(**kwargs) is True
    assert calls == [
        ("init", process_group),
        ("team", process_group),
        ("backward", (4096, 5120, 62080, 1, 17)),
        ("forward", (4096, 62080)),
    ]


@pytest.mark.parametrize("max_tokens", [2048, 8192])
def test_liger_tp_configuration_rejects_capacity_changes(monkeypatch, max_tokens):
    process_group = object()
    device = torch.device("cuda:0")

    monkeypatch.setattr(
        lce,
        "_LIGER_TP_CONFIGURATION",
        ((0,), 4096, 5120, 62080, device),
    )
    monkeypatch.setattr(lce, "_load_lck_runtime", lambda: pytest.fail("workspace must not be reconfigured"))
    monkeypatch.setattr(lce.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(lce.torch.cuda, "get_device_capability", lambda actual: (10, 3))
    monkeypatch.setattr(lce.dist, "get_world_size", lambda group: 1)
    monkeypatch.setattr(lce.dist, "get_global_rank", lambda group, rank: 0)

    with pytest.raises(RuntimeError, match="already configured"):
        lce.configure_liger_tp_flsce(
            max_tokens=max_tokens,
            hidden_size=5120,
            local_vocab_size=62080,
            process_group=process_group,
            device=device,
        )


def test_linear_cross_entropy_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported linear cross entropy backend"):
        lce.linear_cross_entropy(
            torch.randn(2, 3),
            torch.randn(5, 3),
            torch.randint(5, (2,)),
            impl_backend="unknown",
        )
