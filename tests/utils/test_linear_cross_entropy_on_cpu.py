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


def test_liger_delegates_full_tensor_to_public_frontend(monkeypatch):
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

    monkeypatch.setattr(lce, "_require_liger_runtime", lambda: FakeTPFunction)
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
        impl_backend="liger",
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


def test_liger_propagates_public_frontend_gradients(monkeypatch):
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

    monkeypatch.setattr(lce, "_require_liger_runtime", lambda: FakeTPFunction)

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
        impl_backend="liger",
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


def test_liger_runtime_uses_public_ops_frontend(monkeypatch):
    public_function = object()
    liger_module = types.ModuleType("liger_kernel")
    liger_module.__path__ = []
    ops_module = types.ModuleType("liger_kernel.ops")
    ops_module.LigerFusedLinearScaledCrossEntropyTPFunction = public_function
    liger_module.ops = ops_module

    monkeypatch.setattr(lce, "_LIGER_FUNCTION", None)
    monkeypatch.setitem(sys.modules, "liger_kernel", liger_module)
    monkeypatch.setitem(sys.modules, "liger_kernel.ops", ops_module)

    assert lce._require_liger_runtime() is public_function


@pytest.fixture
def public_configuration(monkeypatch):
    calls = []
    module = types.ModuleType("liger_kernel.ops.configure")
    module.FusedLinearCrossEntropyConfig = types.SimpleNamespace
    module.configure = lambda **kwargs: calls.append(kwargs) or True
    monkeypatch.setitem(sys.modules, "liger_kernel.ops.configure", module)
    monkeypatch.setattr(lce.dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(lce.dist, "get_process_group_ranks", lambda group: group)
    monkeypatch.setattr(
        lce.dist,
        "all_gather_object",
        lambda output, ranks: output.__setitem__(slice(None), [(0, 1)] * 2 + [(2, 3)] * 2),
    )
    return module, calls


def _configure(group=(0, 1), max_tokens=4096, device="cuda:0"):
    return lce.configure_liger_flsce(
        max_tokens=max_tokens,
        hidden_size=5120,
        local_vocab_size=62080,
        process_group=group,
        device=torch.device(device),
    )


def test_configuration_delegates_repeated_calls_and_capacity_to_liger(public_configuration):
    _, calls = public_configuration
    for tokens in (4096, 4096, 2048):
        assert _configure(max_tokens=tokens) is True
    assert len(calls) == 3
    assert [call["flsce"].max_tokens for call in calls] == [4096, 4096, 2048]
    for call in calls:
        name = call["flsce"].group
        assert call["process_groups"] == {name: (0, 1)}
        assert call["bootstrap_group"] is lce.dist.group.WORLD
        assert call["device"] == torch.device("cuda:0")
        assert call["flsce"].hidden_size == 5120
        assert call["flsce"].local_vocab_size == 62080
    assert calls[0]["flsce"].group == calls[1]["flsce"].group


def test_configuration_names_identify_global_partition(monkeypatch, public_configuration):
    _, calls = public_configuration
    _configure((0, 1))
    _configure((2, 3))
    assert calls[0]["flsce"].group == calls[1]["flsce"].group
    monkeypatch.setattr(
        lce.dist, "all_gather_object", lambda output, ranks: output.__setitem__(slice(None), [(0, 2), (1, 3)] * 2)
    )
    _configure((0, 2))
    assert calls[2]["flsce"].group != calls[0]["flsce"].group


def test_configuration_preserves_optional_native_fallback(public_configuration):
    module, _ = public_configuration
    module.configure = lambda **kwargs: False
    assert _configure() is False


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "xpu:0"])
@pytest.mark.parametrize("configured", [True, False])
def test_configuration_leaves_device_support_to_liger(monkeypatch, public_configuration, device, configured):
    module, calls = public_configuration

    def hardware_probe(*args, **kwargs):
        pytest.fail("VeRL must not inspect hardware to configure Liger")

    monkeypatch.setattr(lce.torch.cuda, "is_available", hardware_probe)
    monkeypatch.setattr(lce.torch.cuda, "get_device_capability", hardware_probe)
    monkeypatch.setattr(module, "configure", lambda **kwargs: calls.append(kwargs) or configured)
    assert _configure(device=device) is configured
    assert calls[0]["device"] == torch.device(device)


def test_configuration_propagates_installed_runtime_errors(public_configuration):
    module, _ = public_configuration

    def fail(**kwargs):
        raise RuntimeError("capacity cannot grow")

    module.configure = fail
    with pytest.raises(RuntimeError, match="capacity cannot grow"):
        _configure()


def test_configuration_requires_public_api(monkeypatch, public_configuration):
    original_import = builtins.__import__

    def import_without_api(name, *args, **kwargs):
        if name == "liger_kernel.ops.configure":
            raise ModuleNotFoundError(name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_api)
    with pytest.raises(RuntimeError, match="public configure API"):
        _configure()


@pytest.mark.parametrize("backend", ["unknown", "torch"])
def test_linear_cross_entropy_rejects_unknown_backend(backend):
    with pytest.raises(ValueError, match="Unsupported linear cross entropy backend"):
        lce.linear_cross_entropy(
            torch.randn(2, 3),
            torch.randn(5, 3),
            torch.randint(5, (2,)),
            impl_backend=backend,
        )
