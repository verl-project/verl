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
import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

try:  # pragma: no cover - the full verl package needs the runtime deps
    import verl.plugin.platform  # noqa: F401
except ImportError:  # pragma: no cover - minimal CPU environments
    _verl = types.ModuleType("verl")
    _verl.__path__ = []
    _plugin = types.ModuleType("verl.plugin")
    _plugin.__path__ = []
    _platform = types.ModuleType("verl.plugin.platform")

    class _Platform:
        def nvtx_range(self, name):
            return contextlib.nullcontext()

    _platform.get_platform = lambda: _Platform()
    sys.modules["verl"] = _verl
    sys.modules["verl.plugin"] = _plugin
    sys.modules["verl.plugin.platform"] = _platform

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "kernel" / "linear_cross_entropy.py"
_SPEC = importlib.util.spec_from_file_location("_linear_cross_entropy_cce", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
lce = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lce)


def _reference_cce(e, c, targets, bias=None, reduction="mean", return_lse=False, ignore_index=-100):
    """Small autograd-aware stand-in for ``cut_cross_entropy.linear_cross_entropy``."""
    e = e.reshape(-1, e.shape[-1]).to(torch.float64)
    logits = e @ c.to(torch.float64).t()
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    lse = torch.logsumexp(logits, dim=-1)
    picked = logits.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    nll = torch.where(valid, lse - picked, torch.zeros_like(lse))
    lse = torch.where(valid, lse, torch.zeros_like(lse))
    if return_lse:
        return nll, lse
    return nll


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(lce, "_CUT_CROSS_ENTROPY", _reference_cce)
    monkeypatch.setattr(lce, "_check_cut_cross_entropy_device", lambda device: None)


def _inputs(dtype=torch.float64, requires_grad=True):
    torch.manual_seed(0)
    hidden = torch.randn(6, 5, dtype=dtype).requires_grad_(requires_grad)
    weight = torch.randn(11, 5, dtype=dtype).requires_grad_(requires_grad)
    labels = torch.randint(0, 11, (6,))
    return hidden, weight, labels


def _reference(hidden, weight, labels, temperature):
    logits = (hidden.to(torch.float64) @ weight.to(torch.float64).t()) / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1), -(probs * log_probs).sum(dim=-1)


def test_missing_optional_package_reports_install_command(monkeypatch):
    original_import = builtins.__import__

    def import_without_cce(name, *args, **kwargs):
        if name == "cut_cross_entropy":
            raise ModuleNotFoundError(name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(lce, "_CUT_CROSS_ENTROPY", None)
    monkeypatch.setattr(builtins, "__import__", import_without_cce)

    with pytest.raises(RuntimeError, match="cut-cross-entropy") as exc_info:
        lce._require_cut_cross_entropy()
    assert "pip install" in str(exc_info.value)
    assert "apple/ml-cross-entropy" in str(exc_info.value)


def test_non_cuda_device_fails_fast(monkeypatch):
    hidden, weight, labels = _inputs()
    monkeypatch.setattr(lce, "_CUT_CROSS_ENTROPY", _reference_cce)

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        lce.linear_cross_entropy(hidden, weight, labels, impl_backend="cut_cross_entropy")


def test_non_cuda_device_fails_before_importing_optional_package(monkeypatch):
    """NPU/CPU must fail fast instead of reporting a missing optional package."""
    hidden, weight, labels = _inputs()
    monkeypatch.setattr(lce, "_CUT_CROSS_ENTROPY", None)

    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        lce.linear_cross_entropy(hidden, weight, labels, impl_backend="cut_cross_entropy")


def test_multi_label_queries_are_rejected(fake_runtime):
    hidden, weight, labels = _inputs()
    multi_label = torch.stack((labels, labels), dim=-1)

    with pytest.raises(ValueError, match="one label per token"):
        lce.linear_cross_entropy(hidden, weight, multi_label, impl_backend="cut_cross_entropy")


def test_multi_label_queries_are_rejected_at_dispatch(monkeypatch):
    """The dispatcher must reject top-k queries before device/import checks."""
    hidden, weight, labels = _inputs()
    monkeypatch.setattr(lce, "_CUT_CROSS_ENTROPY", None)
    multi_label = torch.stack((labels, labels), dim=-1)

    with pytest.raises(ValueError, match="only supports one label per token"):
        lce.linear_cross_entropy(hidden, weight, multi_label, impl_backend="ml_cross_entropy")


def test_non_none_reduction_is_rejected(fake_runtime):
    hidden, weight, labels = _inputs()

    with pytest.raises(NotImplementedError, match="reduction='none'"):
        lce.linear_cross_entropy(hidden, weight, labels, reduction="mean", impl_backend="cut_cross_entropy")


def test_process_group_is_rejected(fake_runtime):
    hidden, weight, labels = _inputs()

    with pytest.raises(RuntimeError, match="tensor-parallel process groups"):
        lce.linear_cross_entropy(
            hidden,
            weight,
            labels,
            dist_process_group=object(),
            impl_backend="cut_cross_entropy",
        )


@pytest.mark.parametrize("backend", ["cut_cross_entropy", "ml_cross_entropy"])
def test_forward_backward_match_torch_reference(fake_runtime, backend):
    hidden, weight, labels = _inputs()
    temperature = 1.7
    grad_log_probs = torch.linspace(-1.0, 1.0, hidden.shape[0])
    grad_entropy = torch.linspace(0.5, -0.5, hidden.shape[0])

    log_probs, entropy = lce.linear_cross_entropy(hidden, weight, labels, temperature, impl_backend=backend)
    ref_log_probs, ref_entropy = _reference(hidden, weight, labels, temperature)
    torch.testing.assert_close(log_probs, ref_log_probs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy, ref_entropy, atol=1e-5, rtol=1e-5)

    torch.autograd.backward((log_probs, entropy), (grad_log_probs, grad_entropy))

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_hidden, ref_weight, labels, temperature)
    torch.autograd.backward((ref_log_probs, ref_entropy), (grad_log_probs, grad_entropy))

    torch.testing.assert_close(hidden.grad, ref_hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weight.grad, ref_weight.grad, atol=1e-5, rtol=1e-5)


def test_3d_hidden_is_flattened_like_triton(fake_runtime):
    hidden, weight, labels = _inputs()
    hidden = hidden.reshape(2, 3, 5)
    labels = labels.reshape(2, 3)

    log_probs, entropy = lce.linear_cross_entropy(hidden, weight, labels, impl_backend="cut_cross_entropy")

    # The triton single-label path also returns one value per flattened token.
    assert log_probs.shape == (6,)
    assert entropy.shape == (6,)


def test_ignored_labels_get_zero_outputs(fake_runtime):
    hidden, weight, labels = _inputs()
    labels = labels.clone()
    labels[1] = -100

    log_probs, entropy = lce.linear_cross_entropy(hidden, weight, labels, impl_backend="cut_cross_entropy")

    assert log_probs[1].item() == 0.0
    assert entropy[1].item() == 0.0


@pytest.mark.parametrize("backend", ["cut_cross_entropy", "ml_cross_entropy", "CUT_CROSS_ENTROPY"])
def test_vocab_parallel_engines_reject_full_vocab_backend(backend):
    with pytest.raises(ValueError, match="apple/ml-cross-entropy"):
        lce.validate_backend_for_vocab_parallel(backend)


@pytest.mark.parametrize("backend", [None, "triton", "tilelang", "liger", "torch"])
def test_vocab_parallel_validator_allows_other_backends(backend):
    lce.validate_backend_for_vocab_parallel(backend)


def test_temperature_scales_logits(fake_runtime):
    hidden, weight, labels = _inputs(requires_grad=False)

    log_probs, _ = lce.linear_cross_entropy(hidden, weight, labels, 0.5, impl_backend="cut_cross_entropy")
    ref_log_probs, _ = _reference(hidden, weight, labels, 0.5)
    torch.testing.assert_close(log_probs, ref_log_probs, atol=1e-5, rtol=1e-5)
