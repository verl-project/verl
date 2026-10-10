# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run actual pinned release CPU math without importing CUDA-only VeOmni modules.

Set VEOMNI_RELEASE_SOURCE to the unpacked veomni-0.1.11 directory, or install
that release. The source hash rejects other implementations; AST extraction
only avoids package initialization and FA imports, not the actual kernel math.
"""

import ast
import hashlib
import importlib.metadata
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from verl.utils.veomni.chunk_logprobs_compat import _needs_input_grad_backward_adapter


@pytest.fixture
def released_kernel():
    source = os.environ.get("VEOMNI_RELEASE_SOURCE")
    if source:
        path = Path(source) / "veomni/ops/kernels/cross_entropy/chunk_logprobs.py"
    else:
        try:
            dist = importlib.metadata.distribution("veomni")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("Install VeOmni 0.1.11 or set VEOMNI_RELEASE_SOURCE")
        path = Path(dist.locate_file("veomni/ops/kernels/cross_entropy/chunk_logprobs.py"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "b4c1caade0c444eadb920c6f6ae17422e59ccf7403b5499419a530aaa9dfdab3"
    ), "Expected unmodified published VeOmni 0.1.11 kernel"
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)]
    scope = dict(
        torch=torch,
        Optional=Optional,
        _FA_CE_AVAILABLE=False,
        get_parallel_state=lambda: SimpleNamespace(sp_enabled=False),
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    fn = scope["chunk_logprobs_function"]
    fn.kernel = scope["_ChunkedLinearLogProbs"]
    return fn


@pytest.mark.parametrize("strided", [False, True])
def test_released_kernel_matches_dense_forward_and_gradients(released_kernel, strided):
    torch.manual_seed(31)
    leaf = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    hidden = (leaf * 1.3).transpose(0, 1) if strided else leaf * 1.3
    weight = torch.randn(11, 8, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(6).reshape(hidden.shape[:-1]) % 11
    labels.flatten()[-1] = -100
    released_kernel.kernel.backward = staticmethod(_needs_input_grad_backward_adapter(released_kernel.kernel.backward))
    fn = released_kernel
    actual, entropy = fn(hidden, weight, labels, shift_labels=labels, temperature=0.8, chunk_size=2)
    dense = (hidden @ weight.T / 0.8).float().log_softmax(-1)
    valid = labels != -100
    expected = dense.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1).masked_fill(~valid, 0)
    expected_entropy = (-(dense.exp() * dense).sum(-1)).masked_fill(~valid, 0)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(entropy, expected_entropy, atol=2e-6, rtol=2e-6)
    signed = torch.linspace(-1, 1, labels.numel()).reshape_as(labels)
    grads = torch.autograd.grad((actual * signed + 0.17 * entropy).sum(), (leaf, weight), retain_graph=True)
    oracle = torch.autograd.grad((expected * signed + 0.17 * expected_entropy).sum(), (leaf, weight))
    for actual_grad, expected_grad in zip(grads, oracle, strict=True):
        assert actual_grad.norm() > 0
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)


def test_reproduces_unpatched_release_gradient_drop(released_kernel):
    leaf = torch.randn(2, 3, 8, requires_grad=True)
    hidden = (leaf * 2).transpose(0, 1)
    weight = torch.randn(11, 8, requires_grad=True)
    labels = torch.zeros(hidden.shape[:-1], dtype=torch.long)
    sampled, _ = released_kernel(hidden, weight, labels, shift_labels=labels)
    sampled.sum().backward()
    assert leaf.grad is None
    assert weight.grad.norm() > 0


def test_proxy_uses_authoritative_flags_without_copy_or_mutation():
    from verl.utils.veomni.chunk_logprobs_compat import _BackwardContextWithInputGradFlags

    hidden = torch.randn(3, 8)  # Saved flags may be false even for contiguous inputs on torch 2.13.
    weight = torch.randn(11, 8, requires_grad=True)
    labels = torch.zeros(3, dtype=torch.long)
    ctx = SimpleNamespace(saved_tensors=(hidden, weight, labels), needs_input_grad=(True, False), chunk_size=2)
    proxy = _BackwardContextWithInputGradFlags(ctx)
    assert proxy.saved_tensors[0].requires_grad
    assert not proxy.saved_tensors[1].requires_grad
    assert not hidden.requires_grad and weight.requires_grad
    assert proxy.saved_tensors[0].data_ptr() == hidden.data_ptr()
    assert proxy.saved_tensors[1].data_ptr() == weight.data_ptr()
    assert proxy.chunk_size == 2


def test_installer_is_idempotent(monkeypatch, released_kernel):
    import verl.utils.veomni.chunk_logprobs_compat as compat

    module = SimpleNamespace(_ChunkedLinearLogProbs=released_kernel.kernel)
    monkeypatch.setattr(compat.importlib.metadata, "version", lambda name: "0.1.11")
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: module)
    assert compat.install_chunk_logprobs_compat()
    installed = released_kernel.kernel.backward
    assert compat.install_chunk_logprobs_compat()
    assert released_kernel.kernel.backward is installed


def test_other_releases_do_not_import(monkeypatch):
    import verl.utils.veomni.chunk_logprobs_compat as compat

    monkeypatch.setattr(compat.importlib.metadata, "version", lambda name: "0.1.12")
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: pytest.fail("unexpected import"))
    assert not compat.install_chunk_logprobs_compat()


@pytest.mark.parametrize("train_hidden,train_weight", [(True, False), (False, True)])
@pytest.mark.parametrize("entropy_only", [False, True])
def test_frozen_inputs_and_single_output_gradients(released_kernel, train_hidden, train_weight, entropy_only):
    torch.manual_seed(41)
    leaf = torch.randn(2, 3, 8, requires_grad=train_hidden)
    hidden = (leaf * 1.3).transpose(0, 1)
    weight = torch.randn(11, 8, requires_grad=train_weight)
    labels = torch.zeros(hidden.shape[:-1], dtype=torch.long)
    released_kernel.kernel.backward = staticmethod(_needs_input_grad_backward_adapter(released_kernel.kernel.backward))
    sampled, entropy = released_kernel(hidden, weight, labels, shift_labels=labels, chunk_size=2)
    dense = (hidden @ weight.T).log_softmax(-1)
    objective = entropy.sum() if entropy_only else sampled.sum()
    oracle = (-(dense.exp() * dense).sum(-1)).sum() if entropy_only else dense[..., 0].sum()
    trainable = leaf if train_hidden else weight
    actual = torch.autograd.grad(objective, trainable, retain_graph=True)[0]
    expected = torch.autograd.grad(oracle, trainable)[0]
    assert actual.norm() > 0
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
