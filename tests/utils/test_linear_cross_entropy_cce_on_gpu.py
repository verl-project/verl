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

"""GPU parity and memory tests for the apple/ml-cross-entropy backend.

Run on a CUDA machine with ``cut-cross-entropy`` installed::

    uv pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
    pytest tests/utils/test_linear_cross_entropy_cce_on_gpu.py -q
"""

import gc

import pytest
import torch

from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

try:
    import cut_cross_entropy  # noqa: F401

    HAS_CUT_CROSS_ENTROPY = True
except ImportError:
    HAS_CUT_CROSS_ENTROPY = False

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_CUT_CROSS_ENTROPY,
    reason="requires CUDA and the optional cut-cross-entropy package",
)

VOCAB_SIZE = 151936
HIDDEN_SIZE = 4096
DTYPE = torch.bfloat16


def _reference(hidden, weight, labels, temperature=1.0):
    logits = (hidden.float() @ weight.float().t()) / temperature
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    picked = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return picked, entropy


def _make_inputs(num_tokens, requires_grad=True):
    hidden = torch.empty(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device="cuda").uniform_(-0.5, 0.5)
    weight = torch.empty(VOCAB_SIZE, HIDDEN_SIZE, dtype=DTYPE, device="cuda").uniform_(-0.5, 0.5)
    if requires_grad:
        hidden.requires_grad_()
        weight.requires_grad_()
    labels = torch.randint(0, VOCAB_SIZE, (num_tokens,), device="cuda")
    return hidden, weight, labels


def _clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _peak_delta(fn):
    _clear()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return peak - base, output


def test_forward_matches_triton_and_torch():
    hidden, weight, labels = _make_inputs(256)

    lp_cce, entropy_cce = linear_cross_entropy(hidden, weight, labels, 1.0, impl_backend="cut_cross_entropy")
    lp_triton, entropy_triton = linear_cross_entropy(hidden, weight, labels, 1.0, impl_backend="triton")
    lp_torch, entropy_torch = _reference(hidden, weight, labels)

    torch.testing.assert_close(lp_cce, lp_triton, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(entropy_cce, entropy_triton, atol=5e-3, rtol=5e-4)
    torch.testing.assert_close(lp_cce, lp_torch, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(entropy_cce, entropy_torch, atol=5e-3, rtol=5e-4)


def test_backward_matches_triton_and_torch():
    torch.manual_seed(0)
    hidden, weight, labels = _make_inputs(256)
    grad_log_probs = torch.randn(256, dtype=torch.float32, device="cuda")
    grad_entropy = torch.randn(256, dtype=torch.float32, device="cuda")

    lp_cce, entropy_cce = linear_cross_entropy(hidden, weight, labels, 1.0, impl_backend="cut_cross_entropy")
    grad_hidden_cce, grad_weight_cce = torch.autograd.grad(
        (lp_cce, entropy_cce), (hidden, weight), (grad_log_probs, grad_entropy)
    )

    hidden.grad = None
    weight.grad = None
    lp_triton, entropy_triton = linear_cross_entropy(hidden, weight, labels, 1.0, impl_backend="triton")
    grad_hidden_triton, grad_weight_triton = torch.autograd.grad(
        (lp_triton, entropy_triton), (hidden, weight), (grad_log_probs, grad_entropy)
    )

    hidden.grad = None
    weight.grad = None
    lp_torch, entropy_torch = _reference(hidden, weight, labels)
    grad_hidden_torch, grad_weight_torch = torch.autograd.grad(
        (lp_torch, entropy_torch), (hidden, weight), (grad_log_probs, grad_entropy)
    )

    torch.testing.assert_close(grad_hidden_cce, grad_hidden_triton, atol=2e-2, rtol=4e-2)
    torch.testing.assert_close(grad_weight_cce, grad_weight_triton, atol=2e-2, rtol=4e-2)
    torch.testing.assert_close(grad_hidden_cce, grad_hidden_torch, atol=2e-2, rtol=4e-2)
    torch.testing.assert_close(grad_weight_cce, grad_weight_torch, atol=2e-2, rtol=4e-2)


def test_single_optimizer_step():
    torch.manual_seed(0)
    hidden, weight, labels = _make_inputs(256)
    optimizer = torch.optim.SGD([hidden, weight], lr=0.1)
    hidden_before = hidden.detach().clone()

    log_probs, entropy = linear_cross_entropy(hidden, weight, labels, 1.0, impl_backend="cut_cross_entropy")
    loss = -(log_probs.mean() + entropy.mean())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert torch.isfinite(hidden).all() and torch.isfinite(weight).all()
    assert not torch.equal(hidden.detach(), hidden_before)


def test_dense_model_use_fused_kernels_training_step():
    """The FSDP dense-model path must train one step with the CCE backend."""
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    from verl.models.transformers.monkey_patch import patch_forward_with_backends

    _clear()
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=1024,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config).to(device="cuda", dtype=DTYPE)
    patch_forward_with_backends(model, use_fused_kernels=True, fused_kernels_backend="cut_cross_entropy")
    assert model._verl_fused_kernels_backend == "cut_cross_entropy"

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 16), device="cuda")
    output = model(input_ids=input_ids, labels=input_ids, temperature=1.0, return_dict=True)
    loss = -(output.log_probs.mean() + output.entropy.mean())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight).all()

    del model, optimizer
    _clear()


def test_peak_memory_below_unfused_and_not_above_triton():
    num_tokens = 1024
    hidden, weight, labels = _make_inputs(num_tokens, requires_grad=False)

    def unfused():
        logits = hidden.float() @ weight.float().t()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1), -(probs * log_probs).sum(dim=-1)

    # Warm up Triton/CCE autotuning before measuring.
    linear_cross_entropy(hidden, weight, labels, impl_backend="triton")
    linear_cross_entropy(hidden, weight, labels, impl_backend="cut_cross_entropy")

    unfused_delta, _ = _peak_delta(unfused)
    triton_delta, _ = _peak_delta(lambda: linear_cross_entropy(hidden, weight, labels, impl_backend="triton"))
    cce_delta, _ = _peak_delta(lambda: linear_cross_entropy(hidden, weight, labels, impl_backend="cut_cross_entropy"))

    mib = 2**20
    print(
        f"peak delta: unfused={unfused_delta / mib:.1f} MiB, "
        f"triton={triton_delta / mib:.1f} MiB, cce={cce_delta / mib:.1f} MiB"
    )

    assert cce_delta < unfused_delta * 0.5
    # The vocabulary-chunked entropy estimator adds a bounded scratch buffer.
    assert cce_delta <= triton_delta + 64 * 2**20
