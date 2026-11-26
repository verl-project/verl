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
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.kernel.linear_cross_entropy_config import BackwardEnum, _config

ct = pytest.importorskip("cuda.tile")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _inputs(vocab, dtype):
    torch.manual_seed(123)
    x = torch.randn(3, 13, 128, device="cuda", dtype=dtype).requires_grad_()
    w = (torch.randn(vocab, 128, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    labels = torch.randint(vocab, x.shape[:-1], device="cuda")
    labels[0, 0] = vocab - 1
    return x, w, labels


def _reference(x, w, labels, reduction):
    lp = (x.flatten(0, 1).float() @ w.float().T / 0.7).log_softmax(-1)
    selected = lp.gather(1, labels.reshape(-1, 1)).flatten()
    if reduction == "sum":
        selected = selected.sum()
    elif reduction == "mean":
        selected = selected.mean()
    return selected, -(lp.exp() * lp).sum(-1)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize(
    "method", [BackwardEnum._Total_Fuse_MN, BackwardEnum._Total_Separate, BackwardEnum._Split_Dlogits_N]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_outputs_and_gradients(monkeypatch, method, reduction, dtype):
    monkeypatch.setattr(_config, "_backward", method)
    vocab = 9511 if method == BackwardEnum._Split_Dlogits_N else 1031
    x, w, labels = _inputs(vocab, dtype)
    outputs = linear_cross_entropy(x, w, labels, 0.7, reduction, impl_backend="cutile")
    expected = _reference(x, w, labels, reduction)
    upstream = tuple(torch.randn_like(t) for t in expected)
    for actual, ref in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, ref, atol=2e-4, rtol=1e-5)
    for actual, ref in zip(
        torch.autograd.grad(outputs, (x, w), upstream), torch.autograd.grad(expected, (x, w), upstream), strict=True
    ):
        torch.testing.assert_close(actual, ref, atol=0.02 if dtype == torch.bfloat16 else 0.003, rtol=0.025)


def test_single_output_backward_kernels():
    from verl.utils.kernel.cutile import linear_cross_entropy as kernels

    x, w, labels = _inputs(129, torch.float16)
    x = x.view(-1, 128)
    labels = labels.view(-1)
    lp, entropy, maximum, total, mean = kernels.efficient_entropy_forward(x, w, labels, 0, 0.7, None)
    g, h = torch.randn_like(lp), torch.randn_like(entropy)
    expected = _reference(x.view(3, 13, 128), w, labels, "none")
    dx, dw = torch.autograd.grad(expected, (x, w), (g, h))
    for kernel, ref in [(kernels._d_hidden, dx), (kernels._d_weight, dw)]:
        out = torch.empty_like(ref)
        ct.launch(
            torch.cuda.current_stream(),
            (kernels._blocks(out.shape[0]), kernels._blocks(out.shape[1])),
            kernel,
            (x, w, labels, maximum, total, mean, g, h, out, 0, 1 / 0.7, 0),
        )
        torch.testing.assert_close(out, ref, atol=0.003, rtol=0.025)


def _tp_worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, device_id=torch.device("cuda", rank)
    )
    try:
        x, w, labels = _inputs(2050, torch.float16)
        local = w.detach().chunk(2)[rank].contiguous().requires_grad_()
        outputs = linear_cross_entropy(x, local, labels, 0.7, "none", dist.group.WORLD, impl_backend="cutile")
        expected = _reference(x, w, labels, "none")
        upstream = tuple(torch.randn_like(t) for t in expected)
        dx, dw = torch.autograd.grad(outputs, (x, local), upstream)
        dist.all_reduce(dx)
        ref_x, ref_w = torch.autograd.grad(expected, (x, w), upstream)
        for actual, ref in zip(outputs, expected, strict=True):
            torch.testing.assert_close(actual, ref, atol=2e-4, rtol=1e-5)
        torch.testing.assert_close(dx, ref_x, atol=0.003, rtol=0.025)
        torch.testing.assert_close(dw, ref_w.chunk(2)[rank], atol=0.003, rtol=0.025)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_vocab_parallel(tmp_path):
    mp.spawn(_tp_worker, args=((tmp_path / "store").as_uri(),), nprocs=2)


def test_triton_parity_and_dispatch(monkeypatch):
    from verl.utils.kernel import kernels

    x, w, labels = _inputs(1057, torch.bfloat16)
    ref = linear_cross_entropy(x, w, labels, 0.7, impl_backend="triton")
    upstream = tuple(torch.randn_like(t) for t in ref)
    ref_grads = torch.autograd.grad(ref, (x, w), upstream)

    def unexpected_triton_call(*args, **kwargs):
        pytest.fail("cuTile dispatched to Triton")

    monkeypatch.setattr(kernels, "efficient_entropy_forward", unexpected_triton_call)
    monkeypatch.setattr(kernels, "efficient_entropy_backward", unexpected_triton_call)
    actual = linear_cross_entropy(x, w, labels, 0.7, impl_backend="cutile")
    for a, b in zip(actual, ref, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=1e-5)
    for a, b in zip(torch.autograd.grad(actual, (x, w), upstream), ref_grads, strict=True):
        torch.testing.assert_close(a, b, atol=0.02, rtol=0.025)


@pytest.mark.parametrize("output_index", [0, 1])
def test_independent_output_gradients(output_index):
    x, w, labels = _inputs(1031, torch.float32)
    actual = linear_cross_entropy(x, w, labels, 0.7, impl_backend="cutile")[output_index]
    ref = _reference(x, w, labels, "none")[output_index]
    upstream = torch.randn_like(ref)
    for a, b in zip(
        torch.autograd.grad(actual, (x, w), upstream), torch.autograd.grad(ref, (x, w), upstream), strict=True
    ):
        torch.testing.assert_close(a, b, atol=8e-4, rtol=0.002)


def test_chunk_tuning_caches_full_and_tail_shapes(monkeypatch):
    from types import SimpleNamespace

    from verl.utils.kernel.cutile import linear_cross_entropy as kernels

    calls = []

    def select_wide_tile(configs, stream, grid, kernel, args, hints=None, **kwargs):
        if kernel is kernels._forward:
            return SimpleNamespace(best=SimpleNamespace(config=(64, 128, 64)))
        config = next(c for c in configs if (c.tm, c.tn, c.tk) == (128, 256, 32))
        calls.append(tuple(args(config)[8].shape))
        return SimpleNamespace(best=SimpleNamespace(config=config))

    monkeypatch.setattr(kernels, "exhaustive_search", select_wide_tile)
    monkeypatch.setattr(kernels, "_chunk_launch_cache", {})
    monkeypatch.setattr(_config, "_backward", BackwardEnum._Split_Dlogits_N)
    x, w, labels = _inputs(9511, torch.float16)
    for _ in range(2):
        actual = linear_cross_entropy(x, w, labels, 0.7, impl_backend="cutile")
        expected = _reference(x, w, labels, "none")
        upstream = tuple(torch.randn_like(t) for t in expected)
        grads = torch.autograd.grad(actual, (x, w), upstream)
        reference = torch.autograd.grad(expected, (x, w), upstream)
        for grad, ref in zip(grads, reference, strict=True):
            torch.testing.assert_close(grad, ref, atol=0.003, rtol=0.025)
    assert calls == [(39, 9504), (39, 7)]


def test_forward_tuning_caches_by_local_vocab(monkeypatch):
    from types import SimpleNamespace

    from verl.utils.kernel.cutile import linear_cross_entropy as kernels

    widths = []

    def select_tile(configs, stream, grid, kernel, args, **kwargs):
        tile = (64, 128, 64)
        assert kernel is kernels._forward and tile in configs
        widths.append(args(tile)[1].shape[0])
        return SimpleNamespace(best=SimpleNamespace(config=tile))

    monkeypatch.setattr(kernels, "exhaustive_search", select_tile)
    monkeypatch.setattr(kernels, "_forward_launch_cache", {})
    for vocab in (1025, 1025, 1057):
        x, w, labels = _inputs(vocab, torch.bfloat16)
        actual = linear_cross_entropy(x, w, labels, 0.7, impl_backend="cutile")
        for a, b in zip(actual, _reference(x, w, labels, "none"), strict=True):
            torch.testing.assert_close(a, b, atol=2e-4, rtol=1e-5)
    assert widths == [1025, 1057]
