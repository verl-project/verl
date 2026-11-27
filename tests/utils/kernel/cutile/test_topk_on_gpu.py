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

from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

pytest.importorskip("cuda.tile")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _compare(rank, world_size, k, grad_mode):
    torch.cuda.set_device(rank)
    torch.manual_seed(27)
    # Cross the forward split and the backward vocabulary chunk boundary.
    vocab = 9511
    hidden = (0.1 * torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    weights = (0.1 * torch.randn(vocab * world_size, 128, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    shard = weights.detach().chunk(world_size)[rank].requires_grad_()
    ids = torch.randint(vocab * world_size, (2, 7, 2 * k), device="cuda")[..., ::2]
    ids[..., 0] = vocab * world_size - 1
    if k > 1:
        ids[..., 1] = ids[..., 0]
        ids[..., 2] = 0
    group = dist.group.WORLD if world_size > 1 else None
    selected, entropy = linear_cross_entropy(hidden, shard, ids, 0.8, dist_process_group=group, impl_backend="cutile")
    logp = (hidden.float() @ weights.float().T / 0.8).log_softmax(-1)
    expected = logp.gather(-1, ids)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(selected, expected, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(entropy, expected_entropy, atol=2e-4, rtol=1e-4)
    g = torch.randn_like(expected)
    e = torch.randn_like(entropy)

    def loss(lp, ent):
        if grad_mode == "entropy":
            return (ent * e).sum()
        if grad_mode == "selected":
            return (lp * g).sum()
        return (lp * g).sum() + (ent * e).sum()

    dh, dw = torch.autograd.grad(loss(selected, entropy), (hidden, shard))
    ref_h, ref_w = torch.autograd.grad(loss(expected, expected_entropy), (hidden, weights))
    if group is not None:
        dist.all_reduce(dh)
    torch.testing.assert_close(dh, ref_h, atol=0.012, rtol=0.02)
    torch.testing.assert_close(dw, ref_w.chunk(world_size)[rank], atol=0.012, rtol=0.02)


@pytest.mark.parametrize(("k", "grad_mode"), [(1, "selected"), (5, "both"), (129, "selected"), (5, "entropy")])
@pytest.mark.parametrize("token_split", [False, True])
def test_topk(k, grad_mode, token_split, monkeypatch):
    if token_split:
        from verl.utils.kernel import linear_cross_entropy_config as config
        from verl.utils.kernel.cutile import linear_cross_entropy as kernels

        monkeypatch.setattr(config._config, "_backward", config.BackwardEnum._Split_Dlogits_M)
        monkeypatch.setattr(kernels, "_TOKEN_CHUNK_BYTES", 8 * 9511 * 2)
    _compare(0, 1, k, grad_mode)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_single_label_reductions(reduction):
    torch.manual_seed(11)
    hidden = (0.1 * torch.randn(2, 9, 128, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    vocab_size = 19200  # Exercise multiple backward vocabulary chunks for each reduction.
    weight = (0.1 * torch.randn(vocab_size, 128, device="cuda", dtype=torch.bfloat16)).requires_grad_()
    # A policy-label column sliced from a combined query tensor is strided.
    labels = torch.randint(vocab_size, (2, 9, 2), device="cuda")[..., 0]
    query_labels = labels if reduction == "none" else labels.unsqueeze(-1)
    actual, _ = linear_cross_entropy(hidden, weight, query_labels, reduction=reduction, impl_backend="cutile")
    logits = hidden.float().reshape(-1, 128) @ weight.float().T
    reference = -torch.nn.functional.cross_entropy(logits, labels.flatten(), reduction=reduction)
    torch.testing.assert_close(actual, reference, atol=2e-4, rtol=1e-4)
    for grad, ref in zip(
        torch.autograd.grad(actual.sum(), (hidden, weight)),
        torch.autograd.grad(reference.sum(), (hidden, weight)),
        strict=True,
    ):
        torch.testing.assert_close(grad, ref, atol=0.004, rtol=0.02)


@pytest.mark.parametrize("method", ["_Total_Fuse_MN", "_Total_Separate"])
def test_topk_backward_methods(method, monkeypatch):
    from verl.utils.kernel import linear_cross_entropy_config as kernels

    monkeypatch.setattr(kernels._config, "_backward", getattr(kernels.BackwardEnum, method))
    _compare(0, 1, 5, "both")


def _tp_worker(rank, init_method, token_split):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=init_method, rank=rank, world_size=2, device_id=torch.device("cuda", rank)
    )
    try:
        from verl.utils.kernel import linear_cross_entropy_config as config
        from verl.utils.kernel.cutile import linear_cross_entropy as kernels

        if token_split:
            config._config._backward = config.BackwardEnum._Split_Dlogits_M
            kernels._TOKEN_CHUNK_BYTES = 8 * 9511 * 2  # 14 tokens, including a 6-row tail.
        _compare(rank, 2, 5, "both")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
@pytest.mark.parametrize("token_split", [False, True])
def test_topk_tensor_parallel(tmp_path, token_split):
    torch.multiprocessing.spawn(_tp_worker, args=((tmp_path / "init").as_uri(), token_split), nprocs=2)


@pytest.mark.parametrize(
    ("queries", "dtype", "reduction", "message"),
    [
        (0, torch.int64, "none", "positive query count"),
        (3, torch.int32, "none", "int64"),
        (2, torch.int64, "sum", "reduction='none'"),
    ],
)
def test_invalid_queries(queries, dtype, reduction, message):
    x = torch.zeros(3, 128, device="cuda")
    labels = torch.zeros(3, queries, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=message):
        linear_cross_entropy(x, x, labels, reduction=reduction, impl_backend="cutile")
