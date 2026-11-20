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
"""Tests for the input release in ``apply_patch_megatron_recomputation_backward``.

The chains below mimic Megatron's ``checkpointed_forward`` with ``recompute_method=uniform,
recompute_num_layers=1``: one ``CheckpointFunction`` per layer, the output of a segment being
the input of the next. Gradients are always compared bitwise against the same chain run
without checkpointing.
"""

import functools
import os

import pytest
import torch

if os.environ.get("VERL_REQUIRE_MCORE_TESTS"):
    import megatron.core.tensor_parallel.random as mcore_random
else:
    mcore_random = pytest.importorskip("megatron.core.tensor_parallel.random")

from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and not os.environ.get("VERL_REQUIRE_MCORE_TESTS"),
    reason="the patched backward is CUDA only",
)

N_SEG, T, H, E = 4, 256, 256, 8
ALL_SEGMENTS = tuple(range(N_SEG))
ACT_BYTES = T * H * 4  # one fp32 [T, H] activation


@pytest.fixture(autouse=True)
def restore_backward():
    """Each test installs the patch itself; restore Megatron's original backward afterwards."""
    orig = mcore_random.CheckpointFunction.__dict__["backward"]
    yield
    mcore_random.CheckpointFunction.backward = orig


def _layer(hidden, shared=None, *, weight, router, stash=None, seen=None):
    if seen is not None:
        seen.append(hidden)
    out = hidden @ weight
    if shared is not None:
        out = out + shared.sum(0, keepdim=True)
    if stash is not None:
        # Like a MoE token dispatcher keeping ``self.probs``: a tensor of the (recomputed)
        # graph that outlives the backward and keeps that graph, down to its leaves, alive.
        stash.append(torch.softmax(hidden @ router, dim=-1))
    return torch.tanh(out)


def _run_chain(use_checkpoint=True, make_shared=None, share_in=ALL_SEGMENTS, extra_loss=None, **layer_kw):
    """Run ``N_SEG`` segments, feeding ``make_shared(leaf)`` to the segments in ``share_in``.

    ``extra_loss(shared)`` returns ``(loss_term, leaf)`` and is built before the chain.
    Returns the grads of (x, shared leaf, extra leaf) and every segment's hidden input.
    """
    torch.manual_seed(0)
    weights = [torch.randn(H, H, device="cuda") * 0.1 for _ in range(N_SEG)]
    router = torch.randn(H, E, device="cuda")
    x = torch.randn(T, H, device="cuda", requires_grad=True)
    shared_leaf = torch.randn(T // 4, H, device="cuda", requires_grad=True)
    shared = make_shared(shared_leaf) if make_shared is not None else None
    extra = extra_loss(shared) if extra_loss is not None else None
    hidden = x * 1.0  # non-leaf, like real activations
    seg_inputs = []
    for i, w in enumerate(weights):
        seg_inputs.append(hidden)
        fn = functools.partial(_layer, weight=w, router=router, **layer_kw)
        args = (hidden, shared) if shared is not None and i in share_in else (hidden,)
        hidden = mcore_random.CheckpointFunction.apply(fn, False, *args) if use_checkpoint else fn(*args)
    loss = hidden.sum() if extra is None else hidden.sum() + extra[0]
    loss.backward()
    torch.cuda.synchronize()
    grads = (x.grad, shared_leaf.grad if shared is not None else None, extra[1].grad if extra else None)
    return grads, seg_inputs


def _assert_grads_equal(got, ref):
    for g, r in zip(got, ref, strict=True):
        assert (g is None) == (r is None)
        if r is not None:
            assert torch.equal(g, r)


def _retained_by(stash):
    """Bytes kept alive only through ``stash``, i.e. through the outliving recomputed graph."""
    torch.cuda.synchronize()
    with_stash = torch.cuda.memory_allocated()
    stash.clear()
    torch.cuda.synchronize()
    return with_stash - torch.cuda.memory_allocated()


def test_outliving_recompute_graph_does_not_retain_inputs():
    """The MoE leak the patch exists for: segment inputs and their grads must not stay alive."""
    stash = []
    _run_chain(stash=stash)  # Megatron's own backward
    leaked = _retained_by(stash)
    if leaked < N_SEG * ACT_BYTES:
        pytest.skip(f"Megatron's backward no longer retains the recompute inputs ({leaked} B)")

    apply_patch_megatron_recomputation_backward()
    ref, _ = _run_chain(use_checkpoint=False)
    grads, _ = _run_chain(stash=stash)
    _assert_grads_equal(grads, ref)
    assert _retained_by(stash) < ACT_BYTES  # only the stashed probs are left


def test_detached_leaves_released_inputs_intact():
    """Only the detached leaves' own references are dropped; the saved inputs keep their data."""
    apply_patch_megatron_recomputation_backward()
    seen = []
    grads, seg_inputs = _run_chain(seen=seen)
    # The forward pass sees the real inputs, the recompute sees detached leaves.
    detached = [t for t in seen if not any(t is s for s in seg_inputs)]
    assert len(detached) == N_SEG
    assert all(t.numel() == 0 and t.grad is None for t in detached)
    assert all(s.untyped_storage().nbytes() == ACT_BYTES for s in seg_inputs)
    ref, _ = _run_chain(use_checkpoint=False)
    _assert_grads_equal(grads, ref)


def test_input_shared_across_segments():
    """One tensor fed to every segment, like a cross-attention ``context`` or a cross-layer memory."""
    apply_patch_megatron_recomputation_backward()
    make_shared = lambda leaf: leaf * 1.0  # noqa: E731
    ref, _ = _run_chain(use_checkpoint=False, make_shared=make_shared)
    grads, _ = _run_chain(make_shared=make_shared)
    _assert_grads_equal(grads, ref)


def test_context_passed_through_segments():
    """Megatron's ``checkpointed_forward`` hands each chunk the ``context`` the previous chunk returned.

    A checkpoint that returns an input unchanged yields a view of it, so every chunk saves a view
    of the same storage.
    """
    apply_patch_megatron_recomputation_backward()

    def run(use_checkpoint):
        torch.manual_seed(0)
        weights = [torch.randn(H, H, device="cuda") * 0.1 for _ in range(N_SEG)]
        x = torch.randn(T, H, device="cuda", requires_grad=True)
        encoder_out = torch.randn(T // 4, H, device="cuda", requires_grad=True)
        hidden, context = x * 1.0, encoder_out * 1.0
        for w in weights:

            def fn(h, c, w=w):
                return torch.tanh(h @ w + c.sum(0, keepdim=True)), c

            if use_checkpoint:
                hidden, context = mcore_random.CheckpointFunction.apply(fn, False, hidden, context)
            else:
                hidden, context = fn(hidden, context)
        hidden.sum().backward()
        torch.cuda.synchronize()
        return x.grad, encoder_out.grad

    _assert_grads_equal(run(use_checkpoint=True), run(use_checkpoint=False))


def test_view_input_whose_base_is_read_by_a_later_backward():
    """The MTP aliasing case: a checkpoint input is a view of a tensor another node saved.

    Only the last segment gets the view, so its backward runs first. ``base * w`` saves ``base``
    for ``w``'s grad and is built before the chain, so the autograd engine runs it later.
    """
    apply_patch_megatron_recomputation_backward()

    def make_shared(leaf):
        return torch.cat([leaf, leaf])[: leaf.shape[0]]

    def extra_loss(shared):
        w = torch.randn(H, device="cuda", requires_grad=True)
        return (shared._base * w).sum(), w

    kw = dict(make_shared=make_shared, share_in=(N_SEG - 1,), extra_loss=extra_loss)
    ref, _ = _run_chain(use_checkpoint=False, **kw)
    grads, _ = _run_chain(**kw)
    _assert_grads_equal(grads, ref)
