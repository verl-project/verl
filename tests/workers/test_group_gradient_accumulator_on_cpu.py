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
"""Numerical equivalence to a full-batch GRPO update, including unequal response lengths."""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.utils.group_gradient_accumulator import GroupGradientAccumulator, megatron_gradient_buffers


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"])
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_streaming_matches_full_batch_gradient_and_adam_update(mode, order):
    torch.manual_seed(17)
    features = torch.randn(6, 7, 5)
    masks = torch.arange(7)[None, :] < torch.tensor([1, 2, 7, 6, 2, 4])[:, None]
    # Three two-response GRPO groups, rewards differ only within the group.
    rewards = torch.tensor([0.0, 1.0, 2.0, 0.0, -1.0, 3.0]).reshape(3, 2)
    advantages = ((rewards - rewards.mean(1, keepdim=True)) / rewards.std(1, keepdim=True)).flatten()
    initial = torch.randn(5) * 0.05
    reference = torch.nn.Parameter(initial.clone())
    streaming = torch.nn.Parameter(initial.clone())
    old_log_probs = features @ initial + 0.1 * torch.randn(6, 7)

    def loss(parameter, rows):
        log_probs = features[rows] @ parameter
        ratio = (log_probs - old_log_probs[rows]).exp()
        adv = advantages[rows, None]
        clipped = torch.maximum(-adv * ratio, -adv * ratio.clamp(0.8, 1.2))
        # An additional KL-like term shares the configured RL normalizer.
        terms = clipped + 0.02 * log_probs.square()
        mask = masks[rows]
        if mode == "token-mean":
            return (terms * mask).sum() / mask.sum(), mask.sum().item()
        sequence_losses = (terms * mask).sum(-1)
        if mode == "seq-mean-token-mean":
            sequence_losses = sequence_losses / mask.sum(-1)
        if mode == "seq-mean-token-sum-norm":
            sequence_losses = sequence_losses / 7
        return sequence_losses.mean(), len(rows)

    reference_optim = torch.optim.AdamW([reference], lr=0.01)
    streaming_optim = torch.optim.AdamW([streaming], lr=0.01)
    loss(reference, list(range(6)))[0].backward()
    reference_gradient = reference.grad.clone()
    expected_norm = torch.nn.utils.clip_grad_norm_([reference], 0.2)
    reference_optim.step()

    accumulator = GroupGradientAccumulator([str(i) for i in order])
    buffer = torch.zeros_like(streaming)
    for i in order:
        streaming_optim.zero_grad()
        group_loss, weight = loss(streaming, [2 * i, 2 * i + 1])
        group_loss.backward()
        buffer.copy_(streaming.grad)
        accumulator.add(str(i), {"buffer": buffer}, weight)
        # Model/optimizer stay unchanged; engine buffers can be reused immediately.
        buffer.zero_()
        torch.testing.assert_close(streaming, initial, rtol=0, atol=0)
        assert len(streaming_optim.state) == 0
    accumulator.restore({"buffer": buffer})
    torch.testing.assert_close(buffer, reference_gradient, rtol=2e-6, atol=2e-7)
    streaming.grad = buffer
    actual_norm = torch.nn.utils.clip_grad_norm_([streaming], 0.2)
    streaming_optim.step()
    torch.testing.assert_close(actual_norm, expected_norm)
    torch.testing.assert_close(streaming, reference, rtol=2e-6, atol=2e-7)


def test_incomplete_duplicate_foreign_and_repeated_commit_are_rejected():
    accumulator = GroupGradientAccumulator(["a", "b"])
    gradients = {"dense": torch.ones(3)}
    accumulator.add("a", gradients, 2)
    with pytest.raises(RuntimeError, match="1/2"):
        accumulator.restore(gradients)
    for group in ("a", "foreign"):
        with pytest.raises(ValueError, match="Unexpected or duplicate"):
            accumulator.add(group, gradients, 2)
    accumulator.add("b", gradients, 3)
    accumulator.restore(gradients)
    torch.testing.assert_close(gradients["dense"], torch.ones(3))
    with pytest.raises(RuntimeError):
        accumulator.restore(gradients)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_group_normalizers_do_not_modify_accumulation(weight):
    accumulator = GroupGradientAccumulator(["a"])
    with pytest.raises(ValueError, match="finite and positive"):
        accumulator.add("a", {"dense": torch.ones(1)}, weight)
    assert not accumulator.seen and accumulator.weight == 0


def test_layout_change_and_low_precision_gradients_are_rejected():
    accumulator = GroupGradientAccumulator(["a", "b"])
    accumulator.add("a", {"dense": torch.ones(2)}, 1)
    with pytest.raises(ValueError, match="layout changed"):
        accumulator.add("b", {"other": torch.ones(2)}, 1)
    with pytest.raises(ValueError, match="buffer changed"):
        accumulator.add("b", {"dense": torch.ones(3)}, 1)
    with pytest.raises(ValueError, match="FP32"):
        accumulator.add("b", {"dense": torch.ones(2, dtype=torch.bfloat16)}, 1)


def test_native_buffers_preserve_expert_and_partial_parameter_shards():
    # A reduce-scatter can leave the owned shard inside a larger physical buffer.
    dense = SimpleNamespace(grad_data=torch.arange(8, dtype=torch.float32))
    expert = SimpleNamespace(grad_data=torch.arange(4, dtype=torch.float32))
    module = SimpleNamespace(buffers=[dense], expert_parallel_buffers=[expert])
    buffers = megatron_gradient_buffers([module])
    accumulator = GroupGradientAccumulator(["a", "b"])
    accumulator.add("a", buffers, 2)
    dense.grad_data.mul_(3)
    expert.grad_data.mul_(3)
    accumulator.add("b", buffers, 1)
    dense.grad_data.zero_()
    expert.grad_data.zero_()
    accumulator.restore(buffers)
    torch.testing.assert_close(dense.grad_data[2:6], torch.arange(2, 6, dtype=torch.float32) * 5 / 3)
    torch.testing.assert_close(expert.grad_data, torch.arange(4, dtype=torch.float32) * 5 / 3)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_cp_finalization_normalizer_is_invariant_to_group_and_microbatch_boundaries(cp_size):
    from verl.utils.torch_functional import split_token_count

    # Includes counts smaller than CP and zero-loss padding lanes. The old floor-only
    # accounting changes the global divisor when group/micro-batch boundaries move.
    counts = torch.tensor([1, 3, 7, 12, 0, 25])
    counts_per_rank = [split_token_count(counts, cp_size, rank) for rank in range(cp_size)]
    torch.testing.assert_close(torch.stack(counts_per_rank).sum(0), counts)
    group_denominator = sum(share.sum() for share in counts_per_rank)
    full_denominator = sum(split_token_count(counts.sum(), cp_size, rank) for rank in range(cp_size))
    assert group_denominator == full_denominator == counts.sum()
    # The same unnormalized gradient must be divided by the same number.
    gradient_sum = torch.tensor([2.5, -1.25])
    torch.testing.assert_close(gradient_sum / group_denominator, gradient_sum / full_denominator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_host_resident_sums_match_device_resident_sums():
    """The accumulator may keep its FP32 sum in host memory to free GPU memory."""
    torch.manual_seed(23)
    gradients = [torch.randn(4, 5, device="cuda") for _ in range(2)]
    weights = [3.0, 7.0]
    names = ("a", "b")

    restored = {}
    for where in ("cuda", "cpu"):
        sums = {name: torch.zeros_like(grad, device=where) for name, grad in zip(names, gradients, strict=False)}
        accumulator = GroupGradientAccumulator(["g0", "g1"], sums=sums)
        for group_id, grad, weight in zip(("g0", "g1"), gradients, weights, strict=False):
            accumulator.add(group_id, dict(zip(names, (grad, grad), strict=False)), weight)
        buffers = {name: torch.zeros_like(grad) for name, grad in zip(names, gradients, strict=False)}
        accumulator.restore(buffers)
        restored[where] = {name: buffer.cpu() for name, buffer in buffers.items()}

    for name in names:
        assert torch.equal(restored["cpu"][name], restored["cuda"][name])
