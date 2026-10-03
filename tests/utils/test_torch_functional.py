# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.torch_functional import (
    calculate_sum_pi_squared_from_logits,
    distributed_masked_mean,
    distributed_mean_max_min_std,
    expand_as_nested,
    masked_mean,
    scale_logits_by_temperature_,
)


def _worker_mean(rank: int, world_size: int, rendezvous_file: str):
    # 1) set GPU and init NCCL
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    # each rank holds tensor [rank+1]
    local = torch.tensor([float(rank + 1)], device=f"{get_device_name()}:{rank}")
    mean, gmax, gmin, gstd = distributed_mean_max_min_std(local, True, True, True)

    values = [float(i + 1) for i in range(world_size)]
    exp_mean = sum(values) / len(values)
    exp_max = max(values)
    exp_min = min(values)
    var = sum((x - exp_mean) ** 2 for x in values) / (len(values) - 1)
    exp_std = var**0.5

    # all ranks should see the same result
    assert torch.allclose(mean.cpu(), torch.tensor(exp_mean)), f"mean@{rank}"
    assert torch.allclose(gmax.cpu(), torch.tensor(exp_max)), f"max@{rank}"
    assert torch.allclose(gmin.cpu(), torch.tensor(exp_min)), f"min@{rank}"
    assert torch.allclose(gstd.cpu(), torch.tensor(exp_std)), f"std@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize(
    "value,mask,gt",
    [
        ([1.0, 2.0, 3.0, 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 0, 1], 2.5),
        ([1.0, 2.0, float("nan"), 4.0], [1, 0, 1, 0], float("nan")),
    ],
)
def test_masked_mean(value, mask, gt):
    res = masked_mean(torch.tensor(value), torch.tensor(mask))
    gt = torch.tensor(gt)
    assert torch.allclose(res, gt) or (torch.isnan(res) and torch.isnan(gt))


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_mean_max_min_std(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mean")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mean,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


def _worker_mask(rank: int, world_size: int, rendezvous_file: str):
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    # build per‐rank tensor and mask
    local_tensor = torch.tensor([rank * 2 + 1.0, rank * 2 + 2.0], device=f"{get_device_name()}:{rank}")
    if rank == 0:
        mask = torch.tensor([1, 0], device=f"{get_device_name()}:{rank}", dtype=torch.float32)
    else:
        mask = torch.tensor([0, 1], device=f"{get_device_name()}:{rank}", dtype=torch.float32)

    gmean = distributed_masked_mean(local_tensor, mask)

    valid_values = [1.0] + [2 * i + 2.0 for i in range(1, world_size)]
    expected_mean = sum(valid_values) / len(valid_values)
    assert torch.allclose(gmean.cpu(), torch.tensor(expected_mean)), f"masked_mean@{rank}"

    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_masked_mean(world_size, tmp_path):
    rendezvous_file = str(tmp_path / "rdzv_mask")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_worker_mask,
        args=(world_size, rendezvous_file),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("shape", [(8, 17), (3, 5, 32), (1, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_calculate_sum_pi_squared_from_logits(shape, dtype):
    """Σπ² computed from the logsumexp identity must match the naive softmax-then-square."""
    torch.manual_seed(0)
    logits = torch.randn(*shape, dtype=dtype) * 5.0  # broaden range to expose numerical issues

    actual = calculate_sum_pi_squared_from_logits(logits)
    expected = torch.softmax(logits, dim=-1).pow(2).sum(dim=-1)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    # Σπ² is always in (0, 1] for any non-empty distribution.
    assert torch.all(actual > 0)
    assert torch.all(actual <= 1.0 + 1e-5)


def test_calculate_sum_pi_squared_from_logits_extreme_values():
    """Stable under large-magnitude logits where naive exp would overflow."""
    # fp64 so the comparison isn't dominated by fp32 precision near |z|=1000
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -998.0]], dtype=torch.float64)
    actual = calculate_sum_pi_squared_from_logits(logits)
    expected = torch.softmax(logits, dim=-1).pow(2).sum(dim=-1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("temperature", [1.0, 0.7, 1e-10])
def test_scale_logits_by_temperature_inplace_matches_out_of_place(temperature):
    """Temperature scaling reuses logits storage and preserves forward/backward numerics."""
    torch.manual_seed(0)
    hidden_actual = torch.randn(2, 5, 7, dtype=torch.float64, requires_grad=True)
    weight_actual = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    hidden_expected = hidden_actual.detach().clone().requires_grad_(True)
    weight_expected = weight_actual.detach().clone().requires_grad_(True)

    logits_actual = hidden_actual @ weight_actual.T
    logits_expected = hidden_expected @ weight_expected.T
    denominator = torch.full((2, 1, 1), temperature, dtype=torch.float32)
    storage_ptr = logits_actual.data_ptr()

    returned = scale_logits_by_temperature_(logits_actual, denominator)
    expected = logits_expected / denominator.clamp(min=1e-8).to(logits_expected.dtype)

    assert returned is logits_actual
    assert returned.data_ptr() == storage_ptr
    torch.testing.assert_close(returned, expected)

    returned.logsumexp(dim=-1).sum().backward()
    expected.logsumexp(dim=-1).sum().backward()
    torch.testing.assert_close(hidden_actual.grad, hidden_expected.grad)
    torch.testing.assert_close(weight_actual.grad, weight_expected.grad)


def test_scale_logits_by_temperature_supports_squeezed_view():
    """Scaling a ``squeeze(0)`` view keeps autograd correct.

    Every eager call site scales a view rather than the raw model output::

        logits_rmpad = output.logits.squeeze(0)   # (total_nnz, vocab_size)
        scale_logits_by_temperature_(logits_rmpad, temperature.unsqueeze(-1))

    In-place writes through a view are only safe when no saved tensor is
    invalidated. That holds here because the LM head is a matmul, whose
    backward saves its *inputs*, not its output. The torchtitan engine keeps an
    out-of-place divide for the same pattern, so this test pins down which
    behaviour the shared helper is expected to have.
    """
    torch.manual_seed(0)
    hidden = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 6, dtype=torch.float64, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    view = (hidden @ weight.T).unsqueeze(0).squeeze(0)
    reference = (hidden_ref @ weight_ref.T).unsqueeze(0).squeeze(0)
    temperature = torch.full((3, 1), 0.7)
    storage_ptr = view.data_ptr()

    returned = scale_logits_by_temperature_(view, temperature)

    assert returned is view
    assert returned.data_ptr() == storage_ptr
    torch.testing.assert_close(returned, reference / temperature.to(reference.dtype))

    returned.logsumexp(dim=-1).sum().backward()
    (reference / temperature.to(reference.dtype)).logsumexp(dim=-1).sum().backward()
    torch.testing.assert_close(hidden.grad, hidden_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)


def test_scale_logits_by_temperature_rejects_output_saved_for_backward():
    """The helper must not silently corrupt gradients.

    ``div_`` is only valid when the tensor is not saved for its producer's
    backward. Softmax saves its output, so scaling it in place must raise
    rather than return a wrong gradient. This documents that misuse fails
    loudly, which is what makes the optimisation safe to apply broadly.
    """
    torch.manual_seed(0)
    hidden = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    saved_output = torch.softmax(hidden @ weight.T, dim=-1)

    scale_logits_by_temperature_(saved_output, torch.tensor([[0.5]]))
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved_output.sum().backward()


def test_expand_as_nested():
    a = torch.randn(2)
    b = torch.randn(3)
    c = torch.randn(4)
    nested_tensor = torch.nested.as_nested_tensor([a, b, c], layout=torch.jagged)
    tensor = torch.tensor([1, 2, 3])

    output = expand_as_nested(tensor, nested_tensor)

    assert output.values().tolist() == [1, 1, 2, 2, 2, 3, 3, 3, 3]
    assert torch.all(output.offsets() == nested_tensor.offsets()).item()

    # test exceptions
    with pytest.raises(AssertionError):
        expand_as_nested(tensor, tensor)

    other_tensor = torch.tensor([1, 2, 3, 4])

    with pytest.raises(AssertionError):
        expand_as_nested(other_tensor, nested_tensor)

    other_tensor = torch.tensor([[1, 2, 3]])

    with pytest.raises(AssertionError):
        expand_as_nested(other_tensor, nested_tensor)

    with pytest.raises(AssertionError):
        expand_as_nested(tensor, nested_tensor.unsqueeze(-1))
