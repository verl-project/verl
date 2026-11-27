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
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_device_available,
)
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.torch_functional import entropy_from_logits, logprobs_from_logits_v2

DEVICE_NAME = get_device_name()
if DEVICE_NAME == "npu":
    import megatron_adaptor  # noqa: F401

pytestmark = pytest.mark.skipif(
    DEVICE_NAME not in ("cuda", "npu") or not is_device_available(),
    reason="A supported accelerator (CUDA GPU or Ascend NPU) is required",
)


def _require_backend(impl_backend: str):
    if impl_backend == "tilelang":
        pytest.importorskip("tilelang")
    elif impl_backend == "triton":
        pytest.importorskip("triton")


@pytest.mark.parametrize("impl_backend", ["tilelang", "triton"])
@pytest.mark.parametrize("num_splits", [1, 5, 31, 32, 33, 297])
def test_local_reduce_matches_pytorch(impl_backend: str, num_splits: int):
    _require_backend(impl_backend)
    torch.manual_seed(num_splits)
    num_tokens = 65
    partial_max = torch.randn(
        num_tokens,
        num_splits,
        device=DEVICE_NAME,
        dtype=torch.float32,
    )
    partial_sum = torch.rand_like(partial_max)
    partial_weighted_sum = torch.randn_like(partial_max)
    actual_max = torch.empty(num_tokens, device=DEVICE_NAME)
    actual_sum = torch.empty_like(actual_max)
    actual_weighted_sum = torch.empty_like(actual_max)

    if impl_backend == "tilelang":
        from verl.utils.kernel.linear_cross_entropy.tilelang import _get_backend_kernels

        backend_kernels = _get_backend_kernels(DEVICE_NAME)
    else:
        if DEVICE_NAME != "npu":
            pytest.skip("Triton local_reduce helper is only exposed on NPU")
        from verl.utils.kernel.linear_cross_entropy.triton import _get_backend_kernels

        backend_kernels = _get_backend_kernels(DEVICE_NAME)

    kernel = backend_kernels.linear_cross_entropy_fwd_local_reduce(num_splits)
    for shift in (0.0, -25.0):
        shifted_partial_max = partial_max + shift
        kernel(
            shifted_partial_max,
            partial_sum,
            partial_weighted_sum,
            actual_max,
            actual_sum,
            actual_weighted_sum,
        )

        expected_max = shifted_partial_max.max(dim=-1).values
        coefficients = torch.exp(shifted_partial_max - expected_max[:, None])
        expected_sum = torch.sum(coefficients * partial_sum, dim=-1)
        expected_weighted_sum = torch.sum(
            coefficients * partial_weighted_sum,
            dim=-1,
        )
        torch.testing.assert_close(actual_max, expected_max)
        torch.testing.assert_close(actual_sum, expected_sum, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            actual_weighted_sum,
            expected_weighted_sum,
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("impl_backend", ["tilelang", "triton"])
def test_linear_cross_entropy_forward_matches_pytorch(impl_backend: str):
    _require_backend(impl_backend)
    torch.manual_seed(0)
    num_tokens, vocab_size, hidden_size, topk = 1937, 152064, 3584, 4
    temperature = 1.3

    hidden = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=DEVICE_NAME,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    linear = torch.nn.Linear(
        hidden_size,
        vocab_size,
        bias=False,
        device=DEVICE_NAME,
        dtype=torch.bfloat16,
    )
    labels = torch.randint(
        vocab_size,
        (num_tokens, topk),
        device=DEVICE_NAME,
    )
    labels[0, :] = labels[0, 0]

    actual_logprobs, actual_entropy = linear_cross_entropy(
        hidden,
        linear.weight,
        labels,
        temperature,
        impl_backend=impl_backend,
    )
    single_logprobs, single_entropy = linear_cross_entropy(
        hidden,
        linear.weight,
        labels[:, 0],
        temperature,
        impl_backend=impl_backend,
    )

    logits = linear(hidden) / temperature
    expected_logprobs = torch.stack(
        [logprobs_from_logits_v2(logits, labels[:, i]) for i in range(topk)],
        dim=-1,
    )
    expected_entropy = entropy_from_logits(logits)

    assert actual_logprobs.dtype == torch.float32
    assert actual_entropy.dtype == torch.float32
    assert single_logprobs.shape == labels[:, 0].shape
    torch.testing.assert_close(single_logprobs, actual_logprobs[:, 0])
    torch.testing.assert_close(single_entropy, actual_entropy)
    torch.testing.assert_close(
        actual_logprobs[0],
        actual_logprobs[0, 0].expand(topk),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual_logprobs,
        expected_logprobs.float(),
        atol=2e-2,
        rtol=5e-3,
    )
    torch.testing.assert_close(
        actual_entropy,
        expected_entropy.float(),
        atol=5e-2,
        rtol=5e-3,
    )

    logits_fp32 = logits.float()
    expected_logprobs_fp32 = F.log_softmax(logits_fp32, dim=-1).gather(-1, labels)
    probabilities_fp32 = F.softmax(logits_fp32, dim=-1)
    expected_entropy_fp32 = torch.logsumexp(logits_fp32, dim=-1) - torch.sum(
        probabilities_fp32 * logits_fp32,
        dim=-1,
    )
    torch.testing.assert_close(actual_logprobs, expected_logprobs_fp32, atol=2e-3, rtol=2e-4)
    torch.testing.assert_close(actual_entropy, expected_entropy_fp32, atol=2e-4, rtol=2e-5)


@pytest.mark.parametrize("impl_backend", ["tilelang", "triton"])
@pytest.mark.parametrize("use_entropy_grad", [False, True])
def test_linear_cross_entropy_backward_matches_pytorch(impl_backend: str, use_entropy_grad: bool):
    _require_backend(impl_backend)
    torch.manual_seed(1)
    num_tokens, vocab_size, hidden_size, topk = 1530, 32256, 2048, 4
    temperature = 0.8

    hidden = (
        torch.randn(
            num_tokens,
            hidden_size,
            device=DEVICE_NAME,
            dtype=torch.bfloat16,
        )
        * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(
            vocab_size,
            hidden_size,
            device=DEVICE_NAME,
            dtype=torch.bfloat16,
        )
        * 0.1
    ).requires_grad_()
    labels = torch.randint(
        vocab_size,
        (num_tokens, topk),
        device=DEVICE_NAME,
    )
    labels[0, :] = labels[0, 0]
    grad_logprobs = torch.randn(num_tokens, topk, device=DEVICE_NAME)
    grad_entropy = torch.randn(num_tokens, device=DEVICE_NAME)

    hidden_actual = hidden.detach().reshape(1, num_tokens, hidden_size).requires_grad_()
    weight_actual = weight.detach().clone().requires_grad_()
    actual_logprobs, actual_entropy = linear_cross_entropy(
        hidden_actual,
        weight_actual,
        labels.unsqueeze(0),
        temperature,
        impl_backend=impl_backend,
    )
    assert actual_logprobs.shape == (1, num_tokens, topk)
    assert actual_entropy.shape == (1, num_tokens)
    actual_outputs = (actual_logprobs, actual_entropy) if use_entropy_grad else (actual_logprobs,)
    actual_output_grads = (
        (grad_logprobs.unsqueeze(0), grad_entropy.unsqueeze(0)) if use_entropy_grad else (grad_logprobs.unsqueeze(0),)
    )
    actual_grad_hidden, actual_grad_weight = torch.autograd.grad(
        actual_outputs,
        (hidden_actual, weight_actual),
        actual_output_grads,
    )

    hidden_ref = hidden.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    logits_ref = F.linear(hidden_ref, weight_ref) / temperature
    logprobs_ref = torch.stack(
        [logprobs_from_logits_v2(logits_ref, labels[:, i]) for i in range(topk)],
        dim=-1,
    )
    entropy_ref = entropy_from_logits(logits_ref)
    expected_outputs = (logprobs_ref, entropy_ref) if use_entropy_grad else (logprobs_ref,)
    expected_grad_hidden, expected_grad_weight = torch.autograd.grad(
        expected_outputs,
        (hidden_ref, weight_ref),
        (grad_logprobs, grad_entropy) if use_entropy_grad else (grad_logprobs,),
    )

    torch.testing.assert_close(
        actual_grad_hidden.reshape_as(expected_grad_hidden),
        expected_grad_hidden,
        atol=3e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_grad_weight,
        expected_grad_weight,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("impl_backend", ["tilelang", "triton"])
@pytest.mark.parametrize("use_entropy_grad", [False, True])
def test_linear_cross_entropy_single_label_and_unaligned_vocab_boundaries(
    impl_backend: str,
    use_entropy_grad: bool,
):
    _require_backend(impl_backend)
    torch.manual_seed(7)
    # 9503 exercises partial forward splits (9503 % 512 != 0), multiple backward
    # chunks (> 8192), and partial tail tiles (9503 % 128 != 0).
    num_tokens, vocab_size, hidden_size = 131, 9503, 256
    temperature = 0.9

    hidden_base = torch.randn(num_tokens, hidden_size, device=DEVICE_NAME, dtype=torch.bfloat16) * 0.1
    weight_base = torch.randn(vocab_size, hidden_size, device=DEVICE_NAME, dtype=torch.bfloat16) * 0.1
    # Force strictly negative logits on early rows (spread across 32 channels so
    # weights stay O(0.3) in bf16) so out-of-bounds zero-padding or 0-initialized
    # row maxima would introduce large logsumexp errors.
    hidden_base[:8, :32] = -1.0
    weight_base[:, :32] = 0.35 * temperature
    weight_base[vocab_size - 1, :32] = 0.05 * temperature

    # Slice from a 2-column tensor so the 1D single-label input is non-contiguous.
    labels_pair = torch.randint(vocab_size, (num_tokens, 2), device=DEVICE_NAME)
    labels = labels_pair[:, 0]
    boundary_ids = [0, 511, 512, 8191, 8192, vocab_size - 1]
    for idx, tok_id in enumerate(boundary_ids):
        labels[idx] = tok_id

    grad_logprobs = torch.randn(num_tokens, device=DEVICE_NAME)
    grad_entropy = torch.randn(num_tokens, device=DEVICE_NAME)

    hidden_actual = hidden_base.detach().clone().requires_grad_()
    weight_actual = weight_base.detach().clone().requires_grad_()
    actual_logprobs, actual_entropy = linear_cross_entropy(
        hidden_actual,
        weight_actual,
        labels,
        temperature,
        impl_backend=impl_backend,
    )
    assert actual_logprobs.shape == (num_tokens,)
    assert actual_entropy.shape == (num_tokens,)

    hidden_ref = hidden_base.detach().clone().requires_grad_()
    weight_ref = weight_base.detach().clone().requires_grad_()
    logits_fp32 = (hidden_ref.float() @ weight_ref.float().T) / temperature
    expected_logprobs = -F.cross_entropy(logits_fp32, labels, reduction="none")
    probs_fp32 = F.softmax(logits_fp32, dim=-1)
    expected_entropy = torch.logsumexp(logits_fp32, dim=-1) - torch.sum(probs_fp32 * logits_fp32, dim=-1)

    torch.testing.assert_close(actual_logprobs, expected_logprobs, atol=2e-3, rtol=2e-4)
    torch.testing.assert_close(actual_entropy, expected_entropy, atol=2e-3, rtol=2e-4)

    actual_outputs = (actual_logprobs, actual_entropy) if use_entropy_grad else (actual_logprobs,)
    expected_outputs = (expected_logprobs, expected_entropy) if use_entropy_grad else (expected_logprobs,)
    upstream_grads = (grad_logprobs, grad_entropy) if use_entropy_grad else (grad_logprobs,)

    actual_grad_hidden, actual_grad_weight = torch.autograd.grad(
        actual_outputs,
        (hidden_actual, weight_actual),
        upstream_grads,
    )
    expected_grad_hidden, expected_grad_weight = torch.autograd.grad(
        expected_outputs,
        (hidden_ref, weight_ref),
        upstream_grads,
    )

    torch.testing.assert_close(actual_grad_hidden, expected_grad_hidden, atol=1.5e-2, rtol=1.5e-2)
    torch.testing.assert_close(actual_grad_weight, expected_grad_weight, atol=1.5e-2, rtol=1.5e-2)


def _run_tensor_parallel_test(rank: int, world_size: int, init_file: str, impl_backend: str):
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    from megatron.core import ModelParallelConfig
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    from verl.utils.megatron.tensor_parallel import (
        vocab_parallel_entropy,
        vocab_parallel_log_probs_from_logits,
    )

    mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp_group = mpu.get_tensor_model_parallel_group()

    try:
        torch.manual_seed(1234)
        num_tokens, global_vocab_size, hidden_size, topk = 1937, 152064, 3584, 4
        local_vocab_size = global_vocab_size // world_size
        temperature = 1.3

        hidden = (
            torch.randn(
                num_tokens,
                hidden_size,
                device=get_device_name(),
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        labels = torch.randint(
            global_vocab_size,
            (num_tokens, topk),
            device=hidden.device,
        )
        labels[0, :] = local_vocab_size + 7
        labels[1] = torch.tensor(
            [0, local_vocab_size - 1, local_vocab_size, global_vocab_size - 1],
            device=hidden.device,
        )
        dist.broadcast(hidden, src=0, group=tp_group)
        dist.broadcast(labels, src=0, group=tp_group)

        config = ModelParallelConfig(
            params_dtype=torch.bfloat16,
            use_cpu_initialization=False,
            perform_initialization=False,
            gradient_accumulation_fusion=False,
            sequence_parallel=False,
        )
        linear = ColumnParallelLinear(
            hidden_size,
            global_vocab_size,
            config=config,
            init_method=torch.nn.init.xavier_normal_,
            bias=False,
            gather_output=False,
            tp_group=tp_group,
        )
        torch.manual_seed(5678 + rank)
        with torch.no_grad():
            linear.weight.normal_(mean=0.0, std=0.02)
        local_logits, _ = linear(hidden)
        local_logits = local_logits / temperature

        expected_entropy = vocab_parallel_entropy(local_logits)
        expected_logprobs = torch.stack(
            [vocab_parallel_log_probs_from_logits(local_logits, labels[:, i]) for i in range(topk)],
            dim=-1,
        )
        actual_logprobs, actual_entropy = linear_cross_entropy(
            hidden,
            linear.weight,
            labels,
            temperature,
            dist_process_group=tp_group,
            impl_backend=impl_backend,
        )

        torch.testing.assert_close(
            actual_logprobs,
            expected_logprobs.float(),
            atol=2e-2,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            actual_entropy,
            expected_entropy.float(),
            atol=5e-2,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            actual_logprobs[0],
            actual_logprobs[0, 0].expand(topk),
            atol=0,
            rtol=0,
        )

        for use_entropy_grad in (False, True):
            torch.manual_seed(9102)
            grad_logprobs = torch.randn(num_tokens, topk, device=hidden.device)
            grad_entropy = torch.randn(num_tokens, device=hidden.device)
            dist.broadcast(grad_logprobs, src=0, group=tp_group)
            dist.broadcast(grad_entropy, src=0, group=tp_group)
            output_grads = (grad_logprobs, grad_entropy) if use_entropy_grad else (grad_logprobs,)

            hidden_ref = hidden.detach().clone().requires_grad_()
            local_logits_ref, _ = linear(hidden_ref)
            local_logits_ref = local_logits_ref / temperature
            entropy_ref = vocab_parallel_entropy(local_logits_ref)
            logprobs_ref = torch.stack(
                [
                    vocab_parallel_log_probs_from_logits(
                        local_logits_ref,
                        labels[:, i],
                    )
                    for i in range(topk)
                ],
                dim=-1,
            )
            outputs_ref = (logprobs_ref, entropy_ref) if use_entropy_grad else (logprobs_ref,)
            expected_grad_hidden, expected_grad_weight = torch.autograd.grad(
                outputs_ref,
                (hidden_ref, linear.weight),
                output_grads,
            )

            hidden_actual = hidden.detach().clone().requires_grad_()
            weight_actual = linear.weight.detach().clone().requires_grad_()
            logprobs_actual, entropy_actual = linear_cross_entropy(
                hidden_actual,
                weight_actual,
                labels,
                temperature,
                dist_process_group=tp_group,
                impl_backend=impl_backend,
            )
            outputs_actual = (logprobs_actual, entropy_actual) if use_entropy_grad else (logprobs_actual,)
            actual_grad_hidden, actual_grad_weight = torch.autograd.grad(
                outputs_actual,
                (hidden_actual, weight_actual),
                output_grads,
            )
            dist.all_reduce(actual_grad_hidden, group=tp_group)

            torch.testing.assert_close(
                actual_grad_hidden,
                expected_grad_hidden,
                atol=1e-2,
                rtol=1e-2,
            )
            torch.testing.assert_close(
                actual_grad_weight,
                expected_grad_weight,
                atol=1e-2,
                rtol=1e-2,
            )
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("impl_backend", ["tilelang", "triton"])
def test_linear_cross_entropy_matches_tensor_parallel_pytorch(impl_backend: str):
    _require_backend(impl_backend)
    world_size = 2
    if get_torch_device().device_count() < world_size:
        pytest.skip(f"TP test requires at least {world_size} accelerator devices")

    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = f"{temp_dir}/distributed_init"
        mp.spawn(
            _run_tensor_parallel_test,
            args=(world_size, init_file, impl_backend),
            nprocs=world_size,
            join=True,
        )
