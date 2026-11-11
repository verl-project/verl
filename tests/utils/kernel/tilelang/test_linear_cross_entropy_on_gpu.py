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

pytest.importorskip("tilelang")

from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy as dispatched_linear_cross_entropy
from verl.utils.kernel.tilelang.linear_cross_entropy import linear_cross_entropy
from verl.utils.torch_functional import entropy_from_logits, logprobs_from_logits_v2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_linear_cross_entropy_forward_matches_pytorch():
    torch.manual_seed(0)
    num_tokens, vocab_size, hidden_size, topk = 1937, 152064, 3584, 4
    temperature = 1.3

    hidden = (
        torch.randn(
            num_tokens,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    linear = torch.nn.Linear(
        hidden_size,
        vocab_size,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    labels = torch.randint(vocab_size, (num_tokens, topk), device="cuda")
    labels[0, :] = labels[0, 0]

    actual_logprobs, actual_entropy = dispatched_linear_cross_entropy(
        hidden,
        linear.weight,
        labels,
        temperature,
        impl_backend="tilelang",
    )
    single_logprobs, single_entropy = dispatched_linear_cross_entropy(
        hidden,
        linear.weight,
        labels[:, 0],
        temperature,
        impl_backend="tilelang",
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


@pytest.mark.parametrize("use_entropy_grad", [False, True])
def test_linear_cross_entropy_backward_matches_pytorch(use_entropy_grad: bool):
    torch.manual_seed(1)
    num_tokens, vocab_size, hidden_size, topk = 1530, 32256, 2048, 4
    temperature = 0.8

    hidden = (
        torch.randn(
            num_tokens,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    ).requires_grad_()
    weight = (
        torch.randn(
            vocab_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    ).requires_grad_()
    labels = torch.randint(vocab_size, (num_tokens, topk), device="cuda")
    labels[0, :] = labels[0, 0]
    grad_logprobs = torch.randn(num_tokens, topk, device="cuda")
    grad_entropy = torch.randn(num_tokens, device="cuda")

    hidden_actual = hidden.detach().reshape(1, num_tokens, hidden_size).requires_grad_()
    weight_actual = weight.detach().clone().requires_grad_()
    actual_logprobs, actual_entropy = linear_cross_entropy(
        hidden_actual,
        weight_actual,
        labels.unsqueeze(0),
        temperature,
    )
    assert actual_logprobs.shape == (1, num_tokens, topk)
    assert actual_entropy.shape == (1, num_tokens)
    actual_outputs = (actual_logprobs, actual_entropy) if use_entropy_grad else (actual_logprobs,)
    actual_output_grads = (
        (grad_logprobs.unsqueeze(0), grad_entropy.unsqueeze(0))
        if use_entropy_grad
        else (grad_logprobs.unsqueeze(0),)
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
    torch.testing.assert_close(actual_grad_weight, expected_grad_weight, atol=1e-2, rtol=1e-2)


def _run_tensor_parallel_test(rank: int, world_size: int, init_file: str):
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
        labels = torch.randint(global_vocab_size, (num_tokens, topk), device=hidden.device)
        labels[0, :] = local_vocab_size + 7
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
        torch.manual_seed(5678 + dist.get_rank(tp_group))
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
            tp_group,
        )

        torch.testing.assert_close(actual_logprobs, expected_logprobs.float(), atol=2e-2, rtol=5e-3)
        torch.testing.assert_close(actual_entropy, expected_entropy.float(), atol=5e-2, rtol=5e-3)
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
                [vocab_parallel_log_probs_from_logits(local_logits_ref, labels[:, i]) for i in range(topk)],
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
                tp_group,
            )
            outputs_actual = (logprobs_actual, entropy_actual) if use_entropy_grad else (logprobs_actual,)
            actual_grad_hidden, actual_grad_weight = torch.autograd.grad(
                outputs_actual,
                (hidden_actual, weight_actual),
                output_grads,
            )
            dist.all_reduce(actual_grad_hidden, group=tp_group)

            torch.testing.assert_close(actual_grad_hidden, expected_grad_hidden, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(actual_grad_weight, expected_grad_weight, atol=1e-2, rtol=1e-2)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_linear_cross_entropy_matches_tensor_parallel_pytorch():
    world_size = 2
    if get_torch_device().device_count() < world_size:
        pytest.skip(f"TP test requires at least {world_size} accelerator devices")

    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = f"{temp_dir}/distributed_init"
        mp.spawn(
            _run_tensor_parallel_test,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
        )
