#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

import torch
import torch.distributed as dist

try:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
except ImportError:
    # FIXME: remove these manually included paths
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")))
finally:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

import verl.utils.torch_functional as verl_F

compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

MAX_TEST_CASES = os.environ.get("MAX_TEST_CASES", 5)
VERIFY_TORCH_SELF = os.environ.get("VERIFY_TORCH_SELF", False)
LOW_MEMORY = os.environ.get("LOW_MEMORY", False)
LOW_MEMORY_DIV_FACTOR = os.environ.get("LOW_MEMORY_DIV_FACTOR", 16)


def run_torch_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, temperature: float, reduction="none"
) -> list[torch.Tensor]:
    # [num_tokens, vocab_size]
    if len(hidden.shape) > 2:
        hidden = hidden.view(-1, hidden.shape[-1])  # [num_tokens, hidden_size]
    if len(labels.shape) > 1:
        labels = labels.view(-1)
    logits = torch.matmul(
        hidden.to(torch.float32),
        weight.to(torch.float32) if weight.size(0) == hidden.size(1) else weight.T.to(torch.float32),
    )
    logits /= temperature
    pd = torch.nn.functional.softmax(logits, dim=-1)  # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1)  # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1)  # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)  # [num_tokens]
    logprobs = torch.neg(logprobs)
    return logprobs, entropy


class TorchEntropyTP(torch.autograd.Function):
    """
    it is used for testing the correctness of the kernel
    it is not efficient and is not recommended to use in practice
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        dist_process_group: torch.distributed.ProcessGroup,
    ):
        # weight has shape [vocab_size, hidden_size], hidden has shape [num_tokens, hidden_size]
        ctx.original_hidden_shape = hidden.shape
        if len(hidden.shape) > 2:
            hidden = hidden.view(-1, hidden.shape[-1])  # [num_tokens, hidden_size]
        if len(labels.shape) > 1:
            labels = labels.view(-1)

        logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32).T)  # [num_tokens, vocab_size]
        logits /= temperature
        whole_logits = torch.empty(
            (logits.shape[0], logits.shape[1] * dist.get_world_size(dist_process_group)),
            dtype=logits.dtype,
            device=logits.device,
        )
        whole_logits_ref = [
            whole_logits[:, i * logits.shape[1] : (i + 1) * logits.shape[1]]
            for i in range(dist.get_world_size(dist_process_group))
        ]
        dist.all_gather(whole_logits_ref, logits, group=dist_process_group)

        pd = torch.nn.functional.softmax(whole_logits, dim=-1)
        entropy_a = torch.logsumexp(whole_logits, dim=-1)  # [num_tokens]
        entropy_b = torch.sum(pd * whole_logits, dim=-1)  # [num_tokens]
        entropy = entropy_a - entropy_b

        logprobs = torch.nn.functional.cross_entropy(whole_logits, labels, reduction="none")
        logprobs = torch.neg(logprobs)

        ctx.save_for_backward(hidden, weight, labels, whole_logits, entropy_b)
        ctx.dist_process_group = dist_process_group
        ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, g_logprobs: torch.Tensor, g_entropy: torch.Tensor):
        hidden, weight, labels, whole_logits, entropy_b = ctx.saved_tensors
        dist_process_group = ctx.dist_process_group
        temperature = ctx.temperature
        batch_size, hidden_size = hidden.shape
        vocab_size, hidden_size = weight.shape
        rank = dist.get_rank(dist_process_group)

        # Compute softmax probabilities
        maximum, _ = torch.max(whole_logits, dim=-1, keepdim=True)
        exp_logits = torch.exp(whole_logits - maximum)
        accumulate = exp_logits.sum(dim=-1, keepdim=True)
        pd = exp_logits / accumulate

        # Gradient for entropy
        # entropy = entropy_a - entropy_b
        # entropy_a = log(sum(exp(logits)))
        # entropy_b = sum(pd * logits)
        # d_entropy_a/d_logits = pd
        # d_entropy_b/d_logits = pd * (logits - b.unsqueeze(1) + 1)
        # d_entropy/d_logits = d_entropy_a - d_entropy_b
        # d_entropy/d_logits = pd - pd * (logits - b.unsqueeze(1) + 1)
        # d_entropy/d_logits = -pd * (logits - b.unsqueeze(1))
        d_logits_entropy = g_entropy.unsqueeze(1) * (-pd * (whole_logits - entropy_b.unsqueeze(1)))

        # Gradient for logprobs
        # logprobs = -cross_entropy = -log(pd[labels])
        # d_logprobs/d_logits = (pd - one_hot(labels))
        one_hot = torch.zeros_like(whole_logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        g_logprobs = torch.neg(g_logprobs)
        d_logits_logprobs = g_logprobs.unsqueeze(1) * (pd - one_hot)
        # NOTE: This will lead to wrong result
        # d_logits_logprobs = g_logprobs.unsqueeze(1) * (pd - 1) * one_hot

        # Combine gradients
        d_logits = d_logits_entropy + d_logits_logprobs
        d_logits /= temperature

        # Get local slice of gradients
        local_d_logits = d_logits[:, rank * vocab_size : (rank + 1) * vocab_size]

        # Compute gradients for hidden and weight
        d_hidden = torch.matmul(local_d_logits, weight.to(torch.float32))
        d_weight = torch.matmul(local_d_logits.T, hidden.to(torch.float32))
        d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return d_hidden, d_weight, None, None, None


run_torch_entropy_tp = TorchEntropyTP.apply


class TestLinearCrossEntropy_TensorParallel:
    def __init__(self):
        dist.init_process_group(backend="nccl")
        self.group = dist.group.WORLD

        self.local_rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(device)
        print(f"[INFO]: Local rank: {self.local_rank}, World size: {self.world_size}")

    def initialize(self, test_case_idx: int, temperature: float = 1.5):
        self.test_case_idx = test_case_idx
        self.temperature = temperature

    def shutdown(self):
        dist.destroy_process_group()

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    def generate_hyper(self):
        global LOW_MEMORY, LOW_MEMORY_DIV_FACTOR, MAX_TEST_CASES

        self.dtype = torch.bfloat16
        if self.test_case_idx == 0:
            self.batch_size = 1
            self.num_tokens = 1937
            self.hidden_size = 3584
            self.vocab_size = 152064
        elif self.test_case_idx == 1:
            self.batch_size = 1
            self.num_tokens = 2169
            self.hidden_size = 896
            self.vocab_size = 151936
        elif self.test_case_idx == 2:
            self.batch_size = 1
            self.num_tokens = 1530
            self.hidden_size = 2048
            self.vocab_size = 32256
        elif self.test_case_idx == 3:
            self.batch_size = 1
            self.num_tokens = 1388
            self.hidden_size = 4096
            self.vocab_size = 102400
        elif self.test_case_idx == 4:
            self.batch_size = 1
            self.num_tokens = 8192
            self.hidden_size = 4096
            self.vocab_size = 102400
        else:
            raise ValueError(f"Invalid test case index: {self.test_case_idx}")
        if LOW_MEMORY:
            self.vocab_size = int(self.vocab_size / LOW_MEMORY_DIV_FACTOR)
        assert MAX_TEST_CASES <= 5, "MAX_TEST_CASES should be less than or equal to 5."

    def generate_forward_inputs(self):
        hidden = (
            torch.empty((self.batch_size, self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda")
            .uniform_(-0.5, 0.5)
            .requires_grad_()
        )
        weight = (
            torch.empty((self.vocab_size, self.hidden_size), dtype=self.dtype, device="cuda")
            .uniform_(-0.5, 0.5)
            .requires_grad_()
        )
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.num_tokens), device="cuda")
        return hidden, weight, labels

    def generate_backward_inputs(self):
        g_entropy = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5)
        g_logprobs = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-1, 1)
        return g_entropy, g_logprobs

    def verify_torch_itself(self, iterations: int = 5):
        self.cleanup()
        self.generate_hyper()

        for i in range(iterations):
            hidden, weight, labels = self.generate_forward_inputs()

            # NOTE: we need to manually synchronize hidden and labels among Process Group
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            # forward pass
            # Create a tensor to hold the gathered weights from all ranks
            # weight has shape [vocab_size, hidden_size]
            # We want to gather along the first dimension to get [vocab_size * world_size, hidden_size]

            # Create a single contiguous tensor to hold all gathered weights
            whole_weight = torch.empty(
                (self.vocab_size * self.world_size, self.hidden_size), dtype=weight.dtype, device=weight.device
            )

            # Create views into the tensor for each rank's portion
            whole_weight_views = [
                whole_weight[i * self.vocab_size : (i + 1) * self.vocab_size] for i in range(self.world_size)
            ]

            # Perform all_gather operation using the views
            dist.all_gather(whole_weight_views, weight, group=self.group)

            # Set requires_grad for autograd
            whole_weight.requires_grad_()

            (single_logprobs, single_entropy) = run_torch_entropy(hidden, whole_weight, labels, self.temperature)

            (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)

            torch.testing.assert_close(single_logprobs, tp_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(single_entropy, tp_entropy, atol=1e-4, rtol=1e-4)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            (single_d_hidden, single_d_weight) = torch.autograd.grad(
                (single_entropy, single_logprobs), (hidden, whole_weight), (g_entropy, g_logprobs), retain_graph=False
            )

            (tp_d_hidden, tp_d_weight) = torch.autograd.grad(
                (tp_entropy, tp_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
            )
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(tp_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            torch.testing.assert_close(tp_d_hidden, single_d_hidden, atol=1e-2, rtol=1e-4)
            # Extract the corresponding slice from single_d_weight for comparison
            # tp_d_weight has shape [vocab_size, hidden_size]
            # single_d_weight has shape [vocab_size * world_size, hidden_size]
            torch.testing.assert_close(
                tp_d_weight,
                single_d_weight[self.local_rank * self.vocab_size : (self.local_rank + 1) * self.vocab_size],
                atol=1e-2,
                rtol=1e-4,
            )

            # atol=1e-3, rtol=1e-4)
        if self.local_rank == 0:
            print("[PASS] torch TP correctness is verified")

    def check_torch_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)
        torch.cuda.synchronize()
        forward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_tp_hidden, d_tp_weight) = torch.autograd.grad(
            (tp_entropy, tp_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
        )
        torch.cuda.synchronize()
        backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # NOTE: all-reduce on hidden is conducted outside the kernel
        dist.all_reduce(d_tp_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print(f"[INFO]: Torch Forward pass peak memory: {forward_max_memory:.2f} MB")
            print(f"[INFO]: Torch Backward pass peak memory: {backward_max_memory:.2f} MB")

    def verify_kernel_correctness(self, iterations: int = 5):
        self.cleanup()
        self.generate_hyper()

        torch_forward_latency = list()
        torch_backward_latency = list()
        kernel_forward_latency = list()
        kernel_backward_latency = list()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(iterations):
            hidden, weight, labels = self.generate_forward_inputs()

            # NOTE: we need to manually synchronize hidden and labels among Process Group
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            start_event.record()
            (torch_logprobs, torch_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy(
                hidden, weight, labels, self.temperature, "none", self.group
            )
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-1, rtol=1e-2)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-1, rtol=1e-2)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            start_event.record()
            (torch_d_hidden, torch_d_weight) = torch.autograd.grad(
                (torch_entropy, torch_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(torch_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            start_event.record()
            (kernel_d_hidden, kernel_d_weight) = torch.autograd.grad(
                (kernel_entropy, kernel_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            kernel_backward_latency.append(start_event.elapsed_time(end_event))
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(kernel_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            torch.testing.assert_close(torch_d_hidden, kernel_d_hidden, atol=2e-2, rtol=4e-2)
            torch.testing.assert_close(torch_d_weight, kernel_d_weight, atol=2e-2, rtol=4e-2)

        # remove first latency
        torch_forward_latency = torch_forward_latency[1:]
        torch_backward_latency = torch_backward_latency[1:]
        kernel_forward_latency = kernel_forward_latency[1:]
        kernel_backward_latency = kernel_backward_latency[1:]

        if self.local_rank == 0:
            print("\n[PASS]: Verified kernel forward & backward correctness.")

            print(
                f"[INFO]: Forward pass: Torch implementation average time: "
                f"{sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms"
            )
            print(
                f"[INFO]: Backward pass: torch implementation average time: "
                f"{sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms"
            )
            print(
                f"[INFO]: Forward pass: Kernel implementation average time: "
                f"{sum(kernel_forward_latency) / len(kernel_forward_latency):.2f} ms"
            )
            print(
                f"[INFO]: Backward pass: kernel implementation average time: "
                f"{sum(kernel_backward_latency) / len(kernel_backward_latency):.2f} ms"
            )

    def check_kernel_storage(self):
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (kernel_logprobs, kernel_entropy) = linear_cross_entropy(
            hidden, weight, labels, self.temperature, "none", self.group
        )
        torch.cuda.synchronize()
        kernel_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad(
            (kernel_entropy, kernel_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
        )
        torch.cuda.synchronize()
        kernel_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # NOTE: all-reduce on hidden is conducted outside the kernel
        dist.all_reduce(d_kernel_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print(f"[INFO]: Kernel Forward pass peak memory: {kernel_max_memory:.2f} MB")
            print(f"[INFO]: Kernel Backward pass peak memory: {kernel_backward_max_memory:.2f} MB")


if __name__ == "__main__":
    # TP command: torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/kernels/test_linear_cross_entropy_tp.py

    # Check if running with torchrun (distributed mode)
    assert int(os.environ["WORLD_SIZE"]) > 1, (
        "[ERROR]: This test is designed to run in distributed mode with torchrun. Please use torchrun to "
        "execute this script."
    )
    torch.manual_seed(233376 + int(os.environ.get("RANK", 0)))

    # set_backward_method(BackwardEnum._Total_Fuse_MN)
    # set_backward_method(BackwardEnum._Split_Dlogits_N)

    test = TestLinearCrossEntropy_TensorParallel()
    for test_case_idx in range(MAX_TEST_CASES):
        print(f"[INFO] Running test case {test_case_idx}")
        test.initialize(test_case_idx)
        if VERIFY_TORCH_SELF:
            test.verify_torch_itself()
        test.check_torch_storage()
        test.verify_kernel_correctness()
        test.check_kernel_storage()

    test.shutdown()
