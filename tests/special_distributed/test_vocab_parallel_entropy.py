# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""
Test for vocab_parallel_entropy and its chunked version.
This test requires multi-GPU environment with tensor parallelism.

NOTE: This test directly embeds the implementation to avoid heavy dependencies.
"""

import os
import sys
from typing import Optional

import torch
import torch.distributed as dist


# Mock mpu for testing
class MockMPU:
    """Mock tensor parallel group for standalone testing."""

    _group = None

    @classmethod
    def get_tensor_model_parallel_group(cls):
        if cls._group is None:
            cls._group = dist.group.WORLD
        return cls._group


# Direct implementation to avoid importing heavy dependencies
class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=MockMPU.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=MockMPU.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = (softmax_logits * vocab_parallel_logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_softmax_times_logits, group=MockMPU.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


class _VocabParallelEntropyChunked(torch.autograd.Function):
    """Memory-efficient chunked version of VocabParallelEntropy."""

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, chunk_size: int = 1024) -> torch.Tensor:
        total_nnz, local_vocab_size = vocab_parallel_logits.shape
        tp_group = MockMPU.get_tensor_model_parallel_group()

        # Step 1: Compute max in chunks
        logits_max_chunks = []
        for i in range(0, total_nnz, chunk_size):
            chunk = vocab_parallel_logits[i : i + chunk_size]
            chunk_max = chunk.max(dim=-1, keepdim=True).values
            logits_max_chunks.append(chunk_max)

        logits_max = torch.cat(logits_max_chunks, dim=0)
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        del logits_max_chunks

        # Step 2: Compute exp and sum in chunks
        normalized_exp_logits_list = []
        normalized_sum_exp_list = []

        for i in range(0, total_nnz, chunk_size):
            chunk = vocab_parallel_logits[i : i + chunk_size]
            chunk_max = logits_max[i : i + chunk_size]

            normalized_chunk = chunk - chunk_max
            exp_chunk = normalized_chunk.exp_()
            del normalized_chunk

            sum_exp = exp_chunk.sum(dim=-1, keepdim=True)

            normalized_exp_logits_list.append(exp_chunk)
            normalized_sum_exp_list.append(sum_exp)

        normalized_exp_logits = torch.cat(normalized_exp_logits_list, dim=0)
        normalized_sum_exp_logits = torch.cat(normalized_sum_exp_list, dim=0)
        del normalized_exp_logits_list, normalized_sum_exp_list

        dist.all_reduce(normalized_sum_exp_logits, group=tp_group)

        # Step 3: Compute softmax in chunks
        softmax_logits_list = []
        sum_softmax_times_logits_list = []

        for i in range(0, total_nnz, chunk_size):
            exp_chunk = normalized_exp_logits[i : i + chunk_size]
            sum_exp_chunk = normalized_sum_exp_logits[i : i + chunk_size]
            logits_chunk = vocab_parallel_logits[i : i + chunk_size]

            softmax_chunk = exp_chunk.div_(sum_exp_chunk)
            sum_softmax_logits_chunk = (softmax_chunk * logits_chunk).sum(dim=-1, keepdim=True)

            softmax_logits_list.append(softmax_chunk)
            sum_softmax_times_logits_list.append(sum_softmax_logits_chunk)

        softmax_logits = torch.cat(softmax_logits_list, dim=0)
        sum_softmax_times_logits = torch.cat(sum_softmax_times_logits_list, dim=0)
        del softmax_logits_list, sum_softmax_times_logits_list

        dist.all_reduce(sum_softmax_times_logits, group=tp_group)

        # Step 4: Compute final entropy in chunks
        entropy_chunks = []
        for i in range(0, total_nnz, chunk_size):
            max_chunk = logits_max[i : i + chunk_size]
            sum_exp_chunk = normalized_sum_exp_logits[i : i + chunk_size]
            sum_softmax_logits_chunk = sum_softmax_times_logits[i : i + chunk_size]

            entropy_chunk = max_chunk + sum_exp_chunk.log() - sum_softmax_logits_chunk
            entropy_chunks.append(entropy_chunk)

        entropy = torch.cat(entropy_chunks, dim=0).squeeze(dim=-1)

        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        ctx.chunk_size = chunk_size
        return entropy

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        total_nnz = vocab_parallel_logits.shape[0]

        grad_list = []
        for i in range(0, total_nnz, chunk_size):
            logits_chunk = vocab_parallel_logits[i : i + chunk_size].clone()
            softmax_chunk = softmax_logits[i : i + chunk_size].clone()
            sum_softmax_logits_chunk = sum_softmax_times_logits[i : i + chunk_size]
            grad_chunk = grad_output[i : i + chunk_size]

            logits_chunk.sub_(sum_softmax_logits_chunk)
            softmax_chunk.mul_(logits_chunk)
            softmax_chunk.mul_(grad_chunk.unsqueeze(dim=-1))
            logits_chunk.add_(sum_softmax_logits_chunk)
            softmax_chunk.mul_(-1)

            grad_list.append(softmax_chunk)

        grad = torch.cat(grad_list, dim=0)
        return grad, None


def vocab_parallel_entropy(
    vocab_parallel_logits: torch.Tensor,
    entropy_from_logits_with_chunking: Optional[bool] = False,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks.

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)
        chunk_size: If provided, use chunked computation to reduce memory usage.

    Returns: (total_nnz,)
    """
    if entropy_from_logits_with_chunking and vocab_parallel_logits.shape[0] > chunk_size:
        return _VocabParallelEntropyChunked.apply(vocab_parallel_logits, chunk_size)
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


def compute_expected_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy using standard pytorch for verification."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def test_vocab_parallel_entropy():
    """Test basic vocab_parallel_entropy functionality."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    batch_size = 4
    seq_len = 128
    vocab_size = 32000
    local_vocab_size = vocab_size // world_size
    total_nnz = batch_size * seq_len

    vocab_parallel_logits = torch.randn(
        total_nnz, local_vocab_size, device=f"cuda:{local_rank}", dtype=torch.float32, requires_grad=True
    )

    entropy = vocab_parallel_entropy(vocab_parallel_logits)

    assert entropy.shape == (total_nnz,), f"Expected shape {(total_nnz,)}, got {entropy.shape}"
    assert not torch.isnan(entropy).any(), "Entropy contains NaN"
    assert not torch.isinf(entropy).any(), "Entropy contains Inf"
    assert (entropy >= 0).all(), "Entropy should be non-negative"

    loss = entropy.sum()
    loss.backward()

    assert vocab_parallel_logits.grad is not None, "Gradients not computed"
    assert not torch.isnan(vocab_parallel_logits.grad).any(), "Gradients contain NaN"

    print(f"Rank {rank}: Basic test passed!")

    dist.barrier()
    torch.cuda.empty_cache()


def test_vocab_parallel_entropy_chunked():
    """Test chunked version of vocab_parallel_entropy."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    batch_size = 32
    seq_len = 512
    vocab_size = 32000
    local_vocab_size = vocab_size // world_size
    total_nnz = batch_size * seq_len

    vocab_parallel_logits = torch.randn(
        total_nnz, local_vocab_size, device=f"cuda:{local_rank}", dtype=torch.float32, requires_grad=True
    )

    chunk_sizes = [512, 1024, 2048]

    for chunk_size in chunk_sizes:
        vocab_parallel_logits.grad = None
        entropy_chunked = vocab_parallel_entropy(vocab_parallel_logits, True, chunk_size=chunk_size)

        assert entropy_chunked.shape == (total_nnz,), f"Chunk size {chunk_size}: Shape mismatch"
        assert not torch.isnan(entropy_chunked).any(), f"Chunk size {chunk_size}: NaN in entropy"
        assert not torch.isinf(entropy_chunked).any(), f"Chunk size {chunk_size}: Inf in entropy"

        loss = entropy_chunked.sum()
        loss.backward()

        assert vocab_parallel_logits.grad is not None, f"Chunk size {chunk_size}: No gradients"

        print(f"Rank {rank}: Chunk size {chunk_size} test passed!")

        dist.barrier()
        torch.cuda.empty_cache()


def test_vocab_parallel_entropy_consistency():
    """Test that chunked and non-chunked versions produce same results."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    batch_size = 8
    seq_len = 64
    vocab_size = 32000
    local_vocab_size = vocab_size // world_size
    total_nnz = batch_size * seq_len

    torch.manual_seed(42 + rank)

    vocab_parallel_logits = torch.randn(total_nnz, local_vocab_size, device=f"cuda:{local_rank}", dtype=torch.float32)

    entropy_original = vocab_parallel_entropy(vocab_parallel_logits, False, chunk_size=None)
    entropy_chunked = vocab_parallel_entropy(vocab_parallel_logits, True, chunk_size=32)

    max_diff = torch.abs(entropy_original - entropy_chunked).max().item()
    assert max_diff < 1e-5, f"Results differ! Max diff: {max_diff}"

    print(f"Rank {rank}: Consistency test passed! Max diff: {max_diff:.2e}")

    dist.barrier()


def test_vocab_parallel_entropy_memory():
    """Test memory usage of chunked version."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    batch_size = 64
    seq_len = 1024
    vocab_size = 128000
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_vocab_size = vocab_size // world_size
    total_nnz = batch_size * seq_len

    vocab_parallel_logits = torch.randn(total_nnz, local_vocab_size, device=f"cuda:{local_rank}", dtype=torch.float32)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    entropy = vocab_parallel_entropy(vocab_parallel_logits, True, chunk_size=512)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print(f"Rank {local_rank}: Peak memory with chunking: {peak_memory:.2f} GB")
    assert not torch.isnan(entropy).any(), "NaN in entropy"

    dist.barrier()


if __name__ == "__main__":
    print("=" * 60)
    print("Testing vocab_parallel_entropy")
    print("=" * 60)

    try:
        test_vocab_parallel_entropy()
        print("\n" + "=" * 60)

        test_vocab_parallel_entropy_chunked()
        print("\n" + "=" * 60)

        test_vocab_parallel_entropy_consistency()
        print("\n" + "=" * 60)

        test_vocab_parallel_entropy_memory()
        print("\n" + "=" * 60)

        print("All tests passed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
