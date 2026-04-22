# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
Utilities for using tensor_parallel in megatron
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch.nn import init

if TYPE_CHECKING:
    from megatron.core import ModelParallelConfig


def update_kwargs_with_config(dictionary: dict, config: "ModelParallelConfig"):
    dictionary["config"] = config
    return dictionary


def get_default_kwargs_for_model_parallel_config():
    model_parallel_config_kwargs = {
        "params_dtype": torch.float32,
        "use_cpu_initialization": False,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": False,
    }
    return model_parallel_config_kwargs


def get_default_model_parallel_config():
    from megatron.core import ModelParallelConfig

    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


def get_common_default_kwargs_for_parallel_linear():
    default_model_parallel_config = get_default_model_parallel_config()
    common_default_kwargs = {
        "init_method": init.xavier_normal_,
        "stride": 1,
        "keep_master_weight_for_test": False,
        "config": default_model_parallel_config,
    }
    return common_default_kwargs


def get_default_kwargs_for_column_parallel_linear():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    column_parallel_config_kwargs = {
        "async_tensor_model_parallel_allreduce": False,
    }
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    column_default_kwargs = {
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    common_default_kwargs.update(column_default_kwargs)
    return common_default_kwargs


def get_default_kwargs_for_row_parallel_linear():
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    return common_default_kwargs


def get_default_kwargs_for_parallel_embedding():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    embedding_default_kwargs = {
        "init_method": init.xavier_normal_,
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    return embedding_default_kwargs


def is_tensor_parallel_param(param):
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


def get_tensor_parallel_partition_dim(param):
    assert is_tensor_parallel_param(param)
    return param.partition_dim


def get_tensor_parallel_partition_stride(param):
    assert is_tensor_parallel_param(param)
    return param.partition_stride


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


class _VocabParallelEntropyChunked(torch.autograd.Function):
    """Memory-efficient chunked version of VocabParallelEntropy.

    This implementation splits the computation into chunks to reduce peak memory usage,
    which is especially useful when dealing with large batch sizes or long sequences.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        """Compute entropy in chunks to reduce memory usage.

        Args:
            vocab_parallel_logits: (total_nnz, vocab_size // tp_size)
            chunk_size: Number of samples to process in each chunk

        Returns:
            entropy: (total_nnz,)
        """
        total_nnz, local_vocab_size = vocab_parallel_logits.shape
        tp_group = mpu.get_tensor_model_parallel_group()

        # Step 1: Compute max in chunks
        logits_max_chunks = []
        for i in range(0, total_nnz, chunk_size):
            chunk = vocab_parallel_logits[i : i + chunk_size]
            chunk_max = chunk.max(dim=-1, keepdim=True).values
            logits_max_chunks.append(chunk_max)

        logits_max = torch.cat(logits_max_chunks, dim=0)
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        del logits_max_chunks  # Free memory

        # Step 2: Compute exp and sum in chunks
        normalized_exp_logits_list = []
        normalized_sum_exp_list = []

        for i in range(0, total_nnz, chunk_size):
            chunk = vocab_parallel_logits[i : i + chunk_size]
            chunk_max = logits_max[i : i + chunk_size]

            # Normalize and exp
            normalized_chunk = chunk - chunk_max
            exp_chunk = normalized_chunk.exp_()
            del normalized_chunk  # Free immediately

            # Sum for this chunk
            sum_exp = exp_chunk.sum(dim=-1, keepdim=True)

            normalized_exp_logits_list.append(exp_chunk)
            normalized_sum_exp_list.append(sum_exp)

        # Concatenate and all_reduce
        normalized_exp_logits = torch.cat(normalized_exp_logits_list, dim=0)
        normalized_sum_exp_logits = torch.cat(normalized_sum_exp_list, dim=0)
        del normalized_exp_logits_list, normalized_sum_exp_list  # Free memory

        dist.all_reduce(normalized_sum_exp_logits, group=tp_group)

        # Step 3: Compute softmax in chunks
        softmax_logits_list = []
        sum_softmax_times_logits_list = []

        for i in range(0, total_nnz, chunk_size):
            exp_chunk = normalized_exp_logits[i : i + chunk_size]
            sum_exp_chunk = normalized_sum_exp_logits[i : i + chunk_size]
            logits_chunk = vocab_parallel_logits[i : i + chunk_size]

            # Softmax
            softmax_chunk = exp_chunk.div_(sum_exp_chunk)

            # Compute sum(softmax * logits) for this chunk
            sum_softmax_logits_chunk = (softmax_chunk * logits_chunk).sum(dim=-1, keepdim=True)

            softmax_logits_list.append(softmax_chunk)
            sum_softmax_times_logits_list.append(sum_softmax_logits_chunk)

        softmax_logits = torch.cat(softmax_logits_list, dim=0)
        sum_softmax_times_logits = torch.cat(sum_softmax_times_logits_list, dim=0)
        del softmax_logits_list, sum_softmax_times_logits_list  # Free memory

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

        # Save for backward
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        ctx.chunk_size = chunk_size
        return entropy

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        total_nnz = vocab_parallel_logits.shape[0]

        # Process backward in chunks to save memory
        grad_list = []
        for i in range(0, total_nnz, chunk_size):
            logits_chunk = vocab_parallel_logits[i : i + chunk_size].clone()
            softmax_chunk = softmax_logits[i : i + chunk_size].clone()
            sum_softmax_logits_chunk = sum_softmax_times_logits[i : i + chunk_size]
            grad_chunk = grad_output[i : i + chunk_size]

            # Compute grad for this chunk
            logits_chunk.sub_(sum_softmax_logits_chunk)
            softmax_chunk.mul_(logits_chunk)
            softmax_chunk.mul_(grad_chunk.unsqueeze(dim=-1))
            logits_chunk.add_(sum_softmax_logits_chunk)  # Recover
            softmax_chunk.mul_(-1)

            grad_list.append(softmax_chunk)

        grad = torch.cat(grad_list, dim=0)
        return grad, None  # None for chunk_size


def vocab_parallel_entropy(
    vocab_parallel_logits: torch.Tensor,
    entropy_from_logits_with_chunking: Optional[bool] = False,
    chunk_size: Optional[int] = 2048,
) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks.

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)
        chunk_size: If provided, use chunked computation to reduce memory usage.
                   Recommended for large batch sizes or when OOM occurs.
                   Default: None (use original implementation)

    Returns: (total_nnz,)

    """
    if entropy_from_logits_with_chunking and vocab_parallel_logits.shape[0] > chunk_size:
        return _VocabParallelEntropyChunked.apply(vocab_parallel_logits, chunk_size)
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


def vocab_parallel_log_probs_from_logits(logits, labels):
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    from megatron.core import tensor_parallel

    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel
    region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(
        logits=logits_rmpad, labels=input_ids_rmpad_rolled
    )  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output
