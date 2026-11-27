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
from functools import lru_cache

import torch

from verl.utils.device import get_device_name


@lru_cache(maxsize=16)
def _get_cuda_sm_major(device_index: int) -> int:
    major, _ = torch.cuda.get_device_capability(device_index)
    return major


def _cuda_arch_tier(device: torch.device, hidden_size: int, dtype: torch.dtype) -> str:
    if device.type != "cuda" or hidden_size % 8 != 0 or dtype not in (torch.float16, torch.bfloat16):
        return "generic"
    device_index = torch.cuda.current_device() if device.index is None else device.index
    major = _get_cuda_sm_major(device_index)
    if major == 10:
        return "sm100"
    if major == 9:
        return "sm90"
    return "generic"


def _get_backend_kernels(device_type: str):
    if device_type == "cuda":
        from . import lce_cuda

        return lce_cuda
    if device_type == "npu":
        from . import lce_asc

        return lce_asc
    raise ValueError(f"Unsupported device type: {device_type}")


class LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
        dist_process_group: torch.distributed.ProcessGroup = None,
    ):
        """Forward pass.

        Args:
            hidden: hidden states of shape (..., h).
            weight: lm head weights of shape (vocab_size/tp_size, h).
            labels: target tokens of shape (...) or (..., topk).
            temperature: temperature for logits scaling.
            dist_process_group: ColumnParallelLinear process group, megatron only.

        Returns:
            logprobs: logprobs tensor matching labels.
            entropy: entropy tensor matching hidden.shape[:-1].
        """
        if hidden.ndim < 2 or weight.ndim != 2:
            raise ValueError("hidden and weight must have shapes [..., K] and [N, K]")

        original_hidden_shape = hidden.shape
        hidden = hidden.reshape(-1, hidden.shape[-1])
        if labels.ndim == len(original_hidden_shape) - 1 and labels.numel() == hidden.shape[0]:
            single_label = True
            output_shape = labels.shape
            labels = labels.reshape(-1, 1)
        elif labels.ndim == len(original_hidden_shape) and labels[..., 0].numel() == hidden.shape[0]:
            single_label = False
            output_shape = labels.shape[:-1]
            labels = labels.reshape(-1, labels.shape[-1])
        else:
            raise ValueError("labels must contain one or more labels for every hidden row")

        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("hidden and weight must have the same hidden size")
        if hidden.dtype != weight.dtype:
            raise ValueError("hidden and weight must have the same dtype")
        if hidden.device.type != get_device_name() or weight.device != hidden.device or labels.device != hidden.device:
            raise ValueError("hidden, weight, and labels must be on the same accelerator device")
        if labels.dtype != torch.int64:
            raise ValueError("labels must have dtype torch.int64")
        if hidden.shape[0] == 0 or weight.shape[0] == 0 or labels.shape[1] == 0:
            raise ValueError("M, N, and topk must be positive")
        if not isinstance(temperature, float) or temperature <= 0.0:
            raise ValueError("temperature must be a positive float")

        hidden = hidden.contiguous()
        weight = weight.contiguous()
        labels = labels.contiguous()
        backend_kernels = _get_backend_kernels(hidden.device.type)

        tp_rank = 0 if dist_process_group is None else torch.distributed.get_rank(dist_process_group)
        tp_world_size = 1 if dist_process_group is None else torch.distributed.get_world_size(dist_process_group)
        num_tokens, hidden_size = hidden.shape
        local_vocab_size = weight.shape[0]
        vocab_start = tp_rank * local_vocab_size
        topk = labels.shape[1]

        if hidden.device.type == "cuda":
            vocab_per_cta = 1024 if local_vocab_size > 262144 else 512
        else:
            vocab_per_cta = 1024 if num_tokens >= 8192 else 512
        num_splits = (local_vocab_size + vocab_per_cta - 1) // vocab_per_cta

        partial_shape = (num_tokens, num_splits)
        partial_max = torch.empty(partial_shape, dtype=torch.float32, device=hidden.device)
        partial_sum = torch.empty_like(partial_max)
        partial_weighted_sum = torch.empty_like(partial_max)
        target_logits = torch.zeros((num_tokens, topk), dtype=torch.float32, device=hidden.device)
        target_logits_kernel = backend_kernels.linear_cross_entropy_fwd_target_logits(
            local_vocab_size,
            hidden_size,
            topk,
            dtype=hidden.dtype,
        )
        target_logits_kernel(
            hidden,
            weight,
            labels,
            1.0 / temperature,
            vocab_start,
            target_logits,
        )

        arch_tier = _cuda_arch_tier(hidden.device, hidden_size, hidden.dtype)
        if arch_tier == "sm100":
            mainloop = backend_kernels.linear_cross_entropy_fwd_mainloop_sm100(
                local_vocab_size,
                hidden_size,
                dtype=hidden.dtype,
                VOCAB_PER_CTA=vocab_per_cta,
            )
        elif arch_tier == "sm90":
            mainloop = backend_kernels.linear_cross_entropy_fwd_mainloop_sm90(
                local_vocab_size,
                hidden_size,
                dtype=hidden.dtype,
                VOCAB_PER_CTA=vocab_per_cta,
            )
        elif hidden.device.type == "cuda":
            mainloop = backend_kernels.linear_cross_entropy_fwd_mainloop(
                local_vocab_size,
                hidden_size,
                dtype=hidden.dtype,
                BLOCK_M=64,
                BLOCK_N=128,
                BLOCK_K=64,
                VOCAB_PER_CTA=vocab_per_cta,
                num_threads=128,
                num_stages=2,
                min_blocks_per_sm=1,
            )
        else:
            mainloop = backend_kernels.linear_cross_entropy_fwd_mainloop(
                local_vocab_size,
                hidden_size,
                dtype=hidden.dtype,
                BLOCK_M=64,
                BLOCK_N=256,
                BLOCK_K=64,
                VOCAB_PER_CTA=vocab_per_cta,
                num_threads=128,
                num_stages=2,
                min_blocks_per_sm=1,
            )
        mainloop(
            hidden,
            weight,
            1.0 / temperature,
            partial_max,
            partial_sum,
            partial_weighted_sum,
        )

        logprobs = torch.empty_like(target_logits)
        entropy = torch.empty((num_tokens,), dtype=torch.float32, device=hidden.device)
        lse = torch.empty_like(entropy)
        mean_logits = torch.empty_like(entropy)
        if tp_world_size == 1:
            finalize_local = backend_kernels.linear_cross_entropy_fwd_finalize_local(
                num_splits,
                topk,
            )
            finalize_local(
                partial_max,
                partial_sum,
                partial_weighted_sum,
                target_logits,
                logprobs,
                entropy,
                lse,
                mean_logits,
            )
        else:
            local_max = torch.empty(
                (num_tokens,),
                dtype=torch.float32,
                device=hidden.device,
            )
            local_sum = torch.empty_like(local_max)
            local_weighted_sum = torch.empty_like(local_max)
            local_reduce = backend_kernels.linear_cross_entropy_fwd_local_reduce(num_splits)
            local_reduce(
                partial_max,
                partial_sum,
                partial_weighted_sum,
                local_max,
                local_sum,
                local_weighted_sum,
            )

            global_max = local_max.clone()
            torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=dist_process_group)

            reduced = torch.empty(
                (num_tokens, topk + 2),
                dtype=torch.float32,
                device=hidden.device,
            )
            prepare_reduce = backend_kernels.linear_cross_entropy_fwd_prepare_reduce(topk)
            prepare_reduce(
                local_max,
                global_max,
                local_sum,
                local_weighted_sum,
                target_logits,
                reduced,
            )
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM, group=dist_process_group)

            finalize = backend_kernels.linear_cross_entropy_fwd_finalize(topk)
            finalize(
                global_max,
                reduced,
                logprobs,
                entropy,
                lse,
                mean_logits,
            )

        ctx.set_materialize_grads(False)
        ctx.save_for_backward(hidden, weight, labels, lse, mean_logits)
        ctx.temperature = temperature
        ctx.dist_process_group = dist_process_group
        ctx.original_hidden_shape = original_hidden_shape
        ctx.output_shape = output_shape
        ctx.single_label = single_label
        logprobs = logprobs.reshape(*output_shape, topk)
        if single_label:
            logprobs = logprobs.squeeze(-1)
        return logprobs, entropy.reshape(*output_shape)

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor):
        """Backward pass.

        Args:
            grad_logprobs: gradient matching the forward logprobs, or None.
            grad_entropy: gradient matching hidden.shape[:-1], or None if entropy loss is not used.

        Returns:
            grad_hidden: gradient matching the forward hidden shape.
            grad_weight: gradient of the lm head weights tensor of shape (vocab_size/tp_size, h)
        """
        hidden, weight, labels, lse, mean_logits = ctx.saved_tensors
        temperature = ctx.temperature
        dist_process_group = ctx.dist_process_group
        backend_kernels = _get_backend_kernels(hidden.device.type)

        num_tokens, hidden_size = hidden.shape
        local_vocab_size = weight.shape[0]
        topk = labels.shape[1]
        tp_rank = 0 if dist_process_group is None else torch.distributed.get_rank(dist_process_group)
        vocab_start = tp_rank * local_vocab_size

        has_logprobs_grad = grad_logprobs is not None
        has_entropy_grad = grad_entropy is not None
        if has_logprobs_grad:
            expected_shape = ctx.output_shape if ctx.single_label else (*ctx.output_shape, topk)
            if grad_logprobs.shape != expected_shape:
                raise ValueError(f"grad_logprobs must have shape {expected_shape}")
            grad_logprobs = grad_logprobs.reshape(num_tokens, topk).float().contiguous()
        else:
            grad_logprobs = torch.empty((num_tokens, topk), dtype=torch.float32, device=hidden.device)
        if has_entropy_grad:
            if grad_entropy.shape != ctx.output_shape:
                raise ValueError(f"grad_entropy must have shape {ctx.output_shape}")
            grad_entropy = grad_entropy.reshape(num_tokens).float().contiguous()
        else:
            # The no-entropy specialization does not read this argument.
            grad_entropy = lse

        arch_tier = _cuda_arch_tier(hidden.device, hidden_size, hidden.dtype)
        if arch_tier == "sm100":
            block_n = 256
        elif arch_tier == "sm90":
            block_n = 128
        elif num_tokens >= 8192 and topk <= 8 and hidden.device.type != "cuda":
            block_m, block_n, block_k = 128, 256, 64
            num_threads, num_stages, min_blocks_per_sm = 256, 2, 1
        else:
            block_m, block_n, block_k = 64, 128, 64
            num_threads, num_stages = 128, 3
            min_blocks_per_sm = 1 if (hidden.device.type == "cuda" and topk > 8) else 2

        aligned_local_vocab_size = (local_vocab_size + block_n - 1) // block_n * block_n
        if hidden.device.type == "cuda":
            if aligned_local_vocab_size <= 32768 and num_tokens <= 8192:
                max_vocab_chunk = 32768
            elif num_tokens > 32768:
                max_vocab_chunk = 4096
            else:
                max_vocab_chunk = 16384
        else:
            max_vocab_chunk = 8192
        vocab_chunk = min(max_vocab_chunk, aligned_local_vocab_size)

        dlogits = torch.empty(
            (num_tokens, vocab_chunk),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        single_chunk = hidden.device.type == "cuda" and vocab_chunk >= local_vocab_size
        if single_chunk:
            grad_hidden_2d = torch.empty_like(hidden)
        else:
            grad_hidden_accum = torch.empty_like(hidden, dtype=torch.float32)
        grad_weight = torch.empty_like(weight)

        use_fast_cuda_bwd = arch_tier in ("sm100", "sm90")
        if use_fast_cuda_bwd:
            if has_logprobs_grad:
                grad_logprobs_sum = grad_logprobs[:, 0].contiguous() if topk == 1 else grad_logprobs.sum(dim=-1)
            else:
                grad_logprobs_sum = lse
            bwd_fn = (
                backend_kernels.linear_cross_entropy_bwd_dlogits_sm100
                if arch_tier == "sm100"
                else backend_kernels.linear_cross_entropy_bwd_dlogits_sm90
            )
            dlogits_kernel = bwd_fn(
                local_vocab_size,
                hidden_size,
                topk,
                vocab_chunk,
                has_logprobs_grad,
                has_entropy_grad,
                dtype=hidden.dtype,
            )
        else:
            dlogits_kernel = backend_kernels.linear_cross_entropy_bwd_dlogits(
                local_vocab_size,
                hidden_size,
                topk,
                vocab_chunk,
                has_logprobs_grad,
                has_entropy_grad,
                dtype=hidden.dtype,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_threads=num_threads,
                num_stages=num_stages,
                min_blocks_per_sm=min_blocks_per_sm,
            )

        inv_temperature = 1.0 / temperature
        dlogits_valid = dlogits
        for vocab_offset in range(0, local_vocab_size, vocab_chunk):
            if use_fast_cuda_bwd:
                dlogits_kernel(
                    hidden,
                    weight,
                    labels,
                    lse,
                    mean_logits,
                    grad_logprobs,
                    grad_logprobs_sum,
                    grad_entropy,
                    inv_temperature,
                    vocab_start,
                    vocab_offset,
                    dlogits,
                )
            else:
                dlogits_kernel(
                    hidden,
                    weight,
                    labels,
                    lse,
                    mean_logits,
                    grad_logprobs,
                    grad_entropy,
                    inv_temperature,
                    vocab_start,
                    vocab_offset,
                    dlogits,
                )

            valid_vocab = min(vocab_chunk, local_vocab_size - vocab_offset)
            dlogits_valid = dlogits
            if valid_vocab != vocab_chunk:
                dlogits_valid = (
                    dlogits[:, :valid_vocab] if hidden.device.type == "cuda" else dlogits[:, :valid_vocab].contiguous()
                )
            weight_chunk = weight[vocab_offset : vocab_offset + valid_vocab]

            if single_chunk:
                torch.mm(dlogits_valid, weight_chunk, out=grad_hidden_2d)
            else:
                torch.addmm(
                    grad_hidden_accum,
                    dlogits_valid,
                    weight_chunk,
                    beta=int(vocab_offset != 0),
                    out_dtype=torch.float32,
                    out=grad_hidden_accum,
                )
            torch.mm(
                dlogits_valid.T,
                hidden,
                out=grad_weight[vocab_offset : vocab_offset + valid_vocab],
            )

        del dlogits, dlogits_valid
        if single_chunk:
            grad_hidden = grad_hidden_2d.reshape(ctx.original_hidden_shape)
        else:
            grad_hidden = grad_hidden_accum.to(hidden.dtype).reshape(ctx.original_hidden_shape)
        return grad_hidden, grad_weight, None, None, None


linear_cross_entropy = LinearCrossEntropy.apply
