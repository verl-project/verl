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

"""Runtime patches mirroring local Megatron-LM modifications."""

import logging

logger = logging.getLogger(__name__)


def apply_megatron_train_infer_consist_patches() -> None:
    """Apply train-inference consistency patches for Megatron."""
    logger.info("Applying Megatron train-inference consistency patches...")
    _patch_vocab_parallel_cross_entropy_forward()


def _patch_vocab_parallel_cross_entropy_forward() -> None:
    """Patch _VocabParallelCrossEntropy.forward with local NPU-safe logic."""
    try:
        import torch
        from megatron.core.tensor_parallel.cross_entropy import (
            _VocabParallelCrossEntropy,
        )
        from megatron.core.tensor_parallel.utils import VocabUtility
        from megatron.core.parallel_state import (
            get_tensor_model_parallel_group,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        if hasattr(_VocabParallelCrossEntropy, "_orig_forward_train_infer_consist"):
            return

        original_forward = _VocabParallelCrossEntropy.forward

        @staticmethod
        def _patched_forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
            partition_vocab_size = vocab_parallel_logits.size(-1)
            world_size = get_tensor_model_parallel_world_size()
            rank = get_tensor_model_parallel_rank()

            tensor_list = [torch.empty_like(vocab_parallel_logits) for _ in range(world_size)]
            torch.distributed.all_gather(
                tensor_list,
                vocab_parallel_logits,
                group=get_tensor_model_parallel_group(),
            )
            full_vocab_logits = torch.cat(tensor_list, dim=-1)

            log_probs = torch.log_softmax(full_vocab_logits, dim=-1)
            gathered_log_probs = torch.gather(
                log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            loss = -gathered_log_probs

            if label_smoothing > 0:
                assert 1.0 > label_smoothing > 0.0
                full_vocab_size = full_vocab_logits.size(-1)
                smoothing = label_smoothing * full_vocab_size / (full_vocab_size - 1)
                mean_log_probs = log_probs.mean(dim=-1)
                loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

                logits_max = torch.max(vocab_parallel_logits, dim=-1, keepdim=True)[0]
                torch.distributed.all_reduce(
                    logits_max,
                    op=torch.distributed.ReduceOp.MAX,
                    group=get_tensor_model_parallel_group(),
                )
                vocab_parallel_logits_stable = vocab_parallel_logits - logits_max

                get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
                vocab_start_index, vocab_end_index = get_vocab_range(
                    partition_vocab_size, rank, world_size)
                target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
                masked_target = target.clone() - vocab_start_index
                masked_target[target_mask] = 0

                batch_dim = torch.arange(
                    vocab_parallel_logits_stable.size(0),
                    device=vocab_parallel_logits_stable.device,
                ).view(-1, 1)
                seq_dim = torch.arange(
                    vocab_parallel_logits_stable.size(1),
                    device=vocab_parallel_logits_stable.device,
                ).view(1, -1)
                predicted_logits = vocab_parallel_logits_stable[
                    batch_dim, seq_dim, masked_target
                ]
                predicted_logits = predicted_logits * (~target_mask).float()

                exp_logits = torch.exp(vocab_parallel_logits_stable)
                sum_exp_logits = exp_logits.sum(dim=-1)
                torch.distributed.all_reduce(
                    predicted_logits,
                    op=torch.distributed.ReduceOp.SUM,
                    group=get_tensor_model_parallel_group(),
                )
                torch.distributed.all_reduce(
                    sum_exp_logits,
                    op=torch.distributed.ReduceOp.SUM,
                    group=get_tensor_model_parallel_group(),
                )
                softmax_output = exp_logits / sum_exp_logits.unsqueeze(-1)
                ctx.label_smoothing = label_smoothing
                ctx.vocab_size = full_vocab_logits.size(-1)
                ctx.save_for_backward(softmax_output, target_mask, masked_target.view(-1))
            return loss

        _VocabParallelCrossEntropy._orig_forward_train_infer_consist = original_forward
        _VocabParallelCrossEntropy.forward = _patched_forward
    except Exception as e:
        logger.warning("Failed to patch Megatron cross entropy: %s", e)
