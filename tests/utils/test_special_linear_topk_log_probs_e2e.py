# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""End-to-end GPU test for fused Megatron top-k distillation.

This test deliberately keeps only the transformer body synthetic. Everything
after its hidden states is real: jagged THD packing, the Megatron output hook,
the Triton selected-log-probability operator, distillation outputs, THD
unpacking, and autograd backward.

Run with ``torchrun --standalone --nproc-per-node=<tp-size>`` for TP1--TP4.
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core import parallel_state

from verl.models.mcore import model_forward_fused as mff


def _nested(values: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=values.device)
    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


class _MinimalHookModel:
    """A minimal transformer-body stand-in that invokes Megatron's real hook."""

    pre_process = True
    post_process = True

    def __init__(self, embedding: torch.nn.Parameter, output_weight: torch.nn.Parameter):
        self.embedding = embedding
        self.output_layer = SimpleNamespace(weight=output_weight)
        self.config = SimpleNamespace(
            fp8=None,
            experimental_attention_variant=None,
            csa_window_size=None,
            sequence_parallel=False,
        )
        setattr(self, mff._FUSED_FORWARD_MODE_ATTR, mff._HOOK_MODE)
        self.saw_full_logits = False

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        packed_seq_params,
        labels,
        output_processor,
        output_processor_context,
        **kwargs,
    ):
        del attention_mask, position_ids, packed_seq_params, kwargs
        hidden_states = torch.nn.functional.embedding(input_ids, self.embedding)
        output = output_processor(
            hidden_states=hidden_states,
            output_layer=self.output_layer,
            output_weight=None,
            labels=labels,
            context=output_processor_context,
            config=self.config,
            input_ids=input_ids,
        )
        self.saw_full_logits = output.logits is not None
        return output


def _build_inputs(device: torch.device, world_size: int, local_vocab_size: int, topk: int):
    lengths = [5, 3, 7]
    num_tokens = sum(lengths)
    global_vocab_size = local_vocab_size * world_size
    generator = torch.Generator().manual_seed(20260908 + world_size)

    input_values = torch.randint(0, 97, (num_tokens,), generator=generator).to(device)
    teacher_ids = torch.randint(
        0,
        global_vocab_size,
        (num_tokens, topk),
        generator=generator,
    ).to(device)
    shard_boundaries = []
    for shard in range(world_size):
        shard_boundaries.extend([shard * local_vocab_size, (shard + 1) * local_vocab_size - 1])
    shard_boundaries = list(dict.fromkeys(shard_boundaries))
    count = min(topk, len(shard_boundaries))
    teacher_ids[0, :count] = torch.tensor(shard_boundaries[:count], device=device)
    teacher_ids[1, -1] = teacher_ids[1, 0]

    # Use a sub-probability distribution to match real teacher top-k mass.
    teacher_scores = torch.randn(num_tokens, topk, generator=generator)
    teacher_log_probs = (torch.log_softmax(teacher_scores, dim=-1) + torch.log(torch.tensor(0.85))).to(device)
    return (
        lengths,
        input_values,
        teacher_ids,
        teacher_log_probs,
        _nested(input_values, lengths),
        _nested(teacher_ids, lengths),
        _nested(teacher_log_probs, lengths),
    )


def _reference_outputs(hidden, full_weight, teacher_ids, teacher_log_probs, temperature, clamp):
    logits = torch.mm(hidden.float(), full_weight.float().T) / temperature
    student_log_probs = torch.gather(torch.log_softmax(logits, dim=-1), -1, teacher_ids.long())
    student_mass = student_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_log_probs.float().exp().sum(dim=-1)
    student_for_loss = student_log_probs.clamp_min(clamp)
    teacher_for_loss = teacher_log_probs.float().clamp_min(clamp)
    losses = (teacher_for_loss.exp() * (teacher_for_loss - student_for_loss)).sum(dim=-1)
    return losses, student_mass, teacher_mass


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if not 1 <= world_size <= 4:
        raise RuntimeError(f"this test supports TP1 through TP4, got TP{world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)

    hidden_size = 130
    local_vocab_size = 1025
    topk = 16
    temperature = 1.2
    clamp = -10.0
    try:
        (
            lengths,
            input_values,
            teacher_ids,
            teacher_log_probs,
            input_nested,
            teacher_ids_nested,
            teacher_log_probs_nested,
        ) = _build_inputs(device, world_size, local_vocab_size, topk)

        parameter_generator = torch.Generator().manual_seed(31000)
        embedding_seed = (torch.randn(97, hidden_size, generator=parameter_generator, dtype=torch.bfloat16) * 0.2).to(
            device
        )
        torch.manual_seed(32000 + rank)
        local_weight_seed = (
            torch.randn(
                local_vocab_size,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        gathered_weights = [torch.empty_like(local_weight_seed) for _ in range(world_size)]
        dist.all_gather(gathered_weights, local_weight_seed)
        full_weight_seed = torch.cat(gathered_weights, dim=0)

        embedding_actual = torch.nn.Parameter(embedding_seed.detach().clone())
        weight_actual = torch.nn.Parameter(local_weight_seed.detach().clone())
        model = _MinimalHookModel(embedding_actual, weight_actual)

        fused_forward = mff.fused_forward_model_engine()
        actual = fused_forward(
            model=model,
            input_ids=input_nested,
            labels=input_nested,
            multi_modal_inputs={},
            temperature=temperature,
            calculate_entropy=False,
            pad_token_id=0,
            teacher_topk_ids=teacher_ids_nested,
            teacher_topk_log_probs=teacher_log_probs_nested,
            distillation_only=True,
            log_prob_min_clamp=clamp,
        )

        assert set(actual) == {"distillation_losses", "student_mass", "teacher_mass"}
        assert not model.saw_full_logits
        for value in actual.values():
            assert value.is_nested
            assert value.offsets().diff().tolist() == lengths

        embedding_expected = embedding_seed.detach().float().requires_grad_()
        weight_expected = full_weight_seed.detach().float().requires_grad_()
        hidden_expected = torch.nn.functional.embedding(input_values, embedding_expected)
        expected_loss, expected_student_mass, expected_teacher_mass = _reference_outputs(
            hidden_expected,
            weight_expected,
            teacher_ids,
            teacher_log_probs,
            temperature,
            clamp,
        )

        actual_losses = actual["distillation_losses"].values()
        actual_student_mass = actual["student_mass"].values()
        actual_teacher_mass = actual["teacher_mass"].values()
        torch.testing.assert_close(actual_losses, expected_loss, atol=3e-2, rtol=4e-3)
        torch.testing.assert_close(actual_student_mass, expected_student_mass, atol=2e-3, rtol=5e-3)
        torch.testing.assert_close(actual_teacher_mass, expected_teacher_mass, atol=2e-5, rtol=2e-5)

        actual_losses.sum().backward()
        expected_loss.sum().backward()
        dist.all_reduce(embedding_actual.grad, op=dist.ReduceOp.SUM)
        weight_start = rank * local_vocab_size
        weight_end = weight_start + local_vocab_size
        torch.testing.assert_close(
            embedding_actual.grad.float(),
            embedding_expected.grad,
            atol=8e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            weight_actual.grad.float(),
            weight_expected.grad[weight_start:weight_end],
            atol=8e-2,
            rtol=1e-2,
        )

        if rank == 0:
            print(
                f"[PASS] E2E TP{world_size}: nested input -> THD pack -> model hook -> "
                "Triton top-k -> distillation loss -> THD unpack -> backward"
            )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
