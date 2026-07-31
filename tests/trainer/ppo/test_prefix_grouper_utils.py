# Copyright 2026 verl authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import torch
from tensordict import TensorDict

import verl.utils.experimental.torch_functional as experimental_torch_functional
from verl.models.transformers.monkey_patch import apply_prefix_grouper_model_forward_patch
from verl.trainer.ppo.prefix_grouper_utils import (
    attach_prefix_grouper_forward_args,
    build_pg_from_micro_batch,
    response_output_to_nested,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


class _TinyBaseModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, **_):
        return (self.embed_tokens(input_ids),)


class _TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8):
        super().__init__()
        self.model = _TinyBaseModel(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        hidden_states = self.model(input_ids=input_ids, **kwargs)[0]
        return SimpleNamespace(logits=self.lm_head(hidden_states))


def _baseline_response_log_probs(model, prompts, responses):
    input_ids = torch.cat([prompts, responses], dim=1)
    logits = model(input_ids=input_ids).logits
    response_logits = logits[:, prompts.shape[1] - 1 : -1].float()
    return response_logits.log_softmax(dim=-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)


def test_fused_prefix_grouper_matches_repeated_prefix_gradients(monkeypatch):
    monkeypatch.setattr(experimental_torch_functional, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    torch.manual_seed(7)
    prompts = torch.tensor(
        [
            [0, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 0, 6, 7],
            [0, 0, 6, 7],
        ]
    )
    responses = torch.tensor(
        [
            [8, 9, 10],
            [11, 12, 0],
            [13, 14, 15],
            [16, 0, 0],
        ]
    )
    response_mask = responses.ne(0)
    micro_batch = {
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "uid": ["group-0", "group-0", "group-1", "group-1"],
    }

    baseline_model = _TinyCausalLM()
    grouped_model = _TinyCausalLM()
    grouped_model.load_state_dict(baseline_model.state_dict())
    apply_prefix_grouper_model_forward_patch(grouped_model)

    baseline_log_probs = _baseline_response_log_probs(baseline_model, prompts, responses)
    baseline_loss = -(baseline_log_probs * response_mask).sum()
    baseline_loss.backward()

    (
        prefix_grouper,
        concat_input_ids,
        attention_mask,
        position_ids,
        completion_ids,
        completion_mask,
    ) = build_pg_from_micro_batch(micro_batch, pad_token_id=0)
    attach_prefix_grouper_forward_args(
        prefix_grouper=prefix_grouper,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        temperature=1.0,
        calculate_entropy=False,
    )
    grouped_log_probs, _, suffix_mask = grouped_model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        prefix_grouper=prefix_grouper,
        return_prefix_fused_outputs=True,
    )
    grouped_loss = -(grouped_log_probs * suffix_mask).sum()
    grouped_loss.backward()

    assert torch.allclose(
        grouped_log_probs[response_mask],
        baseline_log_probs[response_mask],
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(
        grouped_model.model.embed_tokens.weight.grad,
        baseline_model.model.embed_tokens.weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.allclose(
        grouped_model.lm_head.weight.grad,
        baseline_model.lm_head.weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )


def test_response_output_to_nested_preserves_alignment():
    input_values = torch.arange(11)
    input_ids = torch.nested.nested_tensor_from_jagged(
        input_values,
        torch.tensor([0, 6, 11]),
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    )
    response_output = torch.tensor(
        [
            [10.0, 11.0, 0.0],
            [20.0, 21.0, 22.0],
        ]
    )

    nested = response_output_to_nested(response_output, response_mask, input_ids)

    assert torch.equal(nested.offsets(), input_ids.offsets())
    assert torch.equal(
        nested.values(),
        torch.tensor([0.0, 0.0, 0.0, 10.0, 11.0, 0.0, 0.0, 20.0, 21.0, 22.0, 0.0]),
    )


def test_build_pg_uses_nonzero_pad_token_id():
    prompts = torch.tensor(
        [
            [99, 99, 3, 4],
            [99, 99, 3, 4],
        ]
    )
    responses = torch.tensor(
        [
            [5, 6, 99],
            [7, 99, 99],
        ]
    )
    response_mask = responses.ne(99)

    prefix_grouper, *_ = build_pg_from_micro_batch(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "uid": ["group-0", "group-0"],
        },
        pad_token_id=99,
    )

    assert prefix_grouper.prefix_lens.tolist() == [2]
    assert prefix_grouper.ungrouped_suffix_lens.tolist() == [2, 1]


def test_prefix_grouper_temperature_matches_existing_clamp():
    grouper = SimpleNamespace()
    attach_prefix_grouper_forward_args(
        prefix_grouper=grouper,
        completion_ids=torch.ones(1, 1, dtype=torch.long),
        completion_mask=torch.ones(1, 1, dtype=torch.bool),
        temperature=0.0,
        calculate_entropy=False,
    )
    assert grouper._verl_temperature == 1e-8


def test_fsdp_engine_prepares_and_restores_prefix_grouped_outputs():
    prompts = torch.tensor(
        [
            [0, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 0, 6, 7],
            [0, 0, 6, 7],
        ]
    )
    responses = torch.tensor(
        [
            [8, 9, 10],
            [11, 12, 0],
            [13, 14, 15],
            [16, 0, 0],
        ]
    )
    response_mask = responses.ne(0)
    attention_mask = torch.cat([prompts.ne(0), response_mask], dim=1)
    input_ids = torch.cat([prompts, responses], dim=1)
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(~attention_mask, 0)
    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "temperature": torch.ones(4),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor_data(data, "uid", ["group-0", "group-0", "group-1", "group-1"])
    data = left_right_2_no_padding(data)
    tu.assign_non_tensor(
        data,
        use_remove_padding=False,
        use_fused_kernels=False,
        use_prefix_grouper=True,
        calculate_entropy=True,
        calculate_sum_pi_squared=False,
        distillation_use_topk=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        pad_token_id=0,
    )

    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    model_inputs, output_args = engine.prepare_model_inputs(data)

    assert output_args == {}
    assert model_inputs["input_ids"].shape[0] == 2
    assert model_inputs["return_prefix_fused_outputs"] is True

    response_log_probs = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 0.0],
            [6.0, 7.0, 8.0],
            [9.0, 0.0, 0.0],
        ]
    )
    response_entropy = response_log_probs + 10
    model_output = engine.prepare_model_outputs(
        (response_log_probs, response_entropy, response_mask),
        output_args,
        data,
        logits_processor_func=None,
    )

    restored_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    restored_entropy = no_padding_2_padding(model_output["entropy"], data)
    assert torch.equal(restored_log_probs, response_log_probs)
    assert torch.equal(restored_entropy, response_entropy * response_mask)
