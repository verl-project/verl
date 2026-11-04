# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# ruff: noqa: E402

"""Packed model outputs reuse the ordinary processors and SP gradient contract."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

pytest.importorskip("torch_npu")

from fsdp_turbo.models.kimi.kimi_k3_cp import kimi_k3_model_forward

import verl.utils.tensordict_utils as tu
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.score_centering import score_centering_logits_processor
from verl.utils.device import get_torch_device
from verl.utils.torch_functional import entropy_from_logits
from verl.utils.ulysses import set_ulysses_sequence_parallel_group
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def test_logprobs_respects_logits_device():
    logits = torch.randn(2, 3, 7, requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    labels = torch.tensor([[0, 3, 6], [2, 4, 1]])
    actual = verl_F.logprobs_from_logits(logits, labels)
    expected = reference.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def _engine(world_size):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = world_size > 1
    engine.ulysses_sequence_parallel_size = world_size
    engine.pad_to_length = None
    engine.pass_packed_cu_seqlens = True
    engine.module = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace()))
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    engine.compute_entropy_from_logits = entropy_from_logits
    return engine


def _native_logits(inputs, embedding, weight, group=None):
    """Exercise native CP exchange with a row-local decoder leaf."""

    def decode(inputs_embeds, **kwargs):
        return SimpleNamespace(hidden_states=inputs_embeds, past_key_values=None, attentions=None)

    decode.lm_head = lambda hidden: hidden @ weight
    model = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=False, output_hidden_states=False, use_return_dict=True, ignore_index=-100
        ),
        get_input_embeddings=lambda: lambda ids: torch.nn.functional.embedding(ids, embedding),
        language_model=decode,
    )
    return kimi_k3_model_forward(model, **inputs, context_parallel_group=group)


def _compare_outputs(rank, world_size, score_centering):
    torch.manual_seed(7)
    ids = torch.nested.as_nested_tensor([torch.tensor([2, 3, 4]), torch.tensor([5, 6])], layout=torch.jagged)
    offsets = ids.offsets()
    topk = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    data = TensorDict(
        {
            "input_ids": ids,
            "position_ids": torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1, 2, 0, 1]), offsets),
            "temperature": torch.tensor([0.8, 1.3], dtype=torch.float32),
            "rollout_topk_ids": torch.nested.nested_tensor_from_jagged(topk, offsets),
            "rollout_topk_log_probs": torch.nested.nested_tensor_from_jagged(torch.full((5, 2), -2.0), offsets),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_remove_padding=True,
        score_centering=score_centering,
        calculate_entropy=True,
        calculate_sum_pi_squared=True,
    )
    dense_data = data.clone()
    tu.assign_non_tensor(dense_data, use_remove_padding=False)
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    model_inputs, args = _engine(world_size).prepare_model_inputs(data)
    assert model_inputs["cu_seqlens"].tolist() == [0, 3, 5, 6]
    set_ulysses_sequence_parallel_group(None)
    dense_inputs, dense_args = _engine(1).prepare_model_inputs(dense_data)
    embedding = torch.randn(11, 7, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(7, 11, dtype=torch.float32, requires_grad=True)
    reference_embedding = embedding.detach().clone().requires_grad_()
    reference = weight.detach().clone().requires_grad_()
    parameters = (embedding, weight)
    references = (reference_embedding, reference)
    optimizers = [torch.optim.SGD(values, lr=0.01) for values in (parameters, references)]
    processor = (
        partial(
            score_centering_logits_processor, config=SimpleNamespace(policy_loss=SimpleNamespace(rollout_correction={}))
        )
        if score_centering
        else None
    )
    coefficients = torch.arange(1, 6, dtype=weight.dtype)
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad()
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        actual = _engine(world_size).prepare_model_outputs(
            _native_logits(model_inputs, embedding, weight, dist.group.WORLD),
            args,
            data,
            processor,
        )
        set_ulysses_sequence_parallel_group(None)
        expected = _engine(1).prepare_model_outputs(
            _native_logits(dense_inputs, reference_embedding, reference), dense_args, dense_data, processor
        )
        assert ("sc_correction" in actual) == score_centering
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key].values(), expected[key].values(), rtol=1e-4, atol=1e-4)
        for output in (actual, expected):
            sum((value.values() * coefficients).sum() for value in output.values()).backward()
        for parameter, baseline in zip(parameters, references, strict=True):
            dist.all_reduce(parameter.grad)
            parameter.grad /= world_size
            torch.testing.assert_close(parameter.grad, baseline.grad, rtol=1e-4, atol=1e-4)
        for optimizer in optimizers:
            optimizer.step()
        for parameter, baseline in zip(parameters, references, strict=True):
            torch.testing.assert_close(parameter, baseline, rtol=1e-4, atol=1e-4)


def _worker(rank, rendezvous, score_centering):
    get_torch_device().set_device(rank)
    dist.init_process_group("cpu:gloo,npu:hccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        with torch.device(f"npu:{rank}"):
            _compare_outputs(rank, 2, score_centering)
    finally:
        set_ulysses_sequence_parallel_group(None)
        dist.destroy_process_group()


@pytest.mark.parametrize("score_centering", [False, True])
def test_packed_outputs_use_standard_processors(tmp_path, score_centering):
    mp.spawn(_worker, args=(str(tmp_path / "packed_outputs"), score_centering), nprocs=2, join=True)
