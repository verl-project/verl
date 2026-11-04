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
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    engine.compute_entropy_from_logits = entropy_from_logits
    return engine


def _compare_outputs(rank, world_size, score_centering):
    torch.manual_seed(7)
    ids = torch.nested.as_nested_tensor([torch.tensor([2, 3, 4]), torch.tensor([5, 6])], layout=torch.jagged)
    offsets = ids.offsets()
    topk = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    data = TensorDict(
        {
            "input_ids": ids,
            "rollout_topk_ids": torch.nested.nested_tensor_from_jagged(topk, offsets),
            "rollout_topk_log_probs": torch.nested.nested_tensor_from_jagged(torch.full((5, 2), -2.0), offsets),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_remove_padding=False,
        score_centering=score_centering,
        calculate_entropy=True,
        calculate_sum_pi_squared=True,
    )
    args = dict(
        input_ids_rmpad_rolled=ids.values().roll(-1),
        temperature=torch.tensor([0.8, 1.3], dtype=torch.float32),
        temperature_is_one=False,
    )
    hidden = torch.randn(5, 7, dtype=torch.float32)
    dense_hidden = torch.zeros(2, 3, 7, dtype=hidden.dtype)
    valid = torch.arange(3)[None, :] < offsets.diff()[:, None]
    dense_hidden[valid] = hidden
    chunk = (len(hidden) + world_size - 1) // world_size
    pad_size = chunk * world_size - len(hidden)
    local_hidden = torch.nn.functional.pad(hidden, (0, 0, 0, pad_size))[rank * chunk : (rank + 1) * chunk]
    weight = torch.randn(7, 11, dtype=torch.float32, requires_grad=True)
    reference = weight.detach().clone().requires_grad_()
    optimizers = [torch.optim.SGD([parameter], lr=0.01) for parameter in (weight, reference)]
    processor = (
        partial(
            score_centering_logits_processor, config=SimpleNamespace(policy_loss=SimpleNamespace(rollout_correction={}))
        )
        if score_centering
        else None
    )
    coefficients = torch.arange(1, 6, dtype=hidden.dtype)
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad()
        set_ulysses_sequence_parallel_group(dist.group.WORLD)
        actual = _engine(world_size).prepare_model_outputs(
            SimpleNamespace(
                logits=(local_hidden @ weight).unsqueeze(0),
                logits_layout="packed",
                logits_pad_size=pad_size,
            ),
            args,
            data,
            processor,
        )
        set_ulysses_sequence_parallel_group(None)
        expected = _engine(1).prepare_model_outputs(
            SimpleNamespace(logits=dense_hidden @ reference), args, data, processor
        )
        assert ("sc_correction" in actual) == score_centering
        assert actual.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(actual[key].values(), expected[key].values(), rtol=1e-4, atol=1e-4)
        for output in (actual, expected):
            sum((value.values() * coefficients).sum() for value in output.values()).backward()
        dist.all_reduce(weight.grad)
        weight.grad /= world_size
        torch.testing.assert_close(weight.grad, reference.grad, rtol=1e-4, atol=1e-4)
        for optimizer in optimizers:
            optimizer.step()
        torch.testing.assert_close(weight, reference, rtol=1e-4, atol=1e-4)


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
