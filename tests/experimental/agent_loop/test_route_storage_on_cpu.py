# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

"""Lossless compact routing storage and generic DataProto transport."""

import asyncio
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.experimental.agent_loop import agent_loop as module
from verl.protocol import DataProto, deserialize_tensordict, serialize_tensordict


class Tokenizer:
    pad_token_id = 0
    padding_side = "right"

    def pad(self, values, *, max_length, return_attention_mask, **_kwargs):
        tokens = values["input_ids"]
        padding = [0] * (max_length - len(tokens))
        ids = padding + tokens if self.padding_side == "left" else tokens + padding
        mask = (
            [0] * len(padding) + [1] * len(tokens)
            if self.padding_side == "left"
            else [1] * len(tokens) + [0] * len(padding)
        )
        result = {"input_ids": torch.tensor([ids])}
        if return_attention_mask:
            result["attention_mask"] = torch.tensor([mask])
        return result


async def noop(*_args, **_kwargs):
    return None


def worker():
    result = object.__new__(module.AgentLoopWorker)
    result.rollout_config = SimpleNamespace(prompt_length=8, response_length=16)
    result.tokenizer = Tokenizer()
    result.processor = None
    result.mm_processor_kwargs = None
    result.reward_loop_worker_handles = None
    result._compute_score = noop
    result._compute_teacher_logprobs = noop
    return result


def output(routes):
    return module.AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5, 6, 7],
        response_mask=[1] * 4,
        response_logprobs=[-0.5] * 4,
        routed_experts=routes,
        reward_score=0.5,
        num_turns=2,
        metrics=module.AgentLoopMetrics(),
    )


def routes_for(kind):
    dtype = np.uint8 if "u8" in kind else (np.int32 if "i32" in kind else np.int16)
    values = [0, 127, 128, 255] if dtype == np.uint8 else [-1, 127, 300, 32767]
    routes = np.array(values * 6, dtype=dtype).reshape(6, 2, 2)
    if kind == "readonly_u8":
        routes.flags.writeable = False
    if kind.startswith("tensor"):
        routes = torch.from_numpy(routes)
    if kind == "strided_u8":
        routes = routes[:, :, ::-1]
        # Existing from_numpy does not accept negative strides; use positive
        # non-contiguous source to exercise the supported strided case.
        routes = np.repeat(routes.copy(), 2, axis=0)[::2]
    return routes


KINDS = ["numpy_u8", "readonly_u8", "tensor_u8", "strided_u8", "numpy_i16", "tensor_i16", "numpy_i32", "tensor_i32"]


@pytest.mark.parametrize("kind", KINDS)
def test_route_storage_preserves_values_and_original_source(kind):
    routes = routes_for(kind)
    original = np.asarray(routes).copy()
    expected_dtype = torch.uint8 if "u8" in kind else torch.int16
    as_dict = output(routes).as_dict()["routed_experts"]
    assert as_dict.dtype == expected_dtype
    torch.testing.assert_close(as_dict[:6].to(torch.int16), torch.tensor(original, dtype=torch.int16))
    assert torch.count_nonzero(as_dict[6:]) == 0
    obj = worker()
    internal = asyncio.run(obj._agent_loop_postprocess(output(routes), False, raw_prompt="test"))
    batch = obj._postprocess([internal])
    packed = batch.batch["routed_experts"]
    assert packed.dtype == expected_dtype
    assert packed.numel() * packed.element_size() == (96 if expected_dtype == torch.uint8 else 192)
    torch.testing.assert_close(packed[0, 5:11].to(torch.int16), torch.tensor(original, dtype=torch.int16))
    assert torch.count_nonzero(packed[0, :5]) == torch.count_nonzero(packed[0, 11:]) == 0
    np.testing.assert_array_equal(np.asarray(routes), original)


def test_signed_list_keeps_legacy_storage():
    routes = routes_for("numpy_i16").tolist()
    result = output(routes).as_dict()["routed_experts"]
    assert result.dtype == torch.int16 and result[0, 0, 0] == -1


@pytest.mark.parametrize("codec", ["pickle", "numpy"])
def test_transport_preserves_unsigned_routes(codec):
    packed = []
    for dtype in (np.uint8, np.int16):
        obj = worker()
        internal = asyncio.run(
            obj._agent_loop_postprocess(
                output(np.asarray(routes_for("numpy_u8"), dtype=dtype)), False, raw_prompt="test"
            )
        )
        batch = obj._postprocess([internal])
        batch = DataProto.concat([batch, batch])
        batch = batch.select_idxs([1, 0, 1])
        batch.reorder(torch.tensor([2, 0, 1]))
        batch = batch.slice(0, 2).repeat(2, interleave=True).chunk(2)[0]
        if codec == "pickle":
            batch = pickle.loads(pickle.dumps(batch))
        else:
            batch.batch = deserialize_tensordict(serialize_tensordict(batch.batch))
        packed.append(batch.batch["routed_experts"])
    assert packed[0].dtype == torch.uint8
    assert packed[1].dtype == torch.int16
    assert packed[0].numel() * packed[0].element_size() * 2 == packed[1].numel() * packed[1].element_size()
    torch.testing.assert_close(packed[0].to(torch.int16), packed[1], rtol=0, atol=0)
