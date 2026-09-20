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

"""BF16/NVFP4 replay consumes identical targets from uint8 and int16 routes."""

import asyncio
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("megatron.core")

from tests.experimental.agent_loop.test_route_storage_on_cpu import output, routes_for, worker
from verl.protocol import DataProto, deserialize_tensordict, serialize_tensordict
from verl.utils.megatron import router_replay_utils as rr
from verl.workers.utils.padding import left_right_2_no_padding


class Router:
    def set_target_indices(self, indices, replay_mask=None):
        self.indices = indices
        self.replay_mask = replay_mask


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fp4", [False, True], ids=["bf16", "nvfp4"])
@pytest.mark.parametrize("codec", ["pickle", "numpy"])
def test_transport_filter_padding_and_megatron_consumption_match_int16(monkeypatch, device, fp4, codec):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA replay requires a GPU")
    outputs = []
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
        batch = batch.slice(0, 2)
        batch = batch.repeat(2, interleave=True).chunk(2)[0]
        if codec == "pickle":
            batch = pickle.loads(pickle.dumps(batch))
        else:
            batch.batch = deserialize_tensordict(serialize_tensordict(batch.batch))
        data = left_right_2_no_padding(batch.to_tensordict()).to(device)
        routes = rr.align_r3_router_replay_data(data["routed_experts"], data["input_ids"])
        mask = rr.build_r3_replay_mask(data["input_ids"], data["response_mask"])
        routers = [Router(), Router()]
        monkeypatch.setattr(rr, "iter_model_routers", lambda _model, _routers=routers: iter(enumerate(_routers, 1)))
        monkeypatch.setattr(rr, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
        monkeypatch.setattr(rr, "device_name", device)
        monkeypatch.setattr(rr.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(rr.mpu, "get_context_parallel_world_size", lambda: 1)
        monkeypatch.setattr(rr.mpu, "get_context_parallel_rank", lambda: 0)
        config = SimpleNamespace(fp8=None, fp4="nvfp4" if fp4 else None, num_layers=2)
        rr.set_router_replay_data(routes, None, config, replay_mask=mask, model=object())
        outputs.append([(r.indices.cpu(), r.replay_mask.cpu()) for r in routers])
    for actual, reference in zip(outputs[0], outputs[1], strict=True):
        assert actual[0].dtype == torch.int64
        torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
    # Avoid a vacuous all-padding comparison; the unsigned high ID survives.
    assert any(torch.any(indices == 255) for indices, _mask in outputs[0])
