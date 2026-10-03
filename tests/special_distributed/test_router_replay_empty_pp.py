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
"""Run with torchrun --nproc-per-node=4 -m pytest to exercise real PP collectives."""

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")

from megatron.core import parallel_state as mpu  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group  # noqa: E402
from verl.utils.megatron.router_replay_patch import RouterReplay  # noqa: E402
from verl.utils.megatron.router_replay_utils import (  # noqa: E402
    RouterReplayHelper,
    get_current_rank_layer_info,
    is_moe_layer,
    pp_gather,
    reorder_and_merge_vpp_layers,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires four CUDA devices")


@pytest.fixture(scope="module", autouse=True)
def _parallel_groups():
    _, _, world_size = initialize_global_process_group()
    assert world_size == 4
    mpu.initialize_model_parallel(pipeline_model_parallel_size=4)
    yield
    dist.barrier()
    mpu.destroy_model_parallel()
    destroy_global_process_group()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("vpp_size", [None, 2])
@pytest.mark.parametrize("all_dense", [False, True])
def test_empty_and_uneven_pp_collectives(monkeypatch, nested, vpp_size, all_dense):
    frequency = [0] * 8 if all_dense else [0, 0, 1, 0, 0, 0, 1, 1]
    config = TransformerConfig(
        num_layers=8,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_layer_freq=frequency,
        pipeline_model_parallel_size=4,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=vpp_size,
    )

    def make_routes(layer_numbers):
        lengths = [3, 5] if nested else [5, 5]
        samples = []
        for sample_id, length in enumerate(lengths):
            routes = torch.empty(length, len(layer_numbers), 2, dtype=torch.int16)
            for column, layer in enumerate(layer_numbers):
                routes[:, column, :] = layer * 100 + sample_id * 10 + torch.arange(length).unsqueeze(-1)
            samples.append(routes)
        return torch.nested.as_nested_tensor(samples, layout=torch.jagged) if nested else torch.stack(samples)

    chunks = []
    layers_by_chunk = []
    for vp_stage in range(vpp_size or 1):
        layer_range = get_current_rank_layer_info(config, vp_stage)
        layers = [idx + 1 for idx in range(layer_range["start"], layer_range["end"]) if is_moe_layer(config, idx)]
        chunks.append(make_routes(layers))
        layers_by_chunk.append(layers)
    local_routes = reorder_and_merge_vpp_layers(chunks, 1, vpp_size, 1) if vpp_size else chunks[0]
    gathered = pp_gather(local_routes, config)
    expected = make_routes([idx + 1 for idx, flag in enumerate(frequency) if flag])

    assert gathered.dtype == torch.int16
    assert gathered.device.type == "cpu"
    for actual_sample, expected_sample in zip(gathered.unbind(), expected.unbind(), strict=True):
        torch.testing.assert_close(actual_sample, expected_sample, atol=0, rtol=0)

    for registry in (
        [layer for layers in layers_by_chunk for layer in layers],
        [idx + 1 for idx, flag in enumerate(frequency) if flag],
    ):
        monkeypatch.setattr(RouterReplay, "router_instances", registry)
        for vp_stage, layers in enumerate(layers_by_chunk):
            assert RouterReplayHelper.get_micro_batch_router_list(config, vp_stage) == layers
