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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")

import verl.utils.megatron.router_replay_utils as router_utils
from verl.utils.dynamic_cp_scheduler import DCP_SAMPLE_IDS
from verl.workers.engine.megatron.transformer_impl import _attach_dcp_recorded_routes


def _nested(parts):
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def test_router_record_and_replay_use_dynamic_cp_size(monkeypatch):
    calls = []
    gathered_dtypes = []
    router = SimpleNamespace(recorded_topk_idx=torch.tensor([[1], [2]]))
    monkeypatch.setattr(router_utils.RouterReplayHelper, "get_micro_batch_router_list", lambda *_args: [router])
    monkeypatch.setattr(
        router_utils,
        "gather_from_sequence_parallel_region",
        lambda tensor, **_kwargs: gathered_dtypes.append(tensor.dtype) or tensor,
    )
    monkeypatch.setattr(router_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(router_utils, "get_current_rank_layer_info", lambda *_args: {"start": 0, "end": 1})
    monkeypatch.setattr(router_utils, "device_name", "cpu")

    # The route tensor reaches preprocess_thd_engine only on the replay side, and by
    # then it is back to int16; input_ids arrive as int64.
    def fake_preprocess(value, **kwargs):
        calls.append(kwargs["local_cp_size"])
        if value.dtype == torch.int16:
            value = torch.tensor([[[[1]], [[2]]]], dtype=torch.int16)
        return value, object(), None

    # postprocess_thd_engine is dtype- and trailing-shape-preserving, so it hands
    # back the uint8 payload merge_router_topk_indices fed into the collectives.
    payload = torch.tensor([[[1]], [[2]]], dtype=torch.int16).view(torch.uint8)
    monkeypatch.setattr(router_utils, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(
        router_utils,
        "postprocess_thd_engine",
        lambda output, _packed, _input_ids, _batch_size, **kwargs: (
            calls.append(kwargs["local_cp_size"]) or _nested([payload])
        ),
    )

    input_ids = _nested([torch.arange(2)])
    recorded = []
    config = SimpleNamespace(fp8=None, num_layers=1, moe_layer_freq=1)
    router_utils.merge_router_topk_indices(None, input_ids, recorded, config, local_cp_size=2)

    # int16 has no NCCL datatype, so the collective must see the uint8 view while the
    # recorded routes land back as int16.
    assert gathered_dtypes == [torch.uint8]
    assert recorded[0].dtype == torch.int16

    target = {}
    router.set_target_indices = lambda indices, replay_mask=None: target.update(indices=indices, mask=replay_mask)
    router_utils.set_router_replay_data(recorded[0], None, config, local_cp_size=2)

    assert calls == [2, 2, 2]
    torch.testing.assert_close(target["indices"], torch.tensor([[1], [2]]))


def test_r3_alignment_mask_and_dcp_collection():
    input_ids = _nested([torch.arange(4), torch.arange(3)])
    routes = _nested(
        [
            torch.full((3, 1, 1), 1, dtype=torch.int16),
            torch.full((3, 1, 1), 2, dtype=torch.int16),
        ]
    )
    aligned = router_utils.align_r3_router_replay_data(routes, input_ids)
    mask = router_utils.build_r3_replay_mask(
        input_ids,
        torch.tensor([[1, 1], [0, 0]], dtype=torch.bool),
    )

    assert [part.shape[0] for part in aligned.unbind()] == [4, 3]
    assert [part.tolist() for part in mask.unbind()] == [
        [True, True, True, False],
        [False, False, False],
    ]

    losses = [{DCP_SAMPLE_IDS: [1]}, {DCP_SAMPLE_IDS: [0]}]
    _attach_dcp_recorded_routes(losses, aligned)
    assert [part.shape[0] for part in losses[0]["model_output"]["routed_experts"].unbind()] == [4]
    assert [part.shape[0] for part in losses[1]["model_output"]["routed_experts"].unbind()] == [3]


def test_pp_gather_normalizes_nested_routes_to_cpu(monkeypatch):
    # The nested branch rides all_gather_object, which pickles the tensor, so int16
    # needs no uint8 reinterpretation here.
    local = _nested([torch.tensor([[[1]], [[2]]], dtype=torch.int16)])
    remote = _nested([torch.tensor([[[3]], [[4]]], dtype=torch.int16)])
    monkeypatch.setattr(router_utils.mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)

    def fake_all_gather(output, value, _group):
        assert value.device.type == "cpu"
        output[:] = [value, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)
    config = SimpleNamespace(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=None)
    routes = router_utils.pp_gather(local, config)

    assert routes.device.type == "cpu"
    assert routes.dtype == torch.int16
    assert [part.tolist() for part in routes.unbind()] == [[[[1], [3]], [[2], [4]]]]
