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

import torch

import verl.utils.megatron.router_replay_utils as router_utils


def test_merge_router_topk_indices_uses_dynamic_cp_size(monkeypatch):
    calls = {}
    router = SimpleNamespace(recorded_topk_idx=torch.tensor([[1], [2]], dtype=torch.int64))
    monkeypatch.setattr(router_utils.RouterReplayHelper, "get_micro_batch_router_list", lambda *_args: [router])
    monkeypatch.setattr(router_utils, "gather_from_sequence_parallel_region", lambda tensor, **_kwargs: tensor)
    monkeypatch.setattr(router_utils, "device_name", "cpu")

    packed = object()

    def fake_preprocess(input_ids, **kwargs):
        calls["preprocess"] = kwargs
        return input_ids, packed, None

    def fake_postprocess(output, packed_seq_params, input_ids, batch_size, **kwargs):
        calls["postprocess"] = kwargs
        assert packed_seq_params is packed
        assert batch_size == 1
        return input_ids

    monkeypatch.setattr(router_utils, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(router_utils, "postprocess_thd_engine", fake_postprocess)

    input_ids = torch.nested.as_nested_tensor([torch.arange(4)], layout=torch.jagged)
    outputs = []
    router_utils.merge_router_topk_indices(
        None,
        input_ids,
        outputs,
        SimpleNamespace(fp8=None),
        local_cp_size=2,
    )

    assert calls["preprocess"]["local_cp_size"] == 2
    assert calls["postprocess"]["local_cp_size"] == 2
    assert len(outputs) == 1


def test_set_router_replay_data_uses_dynamic_cp_size(monkeypatch):
    calls = {}

    class FakeRouter:
        def set_target_indices(self, indices, replay_mask=None):
            calls["target"] = indices
            calls["replay_mask"] = replay_mask

    def fake_preprocess(_layers_topk_idx, **kwargs):
        calls["preprocess"] = kwargs
        return torch.tensor([[[[3]], [[4]], [[5]]]], dtype=torch.int64), object(), None

    monkeypatch.setattr(router_utils, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(router_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(router_utils, "get_current_rank_layer_info", lambda *_args: {"start": 0, "end": 1})
    monkeypatch.setattr(
        router_utils.RouterReplayHelper,
        "get_micro_batch_router_list",
        lambda *_args: [FakeRouter()],
    )
    monkeypatch.setattr(router_utils, "device_name", "cpu")

    layers_topk_idx = torch.nested.as_nested_tensor(
        [torch.tensor([[[3]], [[4]], [[5]]], dtype=torch.int64)],
        layout=torch.jagged,
    )
    tf_config = SimpleNamespace(fp8=None, num_layers=1, moe_layer_freq=1)

    router_utils.set_router_replay_data(
        layers_topk_idx,
        None,
        tf_config,
        local_cp_size=3,
    )

    assert calls["preprocess"]["local_cp_size"] == 3
    torch.testing.assert_close(calls["target"], torch.tensor([[3], [4], [5]], dtype=torch.int64))
