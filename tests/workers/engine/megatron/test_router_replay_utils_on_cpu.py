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

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from verl.utils.megatron import router_replay_utils as rr_utils  # noqa: E402
from verl.utils.megatron.router_replay_patch import RouterReplay  # noqa: E402


class _FakeRouter:
    def __init__(self, recorded_topk_idx=None):
        self.recorded_topk_idx = recorded_topk_idx
        self.target_topk_idx = None

    def set_target_indices(self, topk_indices):
        self.target_topk_idx = topk_indices


@pytest.fixture(autouse=True)
def _restore_router_registry():
    old_instances = RouterReplay.router_instances
    RouterReplay.router_instances = []
    yield
    RouterReplay.router_instances = old_instances


def _config(num_layers=48, moe_layer_freq=1):
    return SimpleNamespace(
        fp8=None,
        moe_layer_freq=moe_layer_freq,
        num_layers=num_layers,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
        pipeline_model_parallel_layout=None,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    )


def test_get_micro_batch_router_list_uses_pp_layer_offset(monkeypatch):
    RouterReplay.router_instances = list(range(48))
    tf_config = _config()

    monkeypatch.setattr(rr_utils, "get_current_rank_layer_info", lambda _config, _vp_rank=None: {"start": 0, "end": 24})
    assert rr_utils.RouterReplayHelper.get_micro_batch_router_list(tf_config) == list(range(24))

    monkeypatch.setattr(
        rr_utils, "get_current_rank_layer_info", lambda _config, _vp_rank=None: {"start": 24, "end": 48}
    )
    assert rr_utils.RouterReplayHelper.get_micro_batch_router_list(tf_config) == list(range(24, 48))


def test_get_micro_batch_router_list_supports_local_only_registry(monkeypatch):
    RouterReplay.router_instances = list(range(24))
    tf_config = _config()

    monkeypatch.setattr(
        rr_utils, "get_current_rank_layer_info", lambda _config, _vp_rank=None: {"start": 24, "end": 48}
    )
    assert rr_utils.RouterReplayHelper.get_micro_batch_router_list(tf_config) == list(range(24))


def test_merge_router_topk_indices_prefers_actual_recorded_routers(monkeypatch):
    recorded_a = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
    recorded_b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int64)
    RouterReplay.router_instances = [
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(recorded_a),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(recorded_b),
    ]
    tf_config = _config(num_layers=2)
    attention_mask = torch.ones(1, 3, dtype=torch.bool)
    input_ids = torch.ones(1, 3, dtype=torch.int64)
    merged = []

    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda _config, _vp_rank=None: 2)
    monkeypatch.setattr(rr_utils, "get_current_rank_layer_info", lambda _config, _vp_rank=None: {"start": 0, "end": 2})
    monkeypatch.setattr(rr_utils, "gather_from_sequence_parallel_region", lambda tensor, **_kwargs: tensor)
    monkeypatch.setattr(
        rr_utils,
        "preprocess_packed_seqs",
        lambda input_ids, attention_mask, **_kwargs: (input_ids, object()),
    )
    monkeypatch.setattr(
        rr_utils,
        "postprocess_packed_seqs",
        lambda tensor, _packed_seq_params, _attention_mask, _batch_size, _seq_len, **_kwargs: tensor,
    )

    rr_utils.merge_router_topk_indices(attention_mask, input_ids, merged, tf_config)

    assert tf_config._verl_router_replay_local_router_indices == [2, 5]
    assert len(merged) == 1
    assert merged[0].shape == (1, 3, 2, 2)
    assert torch.equal(merged[0][0, :, 0, :], recorded_a.to(torch.uint8))
    assert torch.equal(merged[0][0, :, 1, :], recorded_b.to(torch.uint8))


def test_thd_sequence_parallel_padding_mask_marks_only_alignment_tail(monkeypatch):
    input_ids = torch.nested.as_nested_tensor([torch.arange(235)], layout=torch.jagged)

    monkeypatch.setattr(rr_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rr_utils.mpu, "get_tensor_model_parallel_world_size", lambda: 2)

    monkeypatch.setattr(rr_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    rank_zero_mask = rr_utils.get_thd_sequence_parallel_padding_mask(input_ids)
    assert rank_zero_mask.shape == (118,)
    assert not rank_zero_mask.any()

    monkeypatch.setattr(rr_utils.mpu, "get_tensor_model_parallel_rank", lambda: 1)
    rank_one_mask = rr_utils.get_thd_sequence_parallel_padding_mask(input_ids)
    assert rank_one_mask.shape == (118,)
    assert rank_one_mask.sum().item() == 1
    assert rank_one_mask[-1].item()


def test_merge_router_topk_indices_hard_fails_when_record_count_mismatches(monkeypatch):
    RouterReplay.router_instances = [
        _FakeRouter(torch.tensor([[1, 2]], dtype=torch.int64)),
        _FakeRouter(),
        _FakeRouter(),
    ]
    tf_config = _config(num_layers=2)

    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda _config, _vp_rank=None: 2)
    monkeypatch.setattr(
        rr_utils.RouterReplayHelper,
        "get_micro_batch_router_list",
        staticmethod(lambda _config, _vp_rank=None: RouterReplay.router_instances[:2]),
    )
    monkeypatch.setattr(rr_utils, "get_current_rank_layer_info", lambda _config, _vp_rank=None: {"start": 0, "end": 2})
    monkeypatch.setattr(rr_utils.mpu, "get_pipeline_model_parallel_rank", lambda: 0)

    with pytest.raises(RuntimeError, match="router replay RECORD did not capture all local routers") as exc_info:
        rr_utils.merge_router_topk_indices(
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.int64),
            [],
            tf_config,
        )

    message = str(exc_info.value)
    assert "missing_local_positions=[1]" in message
    assert "recorded_global_indices=[0]" in message


def test_set_router_replay_data_reuses_cached_recorded_router_indices(monkeypatch):
    routers = [
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
    ]
    RouterReplay.router_instances = routers
    tf_config = _config(num_layers=2)
    tf_config._verl_router_replay_local_router_indices = [2, 5]
    attention_mask = torch.ones(1, 3, dtype=torch.bool)
    routed_experts = torch.tensor(
        [[[[1, 2], [7, 8]], [[3, 4], [9, 10]], [[5, 6], [11, 12]]]],
        dtype=torch.int64,
    )

    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(
        rr_utils,
        "preprocess_packed_seqs",
        lambda tensor, attention_mask, **_kwargs: (tensor, object()),
    )
    monkeypatch.setattr(rr_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(rr_utils, "get_current_rank_layer_info", lambda _config, _vp_rank=None: {"start": 0, "end": 2})

    rr_utils.set_router_replay_data(routed_experts, attention_mask, tf_config)

    assert torch.equal(routers[2].target_topk_idx, routed_experts[0, :, 0, :])
    assert torch.equal(routers[5].target_topk_idx, routed_experts[0, :, 1, :])
    assert routers[0].target_topk_idx is None
    assert routers[1].target_topk_idx is None
