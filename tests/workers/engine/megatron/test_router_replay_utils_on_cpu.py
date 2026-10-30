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
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction  # noqa: E402


class _FakeRouter:
    def __init__(self, recorded_topk_idx=None, router_replay_action=None):
        self.recorded_topk_idx = recorded_topk_idx
        self.router_replay_action = router_replay_action
        self.target_topk_idx = None
        self.target_replay_mask = None

    def set_target_indices(self, topk_indices, replay_mask=None):
        self.target_topk_idx = topk_indices
        self.target_replay_mask = replay_mask


@pytest.fixture(autouse=True)
def _restore_router_registry():
    old_instances = RouterReplay.router_instances
    RouterReplay.router_instances = []
    yield
    RouterReplay.router_instances = old_instances


def _config(num_layers=48, moe_layer_freq=1, virtual_pipeline_model_parallel_size=None):
    return SimpleNamespace(
        fp8=None,
        moe_layer_freq=moe_layer_freq,
        moe_router_topk=2,
        num_layers=num_layers,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_layout=None,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    )


def test_is_moe_layer_uses_hybrid_expert_symbols():
    """Verify hybrid-pattern expert symbols determine which layers use MoE routing."""
    config = _config(num_layers=6)
    config.hybrid_pattern = "VEVEVE"

    assert [rr_utils.is_moe_layer(config, idx) for idx in range(6)] == [False, True, False, True, False, True]


def test_get_micro_batch_router_list_supports_local_only_registry(monkeypatch):
    RouterReplay.router_instances = list(range(24))
    tf_config = _config()

    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda _config, _vp_rank=None: 24)
    assert rr_utils.RouterReplayHelper.get_micro_batch_router_list(tf_config) == list(range(24))


def test_empty_registry_is_a_noop_for_replay_disabled_engine():
    tf_config = _config()

    assert rr_utils.RouterReplayHelper.get_micro_batch_router_list(tf_config) == []
    assert rr_utils.RouterReplayHelper.is_r2_record_action(tf_config) is False
    assert rr_utils.RouterReplayHelper.is_replay_forward_action(tf_config) is False
    assert rr_utils.RouterReplayHelper.is_replay_backward_action(tf_config) is False


def test_action_queries_use_forwarded_model_instead_of_global_registry(monkeypatch):
    stale_router = _FakeRouter(router_replay_action=RouterReplayAction.REPLAY_FORWARD)
    live_router = _FakeRouter(router_replay_action=RouterReplayAction.RECORD)
    RouterReplay.router_instances = [stale_router]
    forwarded_model = object()
    tf_config = _config()

    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda model: iter([(1, live_router)]))

    assert rr_utils.RouterReplayHelper.is_r2_record_action(tf_config, model=forwarded_model)
    assert not rr_utils.RouterReplayHelper.is_replay_forward_action(tf_config, model=forwarded_model)


def test_get_micro_batch_router_list_supports_local_only_vpp_registry(monkeypatch):
    RouterReplay.router_instances = list(range(4))
    tf_config = _config(num_layers=8, virtual_pipeline_model_parallel_size=2)

    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda _config, _vp_rank=None: 2)
    monkeypatch.setattr(
        rr_utils,
        "get_current_rank_layer_info",
        lambda _config, vp_rank=None: {"start": 2 if vp_rank == 0 else 6, "end": 4 if vp_rank == 0 else 8},
    )

    assert rr_utils.RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank=1) == [2, 3]


def test_get_micro_batch_router_list_rejects_short_nonempty_registry(monkeypatch):
    RouterReplay.router_instances = [object()]
    tf_config = _config(num_layers=4)

    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda _config, _vp_rank=None: 2)

    with pytest.raises(RuntimeError, match="registry does not cover"):
        rr_utils.RouterReplayHelper.get_micro_batch_router_list(tf_config)


@pytest.mark.parametrize("frequency", [1, [0, 1, 0, 0, 1, 0, 1, 1], [0] * 8])
@pytest.mark.parametrize("vpp_size", [None, 2])
@pytest.mark.parametrize("pp_rank", [0, 1])
@pytest.mark.parametrize("registry_layout", ["local", "global"])
def test_registry_layouts_select_exact_moe_layers(monkeypatch, frequency, vpp_size, pp_rank, registry_layout):
    config = _config(num_layers=8, moe_layer_freq=frequency, virtual_pipeline_model_parallel_size=vpp_size)

    def layer_range(vp_rank):
        width = 8 // (2 * (vpp_size or 1))
        start = ((vp_rank or 0) * 2 + pp_rank) * width
        return {"start": start, "end": start + width}

    def stage_layers(vp_rank):
        bounds = layer_range(vp_rank)
        return [idx for idx in range(bounds["start"], bounds["end"]) if rr_utils.is_moe_layer(config, idx)]

    monkeypatch.setattr(rr_utils, "get_current_rank_layer_info", lambda _config, vp_rank=None: layer_range(vp_rank))
    monkeypatch.setattr(
        rr_utils, "get_moe_num_layers_to_build", lambda _config, vp_rank=None: len(stage_layers(vp_rank))
    )
    RouterReplay.router_instances = (
        [idx for idx in range(8) if rr_utils.is_moe_layer(config, idx)]
        if registry_layout == "global"
        else [idx for vp_rank in range(vpp_size or 1) for idx in stage_layers(vp_rank)]
    )

    for vp_rank in range(vpp_size or 1):
        assert rr_utils.RouterReplayHelper.get_micro_batch_router_list(config, vp_rank) == stage_layers(vp_rank)


@pytest.mark.parametrize("registry_size", [1, 3, 5, 9])
def test_registry_rejects_incomplete_or_extra_model_layout(monkeypatch, registry_size):
    config = _config(num_layers=8, virtual_pipeline_model_parallel_size=2)
    RouterReplay.router_instances = list(range(registry_size))
    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda _config, vp_rank=None: 2)

    with pytest.raises(RuntimeError, match="registry does not cover"):
        rr_utils.RouterReplayHelper.get_micro_batch_router_list(config, vp_rank=0)


@pytest.mark.parametrize("vp_rank", [-1, 2])
def test_registry_rejects_invalid_vp_rank(vp_rank):
    RouterReplay.router_instances = list(range(4))
    config = _config(num_layers=8, virtual_pipeline_model_parallel_size=2)
    with pytest.raises(ValueError, match="outside the configured VPP size"):
        rr_utils.RouterReplayHelper.get_micro_batch_router_list(config, vp_rank)


@pytest.mark.parametrize("registry_layout", ["local", "global"])
@pytest.mark.parametrize("frequency", [1, [0, 1, 0, 0, 1, 0, 1, 1]])
def test_positional_record_and_replay_use_the_same_routers(monkeypatch, registry_layout, frequency):
    config = _config(num_layers=8, moe_layer_freq=frequency, virtual_pipeline_model_parallel_size=2)
    global_layers = [idx for idx in range(8) if rr_utils.is_moe_layer(config, idx)]
    local_layers = [idx for idx in global_layers if idx in (2, 3, 6, 7)]
    registered_layers = local_layers if registry_layout == "local" else global_layers
    routers = {
        idx: _FakeRouter(torch.full((3, 2), 256 + idx, dtype=torch.int64), RouterReplayAction.RECORD)
        for idx in registered_layers
    }
    RouterReplay.router_instances = list(routers.values())
    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(rr_utils, "get_current_rank_layer_info", lambda _config, vp_rank=None: {"start": 6, "end": 8})
    monkeypatch.setattr(
        rr_utils,
        "get_moe_num_layers_to_build",
        lambda _config, vp_rank=None: sum(idx in ((2, 3) if vp_rank == 0 else (6, 7)) for idx in global_layers),
    )
    monkeypatch.setattr(rr_utils, "gather_from_sequence_parallel_region", lambda tensor, **_kwargs: tensor)
    monkeypatch.setattr(rr_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(rr_utils, "preprocess_packed_seqs", lambda tensor, *_args, **_kwargs: (tensor, object()))
    monkeypatch.setattr(rr_utils, "postprocess_packed_seqs", lambda tensor, *_args, **_kwargs: tensor)
    mask = torch.ones(1, 3, dtype=torch.bool)
    recorded = []
    rr_utils.merge_router_topk_indices(mask, torch.ones(1, 3, dtype=torch.long), recorded, config, vp_rank=1)
    expected_local = torch.stack([routers[idx].recorded_topk_idx for idx in (6, 7)], dim=1).unsqueeze(0)
    assert recorded[0].dtype == torch.int16
    torch.testing.assert_close(recorded[0], expected_local.to(torch.int16))

    global_routes = torch.stack([torch.full((3, 2), 256 + idx) for idx in global_layers], dim=1).unsqueeze(0)
    rr_utils.set_router_replay_data(global_routes, mask, config, vp_rank=1)
    for idx, router in routers.items():
        if idx in (6, 7):
            torch.testing.assert_close(router.target_topk_idx, router.recorded_topk_idx)
        else:
            assert router.target_topk_idx is None


def test_merge_router_topk_indices_uses_forwarded_model_with_leftover_vpp_recordings(monkeypatch):
    stale_a = torch.tensor([[12, 13], [14, 15], [16, 17]], dtype=torch.int64)
    stale_b = torch.tensor([[18, 19], [20, 21], [22, 23]], dtype=torch.int64)
    recorded_a = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
    recorded_b = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int64)
    RouterReplay.router_instances = [
        _FakeRouter(stale_a),
        _FakeRouter(stale_b),
        _FakeRouter(recorded_a),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(recorded_b),
    ]
    forwarded_model = object()
    tf_config = _config(num_layers=2)
    attention_mask = torch.ones(1, 3, dtype=torch.bool)
    input_ids = torch.ones(1, 3, dtype=torch.int64)
    merged = []

    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(
        rr_utils,
        "iter_model_routers",
        lambda model: iter([(2, RouterReplay.router_instances[5]), (1, RouterReplay.router_instances[2])]),
    )
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

    rr_utils.merge_router_topk_indices(
        attention_mask,
        input_ids,
        merged,
        tf_config,
        vp_rank=1,
        model=forwarded_model,
    )

    assert len(merged) == 1
    assert merged[0].shape == (1, 3, 2, 2)
    assert torch.equal(merged[0][0, :, 0, :], recorded_a.to(torch.uint8))
    assert torch.equal(merged[0][0, :, 1, :], recorded_b.to(torch.uint8))


@pytest.mark.parametrize("nested", [False, True])
def test_merge_router_topk_indices_emits_zero_layer_map_for_dense_stage(monkeypatch, nested):
    tf_config = _config(num_layers=2)
    if nested:
        input_ids = torch.nested.as_nested_tensor([torch.ones(3, dtype=torch.int64)], layout=torch.jagged)
        attention_mask = None
    else:
        input_ids = torch.ones(1, 3, dtype=torch.int64)
        attention_mask = torch.ones(1, 3, dtype=torch.bool)
    merged = []

    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter(()))
    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda *_args: 0)

    rr_utils.merge_router_topk_indices(
        attention_mask,
        input_ids,
        merged,
        tf_config,
        model=object(),
    )

    assert len(merged) == 1
    assert merged[0].dtype == torch.int16
    assert merged[0].shape[0] == 1
    assert merged[0].shape[2:] == (0, 2)
    if nested:
        assert merged[0].is_nested
        assert merged[0].unbind()[0].shape == (3, 0, 2)
    else:
        assert merged[0].shape == (1, 3, 0, 2)


def test_merge_router_topk_indices_requires_bshd_attention_mask(monkeypatch):
    router = _FakeRouter(torch.ones(3, 2, dtype=torch.int64))
    tf_config = _config(num_layers=1)

    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter([(1, router)]))
    monkeypatch.setattr(rr_utils, "gather_from_sequence_parallel_region", lambda tensor, **_kwargs: tensor)

    with pytest.raises(RuntimeError, match="RECORD requires attention_mask"):
        rr_utils.merge_router_topk_indices(
            None,
            torch.ones(1, 3, dtype=torch.int64),
            [],
            tf_config,
            model=object(),
        )


def test_merge_router_topk_indices_restores_bshd_sequence_major_rows(monkeypatch):
    """Verify dense sequence-major route rows merge into unpadded per-sample tensors."""
    recorded = torch.tensor([[10], [20], [11], [21], [0], [22], [0], [0]], dtype=torch.int64)
    router = _FakeRouter(recorded)
    input_ids = torch.nested.as_nested_tensor(
        [torch.ones(2, dtype=torch.int64), torch.ones(3, dtype=torch.int64)], layout=torch.jagged
    )
    merged = []

    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter([(1, router)]))
    monkeypatch.setattr(rr_utils, "gather_from_sequence_parallel_region", lambda tensor, **_kwargs: tensor)

    rr_utils.merge_router_topk_indices(
        None,
        input_ids,
        merged,
        _config(num_layers=1),
        model=object(),
        data_format="bshd",
    )

    assert len(merged) == 1
    assert merged[0].is_nested
    sample_0, sample_1 = merged[0].unbind()
    assert torch.equal(sample_0[:, 0, 0], torch.tensor([10, 11], dtype=torch.int16))
    assert torch.equal(sample_1[:, 0, 0], torch.tensor([20, 21, 22], dtype=torch.int16))


def test_empty_model_does_not_hide_missing_moe_routers(monkeypatch):
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter(()))
    monkeypatch.setattr(rr_utils, "get_moe_num_layers_to_build", lambda *_args: 2)
    with pytest.raises(RuntimeError, match="stage that expects MoE layers"):
        rr_utils.merge_router_topk_indices(
            torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.long), [], _config(), model=object()
        )


def test_record_rejects_duplicate_layer_numbers(monkeypatch):
    routers = [(1, _FakeRouter()), (1, _FakeRouter())]
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter(routers))
    with pytest.raises(RuntimeError, match="duplicate layer numbers"):
        rr_utils.merge_router_topk_indices(
            torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.long), [], _config(), model=object()
        )


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
    assert "recorded_local_positions=[0]" in message


def test_set_router_replay_data_uses_forwarded_vp_model(monkeypatch):
    routers = [
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
        _FakeRouter(),
    ]
    RouterReplay.router_instances = routers
    tf_config = _config(num_layers=4, virtual_pipeline_model_parallel_size=2)
    forwarded_model = object()
    attention_mask = torch.ones(1, 3, dtype=torch.bool)
    routed_experts = torch.tensor(
        [
            [
                [[1, 2], [3, 4], [5, 6], [7, 8]],
                [[9, 10], [11, 12], [13, 14], [15, 16]],
                [[17, 18], [19, 20], [21, 22], [23, 24]],
            ]
        ],
        dtype=torch.int64,
    )

    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(
        rr_utils,
        "preprocess_packed_seqs",
        lambda tensor, attention_mask, **_kwargs: (tensor, object()),
    )
    monkeypatch.setattr(rr_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda model: iter([(3, routers[2]), (4, routers[5])]))

    rr_utils.set_router_replay_data(
        routed_experts,
        attention_mask,
        tf_config,
        vp_rank=1,
        model=forwarded_model,
    )

    assert torch.equal(routers[2].target_topk_idx, routed_experts[0, :, 2, :])
    assert torch.equal(routers[5].target_topk_idx, routed_experts[0, :, 3, :])
    untouched = set(range(len(routers))) - {2, 5}
    assert all(routers[idx].target_topk_idx is None for idx in untouched)


def test_set_router_replay_data_rejects_missing_routes():
    with pytest.raises(RuntimeError, match="requires routed_experts"):
        rr_utils.set_router_replay_data(None, torch.ones(1, 1, dtype=torch.bool), _config())


def test_set_router_replay_data_requires_bshd_attention_mask():
    routes = torch.ones(1, 3, 1, 2, dtype=torch.int16)

    with pytest.raises(RuntimeError, match="REPLAY requires attention_mask"):
        rr_utils.set_router_replay_data(routes, None, _config(num_layers=1), model=object())


def test_set_router_replay_data_pads_bshd_in_sequence_major_order(monkeypatch):
    """Verify dense replay routes and masks are padded and flattened in sequence-major order."""
    router = _FakeRouter()
    routes = torch.nested.as_nested_tensor(
        [
            torch.tensor([[[10]], [[11]]], dtype=torch.int16),
            torch.tensor([[[20]], [[21]], [[22]]], dtype=torch.int16),
        ],
        layout=torch.jagged,
    )
    replay_mask = torch.nested.as_nested_tensor(
        [torch.tensor([True, True]), torch.tensor([True, False, True])], layout=torch.jagged
    )

    monkeypatch.setattr(rr_utils.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rr_utils.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(rr_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter([(1, router)]))

    rr_utils.set_router_replay_data(
        routes,
        None,
        _config(num_layers=1),
        replay_mask=replay_mask,
        model=object(),
        data_format="bshd",
        forced_max_seqlen=4,
    )

    assert torch.equal(router.target_topk_idx[:, 0], torch.tensor([10, 20, 11, 21, 0, 22, 0, 0]))
    assert torch.equal(
        router.target_replay_mask,
        torch.tensor([True, True, True, False, False, True, False, False]),
    )


def _patch_replay_plumbing(monkeypatch, layer_routers):
    """Stub packing, parallel scattering, and router discovery for CPU replay tests."""
    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(rr_utils, "preprocess_packed_seqs", lambda tensor, _mask, **_kwargs: (tensor, object()))
    monkeypatch.setattr(rr_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda model: iter(layer_routers))


def test_set_router_replay_data_rejects_truncated_route_tensor(monkeypatch):
    """A route tensor that covers only some of the model's layers cannot be mapped onto the
    module/MoE/block layouts and is rejected before any router is addressed."""
    router = _FakeRouter()
    tf_config = _config(num_layers=4)
    routed_experts = torch.ones(1, 3, 2, 1, dtype=torch.int64)
    _patch_replay_plumbing(monkeypatch, [(4, router)])

    with pytest.raises(ValueError, match="Cannot map 2 route layers to 4 modules"):
        rr_utils.set_router_replay_data(
            routed_experts,
            torch.ones(1, 3, dtype=torch.bool),
            tf_config,
            model=object(),
        )
    assert router.target_topk_idx is None


def test_set_router_replay_data_rejects_incomplete_model_routes(monkeypatch):
    """A complete route tensor that still leaves a forwarded router without a row is a hard failure."""
    router = _FakeRouter()
    tf_config = _config(num_layers=4)
    routed_experts = torch.ones(1, 3, 4, 1, dtype=torch.int64)
    _patch_replay_plumbing(monkeypatch, [(5, router)])

    with pytest.raises(RuntimeError, match="does not cover every forwarded MoE layer"):
        rr_utils.set_router_replay_data(
            routed_experts,
            torch.ones(1, 3, dtype=torch.bool),
            tf_config,
            model=object(),
        )


@pytest.mark.parametrize("action", [RouterReplayAction.REPLAY_FORWARD, RouterReplayAction.REPLAY_BACKWARD])
@pytest.mark.parametrize("mask", [[True, False, True, True, False], [True] * 5, [False] * 5])
def test_replay_topk_keeps_fixed_shape_and_native_routes(action, mask):
    """Verify native top-k keeps a fixed row count and only replaces uncovered routes."""
    router = RouterReplay()
    router.set_router_replay_action(action)
    target_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [1, 6], [0, 7]])
    replay_mask = torch.tensor(mask)
    router.set_target_indices(target_indices, replay_mask=replay_mask)
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.1, 0.2, 0.7, 0.3, 0.4, 0.5, 0.6],
            [0.8, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.9, 0.1, 0.4, 0.5, 0.6, 0.7],
            [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
        ]
    )
    native_rows_seen = []

    def compute_topk(values, topk, num_groups=None, group_topk=None):
        """Record the fixed row count and compute the native top-k experts."""
        del num_groups, group_topk
        native_rows_seen.append(values.shape[0])
        return torch.topk(values, k=topk, dim=1)

    probs, indices = router.get_replay_topk(scores, 2, default_compute_topk=compute_topk)
    expected = target_indices.clone()
    expected[~replay_mask] = torch.topk(scores[~replay_mask], k=2, dim=1).indices
    assert native_rows_seen == [5]
    torch.testing.assert_close(indices, expected)
    torch.testing.assert_close(probs, scores.gather(1, expected))


@pytest.mark.parametrize("missing_final", [False, True])
def test_compact_image_routes_are_consumed_at_each_span_start(missing_final):
    """Two unequal image spans must each consume their own compact placeholder route."""
    vision_types = [-1, 0, 1, 1, 2, 3, -1, 0, 1, 2, 3, -1, -1]
    input_ids = torch.nested.as_nested_tensor([torch.arange(len(vision_types))], layout=torch.jagged)
    # Text 10, image 11, text 12, image 13, text 14, optional final text 15.
    routes = torch.nested.as_nested_tensor(
        [torch.arange(10, 16 - int(missing_final), dtype=torch.int16).reshape(-1, 1, 1)], layout=torch.jagged
    )
    aligned = rr_utils.align_r3_router_replay_data(
        routes, input_ids, {"vision_token_types": torch.tensor([vision_types])}
    )
    assert aligned.unbind()[0][:, 0, 0].tolist() == [10, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 14, 0 if missing_final else 15]


@pytest.mark.parametrize(
    "tp_size,cp_size,cp_rank,fp8,padded_length",
    [
        (1, 1, 0, None, 9),
        (2, 1, 0, None, 10),
        (2, 2, 0, None, 16),
        (2, 2, 1, None, 16),
        (2, 1, 0, "hybrid", 256),
        (2, 2, 1, "hybrid", 512),
    ],
)
@pytest.mark.parametrize("forced_max_seqlen", [None, 17])
@pytest.mark.parametrize("dense", [False, True])
def test_bshd_replay_matches_model_padding_and_parallel_rows(
    monkeypatch, tp_size, cp_size, cp_rank, fp8, padded_length, forced_max_seqlen, dense
):
    """Routes and masks must match model padding, CP ownership, and TP scatter for raw maxima."""
    from verl.models.mcore.util import preprocess_bshd_engine

    monkeypatch.setattr(rr_utils.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(rr_utils.mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(rr_utils.mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    router = _FakeRouter()
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter([(1, router)]))
    # Select the last TP rank so padding cannot accidentally be omitted before scatter.
    monkeypatch.setattr(rr_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor.chunk(tp_size)[-1])
    lengths = [5, 7, 9] if not dense else [9, 9, 9]
    parts = [torch.arange(1, length + 1, dtype=torch.int16) + 20 * i for i, length in enumerate(lengths)]
    mask_parts = [part % 2 == 0 for part in parts]
    model_ids = torch.nested.as_nested_tensor(parts, layout=torch.jagged)
    model_rows, _, positions = preprocess_bshd_engine(
        model_ids, use_fp8_padding=fp8 is not None, forced_max_seqlen=forced_max_seqlen
    )
    if forced_max_seqlen is not None and fp8 is None:
        padded_length = {(1, 1): 17, (2, 1): 18, (2, 2): 24}[tp_size, cp_size]
    assert model_rows.shape == (3, padded_length // cp_size)
    # Independently construct the global padded routes, then select the model's CP positions.
    expected = torch.zeros(3, padded_length, dtype=torch.int16)
    expected_mask = torch.zeros(3, padded_length, dtype=torch.bool)
    for sample, part in enumerate(parts):
        expected[sample, : len(part)] = part
        expected_mask[sample, : len(part)] = mask_parts[sample]
    expected = expected[:, positions[0]].T.flatten().chunk(tp_size)[-1]
    expected_mask = expected_mask[:, positions[0]].T.flatten().chunk(tp_size)[-1]
    torch.testing.assert_close(model_rows.T.flatten().chunk(tp_size)[-1], expected)

    route_parts = [part.reshape(-1, 1, 1) for part in parts]
    if dense:
        routes, mask = torch.stack(route_parts), torch.stack(mask_parts)
    else:
        routes = torch.nested.as_nested_tensor(route_parts, layout=torch.jagged)
        mask = torch.nested.as_nested_tensor(mask_parts, layout=torch.jagged)
    config = _config(num_layers=1)
    config.fp8 = fp8
    rr_utils.set_router_replay_data(
        routes,
        torch.ones(3, 9, dtype=torch.bool) if dense else None,
        config,
        replay_mask=mask,
        model=object(),
        data_format="bshd",
        forced_max_seqlen=forced_max_seqlen,
    )
    torch.testing.assert_close(router.target_topk_idx[:, 0], expected.long())
    torch.testing.assert_close(router.target_replay_mask, expected_mask)


@pytest.mark.parametrize("version", [None, "v4", "v4.1"])
@pytest.mark.parametrize("cp_size,local_cp_size", [(1, None), (2, None), (2, 1), (1, 2)])
@pytest.mark.parametrize("fused", [False, True])
def test_thd_model_record_and_replay_share_csa_padding(monkeypatch, version, cp_size, local_cp_size, fused):
    """Record and replay must honor the active forward's CSA padding, including dynamic CP overrides."""
    from verl.models.mcore import model_forward, model_forward_fused

    expected_min = 8 if fused or (version == "v4.1" and (local_cp_size or cp_size) > 1) else None
    calls = []
    config = _config(num_layers=1)
    config.dsv4_version = version
    config.context_parallel_size = cp_size
    config.experimental_attention_variant = "dsv4_hybrid"
    config.csa_window_size = 8
    input_ids = torch.nested.as_nested_tensor([torch.arange(3)], layout=torch.jagged)
    routes = torch.nested.as_nested_tensor([torch.ones(3, 1, 1, dtype=torch.int16)], layout=torch.jagged)
    packed = SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 3], dtype=torch.int32))

    def preprocess(value, **kwargs):
        """Capture the forward's packing contract without requiring distributed collectives."""
        calls.append(kwargs["min_local_rows"])
        return (value.unbind()[0].unsqueeze(0), packed, None)

    model = SimpleNamespace(config=config, pre_process=True, post_process=False)

    # SimpleNamespace has no __call__, so use a tiny module that accepts both forward interfaces.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.__dict__.update(model.__dict__)

        def forward(self, **kwargs):
            return kwargs["input_ids"]

    monkeypatch.setattr(model_forward, "preprocess_thd_engine", preprocess)
    monkeypatch.setattr(model_forward_fused, "preprocess_thd_engine", preprocess)
    monkeypatch.setattr(rr_utils, "preprocess_thd_engine", preprocess)
    if fused:
        model_forward_fused.fused_forward_model_engine()(
            Model(), input_ids, input_ids, {}, 1.0, False, 0, local_cp_size=local_cp_size
        )
    else:
        model_forward.gptmodel_forward_model_engine(Model(), input_ids, {}, local_cp_size=local_cp_size)
    # Fused forward packs labels as well as inputs.
    assert calls == [expected_min] * (2 if fused else 1)
    calls.clear()
    router = _FakeRouter(torch.ones(3, 1, dtype=torch.int64))
    monkeypatch.setattr(rr_utils, "iter_model_routers", lambda _model: iter([(1, router)]))
    monkeypatch.setattr(rr_utils, "device_name", "cpu")
    monkeypatch.setattr(rr_utils, "gather_from_sequence_parallel_region", lambda value, **_kwargs: value)
    monkeypatch.setattr(rr_utils, "scatter_to_sequence_parallel_region", lambda value: value)
    payload = routes.unbind()[0].view(torch.uint8)
    monkeypatch.setattr(
        rr_utils,
        "postprocess_thd_engine",
        lambda *_args, **_kwargs: torch.nested.as_nested_tensor([payload], layout=torch.jagged),
    )
    recorded = []
    rr_utils.merge_router_topk_indices(
        None, input_ids, recorded, config, model=object(), local_cp_size=local_cp_size, use_fused_kernels=fused
    )
    rr_utils.set_router_replay_data(
        recorded[0], None, config, model=object(), local_cp_size=local_cp_size, use_fused_kernels=fused
    )
    assert calls == [expected_min, expected_min]
    torch.testing.assert_close(router.target_topk_idx, routes.unbind()[0][:, 0].long())
