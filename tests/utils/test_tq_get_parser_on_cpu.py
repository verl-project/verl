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
"""Each rank reads only the routes it replays from TransferQueue, through a get parser."""

import uuid
from functools import partial

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.tq_get_parsers import slice_for_rank


def test_slice_for_rank_keeps_stored_values():
    routes = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
    data = {"routed_experts": [routes, None], "input_ids": [1, 2]}

    parsed = slice_for_rank(dict(data), None, route_layers=[1, 3])
    torch.testing.assert_close(parsed["routed_experts"][0], routes[:, [1, 3]])
    assert parsed["routed_experts"][1] is None
    assert parsed["input_ids"] == [1, 2]
    assert routes.shape == (2, 4, 2)

    with pytest.raises(ValueError, match="exceed"):
        slice_for_rank(dict(data), None, route_layers=[4])


def test_slice_for_rank_copies_the_layers_of_a_pipeline_stage():
    """Consecutive layers take the fast path, which must copy rather than return a view of all layers."""
    routes = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
    layers = [1, 2]

    parsed = slice_for_rank({"routed_experts": [routes]}, None, route_layers=layers)["routed_experts"][0]
    torch.testing.assert_close(parsed, routes[:, layers])
    assert parsed.untyped_storage().nbytes() == parsed.numel() * parsed.element_size()


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_rows_match_the_thd_packing(cp_size, monkeypatch):
    """Concatenating the per-sample slices must rebuild exactly what the packing gives this rank."""
    util = pytest.importorskip("verl.models.mcore.util")
    if getattr(util.mpu, "__name__", "") != "megatron.core.parallel_state":
        pytest.skip("another test stubbed megatron out, so the real packing is unavailable")
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: cp_size)

    lengths = [7, 16, 33]
    routes = [torch.arange(n * 3 * 2, dtype=torch.int16).reshape(n, 3, 2) for n in lengths]
    # The rollout of the last sample missed the route of its final token, as R3 alignment allows.
    stored = routes[:-1] + [routes[-1][:-1]]
    aligned = torch.nested.as_nested_tensor(
        routes[:-1] + [torch.cat([stored[-1], torch.zeros_like(routes[-1][:1])])], layout=torch.jagged
    )
    data = {"routed_experts": stored, "input_ids": [torch.zeros(n) for n in lengths]}

    for cp_rank in range(cp_size):
        monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda rank=cp_rank: rank)
        packed, _, _ = util.preprocess_thd_engine(aligned, pre_process=True)
        parsed = slice_for_rank(dict(data), None, cp_rows=(cp_rank, cp_size, cp_size * 2))
        torch.testing.assert_close(torch.cat(parsed["routed_experts"]), packed.squeeze(0))


def test_cp_rows_cover_every_row_exactly_once():
    routes = torch.arange(7 * 1 * 1, dtype=torch.int16).reshape(7, 1, 1)
    data = {"routed_experts": [routes], "input_ids": [torch.zeros(7)]}
    seen = [
        row
        for cp_rank in range(4)
        for row in slice_for_rank(dict(data), None, cp_rows=(cp_rank, 4, 8))["routed_experts"][0]
    ]
    assert sorted(int(row) for row in seen) == [0, 0, 1, 2, 3, 4, 5, 6]  # the 7 rows plus one of padding


def test_cp_rows_need_the_aligned_length():
    routes = torch.zeros(7, 1, 1, dtype=torch.int16)
    with pytest.raises(ValueError, match="input_ids"):
        slice_for_rank({"routed_experts": [routes]}, None, cp_rows=(0, 4, 8))
    with pytest.raises(ValueError, match="route rows"):
        slice_for_rank({"routed_experts": [routes], "input_ids": [torch.zeros(9)]}, None, cp_rows=(0, 4, 8))


@pytest.fixture(scope="module")
def tq_init():
    """The running TransferQueue; only the integration test below needs one."""
    tq = pytest.importorskip("transfer_queue")
    tq.init()
    yield tq
    tq.close()


def test_tqbridge_fetches_the_rank_share(tq_init):
    tq = tq_init

    from verl.utils import transferqueue_utils as tqu

    if not tqu.tq_get_parser_supported():
        pytest.skip("TransferQueue has no get parser")

    keys = [uuid.uuid4().hex for _ in range(3)]
    routes = [torch.arange(n * 4 * 2, dtype=torch.int16).reshape(n, 4, 2) for n in (2, 3, 1)]
    fields = tu.get_tensordict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.arange(len(r)) for r in routes], layout=torch.jagged),
            "routed_experts": torch.nested.as_nested_tensor(routes, layout=torch.jagged),
        }
    )
    spec = tqu.TQGetSpec(
        data_parser=partial(slice_for_rank, route_layers=[0, 2]),
        non_tensor={"routed_experts_layers": [0, 2]},
    )

    class _Worker:
        def get_tq_get_spec(self, method_name):
            assert method_name == "infer"
            return spec

        @tqu.tqbridge()
        def infer(self, data: TensorDict):
            self.data = data

    try:
        meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields)
        worker = _Worker()
        worker.infer(meta)
        data = worker.data
        for got, ref in zip(data["routed_experts"].unbind(), routes, strict=True):
            torch.testing.assert_close(got, ref[:, [0, 2]])
        assert tu.get_non_tensor_data(data, "routed_experts_layers", None) == [0, 2]
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
