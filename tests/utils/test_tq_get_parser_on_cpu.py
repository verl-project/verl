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
"""Each rank reads only its part of a sample from TransferQueue through a get parser."""

import uuid
from functools import partial

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.megatron.qwen_vl_vision_inputs import _local_cp_split, _local_vision_cp_data, plan_cp_vision_rows
from verl.utils.tq_get_parsers import PIXEL_KEYS, select_multi_modal_rows, slice_for_rank

HIDDEN = 3
MERGE = 4


def _bridge_cp_vision(pixel_values, grid_thw, cp_rank, cp_size):
    """Megatron-Bridge Qwen3-VL ``qwen3vl_cp_split`` + ``get_vision_cp_data`` on the full pixels."""
    frames = grid_thw[torch.arange(grid_thw.shape[0]).repeat_interleave(grid_thw[:, 0])].clone()
    frames[:, 0] = 1
    per_rank, remain = divmod(frames.shape[0], cp_size)
    cp_img_num = [per_rank + (1 if i < remain else 0) for i in range(cp_size)]
    seqlens = torch.repeat_interleave(frames[:, 1] * frames[:, 2], frames[:, 0])
    start = sum(cp_img_num[:cp_rank])
    end = start + cp_img_num[cp_rank]
    seqlens_list, begin = [], 0
    for count in cp_img_num:
        seqlens_list.append(seqlens[begin : begin + count] // MERGE)
        begin += count
    return pixel_values[seqlens[:start].sum() : seqlens[:end].sum()], frames[start:end], seqlens_list


def _sample(grids, key="image_grid_thw", seed=0):
    if not grids:
        return {}
    grid = torch.tensor(grids, dtype=torch.int64)
    rows = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    generator = torch.Generator().manual_seed(seed)
    pixel_key = "pixel_values" if key == "image_grid_thw" else "pixel_values_videos"
    return {pixel_key: torch.randn(rows, HIDDEN, generator=generator), key: grid}


@pytest.mark.parametrize("cp_size", [2, 3, 4])
@pytest.mark.parametrize("key", ["image_grid_thw", "video_grid_thw"])
def test_cp_vision_rows_match_bridge_split(cp_size, key):
    t = 1 if key == "image_grid_thw" else 3
    samples = [
        _sample([[t, 2, 4], [1, 4, 4]], key, seed=0),
        {},
        _sample([[t, 2, 2]], key, seed=1),
        _sample([[1, 2, 6], [t, 4, 2], [1, 2, 2]], key, seed=2),
    ]
    pixel_key = "pixel_values" if key == "image_grid_thw" else "pixel_values_videos"
    full_pixels = torch.cat([s[pixel_key] for s in samples if s])
    full_grid = torch.cat([s[key] for s in samples if s])
    grids = [{k: v for k, v in s.items() if k not in PIXEL_KEYS} for s in samples]

    for cp_rank in range(cp_size):
        plan = plan_cp_vision_rows(grids, cp_rank, cp_size)
        local = [s[k][start:end] for s, ranges in zip(samples, plan, strict=True) for k, start, end in ranges]
        local_pixels = torch.cat(local) if local else full_pixels[:0]
        split = _local_cp_split(cp_size, local_pixels, full_grid)
        vision_data, vision_grid, seqlens_list = _local_vision_cp_data(*split[:2], MERGE, *split[2:], cp_rank, cp_size)

        ref_data, ref_grid, ref_seqlens = _bridge_cp_vision(full_pixels, full_grid, cp_rank, cp_size)
        torch.testing.assert_close(vision_data, ref_data)
        torch.testing.assert_close(vision_grid, ref_grid)
        for got, ref in zip(seqlens_list, ref_seqlens, strict=True):
            torch.testing.assert_close(got, ref)


def test_cp_vision_rows_need_all_pixels_when_images_and_videos_mix():
    grids = [{"image_grid_thw": torch.tensor([[1, 2, 2]])}, {"video_grid_thw": torch.tensor([[2, 2, 2]])}]
    assert plan_cp_vision_rows(grids, 0, 2) is None
    assert plan_cp_vision_rows([{}, None], 1, 2) == [[], []]


def test_local_vision_cp_data_rejects_full_pixels():
    grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
    split = _local_cp_split(2, torch.zeros(12, HIDDEN), grid)
    with pytest.raises(RuntimeError, match="pixel rows"):
        _local_vision_cp_data(*split[:2], MERGE, *split[2:], 0, 2)


def test_slice_for_rank_keeps_stored_values():
    routes = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
    inputs = {"pixel_values": torch.ones(4, HIDDEN), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    data = {"multi_modal_inputs": [inputs, None], "routed_experts": [routes, None], "input_ids": [1, 2]}

    parsed = slice_for_rank(dict(data), None, drop_multi_modal_keys=PIXEL_KEYS, route_layers=[1, 3])
    assert parsed["multi_modal_inputs"] == [{"image_grid_thw": inputs["image_grid_thw"]}, None]
    torch.testing.assert_close(parsed["routed_experts"][0], routes[:, [1, 3]])
    assert parsed["routed_experts"][1] is None
    assert parsed["input_ids"] == [1, 2]
    assert "pixel_values" in inputs and routes.shape == (2, 4, 2)

    with pytest.raises(ValueError, match="exceed"):
        slice_for_rank(dict(data), None, route_layers=[4])


def test_slice_for_rank_copies_the_layers_of_a_pipeline_stage():
    """Consecutive layers take the fast path, which must copy rather than return a view of all layers."""
    routes = torch.arange(2 * 4 * 2).reshape(2, 4, 2)
    layers = [1, 2]

    parsed = slice_for_rank({"routed_experts": [routes]}, None, route_layers=layers)["routed_experts"][0]
    torch.testing.assert_close(parsed, routes[:, layers])
    assert parsed.untyped_storage().nbytes() == parsed.numel() * parsed.element_size()


def test_select_multi_modal_rows():
    pixels = torch.arange(6 * HIDDEN, dtype=torch.float32).reshape(6, HIDDEN)
    data = {"multi_modal_inputs": [{"pixel_values": pixels}, {"pixel_values": pixels}]}
    parsed = select_multi_modal_rows(data, [[["pixel_values", 2, 5]], [["pixel_values", 0, None]]])
    torch.testing.assert_close(parsed["multi_modal_inputs"][0]["pixel_values"], pixels[2:5])
    torch.testing.assert_close(parsed["multi_modal_inputs"][1]["pixel_values"], pixels)


tq = pytest.importorskip("transfer_queue")


@pytest.fixture(scope="module")
def tq_init():
    tq.init()
    yield
    tq.close()


def test_tqbridge_fetches_the_rank_share(tq_init):
    from verl.utils import transferqueue_utils as tqu

    if not tqu.tq_get_parser_supported():
        pytest.skip("TransferQueue has no get parser")

    keys = [uuid.uuid4().hex for _ in range(3)]
    routes = [torch.arange(n * 4 * 2, dtype=torch.int16).reshape(n, 4, 2) for n in (2, 3, 1)]
    samples = [_sample([[1, 2, 4]], seed=0), _sample([[1, 2, 2], [1, 4, 2]], seed=1), {}]
    fields = tu.get_tensordict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.arange(len(r)) for r in routes], layout=torch.jagged),
            "routed_experts": torch.nested.as_nested_tensor(routes, layout=torch.jagged),
            "multi_modal_inputs": samples,
        }
    )
    meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields)

    spec = tqu.TQGetSpec(
        data_parser=partial(slice_for_rank, drop_multi_modal_keys=PIXEL_KEYS, route_layers=[0, 2]),
        non_tensor={"routed_experts_layers": [0, 2]},
        attach_keys=True,
    )

    class _Worker:
        def get_tq_get_spec(self, method_name):
            assert method_name == "infer"
            return spec

        @tqu.tqbridge()
        def infer(self, data: TensorDict):
            self.data = data

    worker = _Worker()
    worker.infer(meta)
    data = worker.data
    try:
        for got, ref in zip(data["routed_experts"].unbind(), routes, strict=True):
            torch.testing.assert_close(got, ref[:, [0, 2]])
        multi_modal_inputs = tu.get(data, "multi_modal_inputs")
        assert all(not set(PIXEL_KEYS) & set(inputs or {}) for inputs in multi_modal_inputs)
        torch.testing.assert_close(multi_modal_inputs[1]["image_grid_thw"], samples[1]["image_grid_thw"])
        assert tu.get_non_tensor_data(data, "routed_experts_layers", None) == [0, 2]
        assert tu.get(data, tqu.TQ_KEY_FIELD) == keys
        assert tu.get_non_tensor_data(data, tqu.TQ_PARTITION_KEY, None) == "train"

        plan = plan_cp_vision_rows(multi_modal_inputs, 1, 2)
        rows = [i for i, ranges in enumerate(plan) if ranges]
        fetched = tqu.kv_batch_get_parsed(
            keys=[keys[i] for i in rows],
            partition_id="train",
            select_fields="multi_modal_inputs",
            data_parser=select_multi_modal_rows,
            parser_args=[plan[i] for i in rows],
        )
        for i, pixels in zip(rows, tu.get(fetched, "multi_modal_inputs"), strict=True):
            ((key, start, end),) = plan[i]
            torch.testing.assert_close(pixels[key], samples[i][key][start:end])
    finally:
        tq.kv_clear(keys=keys, partition_id="train")


def test_parsers_read_linked_multi_modal_inputs(tq_init):
    """Trajectories of a GRPO prompt link its ``multi_modal_inputs``; parsing them works the same way."""
    from verl.utils import transferqueue_utils as tqu

    if not tqu.tq_get_parser_supported() or not hasattr(tq, "kv_link"):
        pytest.skip("TransferQueue has no get parser or public links")

    public_key = uuid.uuid4().hex
    keys = [f"{public_key}_{i}" for i in range(3)]
    sample = _sample([[1, 2, 4], [1, 4, 2]], seed=3)
    tq.kv_batch_put(
        keys=[public_key],
        partition_id=tq.PUBLIC_PARTITION,
        fields=tu.list_of_dict_to_tensordict([{"multi_modal_inputs": sample}]),
    )
    tq.kv_link(keys=keys, partition_id="train", public_key=public_key, fields="multi_modal_inputs")
    tq.kv_batch_put(keys=keys, partition_id="train", fields=tu.get_tensordict({"input_ids": torch.ones(3, 2)}))
    try:
        data = tq.kv_batch_get(
            keys,
            "train",
            select_fields="multi_modal_inputs",
            data_parser=partial(slice_for_rank, drop_multi_modal_keys=PIXEL_KEYS),
        )
        assert all(set(inputs) == {"image_grid_thw"} for inputs in tu.get(data, "multi_modal_inputs"))

        fetched = tqu.kv_batch_get_parsed(
            keys=keys,
            partition_id="train",
            select_fields="multi_modal_inputs",
            data_parser=select_multi_modal_rows,
            parser_args=[[["pixel_values", 0, 8]], [["pixel_values", 8, 16]], [["pixel_values", 0, 8]]],
        )
        pixels = [inputs["pixel_values"] for inputs in tu.get(fetched, "multi_modal_inputs")]
        torch.testing.assert_close(pixels[0], sample["pixel_values"][:8])
        torch.testing.assert_close(pixels[1], sample["pixel_values"][8:])
        torch.testing.assert_close(pixels[2], sample["pixel_values"][:8])
    finally:
        tq.kv_clear(keys=keys, partition_id="train")
        tq.kv_clear(keys=[public_key], partition_id=tq.PUBLIC_PARTITION)
