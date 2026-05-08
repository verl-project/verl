from __future__ import annotations

import torch

from tests.agent.support import FakeProcessor
from verl.agent.framework.assembler import TrajectoryAssembler
from verl.agent.framework.types import Trajectory
from verl.utils import tensordict_utils as tu


def _build_trajectory(
    *,
    uid: str,
    session_id: str,
    trajectory_id: int,
    prompt_ids: list[int],
    response_ids: list[int],
    response_mask: list[int],
    multi_modal_data: dict | None = None,
) -> Trajectory:
    return Trajectory(
        uid=uid,
        session_id=session_id,
        trajectory_id=trajectory_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        reward_score=1.0,
        multi_modal_data=multi_modal_data,
    )


def test_compute_multi_modal_inputs_returns_empty_dict_without_processor():
    from verl.agent.framework.multi_modal_postprocess import compute_multi_modal_inputs

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert compute_multi_modal_inputs(None, input_ids, {"images": ["image://a.png"]}) == {}


def test_compute_multi_modal_inputs_returns_image_tensors_and_images_seqlens():
    from verl.agent.framework.multi_modal_postprocess import compute_multi_modal_inputs

    processor = FakeProcessor()
    input_ids = torch.tensor([[11, processor.image_token_id, 12]], dtype=torch.long)

    multi_modal_inputs = compute_multi_modal_inputs(
        processor,
        input_ids,
        {"images": ["image://a.png"]},
    )

    assert "input_ids" not in multi_modal_inputs
    assert "attention_mask" not in multi_modal_inputs
    assert tuple(multi_modal_inputs["pixel_values"].shape) == (1, 3, 2, 2)
    assert multi_modal_inputs["image_grid_thw"].tolist() == [[1, 2, 3]]
    assert multi_modal_inputs["images_seqlens"].tolist() == [6]
    assert "mm_token_type_ids" in multi_modal_inputs


def test_compute_position_ids_returns_text_shape_without_processor():
    from verl.agent.framework.multi_modal_postprocess import compute_position_ids

    input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    position_ids = compute_position_ids(None, input_ids, attention_mask, {})

    assert tuple(position_ids.shape) == (1, 4)
    assert position_ids.tolist() == [[0, 0, 1, 2]]


def test_compute_position_ids_returns_multimodal_shape_with_processor():
    from verl.agent.framework.multi_modal_postprocess import compute_position_ids

    processor = FakeProcessor()
    input_ids = torch.tensor(
        [[11, processor.image_token_id, processor.video_token_id, 12]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {
        "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[1, 3, 4]], dtype=torch.long),
        "mm_token_type_ids": torch.ones_like(input_ids),
    }

    position_ids = compute_position_ids(processor, input_ids, attention_mask, multi_modal_inputs)

    assert tuple(position_ids.shape) == (1, 4, 4)
    assert position_ids[0, 0].tolist() == [0, 1, 2, 3]
    assert processor.last_get_rope_index_call["mm_token_type_ids"].tolist() == [[0, 1, 2, 0]]


def test_apply_multi_modal_postprocess_aligns_non_tensor_outputs_per_sample():
    from verl.agent.framework.multi_modal_postprocess import apply_multi_modal_postprocess

    processor = FakeProcessor()
    trajectories = [
        _build_trajectory(
            uid="sample-0",
            session_id="session-0",
            trajectory_id=0,
            prompt_ids=[10, processor.image_token_id],
            response_ids=[20],
            response_mask=[1],
            multi_modal_data={"images": ["image://a.png"]},
        ),
        _build_trajectory(
            uid="sample-1",
            session_id="session-1",
            trajectory_id=0,
            prompt_ids=[30],
            response_ids=[40, 41],
            response_mask=[1, 1],
            multi_modal_data=None,
        ),
    ]
    batch = TrajectoryAssembler(pad_token_id=0).assemble(trajectories)

    output = apply_multi_modal_postprocess(batch, trajectories, processor)

    multi_modal_data = tu.get(output, "multi_modal_data")
    multi_modal_inputs = tu.get(output, "multi_modal_inputs")

    assert multi_modal_data == [{"images": ["image://a.png"]}, None]
    assert tuple(output["position_ids"].shape) == (2, 4, 4)
    assert tuple(multi_modal_inputs[0]["pixel_values"].shape) == (1, 3, 2, 2)
    assert multi_modal_inputs[0]["images_seqlens"].tolist() == [6]
    assert multi_modal_inputs[1] == {}
