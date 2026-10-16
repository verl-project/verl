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

from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from verl.models.transformers.qwen3_5 import _get_input_embeds
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

_IMAGE_TOKEN_ID = 29
_VIDEO_TOKEN_ID = 30


class _VisualOutput:
    def __init__(self, pooler_output: torch.Tensor) -> None:
        self.pooler_output = pooler_output


class _RecordingVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.proj.weight.dtype

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> _VisualOutput:
        self.calls.append((pixel_values.detach().clone(), grid_thw.detach().clone()))
        return _VisualOutput(self.proj(pixel_values))


class _EmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.visual = _RecordingVisual()
        self.config = SimpleNamespace(
            image_token_id=_IMAGE_TOKEN_ID,
            video_token_id=_VIDEO_TOKEN_ID,
            vision_config=SimpleNamespace(
                spatial_merge_size=1,
                in_channels=1,
                temporal_patch_size=1,
                patch_size=2,
            ),
        )

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding


def _legacy_separate_visual_calls(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    pixel_values_videos: torch.Tensor,
    image_grid_thw: torch.Tensor,
    video_grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Reference the pre-fix math while deliberately retaining two calls."""
    inputs_embeds = model.get_input_embeddings()(input_ids)
    for values, grid, token_id in (
        (pixel_values, image_grid_thw, model.config.image_token_id),
        (pixel_values_videos, video_grid_thw, model.config.video_token_id),
    ):
        media_embeds = model.visual(values.type(model.visual.dtype), grid_thw=grid).pooler_output
        mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, media_embeds.to(inputs_embeds.dtype))
    return inputs_embeds


def _mixed_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[_IMAGE_TOKEN_ID, 2, _VIDEO_TOKEN_ID, _IMAGE_TOKEN_ID]]),
        "pixel_values": torch.tensor(
            [[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]],
            requires_grad=True,
        ),
        "pixel_values_videos": torch.tensor([[-0.5, -1.0, -1.5, -2.0]], requires_grad=True),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
        "video_grid_thw": torch.tensor([[1, 1, 1]]),
    }


def test_mixed_image_video_uses_one_visual_call_and_matches_separate_math():
    torch.manual_seed(2026)
    combined_model = _EmbeddingModel()
    torch.manual_seed(2026)
    separate_model = _EmbeddingModel()

    combined_inputs = _mixed_inputs()
    separate_inputs = {
        key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in _mixed_inputs().items()
    }

    combined_embeds = _get_input_embeds(combined_model, **combined_inputs)["inputs_embeds"]
    separate_embeds = _legacy_separate_visual_calls(separate_model, **separate_inputs)

    assert len(combined_model.visual.calls) == 1
    assert len(separate_model.visual.calls) == 2
    torch.testing.assert_close(
        combined_model.visual.calls[0][0],
        torch.cat((combined_inputs["pixel_values"], combined_inputs["pixel_values_videos"]), dim=0),
    )
    torch.testing.assert_close(
        combined_model.visual.calls[0][1],
        torch.cat((combined_inputs["image_grid_thw"], combined_inputs["video_grid_thw"]), dim=0),
    )
    torch.testing.assert_close(combined_embeds, separate_embeds)

    combined_embeds.square().sum().backward()
    separate_embeds.square().sum().backward()
    for combined_parameter, separate_parameter in zip(
        combined_model.parameters(), separate_model.parameters(), strict=True
    ):
        torch.testing.assert_close(combined_parameter.grad, separate_parameter.grad)
    torch.testing.assert_close(combined_inputs["pixel_values"].grad, separate_inputs["pixel_values"].grad)
    torch.testing.assert_close(
        combined_inputs["pixel_values_videos"].grad,
        separate_inputs["pixel_values_videos"].grad,
    )


def test_real_qwen35_visual_combined_media_matches_separate_calls():
    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=1,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=8,
        num_position_embeddings=64,
    )
    config._attn_implementation = "eager"

    class RealVisionEmbeddingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(64, config.out_hidden_size)
            self.visual = Qwen3_5VisionModel(config)
            self.config = SimpleNamespace(
                image_token_id=_IMAGE_TOKEN_ID,
                video_token_id=_VIDEO_TOKEN_ID,
                vision_config=config,
            )

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return self.embedding

    torch.manual_seed(2026)
    combined_model = RealVisionEmbeddingModel()
    separate_model = RealVisionEmbeddingModel()
    separate_model.load_state_dict(combined_model.state_dict())

    image_grid_thw = torch.tensor([[1, 2, 2], [1, 2, 4]])
    video_grid_thw = torch.tensor([[2, 2, 2], [1, 4, 2]])
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    image_values = torch.randn(int(image_grid_thw.prod(dim=-1).sum()), patch_dim, requires_grad=True)
    video_values = torch.randn(int(video_grid_thw.prod(dim=-1).sum()), patch_dim, requires_grad=True)
    separate_image_values = image_values.detach().clone().requires_grad_()
    separate_video_values = video_values.detach().clone().requires_grad_()

    input_ids = torch.tensor(
        [
            [
                _IMAGE_TOKEN_ID,
                _VIDEO_TOKEN_ID,
                _IMAGE_TOKEN_ID,
                1,
                _VIDEO_TOKEN_ID,
                _VIDEO_TOKEN_ID,
                _IMAGE_TOKEN_ID,
                _VIDEO_TOKEN_ID,
            ]
        ]
    )
    combined_output = _get_input_embeds(
        combined_model,
        input_ids=input_ids,
        pixel_values=image_values,
        pixel_values_videos=video_values,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )["inputs_embeds"]
    separate_output = _legacy_separate_visual_calls(
        separate_model,
        input_ids=input_ids,
        pixel_values=separate_image_values,
        pixel_values_videos=separate_video_values,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    torch.testing.assert_close(combined_output, separate_output)

    combined_output.square().sum().backward()
    separate_output.square().sum().backward()
    torch.testing.assert_close(image_values.grad, separate_image_values.grad)
    torch.testing.assert_close(video_values.grad, separate_video_values.grad)
    for combined_parameter, separate_parameter in zip(
        combined_model.parameters(), separate_model.parameters(), strict=True
    ):
        assert combined_parameter.grad is not None
        assert separate_parameter.grad is not None
        torch.testing.assert_close(combined_parameter.grad, separate_parameter.grad)


@pytest.mark.parametrize("modality", ["image", "video", "text"])
def test_every_modality_combination_enters_visual_once(modality: str):
    model = _EmbeddingModel()
    kwargs: dict[str, torch.Tensor] = {"input_ids": torch.tensor([[1, 2]])}
    if modality == "image":
        kwargs.update(
            input_ids=torch.tensor([[_IMAGE_TOKEN_ID, 2]]),
            pixel_values=torch.ones(1, 4),
            image_grid_thw=torch.tensor([[1, 1, 1]]),
        )
    elif modality == "video":
        kwargs.update(
            input_ids=torch.tensor([[_VIDEO_TOKEN_ID, 2]]),
            pixel_values_videos=torch.ones(1, 4),
            video_grid_thw=torch.tensor([[1, 1, 1]]),
        )

    _get_input_embeds(model, **kwargs)
    assert len(model.visual.calls) == 1


class _TinyVisualBlock(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(input_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(inputs))


class _TinyVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            (
                _TinyVisualBlock(4, 5),
                _TinyVisualBlock(5, 6),
                _TinyVisualBlock(6, 7),
            )
        )
        self.merger = torch.nn.Linear(7, 4)

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].proj.weight.dtype

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> _VisualOutput:
        del grid_thw
        for block in self.blocks:
            pixel_values = block(pixel_values)
        return _VisualOutput(self.merger(pixel_values))


class _TinyQwen35(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.visual = _TinyVisual()
        self.lm_head = torch.nn.Linear(4, 2)
        self.config = SimpleNamespace(
            image_token_id=_IMAGE_TOKEN_ID,
            video_token_id=_VIDEO_TOKEN_ID,
            vision_config=SimpleNamespace(
                spatial_merge_size=1,
                in_channels=1,
                temporal_patch_size=1,
                patch_size=2,
            ),
        )

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(self, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs_embeds = _get_input_embeds(self, **model_inputs)["inputs_embeds"]
        return self.lm_head(inputs_embeds).square().mean()


def _distributed_inputs(rank: int, micro_batch: int, step: int) -> dict[str, torch.Tensor]:
    offset = float(step + 1) / 10
    if rank == 0 and micro_batch == 0:
        return {
            "input_ids": torch.tensor([[_IMAGE_TOKEN_ID, _IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, 1]]),
            "pixel_values": torch.arange(8, dtype=torch.float32).reshape(2, 4) / 7 + offset,
            "pixel_values_videos": torch.arange(4, dtype=torch.float32).reshape(1, 4) / 5 + offset,
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
        }
    if rank == 1 and micro_batch == 0:
        return {
            "input_ids": torch.tensor([[_IMAGE_TOKEN_ID, 2, _IMAGE_TOKEN_ID, 3, 4]]),
            "pixel_values": torch.arange(8, dtype=torch.float32).reshape(2, 4) / 9 + offset,
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
    if rank == 1:
        return {
            "input_ids": torch.tensor([[_VIDEO_TOKEN_ID, 5, _VIDEO_TOKEN_ID, 6]]),
            "pixel_values_videos": torch.arange(8, dtype=torch.float32).reshape(2, 4) / 11 + offset,
            "video_grid_thw": torch.tensor([[1, 1, 2]]),
        }
    return {"input_ids": torch.tensor([[7, 8, 9]])}


def _build_sharded_model(mesh) -> _TinyQwen35:
    torch.manual_seed(2026)
    model = _TinyQwen35()
    fully_shard(model.embedding, mesh=mesh, reshard_after_forward=False)
    for block in model.visual.blocks:
        fully_shard(block, mesh=mesh, reshard_after_forward=False)
    fully_shard(model.lm_head, mesh=mesh, reshard_after_forward=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    return model


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.full_tensor().detach().clone()
    return tensor.detach().clone()


def _train_two_steps(model: _TinyQwen35, rank: int, defer_sync: bool):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    engine = object.__new__(FSDPEngine)
    engine.module = model
    losses = []
    gradients = []

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for micro_batch in range(2):
            sync_context = (
                engine._gradient_sync_context(is_last_micro_batch=micro_batch == 1) if defer_sync else nullcontext()
            )
            with sync_context:
                loss = model(_distributed_inputs(rank, micro_batch, step))
                loss.backward()
            step_losses.append(loss.detach())

        losses.append(torch.stack(step_losses))
        gradients.append([_full_tensor(parameter.grad) for parameter in model.parameters()])
        for parameter in model.visual.parameters():
            visual_gradient = _full_tensor(parameter.grad)
            assert torch.isfinite(visual_gradient).all()
            assert torch.count_nonzero(visual_gradient) > 0
        optimizer.step()

    return losses, gradients, [_full_tensor(parameter) for parameter in model.parameters()]


def _distributed_worker(rank: int, world_size: int, rendezvous_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        baseline = _build_sharded_model(mesh)
        deferred = _build_sharded_model(mesh)
        initial_parameters = [_full_tensor(parameter) for parameter in baseline.parameters()]

        baseline_losses, baseline_gradients, baseline_parameters = _train_two_steps(baseline, rank, defer_sync=False)
        deferred_losses, deferred_gradients, deferred_parameters = _train_two_steps(deferred, rank, defer_sync=True)

        for baseline_loss, deferred_loss in zip(baseline_losses, deferred_losses, strict=True):
            torch.testing.assert_close(baseline_loss, deferred_loss, rtol=1e-6, atol=1e-6)
        for baseline_step, deferred_step in zip(baseline_gradients, deferred_gradients, strict=True):
            for baseline_gradient, deferred_gradient in zip(baseline_step, deferred_step, strict=True):
                assert torch.isfinite(baseline_gradient).all()
                torch.testing.assert_close(baseline_gradient, deferred_gradient, rtol=1e-5, atol=1e-6)

        update_norm = 0.0
        for initial, baseline_parameter, deferred_parameter in zip(
            initial_parameters, baseline_parameters, deferred_parameters, strict=True
        ):
            assert torch.isfinite(baseline_parameter).all()
            torch.testing.assert_close(baseline_parameter, deferred_parameter, rtol=1e-5, atol=1e-6)
            update_norm += (baseline_parameter - initial).float().square().sum().item()
        assert update_norm > 0
    finally:
        dist.destroy_process_group()


def test_two_rank_mixed_modal_dynamic_micro_batches_complete(tmp_path):
    rendezvous_file = str(tmp_path / "qwen35_mixed_modal_rdzv")
    process_context = mp.spawn(_distributed_worker, args=(2, rendezvous_file), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not process_context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("two-rank mixed-modal FSDP test exceeded 60 seconds")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        for process in process_context.processes:
            process.join(timeout=5)
