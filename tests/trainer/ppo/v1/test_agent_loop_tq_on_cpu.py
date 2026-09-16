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

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    async def run():
        settled = asyncio.Event()

        async def fail():
            raise RuntimeError("session failed")

        async def finish_later():
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())


def test_deepseek_adjacent_images_keep_separate_spans():
    import torch

    from verl.utils.tokenizer.deepseek import expand_image_tokens

    ids, types = expand_image_tokens(torch.tensor([10, 129264, 129264, 12]), torch.tensor([[3, 3], [3, 6]]), 129264, 3)
    torch.testing.assert_close(ids, torch.tensor([[10] + [129264] * 9 + [12]]))
    torch.testing.assert_close(types, torch.tensor([[-1, 0, 1, 2, 3, 0, 1, 1, 2, 3, -1]]))


def test_deepseek_text_only_processor_outputs_keep_aligned_empty_image_inputs(monkeypatch):
    from types import SimpleNamespace

    import torch
    from transformers import BatchFeature

    from verl.trainer.ppo.v1 import agent_loop_tq
    from verl.utils.model import extract_multi_modal_inputs

    worker = object.__new__(agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class)
    worker.processor = SimpleNamespace(
        image_token_id=129264,
        config=SimpleNamespace(
            model_type="deepseek_v41",
            image_token_id=129264,
            vision_config=SimpleNamespace(patch_size=14, downsample_ratio=3),
        ),
    )
    worker.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "decoded text")
    worker._get_mm_processor_kwargs = lambda *args: {}
    monkeypatch.setattr(
        agent_loop_tq,
        "build_multimodal_processor_inputs",
        lambda *args, **kwargs: BatchFeature({"vision_token_types": torch.tensor([[-1]])}),
    )
    result = worker._compute_multi_modal_inputs(
        SimpleNamespace(multi_modal_data=None, mm_processor_kwargs={}), torch.tensor([0, 5, 6])
    )
    torch.testing.assert_close(result["vision_token_types"], torch.full((1, 3), -1))
    torch.testing.assert_close(result["_expanded_input_ids"], torch.tensor([[0, 5, 6]]))
    assert result["pixel_values"].shape == (0, 3, 14, 14)
    assert result["image_grid_hws"].shape == (0, 2)
    # Real processor patches are BF16; text-only rows must batch with them.
    batch = agent_loop_tq.list_of_dict_to_tensordict(
        [
            {"multi_modal_inputs": {"pixel_values": result["pixel_values"]}},
            {"multi_modal_inputs": {"pixel_values": torch.zeros((1, 3, 14, 14), dtype=torch.bfloat16)}},
        ]
    )
    assert extract_multi_modal_inputs(batch["multi_modal_inputs"])["pixel_values"].dtype == torch.bfloat16


def test_deepseek_multimodal_inputs_survive_storage_and_padding(monkeypatch):
    from unittest.mock import AsyncMock

    import torch
    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
    from transfer_queue.storage.simple_storage import StorageUnitData
    from transfer_queue.utils.serial_utils import decode, encode

    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
    from verl.trainer.ppo import padding_utils
    from verl.trainer.ppo.v1 import agent_loop_tq
    from verl.utils.model import extract_multi_modal_inputs
    from verl.utils.tokenizer.deepseek import expand_image_tokens

    worker = object.__new__(agent_loop_tq.AgentLoopWorkerTQ.__ray_metadata__.modified_class)
    worker.processor = None
    worker._compute_score = AsyncMock()
    worker._compute_teacher_logprobs = AsyncMock()
    put = AsyncMock()
    monkeypatch.setattr(agent_loop_tq.tq, "async_kv_batch_put", put)

    def image_inputs(output, input_ids):
        has_image = 129264 in output.prompt_ids
        grids = torch.tensor([[3, 3]]) if has_image else torch.empty((0, 2), dtype=torch.long)
        expanded_ids, types = expand_image_tokens(input_ids, grids, 129264, 3)
        return {
            "pixel_values": torch.ones((9 if has_image else 0, 3, 2, 2), dtype=torch.bfloat16),
            "image_grid_hws": grids,
            "vision_token_types": types,
            "_expanded_input_ids": expanded_ids,
        }

    worker._compute_multi_modal_inputs = image_inputs
    outputs = [
        AgentLoopOutput(
            prompt_ids=prompt,
            response_ids=[13, 14],
            response_mask=[1, 0],
            metrics=AgentLoopMetrics(),
            extra_fields={},
        )
        for prompt in ([10, 129264, 12], [20])
    ]
    asyncio.run(worker._agent_loop_postprocess(outputs, False, uid="sample", session_id=0, global_steps=1))
    fields = put.call_args.kwargs["fields"]
    assert not {"pixel_values", "image_grid_hws", "vision_token_types"}.intersection(fields.keys())
    torch.testing.assert_close(fields["responses"][0], torch.tensor([13, 14]))
    assert fields["prompts"][0].numel() == 6

    padding, _ = padding_utils.construct_minimal_padding_template(
        fields[0].to_dict(), put.call_args.kwargs["tags"][0], eos_token_id=0
    )
    samples = [fields[0].to_dict(), fields[1].to_dict(), padding]
    batch = agent_loop_tq.list_of_dict_to_tensordict(samples)
    # Exercise SimpleStorage's actual selection, wire encoding, and result packing.
    selected = AsyncSimpleStorageManager._select_by_positions(batch["multi_modal_inputs"], [2, 0, 1])
    storage = StorageUnitData(storage_size=len(samples))
    storage.put_data(decode(encode({"multi_modal_inputs": selected})), [2, 0, 1])
    received = decode(encode(storage.get_data(["multi_modal_inputs"], [0, 1, 2])))
    packed = AsyncSimpleStorageManager._pack_field_values(received["multi_modal_inputs"])
    merged = extract_multi_modal_inputs(packed)

    torch.testing.assert_close(merged["pixel_values"], torch.ones((9, 3, 2, 2), dtype=torch.bfloat16))
    torch.testing.assert_close(merged["image_grid_hws"], torch.tensor([[3, 3]]))
    expected_types = torch.full((3, padding_utils.SYNTHETIC_PADDING_SEQ_LEN), -1, dtype=torch.long)
    expected_types[0, 1:5] = torch.tensor([0, 1, 2, 3])
    torch.testing.assert_close(merged["vision_token_types"], expected_types)
