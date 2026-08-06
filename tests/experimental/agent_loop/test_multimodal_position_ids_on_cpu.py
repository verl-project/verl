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

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker


def test_compute_position_ids_accepts_processor_rope_kwargs_hook():
    class Processor:
        def get_rope_index_kwargs(self, multi_modal_inputs):
            return {"audio_seqlens": multi_modal_inputs["feature_attention_mask"].sum(-1)}

        def get_rope_index(
            self,
            *,
            input_ids,
            attention_mask,
            image_grid_thw,
            video_grid_thw,
            audio_seqlens,
        ):
            del attention_mask, image_grid_thw, video_grid_thw
            torch.testing.assert_close(audio_seqlens, torch.tensor([2]))
            return torch.zeros((3, 1, input_ids.shape[-1]), dtype=torch.long), None

    worker = type("Worker", (), {"processor": Processor()})()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {"feature_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long)}

    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids,
        attention_mask,
        multi_modal_inputs,
    )

    assert position_ids.shape == (1, 4, 3)
    torch.testing.assert_close(position_ids[:, 0], torch.tensor([[0, 1, 2]]))
