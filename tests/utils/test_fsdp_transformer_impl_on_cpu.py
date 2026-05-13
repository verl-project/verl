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

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def test_prepare_model_inputs_preserves_offsets_for_gemma4_mm_token_type_ids():
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False

    input_ids = torch.nested.as_nested_tensor(
        [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])],
        layout=torch.jagged,
    )
    position_ids = torch.nested.as_nested_tensor(
        [torch.arange(3), torch.arange(3)],
        layout=torch.jagged,
    )
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(3, dtype=torch.int64), torch.ones(3, dtype=torch.int64)],
        layout=torch.jagged,
    )

    micro_batch = TensorDict(
        {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
            "temperature": torch.ones(2),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=False,
        pad_mode=DatasetPadMode.NO_PADDING,
        multi_modal_inputs=[
            {"mm_token_type_ids": torch.tensor([[0, 0, 0]], dtype=torch.long)},
            {"mm_token_type_ids": torch.tensor([[1, 1, 1]], dtype=torch.long)},
        ],
    )

    model_inputs, output_args = engine.prepare_model_inputs(micro_batch)

    assert "mm_token_type_ids" in model_inputs
    assert model_inputs["mm_token_type_ids"].shape == (2, 3)
    torch.testing.assert_close(
        model_inputs["mm_token_type_ids"],
        torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long),
    )
    torch.testing.assert_close(
        output_args["input_ids_rmpad_rolled"],
        torch.tensor([2, 3, 1, 5, 6, 4]),
    )
