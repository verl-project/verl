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

"""CPU coverage for score centering's FSDP engine wiring."""

from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


@pytest.mark.parametrize("use_remove_padding", [True, False])
@pytest.mark.parametrize("score_centering", [True, False])
def test_prepare_model_outputs_keeps_logits_intact_for_score_centering(use_remove_padding, score_centering):
    """The score centering hook saves the logits and re-reads them in its backward, so the engine
    must not run the flash-attn cross-entropy backward that writes the gradient into the logits."""
    seq_lengths = torch.tensor([3, 2])
    total_nnz, vocab_size = int(seq_lengths.sum()), 8
    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)])
    input_ids = torch.nested.nested_tensor_from_jagged(torch.randint(0, vocab_size, (total_nnz,)), offsets=cu_seqlens)
    output = type("Output", (), {})()
    output_args = {"input_ids_rmpad_rolled": torch.randint(0, vocab_size, (total_nnz,))}
    if use_remove_padding:
        output.logits = torch.randn(1, total_nnz, vocab_size)
        output_args.update(temperature_rmpad=torch.ones(total_nnz), pad_size=0)
    else:
        output.logits = torch.randn(2, 3, vocab_size)
        output_args["temperature"] = torch.ones(2)

    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        score_centering=score_centering,
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False

    def logits_processor(student_logits, data):
        return {"sc_correction": torch.zeros(student_logits.shape[:2])}

    with patch(
        "verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits", return_value=torch.zeros(total_nnz)
    ) as mock_logprobs:
        FSDPEngineWithLMHead.prepare_model_outputs(engine, output, output_args, micro_batch, logits_processor)

    assert mock_logprobs.call_args.kwargs.get("inplace_backward", True) is not score_centering
