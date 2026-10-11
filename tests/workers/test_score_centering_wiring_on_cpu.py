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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead


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


@pytest.mark.parametrize("score_centering", [True, False])
def test_megatron_logits_processor_runs_score_centering_hook_and_log_probs(score_centering):
    total_nnz, vocab_size = 5, 8
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(entropy_from_logits_with_chunking=False, entropy_from_logits_chunk_size=1024)
    engine.tf_config = SimpleNamespace(fp8=None)
    calls = []

    def logits_processor(student_logits, data, data_format, **packing):
        calls.append(data_format)
        return {"sc_correction": torch.zeros(student_logits.shape[:2])}

    with patch(
        "verl.workers.engine.megatron.transformer_impl.vocab_parallel_log_probs_from_logits",
        return_value=torch.zeros(1, total_nnz),
    ) as mock_log_probs:
        ret = engine._lm_head_logits_processor(
            torch.randn(1, total_nnz, vocab_size),
            torch.randint(0, vocab_size, (1, total_nnz)),
            torch.ones(1, total_nnz),
            calculate_sum_pi_squared=False,
            calculate_entropy=False,
            distillation_use_topk=False,
            distillation_only=False,
            score_centering=score_centering,
            logits_processor_func=logits_processor,
            batch=TensorDict({}, batch_size=[]),
            data_format="thd",
        )

    mock_log_probs.assert_called_once()
    assert "log_probs" in ret
    assert calls == (["thd"] if score_centering else [])
    assert ("sc_correction" in ret) is score_centering


def test_megatron_log_probs_do_not_touch_the_logits_score_centering_saved():
    """Megatron's vocab-parallel cross entropy shifts fp32 logits in place; the score centering hook
    re-reads the logits it was given in its backward, so the engine must hand the cross entropy a copy."""
    total_nnz, vocab_size = 5, 8
    engine = object.__new__(MegatronEngineWithLMHead)
    engine.engine_config = SimpleNamespace(entropy_from_logits_with_chunking=False, entropy_from_logits_chunk_size=1024)
    engine.tf_config = SimpleNamespace(fp8=None)
    seen = {}

    def logits_processor(student_logits, data, data_format, **packing):
        seen["logits"] = student_logits
        seen["snapshot"] = student_logits.clone()
        return {"sc_correction": torch.zeros(student_logits.shape[:2])}

    def shifting_log_probs(logits, label):
        logits.sub_(logits.max(dim=-1, keepdim=True).values)
        return torch.zeros(logits.shape[:2])

    with patch(
        "verl.workers.engine.megatron.transformer_impl.vocab_parallel_log_probs_from_logits",
        side_effect=shifting_log_probs,
    ):
        engine._lm_head_logits_processor(
            torch.randn(1, total_nnz, vocab_size, dtype=torch.float32),
            torch.randint(0, vocab_size, (1, total_nnz)),
            torch.ones(1, total_nnz),
            calculate_sum_pi_squared=False,
            calculate_entropy=False,
            distillation_use_topk=False,
            distillation_only=False,
            score_centering=True,
            logits_processor_func=logits_processor,
            batch=TensorDict({}, batch_size=[]),
            data_format="thd",
        )

    torch.testing.assert_close(seen["logits"], seen["snapshot"])
