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
"""Regression guard for verl#6293.

The use_remove_padding=False branch of
FSDPEngineWithLMHead.prepare_model_outputs previously lacked the
distillation_use_topk handling that the use_remove_padding=True branch had,
so distillation outputs were silently dropped from model_output and the
downstream loss raised KeyError. This test invokes prepare_model_outputs on
a stub engine for both branches with distillation_use_topk=True and asserts
the distillation keys produced by logits_processor_func are propagated into
model_output as nested tensors in both cases.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

_VOCAB_SIZE = 8
_DISTILLATION_KEYS = ("distillation_losses", "student_mass")


def _make_engine_stub():
    """Bypass FSDPEngineWithLMHead.__init__; set only attributes that
    prepare_model_outputs touches in this test path (no SP, no fused kernels,
    no entropy)."""
    eng = object.__new__(FSDPEngineWithLMHead)
    eng.use_ulysses_sp = False

    class _EngineCfg:
        entropy_checkpointing = False

    eng.engine_config = _EngineCfg()
    return eng


def _make_logits_processor(keys):
    """Fake top-k distillation processor: returns one (1, total_nnz) tensor per key.

    The real processor (verl/trainer/distillation/losses.py) returns
    student_logits.shape[:2]; we mimic that contract.
    """

    def _proc(student_logits, data):
        n = student_logits.shape[1]
        return {k: torch.full((1, n), float(i + 1)) for i, k in enumerate(keys)}

    return _proc


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_distillation_outputs_emitted_in_both_padding_modes(use_remove_padding):
    """distillation_use_topk=True must populate distillation outputs into
    model_output regardless of use_remove_padding. See verl#6293."""
    bsz = 2
    seq_lengths_list = [3, 2]
    seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int64)
    total_nnz = int(seq_lengths.sum())

    cu_seqlens = torch.cat([torch.tensor([0]), seq_lengths.cumsum(0)]).to(torch.int64)

    flat_input_ids = torch.randint(0, _VOCAB_SIZE, (total_nnz,))
    input_ids_nested = torch.nested.nested_tensor_from_jagged(flat_input_ids, offsets=cu_seqlens)

    input_ids_rmpad_rolled = torch.randint(0, _VOCAB_SIZE, (total_nnz,))

    class _Output:
        pass

    output = _Output()

    if use_remove_padding:
        # True branch: output.logits shape (1, total_nnz, V), squeeze(0) -> (total_nnz, V).
        output.logits = torch.randn(1, total_nnz, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature_rmpad": torch.ones(total_nnz),
        }
    else:
        # False branch: output.logits shape (bsz, max_seqlen, V).
        max_seqlen = max(seq_lengths_list)
        output.logits = torch.randn(bsz, max_seqlen, _VOCAB_SIZE)
        output_args = {
            "input_ids_rmpad_rolled": input_ids_rmpad_rolled,
            "temperature": torch.ones(bsz),
        }

    micro_batch = TensorDict({"input_ids": input_ids_nested}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch,
        use_remove_padding=use_remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        calculate_sum_pi_squared=False,
        distillation_use_topk=True,
        max_response_length=max(seq_lengths_list),
    )

    eng = _make_engine_stub()
    model_output = FSDPEngineWithLMHead.prepare_model_outputs(
        eng,
        output=output,
        output_args=output_args,
        micro_batch=micro_batch,
        logits_processor_func=_make_logits_processor(_DISTILLATION_KEYS),
    )

    assert "log_probs" in model_output, (
        f"log_probs missing (use_remove_padding={use_remove_padding}); keys: {list(model_output.keys())}"
    )

    for k in _DISTILLATION_KEYS:
        assert k in model_output, (
            f"Distillation key '{k}' missing from model_output "
            f"(use_remove_padding={use_remove_padding}); "
            f"keys: {list(model_output.keys())}"
        )
        assert model_output[k].is_nested, (
            f"Expected '{k}' to be a nested tensor (use_remove_padding={use_remove_padding}); "
            f"got {type(model_output[k])}"
        )
