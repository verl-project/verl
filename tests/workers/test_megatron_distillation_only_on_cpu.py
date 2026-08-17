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
"""Unit tests for Megatron supervised top-k distillation (distillation_only) path.

``MegatronEngineWithLMHead._lm_head_logits_processor`` must propagate distillation
outputs from ``logits_processor_func`` and omit ``log_probs`` when
``distillation_only=True``.

``vocab_parallel_log_probs_from_logits`` is patched out: it requires Megatron TP
groups and cannot run on CPU in CI. The contract under test is call-skipping and
key propagation, not numerical log-prob correctness.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

_VOCAB_SIZE = 8
_DISTILLATION_KEYS = ("distillation_losses", "student_mass", "teacher_mass")


def _make_engine_stub():
    eng = object.__new__(MegatronEngineWithLMHead)
    eng.engine_config = SimpleNamespace(
        entropy_from_logits_with_chunking=False,
        entropy_from_logits_chunk_size=1024,
    )
    return eng


def _make_logits_processor(keys):
    def _proc(student_logits, data, data_format):
        n = student_logits.shape[1]
        return {k: torch.full((1, n), float(i + 1)) for i, k in enumerate(keys)}

    return _proc


def _run_logits_processor(eng, *, distillation_use_topk, distillation_only):
    total_nnz = 5 if distillation_use_topk else 4
    logits = torch.randn(1, total_nnz, _VOCAB_SIZE)
    label = torch.randint(0, _VOCAB_SIZE, (1, total_nnz))
    temperature = torch.ones(1, total_nnz)
    batch = TensorDict({}, batch_size=[])

    with patch(
        "verl.workers.engine.megatron.transformer_impl.vocab_parallel_log_probs_from_logits",
        return_value=torch.zeros(1, total_nnz),
    ) as mock_log_probs:
        ret = eng._lm_head_logits_processor(
            logits.clone(),
            label,
            temperature.clone(),
            calculate_sum_pi_squared=False,
            calculate_entropy=False,
            distillation_use_topk=distillation_use_topk,
            distillation_only=distillation_only,
            logits_processor_func=_make_logits_processor(_DISTILLATION_KEYS),
            batch=batch,
            data_format="thd",
        )

    return mock_log_probs, ret, total_nnz


@pytest.mark.parametrize("distillation_only", [False, True])
def test_megatron_logits_processor_distillation_only_skips_log_probs(distillation_only):
    eng = _make_engine_stub()
    mock_log_probs, ret, total_nnz = _run_logits_processor(
        eng, distillation_use_topk=True, distillation_only=distillation_only
    )

    if distillation_only:
        mock_log_probs.assert_not_called()
        assert "log_probs" not in ret
    else:
        mock_log_probs.assert_called_once()
        assert "log_probs" in ret

    for k in _DISTILLATION_KEYS:
        assert k in ret, f"Missing distillation key {k!r}; got {list(ret.keys())}"
        assert ret[k].shape == (1, total_nnz), f"{k} shape {ret[k].shape}"


def test_megatron_logits_processor_without_topk_always_computes_log_probs():
    """When distillation_use_topk=False, log_probs are always computed.

    distillation_only is only set by the trainer when use_topk=True, so we do not
    parametrize distillation_only=True here (that combo cannot occur in production).
    """
    eng = _make_engine_stub()
    mock_log_probs, ret, _ = _run_logits_processor(eng, distillation_use_topk=False, distillation_only=False)

    mock_log_probs.assert_called_once()
    assert "log_probs" in ret
    for k in _DISTILLATION_KEYS:
        assert k not in ret
