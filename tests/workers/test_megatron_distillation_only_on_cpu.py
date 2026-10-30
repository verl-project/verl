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

from verl.utils import tensordict_utils as tu
from verl.utils.dynamic_cp_scheduler import DCP_PADDING_MASK
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

_VOCAB_SIZE = 8
_DISTILLATION_KEYS = ("distillation_losses", "student_mass", "teacher_mass")


def _make_engine_stub(tf_config=None):
    eng = object.__new__(MegatronEngineWithLMHead)
    eng.engine_config = SimpleNamespace(
        entropy_from_logits_with_chunking=False,
        entropy_from_logits_chunk_size=1024,
    )
    eng.tf_config = tf_config if tf_config is not None else SimpleNamespace(fp8=None)
    return eng


def _make_logits_processor(keys, packings=None):
    def _proc(student_logits, data, data_format, quant_align_size=None, cp_layout="zigzag", local_cp_size=None):
        if packings is not None:
            packings.append(
                {"quant_align_size": quant_align_size, "cp_layout": cp_layout, "local_cp_size": local_cp_size}
            )
        n = student_logits.shape[1]
        return {k: torch.full((1, n), float(i + 1)) for i, k in enumerate(keys)}

    return _proc


def _run_logits_processor(eng, *, distillation_use_topk, distillation_only, logits_processor_func=None, **packing):
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
            logits_processor_func=logits_processor_func or _make_logits_processor(_DISTILLATION_KEYS),
            batch=batch,
            data_format="thd",
            **packing,
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


@pytest.mark.parametrize(
    "packing, cp_layout, local_cp_size",
    [({}, "zigzag", None), ({"cp_layout": "contiguous", "local_cp_size": 2}, "contiguous", 2)],
)
@pytest.mark.parametrize(
    "tf_config, expected",
    [
        (SimpleNamespace(fp8="e4m3", fp8_recipe="mxfp8"), 32),
        (SimpleNamespace(fp8="hybrid", fp8_recipe="blockwise"), 16),
        (SimpleNamespace(fp8=None, fp8_recipe="mxfp8"), None),
    ],
)
def test_megatron_logits_processor_passes_the_student_packing_to_topk_loss(
    tf_config, expected, packing, cp_layout, local_cp_size
):
    """The top-k loss packs the teacher tensors itself, so it needs the padding and CP split of the student inputs."""
    eng = _make_engine_stub(tf_config)
    packings = []
    _run_logits_processor(
        eng,
        distillation_use_topk=True,
        distillation_only=True,
        logits_processor_func=_make_logits_processor(_DISTILLATION_KEYS, packings),
        **packing,
    )
    assert packings == [{"quant_align_size": expected, "cp_layout": cp_layout, "local_cp_size": local_cp_size}]


def test_megatron_forward_step_gives_the_topk_loss_the_student_cp_split():
    """forward_step packs the student with the model's CP layout (contiguous for dsv4_hybrid) and the micro-batch's
    dynamic CP group; the logits processor it hands to the model forward must carry both to the top-k loss."""
    eng = object.__new__(MegatronEngineWithLMHead)
    eng.engine_config = SimpleNamespace(
        use_fused_kernels=False,
        pad_to_length=False,
        pad_to_length_bucket=None,
        use_remove_padding=True,
        dynamic_context_parallel=True,
    )
    eng.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        mtp=SimpleNamespace(enable=False, enable_train=False),
    )
    eng.tf_config = SimpleNamespace(fp8=None)
    eng.enable_routing_replay = False

    input_ids = torch.nested.as_nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
    loss_mask = torch.nested.as_nested_tensor([torch.ones(3), torch.ones(2)], layout=torch.jagged)
    batch = TensorDict({"input_ids": input_ids, "loss_mask": loss_mask, "temperature": torch.ones(2)}, batch_size=[2])
    tu.assign_non_tensor(batch, distillation_use_topk=True, distillation_only=True, local_cp_size=2)
    tu.assign_non_tensor(batch, **{DCP_PADDING_MASK: torch.zeros(5, dtype=torch.bool)})
    model = SimpleNamespace(config=SimpleNamespace(experimental_attention_variant="dsv4_hybrid"))
    captured = {}

    def forward_fn(*args, **kwargs):
        captured.update(kwargs)
        return {}

    with (
        patch("verl.workers.engine.megatron.transformer_impl.get_device_id", return_value="cpu"),
        patch("verl.models.mcore.get_mcore_engine_forward_fn", return_value=forward_fn),
    ):
        eng.forward_step(iter([batch]), model, _make_logits_processor(_DISTILLATION_KEYS), lambda *a, **k: None)

    assert captured["cp_layout"] == "contiguous" and captured["local_cp_size"] == 2  # the student's split
    processor = captured["logits_processor"]
    assert processor.keywords["cp_layout"] == "contiguous"
    assert processor.keywords["local_cp_size"] == 2
