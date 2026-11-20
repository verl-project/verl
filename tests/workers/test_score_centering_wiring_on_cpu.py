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
    engine.engine_config = SimpleNamespace(batch_invariant=False)

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
        calls.append((data_format, packing))
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
            pad_to_length_bucket=512,
        )

    mock_log_probs.assert_called_once()
    assert "log_probs" in ret
    expected_packing = {
        "quant_align_size": None,
        "min_local_rows": None,
        "pad_to_length_bucket": 512,
        "cp_layout": "zigzag",
        "local_cp_size": None,
    }
    assert calls == ([("thd", expected_packing)] if score_centering else [])
    assert ("sc_correction" in ret) is score_centering


@pytest.mark.parametrize("use_fused_kernels", [False, True])
def test_megatron_forward_step_supports_pad_to_length_with_score_centering(use_fused_kernels):
    eng = object.__new__(MegatronEngineWithLMHead)
    eng.engine_config = SimpleNamespace(
        use_fused_kernels=use_fused_kernels,
        pad_to_length=True,
        pad_to_length_bucket=512,
        use_remove_padding=True,
        dynamic_context_parallel=False,
    )
    eng.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        mtp=SimpleNamespace(enable=False, enable_train=False),
    )
    eng.tf_config = SimpleNamespace(fp8=None)
    eng.enable_routing_replay = True

    input_ids = torch.nested.as_nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
    loss_mask = torch.nested.as_nested_tensor([torch.ones(3), torch.ones(2)], layout=torch.jagged)
    rollout_topk_ids = torch.nested.as_nested_tensor(
        [torch.tensor([[1, 2], [2, 3], [3, 4]]), torch.tensor([[4, 5], [5, 6]])],
        layout=torch.jagged,
    )
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "temperature": torch.ones(2),
            "rollout_topk_ids": rollout_topk_ids,
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(batch, score_centering=True)
    model = SimpleNamespace(config=SimpleNamespace(experimental_attention_variant=None))
    captured = {}

    def forward_fn(*args, **kwargs):
        captured.update(kwargs)
        return {}

    with (
        patch("verl.workers.engine.megatron.transformer_impl.get_device_id", return_value="cpu"),
        patch("verl.models.mcore.get_mcore_engine_forward_fn", return_value=forward_fn),
        patch("verl.models.mcore.get_mcore_forward_fused_model_engine_fn", return_value=forward_fn),
        patch(
            "verl.workers.engine.megatron.transformer_impl.RouterReplayHelper.is_replay_backward_action",
            return_value=False,
        ),
        patch(
            "verl.workers.engine.megatron.transformer_impl.RouterReplayHelper.is_replay_forward_action",
            return_value=False,
        ),
    ):
        eng.forward_step(iter([batch]), model, lambda **_kwargs: {}, lambda *args, **kwargs: None)

    assert captured["pad_to_length_bucket"] == 512
    processor_key = "auxiliary_logits_processor" if use_fused_kernels else "logits_processor"
    assert captured[processor_key].keywords["pad_to_length_bucket"] == 512


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
