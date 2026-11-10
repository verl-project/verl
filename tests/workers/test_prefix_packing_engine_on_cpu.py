# Copyright 2025 Individual Contributor: Deepro Choudhury
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
"""CPU tests for the FSDP engine wiring of ``use_prefix_packing``.

The engine is built with ``object.__new__`` and the model output is random, so only the index plumbing is checked.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead
from verl.workers.utils.padding import no_padding_2_padding
from verl.workers.utils.prefix_packing import check_prefix_packing_supported

VOCAB = 16
NUM_LAYERS = 2
PROMPTS = [[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8], [7, 8]]
RESPONSES = [[10, 11], [12], [13, 14, 15], [9], [9, 10]]


def _make_batch(prompts=PROMPTS, include_prompts=True, position_ids_3d=False, **non_tensor) -> TensorDict:
    rows = [torch.tensor(p + r) for p, r in zip(prompts, RESPONSES, strict=True)]
    pos_rows = [torch.arange(len(r)) for r in rows]
    if position_ids_3d:
        pos_rows = [p.unsqueeze(0).expand(4, -1).clone() for p in pos_rows]
    max_resp = max(len(r) for r in RESPONSES)
    response_mask = torch.zeros(len(rows), max_resp, dtype=torch.long)
    for i, r in enumerate(RESPONSES):
        response_mask[i, : len(r)] = 1
    data = {
        "input_ids": torch.nested.as_nested_tensor(rows, layout=torch.jagged),
        "position_ids": torch.nested.as_nested_tensor(pos_rows, layout=torch.jagged),
        "response_mask": response_mask,
        "responses": torch.nested.as_nested_tensor([torch.tensor(r) for r in RESPONSES], layout=torch.jagged),
    }
    if include_prompts:
        data["prompts"] = torch.nested.as_nested_tensor([torch.tensor(p) for p in prompts], layout=torch.jagged)
    micro_batch = TensorDict(data, batch_size=[])
    flags = dict(
        use_remove_padding=True,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        temperature=1.0,
        max_response_len=max_resp,
    )
    tu.assign_non_tensor(micro_batch, **{**flags, **non_tensor})
    return micro_batch


def _make_engine(cls=FSDPEngineWithLMHead, pad_to_length=False):
    engine = object.__new__(cls)
    engine.use_prefix_packing = True
    engine.prefix_packing_num_layers = NUM_LAYERS
    engine.use_ulysses_sp = False
    engine.use_remove_padding = True
    engine.pass_packed_cu_seqlens = False
    engine.pad_to_length = pad_to_length
    engine.pad_to_length_bucket = 16
    engine.module = SimpleNamespace()
    return engine


class TestPrepareModelInputs:
    def test_packs_shared_prompts(self):
        micro_batch = _make_batch()
        model_inputs, output_args = FSDPEngineWithLMHead.prepare_model_inputs(_make_engine(), micro_batch)
        plan = output_args["prefix_packing_plan"]
        flat = micro_batch["input_ids"].values()
        assert plan.num_packed == plan.num_unpacked - 3
        assert torch.equal(model_inputs["input_ids"][0], flat[plan.pack_index])
        assert torch.equal(model_inputs["position_ids"][0], micro_batch["position_ids"].values()[plan.pack_index])
        assert model_inputs["prefix_packing"] is output_args["prefix_packing_meta"]
        assert torch.equal(output_args["input_ids_rmpad_rolled"], torch.roll(flat[plan.pack_index], -1))
        assert output_args["temperature_rmpad"].shape == (plan.num_packed,)

    def test_nothing_shared_runs_unpacked(self):
        micro_batch = _make_batch(prompts=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2], [3, 4]])
        model_inputs, output_args = FSDPEngineWithLMHead.prepare_model_inputs(_make_engine(), micro_batch)
        assert "prefix_packing" not in model_inputs and "prefix_packing_plan" not in output_args
        assert torch.equal(model_inputs["input_ids"][0], micro_batch["input_ids"].values())

    def test_pad_to_length_adds_tail_segment(self):
        engine = _make_engine(pad_to_length=True)
        model_inputs, output_args = FSDPEngineWithLMHead.prepare_model_inputs(engine, _make_batch())
        meta = output_args["prefix_packing_meta"]
        total = model_inputs["input_ids"].shape[1]
        assert total % 16 == 0
        assert output_args["pad_size"] == total - output_args["prefix_packing_plan"].num_packed
        assert meta.num_rows == total
        assert meta.cu_seqlens_q[-1] - meta.cu_seqlens_q[-2] == output_args["pad_size"]

    def test_text_only_multi_modal_placeholders_do_not_block(self):
        micro_batch = _make_batch()
        tu.assign_non_tensor(micro_batch, multi_modal_inputs=[{} for _ in PROMPTS])
        model_inputs, _ = FSDPEngineWithLMHead.prepare_model_inputs(_make_engine(), micro_batch)
        assert "prefix_packing" in model_inputs

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(distillation_use_topk=True), "distillation"),
            (dict(score_centering=True), "score centering"),
            (dict(position_ids_3d=True), "multi-modal"),
            (dict(multi_modal_inputs=[{"pixel_values": torch.zeros(1, 4)} for _ in PROMPTS]), "multi-modal"),
            (dict(include_prompts=False), "prompts"),
        ],
    )
    def test_rejects_unsupported_batches(self, kwargs, match):
        with pytest.raises(RuntimeError, match=match):
            FSDPEngineWithLMHead.prepare_model_inputs(_make_engine(), _make_batch(**kwargs))

    def test_rejects_router_replay(self):
        micro_batch = _make_batch()
        micro_batch["routed_experts"] = micro_batch["input_ids"].clone()
        with pytest.raises(RuntimeError, match="router replay"):
            FSDPEngineWithLMHead.prepare_model_inputs(_make_engine(), micro_batch)


class TestPrepareModelOutputs:
    def _run(self, cls, pad_to_length=False, layers_seen=NUM_LAYERS, last_dim=VOCAB):
        micro_batch = _make_batch()
        engine = _make_engine(cls, pad_to_length=pad_to_length)
        model_inputs, output_args = cls.prepare_model_inputs(engine, micro_batch)
        output_args["prefix_packing_meta"].counter.value = layers_seen
        logits = torch.randn(
            1, model_inputs["input_ids"].shape[1], last_dim, generator=torch.Generator().manual_seed(0)
        )
        out = cls.prepare_model_outputs(engine, SimpleNamespace(logits=logits), output_args, micro_batch, None)
        return micro_batch, output_args, logits, out

    @pytest.mark.parametrize("pad_to_length", [False, True])
    def test_log_probs_unpacked(self, pad_to_length):
        micro_batch, output_args, logits, out = self._run(FSDPEngineWithLMHead, pad_to_length)
        plan = output_args["prefix_packing_plan"]
        n = plan.num_packed
        packed_lp = logprobs_from_logits(logits[0, :n], output_args["input_ids_rmpad_rolled"][:n])
        assert torch.equal(out["log_probs"].offsets(), micro_batch["input_ids"].offsets())
        torch.testing.assert_close(out["log_probs"].values(), packed_lp[plan.unpack_index])

        response_lp = no_padding_2_padding(out["log_probs"], micro_batch)
        offsets = micro_batch["input_ids"].offsets().tolist()
        for i, (prompt, response) in enumerate(zip(PROMPTS, RESPONSES, strict=True)):
            for j in range(len(response)):
                row = plan.unpack_index[offsets[i] + len(prompt) - 1 + j]
                torch.testing.assert_close(response_lp[i, j], packed_lp[row])

    def test_values_unpacked(self):
        _, output_args, values, out = self._run(FSDPEngineWithValueHead, last_dim=1)
        torch.testing.assert_close(
            out["values"].values(), values[0, :, 0][output_args["prefix_packing_plan"].unpack_index]
        )

    def test_layer_guard(self):
        with pytest.raises(RuntimeError, match="1/2"):
            self._run(FSDPEngineWithLMHead, layers_seen=1)


class TestSupportCheck:
    @staticmethod
    def _check(attn="flash_attention_2", vision=False, cuda=True, top_left_mask=False, **kwargs):
        config = SimpleNamespace(_attn_implementation=attn)
        if vision:
            config.vision_config = {}
        kwargs = {"use_remove_padding": True, "ulysses_sp_size": 1, "pass_packed_cu_seqlens": False, **kwargs}
        with (
            patch("verl.workers.utils.prefix_packing.is_cuda_available", cuda),
            patch("verl.workers.utils.prefix_packing.flash_attn_supports_top_left_mask", return_value=top_left_mask),
        ):
            check_prefix_packing_supported(config, **kwargs)

    def test_supported(self):
        self._check()

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(attn="sdpa"), "attn_implementation"),
            (dict(ulysses_sp_size=2), "Ulysses"),
            (dict(vision=True), "vision-language"),
            (dict(pass_packed_cu_seqlens=True), "cu_seqlens"),
            (dict(use_remove_padding=False), "use_remove_padding=False"),
            (dict(cuda=False), "non-CUDA"),
            (dict(top_left_mask=True), "flash-attn < 2.1"),
        ],
    )
    def test_unsupported(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._check(**kwargs)
