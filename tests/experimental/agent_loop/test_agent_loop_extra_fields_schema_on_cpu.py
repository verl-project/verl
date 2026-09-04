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

from __future__ import annotations

import warnings
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import verl.experimental.agent_loop.agent_loop as agent_loop_module
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.rollout.replica import TokenOutput


class _FakeServerManager:
    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        del request_id, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs, kwargs
        # Return a short, deterministic "generation" for testing.
        return TokenOutput(token_ids=prompt_ids[-1:] + [11, 12, 13], log_probs=[0.0, 0.0, 0.0, 0.0])

    async def generate_for_partial(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[list[int], list[float], bool]:
        del request_id, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        # Return a short partial generation and "not cancelled".
        response_ids = prompt_ids[-1:] + [21, 22]
        response_logprobs = [0.0] * len(response_ids)
        return response_ids, response_logprobs, False


class _FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        # Minimal tokenization: return a small prompt.
        return [101, 102]

    def pad(
        self,
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        input_ids = encoded_inputs["input_ids"]
        if len(input_ids) > max_length:
            if self.padding_side == "left":
                input_ids = input_ids[-max_length:]
            else:
                input_ids = input_ids[:max_length]

        pad_len = max_length - len(input_ids)
        if self.padding_side == "left":
            padded_ids = [0] * pad_len + input_ids
            attention_mask = [0] * pad_len + [1] * len(input_ids)
        else:
            padded_ids = input_ids + [0] * pad_len
            attention_mask = [1] * len(input_ids) + [0] * pad_len

        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        del ids, skip_special_tokens
        return "<decoded>"


def _make_agent_loop_output() -> AgentLoopOutput:
    return AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
    )


def test_agent_loop_output_as_dict_handles_default_extra_fields():
    fields = _make_agent_loop_output().as_dict()

    assert fields["extra_fields"] == {}


def test_agent_loop_output_as_dict_preserves_loss_weight():
    output = _make_agent_loop_output()
    output.loss_weight = 0.25

    fields = output.as_dict()

    assert fields["loss_weight"] == pytest.approx(0.25)


def test_agent_loop_output_as_dict_preserves_mutated_default_extra_fields():
    output = _make_agent_loop_output()
    output.extra_fields["raw_prompt"] = [{"role": "user", "content": "hello"}]

    fields = output.as_dict()

    assert fields["extra_fields"] == {"raw_prompt": [{"role": "user", "content": "hello"}]}


def test_agent_loop_output_as_dict_promotes_teacher_fields_without_mutating_model():
    output = _make_agent_loop_output()
    output.extra_fields.update(
        {
            "raw_prompt": [{"role": "user", "content": "hello"}],
            "teacher_ids": [201, 202],
            "teacher_logprobs": [-0.1, -0.2],
        }
    )

    fields = output.as_dict()

    assert fields["teacher_ids"] == [201, 202]
    assert fields["teacher_logprobs"] == [-0.1, -0.2]
    assert fields["extra_fields"] == {"raw_prompt": [{"role": "user", "content": "hello"}]}
    assert output.extra_fields["teacher_ids"] == [201, 202]
    assert output.extra_fields["teacher_logprobs"] == [-0.1, -0.2]


@pytest.mark.asyncio
async def test_agent_loop_worker_passes_only_hf_model_type_through_hydra(monkeypatch):
    captured_kwargs: dict[str, Any] = {}
    expected_output = object()

    class _FakeAgentLoop:
        async def run(self, sampling_params: dict[str, Any], **kwargs):
            del sampling_params, kwargs
            return expected_output

    def fake_instantiate(*, config, **kwargs):
        del config
        captured_kwargs.update(kwargs)
        return _FakeAgentLoop()

    monkeypatch.setattr(agent_loop_module.hydra.utils, "instantiate", fake_instantiate)
    monkeypatch.setattr(agent_loop_module, "rollout_trace_attr", lambda **kwargs: nullcontext())
    monkeypatch.setitem(agent_loop_module._agent_loop_registry, "scalar_model_type_test", {"_target_": "unused"})

    worker = object.__new__(AgentLoopWorker)
    worker.config = OmegaConf.create({"data": {}})
    worker.llm_client = object()
    worker.tokenizer = _FakeTokenizer()
    worker.processor = None
    worker.hf_model_type = "qwen2_5_vl"
    worker.dataset_cls = RLHFDataset
    worker.tools = []

    async def passthrough_postprocess(output, validate, **kwargs):
        del validate, kwargs
        return output

    worker._agent_loop_postprocess = passthrough_postprocess
    result = await worker._run_agent_loop(
        {},
        {"step": 0, "sample_index": 0, "rollout_n": 0, "validate": False},
        agent_name="scalar_model_type_test",
    )

    assert result is expected_output
    assert captured_kwargs["hf_model_type"] == "qwen2_5_vl"
    assert isinstance(captured_kwargs["hf_model_type"], str)
    assert "hf_config" not in captured_kwargs


def _pad_1d(ids: list[int], *, length: int, pad_id: int = 0) -> list[int]:
    if len(ids) > length:
        return ids[:length]
    return ids + [pad_id] * (length - len(ids))


def _to_internal(
    *,
    output_prompt_ids: list[int],
    output_response_ids: list[int],
    output_response_mask: list[int],
    metrics: AgentLoopMetrics,
    extra_fields: dict[str, Any],
    num_turns: int,
    prompt_len: int,
    response_len: int,
) -> _InternalAgentLoopOutput:
    prompt_ids = _pad_1d(output_prompt_ids, length=prompt_len, pad_id=0)
    response_ids = _pad_1d(output_response_ids, length=response_len, pad_id=0)
    response_mask = _pad_1d(output_response_mask, length=response_len, pad_id=0)

    seq_len = prompt_len + response_len
    attention_mask = _pad_1d([1] * len(output_prompt_ids), length=prompt_len, pad_id=0) + _pad_1d(
        [1] * len(output_response_ids),
        length=response_len,
        pad_id=0,
    )
    input_ids = prompt_ids + response_ids
    position_ids = list(range(seq_len))

    def t(x: list[int]) -> torch.Tensor:
        return torch.tensor([x], dtype=torch.long)

    return _InternalAgentLoopOutput(
        prompt_ids=t(prompt_ids),
        response_ids=t(response_ids),
        response_mask=t(response_mask),
        attention_mask=t(attention_mask),
        input_ids=t(input_ids),
        position_ids=t(position_ids),
        response_logprobs=None,
        routed_experts=None,
        multi_modal_inputs=None,
        multi_modal_data=None,
        reward_score=None,
        num_turns=num_turns,
        metrics=metrics,
        extra_fields=extra_fields,
    )


@pytest.mark.asyncio
async def test_agent_loop_extra_fields_schema_stable_for_training_concat_on_cpu():
    # Minimal config surface used by the agent loops.
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 16, "response_length": 16, "multi_turn": {"tool_config_path": None}},
                # AgentLoopBase unconditionally builds the Continuous Token builder; without an
                # HF model type or multimodal processor it falls back to the default builder.
                "model": {"path": "dummy-model", "tokenizer_path": "dummy-model"},
            },
            "data": {
                "tool_config_path": None,
                "apply_chat_template_kwargs": {},
            },
        }
    )

    server_manager = _FakeServerManager()
    tokenizer = _FakeTokenizer()
    processor = None

    trainer_config = DictConfigWrap(config)
    data_config = DictConfigWrap(config.data)

    single_turn = SingleTurnAgentLoop(
        trainer_config=trainer_config,
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=processor,
        dataset_cls=RLHFDataset,
        data_config=data_config,
    )

    raw_prompt = [{"role": "user", "content": "hi"}]
    sampling_params: dict[str, Any] = {}

    out = await single_turn.run(sampling_params=sampling_params, raw_prompt=raw_prompt)

    # Agent loop outputs should always contain these fields with consistent types.
    assert out.extra_fields["turn_scores"] == []
    assert out.extra_fields["tool_rewards"] == []

    internal_a = _to_internal(
        output_prompt_ids=out.prompt_ids,
        output_response_ids=out.response_ids,
        output_response_mask=out.response_mask,
        metrics=out.metrics,
        extra_fields=out.extra_fields,
        num_turns=out.num_turns,
        prompt_len=len(out.prompt_ids),
        response_len=len(out.response_ids),
    )

    # Mimic two "worker chunks" and concatenate as in training.
    dummy_worker = type(
        "_DummyWorker",
        (),
        {"reward_loop_worker_handles": None, "distillation_enabled": False},
    )()
    merged = AgentLoopWorker._postprocess(
        dummy_worker,
        inputs=[internal_a],
        input_non_tensor_batch={
            "index": np.array([0], dtype=object),
            "agent_name": np.array(["single_turn_agent"], dtype=object),
        },
    )

    # Stable schema: present regardless of which loop produced a sample.
    stable_keys = (
        "turn_scores",
        "tool_rewards",
        "min_global_steps",
        "max_global_steps",
        "extras",
    )
    for key in stable_keys:
        assert key in merged.non_tensor_batch, f"missing key in merged batch: {key}"
        assert merged.non_tensor_batch[key].shape == (1,), (
            f"invalid shape for {key}: {merged.non_tensor_batch[key].shape}"
        )

    # And the list-typed fields are actually lists (not missing / scalar).
    assert merged.non_tensor_batch["turn_scores"][0] == []
    assert merged.non_tensor_batch["tool_rewards"][0] == []


@pytest.mark.asyncio
async def test_agent_loop_postprocess_accepts_read_only_routed_experts_on_cpu():
    class _DummyWorker:
        _compute_multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs
        _compute_position_ids = AgentLoopWorker._compute_position_ids
        _get_mm_processor_kwargs = AgentLoopWorker._get_mm_processor_kwargs
        _compute_score = AgentLoopWorker._compute_score
        _compute_teacher_logprobs = AgentLoopWorker._compute_teacher_logprobs
        _pad_token_ids = AgentLoopWorker._pad_token_ids
        distillation_enabled = False

        def __init__(self):
            self.tokenizer = _FakeTokenizer()
            self.rollout_config = OmegaConf.create({"prompt_length": 4, "response_length": 4})
            self.processor = None
            self.mm_processor_kwargs = {}
            self.reward_loop_worker_handles = None

    routed_experts = np.arange(8, dtype=np.int64).reshape(4, 2, 1)
    routed_experts.setflags(write=False)
    assert not routed_experts.flags.writeable

    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        routed_experts=routed_experts,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="The given NumPy array is not writable.*",
            category=UserWarning,
        )
        internal = await AgentLoopWorker._agent_loop_postprocess(
            _DummyWorker(),
            output,
            validate=False,
            raw_prompt=[{"role": "user", "content": "hi"}],
        )

    # Rollout is where routed_experts gets its int16 storage dtype, whatever dtype
    # the backend handed over.
    expected = torch.tensor(routed_experts.copy()).to(torch.int16).unsqueeze(0)
    assert internal.routed_experts is not None
    assert internal.routed_experts.dtype == torch.int16
    assert internal.routed_experts.shape == (1, 8, 2, 1)
    torch.testing.assert_close(internal.routed_experts[:, 2:6], expected)
    assert torch.count_nonzero(internal.routed_experts[:, :2]) == 0
    assert torch.count_nonzero(internal.routed_experts[:, 6:]) == 0


class _FakeTokenizerCustomPad:
    """A minimal tokenizer with a non-zero pad_token_id for testing."""

    pad_token_id = 42
    padding_side = "right"

    def pad(
        self,
        encoded_inputs: dict[str, list[int]],
        *,
        padding: str,
        max_length: int,
        return_tensors: str,
        return_attention_mask: bool,
    ) -> dict[str, torch.Tensor]:
        del padding, return_tensors
        input_ids = encoded_inputs["input_ids"]
        pad_len = max_length - len(input_ids)
        padded_ids = input_ids + [0] * pad_len
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        output = {"input_ids": torch.tensor([padded_ids], dtype=torch.long)}
        if return_attention_mask:
            output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
        return output


@pytest.mark.asyncio
async def test_agent_loop_pad_token_ids_empty_with_non_zero_pad_id():
    """Regression test: empty token list uses tokenizer's pad_token_id (not hardcoded 0)."""
    worker = type(
        "_DummyWorker",
        (),
        {
            "_pad_token_ids": AgentLoopWorker._pad_token_ids,
            "tokenizer": _FakeTokenizerCustomPad(),
            "processor": None,
            "mm_processor_kwargs": {},
        },
    )()

    result = worker._pad_token_ids(
        tokens=[],
        max_length=8,
        padding_side="right",
        return_attention_mask=True,
    )

    # input_ids should use the tokenizer's pad_token_id (42), not 0
    expected_input_ids = torch.full((1, 8), 42, dtype=torch.long)
    torch.testing.assert_close(result["input_ids"], expected_input_ids)

    # attention_mask should be all zeros
    expected_attention_mask = torch.zeros((1, 8), dtype=torch.long)
    torch.testing.assert_close(result["attention_mask"], expected_attention_mask)
