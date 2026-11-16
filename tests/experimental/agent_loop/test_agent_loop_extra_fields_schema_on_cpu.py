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
)
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


def test_agent_loop_output_as_dict_promotes_rollout_topk_fields():
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[4, 9], [5, 8]],
            "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]],
        },
    )

    fields = output.as_dict()

    # Unpadded full-sequence layout: row P-1+r holds response token r's head, other rows the dummy head.
    assert fields["rollout_topk_ids"].shape == (5, 2) and fields["rollout_topk_ids"].dtype == torch.int32
    assert fields["rollout_topk_log_probs"].shape == (5, 2) and fields["rollout_topk_log_probs"].dtype == torch.float32
    assert fields["rollout_topk_ids"][2].tolist() == [4, 9] and fields["rollout_topk_ids"][3].tolist() == [5, 8]
    assert fields["rollout_topk_ids"][0].tolist() == [0, 1]
    assert fields["rollout_support_complete"].tolist() == [True, True, False, False, True]
    assert fields["extra_fields"] == {}
    assert output.extra_fields["response_topk_ids"] == [[4, 9], [5, 8]]


@pytest.mark.parametrize(
    "response_complete",
    [[True, False], np.array([True, False], dtype=np.bool_), torch.tensor([True, False])],
    ids=["bool-list", "numpy-bool", "torch-bool"],
)
def test_agent_loop_output_aligns_provider_support_evidence_with_prediction_rows(response_complete):
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[4, 9], [5, 8]],
            "response_topk_log_probs": [[0.0, float("-inf")], [-0.2, -1.5]],
            "response_support_complete": response_complete,
        },
    )
    fields = output.as_dict()
    assert fields["rollout_support_complete"].dtype == torch.bool
    assert fields["rollout_support_complete"].tolist() == [True, True, True, False, True]
    assert fields["extra_fields"] == {}


@pytest.mark.parametrize(
    "response_complete",
    [[1, 0], np.array([0.25, 0.0], dtype=np.float32), [float("nan"), 0.0], [True, 1], [True, 0.25]],
    ids=["integers", "floating-array", "nan", "mixed-bool-int", "mixed-bool-float"],
)
def test_agent_loop_output_rejects_nonboolean_provider_support_evidence(response_complete):
    output = _make_agent_loop_output()
    output.extra_fields.update(
        {
            "response_topk_ids": [[11, 13], [12, 14]],
            "response_topk_log_probs": [[0.0, float("-inf")], [-0.2, -1.5]],
            "response_support_complete": response_complete,
        }
    )

    with pytest.raises(ValueError, match="one bool per response token"):
        output.as_dict()


def test_agent_loop_output_rejects_incomplete_provider_evidence_records():
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": [[3, 9], [4, 8]],
            "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]],
            "response_support_complete": [True],
        },
    )
    with pytest.raises(ValueError, match="one bool per response token"):
        output.as_dict()


def test_agent_loop_output_as_dict_rejects_multi_turn_rollout_topk():
    output = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=[2, 3],
        response_mask=[1, 1],
        num_turns=4,
        metrics=AgentLoopMetrics(),
        extra_fields={"response_topk_ids": [[2, 9], [3, 8]], "response_topk_log_probs": [[-0.1, -2.0], [-0.2, -1.5]]},
    )

    with pytest.raises(ValueError, match="single-turn"):
        output.as_dict()


def test_agent_loop_output_as_dict_rejects_misaligned_rollout_topk():
    output = AgentLoopOutput(
        prompt_ids=[1, 2],
        response_ids=[3, 4],
        response_mask=[1, 1],
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "response_topk_ids": np.array([[3, 9], [4, 8], [5, 7]], dtype=np.int32),
            "response_topk_log_probs": np.full((3, 2), -0.5, dtype=np.float32),
        },
    )

    with pytest.raises(ValueError, match="cover 3 tokens"):
        output.as_dict()


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


def test_agent_loop_output_accepts_read_only_routed_experts_on_cpu():
    routed_experts = np.arange(8, dtype=np.int64).reshape(4, 2, 1)
    routed_experts.setflags(write=False)
    output = AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=[11, 12],
        response_mask=[1, 1],
        routed_experts=routed_experts,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="The given NumPy array is not writable.*", category=UserWarning)
        fields = output.as_dict()
    assert fields["routed_experts"].dtype == torch.int16
    assert fields["routed_experts"].shape == (4, 2, 1)
    torch.testing.assert_close(fields["routed_experts"], torch.tensor(routed_experts.copy(), dtype=torch.int16))
